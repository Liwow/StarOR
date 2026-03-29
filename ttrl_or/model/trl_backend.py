from __future__ import annotations

import gc
import json
import inspect
import math
import os
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ttrl_or.config import DatasetConfig, GRPOConfig
from ttrl_or.model.backend import PolicyBackend
from ttrl_or.types import Generation, OptimizationTask, Stage, TrainingSample


@dataclass(slots=True)
class TRLPolicyBackend(PolicyBackend):
    """
    Training backend using Hugging Face + TRL GRPOTrainer.

    Key behavior:
    - Optional base-model reuse across task episodes.
    - Optional LoRA reset at episode start (without reloading base model).
    """

    model_name_or_path: str
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 1024
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    reuse_base_model_across_tasks: bool = True
    reset_lora_on_begin_episode: bool = True
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    _tokenizer: Any = field(init=False, default=None, repr=False)
    _model: Any = field(init=False, default=None, repr=False)
    _episode_key: str = field(init=False, default="", repr=False)
    _grpo_call_index: int = field(init=False, default=0, repr=False)
    _warned_vllm_unsupported: bool = field(init=False, default=False, repr=False)
    _force_disable_vllm_for_episode: bool = field(init=False, default=False, repr=False)
    _force_disable_vllm_for_run: bool = field(init=False, default=False, repr=False)
    _warned_vllm_disabled_for_run: bool = field(init=False, default=False, repr=False)

    def begin_episode(self, task: OptimizationTask) -> None:
        self._episode_key = task.task_id
        self._grpo_call_index = 0
        self._warned_vllm_unsupported = False
        self._force_disable_vllm_for_episode = False

        if self._model is None or self._tokenizer is None:
            self._load_fresh_episode_model()
        elif self.reset_lora_on_begin_episode:
            self._reset_lora_state()

        if self._model is not None:
            self._model.train()

    def end_episode(self) -> None:
        self._episode_key = ""
        self._grpo_call_index = 0
        self._warned_vllm_unsupported = False
        self._force_disable_vllm_for_episode = False

        if not self.reuse_base_model_across_tasks:
            self._unload_model()

    def generate(self, stage: Stage, prompt: str, n: int) -> list[Generation]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Episode not initialized. Call begin_episode() before generate().")

        self._model.eval()
        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = self._infer_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=n,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        seqs = output.sequences
        scores = output.scores

        generations: list[Generation] = []
        for idx in range(seqs.shape[0]):
            full_ids = seqs[idx]
            comp_ids = full_ids[prompt_len:]
            completion = self._tokenizer.decode(comp_ids, skip_special_tokens=True)
            prior = self._sequence_prior(idx, comp_ids, scores)
            generations.append(
                Generation(
                    text=completion,
                    prior=prior,
                    metadata={"backend": "trl", "stage": stage.value},
                )
            )
        return generations

    def grpo_rollout_group(
        self,
        stage: Stage,
        prompt: str,
        config: GRPOConfig,
        reward_callback,
    ) -> tuple[list[Generation], dict[str, Any]]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Episode not initialized. Call begin_episode() before grpo_rollout_group().")

        trl = _import_trl()
        datasets = _import_datasets()

        k = int(config.num_generations)
        if k < 2:
            return [], {
                "updated": False,
                "stage": stage.value,
                "backend": "trl",
                "reason": "grpo_requires_num_generations_ge_2",
                "requested_num_generations": k,
            }

        train_dataset = datasets.Dataset.from_list([{"prompt": prompt}])
        captured: list[Generation] = []

        def reward_func(prompts, completions, **kwargs):
            prompts_norm = [_normalize_text(p) for p in prompts]
            completions_norm = [_normalize_text(c) for c in completions]

            batch_score = getattr(reward_callback, "batch_score", None)
            rewards: list[float] = []

            if callable(batch_score) and prompts_norm:
                unique_prompts = {p for p in prompts_norm}
                if len(unique_prompts) == 1:
                    rewards = [float(r) for r in list(batch_score(prompts_norm[0], completions_norm))]
                else:
                    for p_text, c_text in zip(prompts_norm, completions_norm, strict=False):
                        one = list(batch_score(p_text, [c_text]))
                        rewards.append(float(one[0]) if one else 0.0)
            else:
                for ridx, (prompt_text, completion_text) in enumerate(zip(prompts_norm, completions_norm, strict=False)):
                    rewards.append(float(reward_callback(prompt_text, completion_text, ridx % max(1, k))))

            if len(rewards) != len(completions_norm):
                rewards = rewards[: len(completions_norm)] + [0.0] * max(0, len(completions_norm) - len(rewards))

            if len(captured) < k:
                prior = 1.0 / float(max(1, k))
                for ridx, c_text in enumerate(completions_norm[:k]):
                    if len(captured) >= k:
                        break
                    captured.append(
                        Generation(
                            text=c_text,
                            prior=prior,
                            metadata={
                                "stage": stage.value,
                                "rollout_index": ridx,
                                "reward_total": float(rewards[ridx]),
                            },
                        )
                    )

            return [float(r) for r in rewards]

        output_dir = self._stage_output_dir(stage)
        trl_args = self._build_trl_grpo_args(config, output_dir, num_generations=k)

        self._grpo_call_index += 1
        call_index = int(self._grpo_call_index)
        rank = _env_int("RANK", 0)
        world_size = max(1, _env_int("WORLD_SIZE", 1))
        prompt_tokens = int(self._tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1])
        mem_before = self._cuda_mem_snapshot()

        if rank == 0:
            print(
                f"[GRPO] start task={self._episode_key} call={call_index} "
                f"rank={rank}/{world_size} stage={stage.value} "
                f"num_generations={k} prompt_tokens={prompt_tokens} "
                f"use_vllm={bool(config.use_vllm)} vllm_mode={config.vllm_mode}"
            )

        trainer_kwargs = {
            "model": self._model,
            "args": trl_args,
            "reward_funcs": reward_func,
            "train_dataset": train_dataset,
        }
        trainer_sig = inspect.signature(trl.GRPOTrainer.__init__)
        if "processing_class" in trainer_sig.parameters:
            trainer_kwargs["processing_class"] = self._tokenizer
        elif "tokenizer" in trainer_sig.parameters:
            trainer_kwargs["tokenizer"] = self._tokenizer

        train_result, used_fallback = self._train_with_optional_vllm_fallback(
            trainer_kwargs=trainer_kwargs,
            config=config,
            stage=stage,
            output_dir=output_dir,
            num_generations=k,
            on_fallback_reset=captured.clear,
        )

        metrics = dict(getattr(train_result, "metrics", {}) or {})
        mem_after = self._cuda_mem_snapshot()
        if rank == 0:
            print(
                f"[GRPO] done task={self._episode_key} call={call_index} "
                f"rank={rank}/{world_size} stage={stage.value} "
                f"num_samples={len(captured)} train_loss={metrics.get('train_loss', 'n/a')} "
                f"train_runtime={metrics.get('train_runtime', 'n/a')} "
                f"cuda_alloc_mb={mem_before.get('allocated_mb', 'n/a')}->"
                f"{mem_after.get('allocated_mb', 'n/a')} "
                f"cuda_reserved_mb={mem_before.get('reserved_mb', 'n/a')}->"
                f"{mem_after.get('reserved_mb', 'n/a')}"
            )

        report: dict[str, Any] = {
            "updated": True,
            "stage": stage.value,
            "backend": "trl",
            "num_updates": 1,
            "num_groups": 1,
            "num_samples": len(captured),
            "num_generations": k,
            "generation_batch_size": int(trainer_kwargs["args"].generation_batch_size),
            "max_steps": int(getattr(trainer_kwargs["args"], "max_steps", 1)),
            "group_mode": "internal_rollout_strict",
            "grpo_call_index": call_index,
            "rank": rank,
            "world_size": world_size,
            "fallback_disable_vllm": used_fallback,
            "cuda_mem_before": mem_before,
            "cuda_mem_after": mem_after,
        }
        if "train_loss" in metrics:
            report["train_loss"] = float(metrics["train_loss"])
        if "train_runtime" in metrics:
            report["train_runtime"] = float(metrics["train_runtime"])
        if "train_steps_per_second" in metrics:
            report["train_steps_per_second"] = float(metrics["train_steps_per_second"])

        return captured, report

    def grpo_update(self, samples: list[TrainingSample], config: GRPOConfig, stage: Stage) -> dict[str, Any]:
        if not samples:
            return {"updated": False, "stage": stage.value, "num_samples": 0, "backend": "trl"}

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Episode not initialized. Call begin_episode() before grpo_update().")

        trl = _import_trl()
        datasets = _import_datasets()

        prompts = {_normalize_text(s.prompt) for s in samples}
        if len(prompts) != 1:
            return {
                "updated": False,
                "stage": stage.value,
                "num_samples": len(samples),
                "backend": "trl",
                "reason": "expect_single_prompt_for_manual_grpo_update",
                "num_prompts": len(prompts),
            }

        ordered = sorted(samples, key=lambda s: int((s.metadata or {}).get("rollout_index", 0)))
        rewards = [float(s.reward) for s in ordered]
        if len(rewards) < 2:
            return {
                "updated": False,
                "stage": stage.value,
                "num_samples": len(samples),
                "backend": "trl",
                "reason": "grpo_requires_at_least_2_rewards",
            }

        prompt = next(iter(prompts))
        train_dataset = datasets.Dataset.from_list([{"prompt": prompt}])

        cursor = 0

        def reward_func(prompts_batch, completions, **kwargs):
            nonlocal cursor
            out: list[float] = []
            for _ in zip(prompts_batch, completions, strict=False):
                idx = cursor % len(rewards)
                out.append(float(rewards[idx]))
                cursor += 1
            return out

        output_dir = self._stage_output_dir(stage)
        num_generations_used = int(len(rewards))
        trl_args = self._build_trl_grpo_args(config, output_dir, num_generations=num_generations_used)

        trainer_kwargs = {
            "model": self._model,
            "args": trl_args,
            "reward_funcs": reward_func,
            "train_dataset": train_dataset,
        }
        trainer_sig = inspect.signature(trl.GRPOTrainer.__init__)
        if "processing_class" in trainer_sig.parameters:
            trainer_kwargs["processing_class"] = self._tokenizer
        elif "tokenizer" in trainer_sig.parameters:
            trainer_kwargs["tokenizer"] = self._tokenizer

        train_result, used_fallback = self._train_with_optional_vllm_fallback(
            trainer_kwargs=trainer_kwargs,
            config=config,
            stage=stage,
            output_dir=output_dir,
            num_generations=num_generations_used,
            on_fallback_reset=None,
        )

        metrics = dict(getattr(train_result, "metrics", {}) or {})
        report: dict[str, Any] = {
            "updated": True,
            "stage": stage.value,
            "num_updates": 1,
            "num_samples": len(samples),
            "backend": "trl",
            "num_generations": num_generations_used,
            "fallback_disable_vllm": used_fallback,
            "group_mode": "manual_reward_binding",
        }
        if "train_loss" in metrics:
            report["train_loss"] = float(metrics["train_loss"])
        if "train_runtime" in metrics:
            report["train_runtime"] = float(metrics["train_runtime"])
        if "train_steps_per_second" in metrics:
            report["train_steps_per_second"] = float(metrics["train_steps_per_second"])
        return report

    def generate_mapping_from_description(
        self,
        description: str,
        dataset_config: DatasetConfig,
    ) -> dict[str, Any] | str | None:
        if self._model is None or self._tokenizer is None:
            return None

        prompt = _build_mapping_extraction_prompt(
            description=description,
            max_numeric_features=dataset_config.max_numeric_features,
            key_param_top_k=dataset_config.key_param_top_k,
        )

        self._model.eval()
        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = self._infer_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}

        temperature = max(0.0, float(dataset_config.mapping_llm_temperature))
        do_sample = temperature > 0.0

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(dataset_config.mapping_llm_max_new_tokens),
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = float(dataset_config.mapping_llm_top_p)
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output = self._model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        completion_ids = output[0][prompt_len:]
        completion = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
        return completion.strip()

    def generate_auxiliary_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        prefer_vllm: bool = False,
        vllm_mode: str = "",
    ) -> str | None:
        if bool(prefer_vllm):
            out = self._generate_auxiliary_vllm_server(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                vllm_mode=vllm_mode,
            )
            if out:
                return out

        if self._model is None or self._tokenizer is None:
            # Lazy-load base model once for pre-MCTS auxiliary planning.
            self._load_fresh_episode_model()
        if self._model is None or self._tokenizer is None:
            return None

        self._model.eval()
        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = self._infer_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}

        temp = max(0.0, float(temperature))
        do_sample = temp > 0.0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = float(top_p)
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output = self._model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        completion_ids = output[0][prompt_len:]
        completion = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
        return completion.strip()

    def _generate_auxiliary_vllm_server(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        vllm_mode: str,
    ) -> str | None:
        # Prefer external vLLM only for server mode.
        if str(vllm_mode or "").lower() != "server":
            return None
        base_url = (os.environ.get("VLLM_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")
        model_name = (os.environ.get("VLLM_MODEL_NAME") or self.model_name_or_path).strip()
        url = f"{base_url}/v1/completions"

        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": int(max_new_tokens),
            "temperature": float(max(0.0, temperature)),
            "top_p": float(top_p),
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # nosec B310
                body = resp.read().decode("utf-8", errors="ignore")
        except Exception:
            return None

        try:
            parsed = json.loads(body)
        except Exception:
            return None

        choices = parsed.get("choices") if isinstance(parsed, dict) else None
        if not isinstance(choices, list) or not choices:
            return None

        first = choices[0] if isinstance(choices[0], dict) else {}
        text = first.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        return None

    def generate_test_instances(self, task: OptimizationTask, k: int) -> list[dict[str, Any]]:
        from ttrl_or.reward.perturbation import generate_perturbed_instances_from_map

        tests = generate_perturbed_instances_from_map(task.instance, task.perturbation_map, k)
        if tests:
            return tests

        fallback: list[dict[str, Any]] = []
        for i in range(k):
            case: dict[str, Any] = {}
            scale = 1.0 + 0.1 * ((i % 3) - 1)
            for key, value in task.instance.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    case[key] = round(float(value) * scale, 6)
                else:
                    case[key] = value
            fallback.append(case)
        return fallback

    def _train_with_optional_vllm_fallback(
        self,
        trainer_kwargs: dict[str, Any],
        config: GRPOConfig,
        stage: Stage,
        output_dir: str,
        num_generations: int,
        on_fallback_reset,
    ) -> tuple[Any, bool]:
        trl = _import_trl()
        rank = _env_int("RANK", 0)

        trainer = None
        try:
            trainer = trl.GRPOTrainer(**trainer_kwargs)
            train_result = trainer.train()
            return train_result, False
        except Exception as exc:
            should_fallback = (
                bool(config.use_vllm)
                and bool(config.vllm_fallback_disable_on_error)
                and _looks_like_vllm_comm_error(exc)
            )
            if not should_fallback:
                raise

            if rank == 0:
                print(
                    "[GRPO][WARN] vLLM communication failed; fallback to use_vllm=False for this episode. "
                    f"stage={stage.value} reason={type(exc).__name__}: {exc}"
                )

            if callable(on_fallback_reset):
                on_fallback_reset()

            self._force_disable_vllm_for_episode = True
            self._force_disable_vllm_for_run = True
            self._warned_vllm_disabled_for_run = False

            trl_args = self._build_trl_grpo_args(
                config,
                output_dir,
                num_generations=num_generations,
                force_use_vllm=False,
            )
            fallback_kwargs = dict(trainer_kwargs)
            fallback_kwargs["args"] = trl_args

            fallback_trainer = None
            try:
                fallback_trainer = trl.GRPOTrainer(**fallback_kwargs)
                train_result = fallback_trainer.train()
                return train_result, True
            finally:
                self._cleanup_trainer_vllm(fallback_trainer)
        finally:
            self._cleanup_trainer_vllm(trainer)

    @staticmethod
    def _cleanup_trainer_vllm(trainer: Any) -> None:
        if trainer is None:
            return

        try:
            vgen = getattr(trainer, "vllm_generation", None)
            if vgen is not None:
                for fn_name in ("close_communicator", "close", "shutdown"):
                    fn = getattr(vgen, fn_name, None)
                    if callable(fn):
                        try:
                            fn()
                        except Exception:
                            pass

                client = getattr(vgen, "vllm_client", None)
                if client is not None:
                    for fn_name in ("close_communicator", "close", "shutdown"):
                        cfn = getattr(client, fn_name, None)
                        if callable(cfn):
                            try:
                                cfn()
                            except Exception:
                                pass
        except Exception:
            pass
        finally:
            try:
                if hasattr(trainer, "vllm_generation"):
                    trainer.vllm_generation = None
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _build_trl_grpo_args(
        self,
        config: GRPOConfig,
        output_dir: str,
        num_generations: int | None = None,
        force_use_vllm: bool | None = None,
    ):
        trl = _import_trl()

        used_num_generations = int(num_generations if num_generations is not None else config.num_generations)
        used_generation_batch_size = self._resolve_generation_batch_size(
            configured_generation_batch_size=int(config.generation_batch_size),
            per_device_train_batch_size=int(config.per_device_train_batch_size),
            num_generations=used_num_generations,
        )

        rank = _env_int("RANK", 0)
        world_size = _world_size()
        effective_use_vllm = bool(config.use_vllm) if force_use_vllm is None else bool(force_use_vllm)
        if (self._force_disable_vllm_for_episode or self._force_disable_vllm_for_run) and force_use_vllm is None:
            if self._force_disable_vllm_for_run and bool(config.use_vllm) and rank == 0 and not self._warned_vllm_disabled_for_run:
                print("[GRPO][WARN] vLLM has been disabled for the remaining run due to previous communicator/NCCL failure.")
                self._warned_vllm_disabled_for_run = True
            effective_use_vllm = False

        # Multi-process + colocate tends to start duplicate vLLM engines per rank.
        # Disable this combination to avoid doubled memory/process contention.
        if effective_use_vllm and world_size > 1 and str(config.vllm_mode).lower() == "colocate":
            if rank == 0:
                print(
                    "[GRPO][WARN] Disabling use_vllm for multi-process colocate mode "
                    f"(world_size={world_size}, vllm_mode={config.vllm_mode}). "
                    "Use server mode for external vLLM, or keep use_vllm=False for stable multi-GPU training."
                )
            effective_use_vllm = False

        effective_vllm_tp = _effective_vllm_tensor_parallel_size(
            requested_tp=int(config.vllm_tensor_parallel_size),
            use_vllm=effective_use_vllm,
            rank=rank,
            world_size=world_size,
        )

        kwargs = {
            "output_dir": output_dir,
            "learning_rate": config.learning_rate,
            "beta": config.kl_coef,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "num_generations": used_num_generations,
            "generation_batch_size": used_generation_batch_size,
            "max_prompt_length": config.max_prompt_length,
            "max_completion_length": config.max_completion_length,
            "num_train_epochs": float(config.train_epochs),
            "max_steps": -1,
            "epsilon": float(config.clip_epsilon),
            "use_vllm": effective_use_vllm,
            "vllm_mode": config.vllm_mode,
            "vllm_gpu_memory_utilization": config.vllm_gpu_memory_utilization,
            "vllm_tensor_parallel_size": effective_vllm_tp,
            "vllm_max_model_len": config.vllm_max_model_len,
            "report_to": [],
            "save_strategy": "no",
            "logging_steps": 1,
        }

        if config.clip_epsilon_high is not None:
            kwargs["epsilon_high"] = float(config.clip_epsilon_high)

        init_sig = inspect.signature(trl.GRPOConfig.__init__)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in init_sig.parameters}
        if bool(config.use_vllm):
            missing = [k for k in ("use_vllm", "vllm_mode") if k not in filtered_kwargs]
            if missing and not self._warned_vllm_unsupported:
                print(f"[GRPO][WARN] TRL GRPOConfig does not support {missing}; vLLM path may be inactive.")
                self._warned_vllm_unsupported = True
        return trl.GRPOConfig(**filtered_kwargs)

    @staticmethod
    def _resolve_generation_batch_size(
        configured_generation_batch_size: int,
        per_device_train_batch_size: int,
        num_generations: int,
    ) -> int:
        k = max(1, int(num_generations))

        if configured_generation_batch_size > 0:
            gen_bs = int(configured_generation_batch_size)
        else:
            gen_bs = max(k, int(per_device_train_batch_size))

        if gen_bs % k != 0:
            gen_bs = ((gen_bs + k - 1) // k) * k

        return max(k, gen_bs)

    def _load_fresh_episode_model(self) -> None:
        transformers = _import_transformers()
        peft = _import_peft()

        dtype = self._resolve_torch_dtype()
        rank = _env_int("RANK", 0)
        local_rank = _local_rank()
        world_size = _world_size()

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
        }
        if dtype != "auto":
            model_kwargs["dtype"] = dtype

        if torch.cuda.is_available():
            # In multi-process training, each rank must bind to one GPU only.
            if world_size > 1:
                local_rank = min(local_rank, max(0, torch.cuda.device_count() - 1))
                torch.cuda.set_device(local_rank)
                model_kwargs["device_map"] = {"": local_rank}
                if rank == 0:
                    print(
                        "[MODEL] multi-process load: binding each rank to one GPU "
                        f"(world_size={world_size})."
                    )
            else:
                model_kwargs["device_map"] = "auto"

        try:
            base_model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **model_kwargs)
        except TypeError as exc:
            # Backward compatibility for transformers versions that still expect torch_dtype.
            msg = str(exc)
            if "dtype" in model_kwargs and ("unexpected keyword" in msg or "got an unexpected" in msg):
                fallback_kwargs = dict(model_kwargs)
                fallback_kwargs.pop("dtype", None)
                if dtype != "auto":
                    fallback_kwargs["torch_dtype"] = dtype
                base_model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    **fallback_kwargs,
                )
            else:
                raise

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        lora_config = peft.LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=list(self.lora_target_modules),
            bias=self.lora_bias,
            task_type="CAUSAL_LM",
        )

        model = peft.get_peft_model(base_model, lora_config)
        model.train()

        self._model = model
        self._tokenizer = tokenizer

    def _reset_lora_state(self) -> None:
        if self._model is None:
            return

        reset_a = 0
        reset_b = 0

        for name, param in self._model.named_parameters():
            if not param.requires_grad:
                continue
            lname = name.lower()
            if "lora_a" in lname:
                with torch.no_grad():
                    if param.ndim >= 2:
                        torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                    else:
                        fan_in = max(1, int(param.numel()))
                        bound = 1.0 / math.sqrt(fan_in)
                        torch.nn.init.uniform_(param, -bound, bound)
                reset_a += 1
            elif "lora_b" in lname:
                with torch.no_grad():
                    torch.nn.init.zeros_(param)
                reset_b += 1

        if reset_a == 0 and reset_b == 0:
            print("[GRPO][WARN] reset_lora_on_begin_episode=True, but no LoRA trainable params were reset.")

    def _unload_model(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _cuda_mem_snapshot() -> dict[str, float]:
        if not torch.cuda.is_available():
            return {}

        try:
            device = torch.cuda.current_device()
            allocated = float(torch.cuda.memory_allocated(device) / (1024 ** 2))
            reserved = float(torch.cuda.memory_reserved(device) / (1024 ** 2))
            peak = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
            return {
                "device": float(device),
                "allocated_mb": round(allocated, 2),
                "reserved_mb": round(reserved, 2),
                "peak_allocated_mb": round(peak, 2),
            }
        except Exception:
            return {}

    def _resolve_torch_dtype(self):
        if self.torch_dtype == "auto":
            return "auto"
        if not hasattr(torch, self.torch_dtype):
            raise ValueError(f"Unknown torch dtype: {self.torch_dtype}")
        return getattr(torch, self.torch_dtype)

    def _infer_device(self) -> torch.device:
        if hasattr(self._model, "device"):
            return self._model.device
        return next(self._model.parameters()).device

    def _sequence_prior(self, seq_index: int, completion_ids: torch.Tensor, scores: list[torch.Tensor]) -> float:
        if not scores:
            return 1e-6

        log_probs: list[float] = []
        eos_id = self._tokenizer.eos_token_id
        pad_id = self._tokenizer.pad_token_id

        max_steps = min(len(scores), completion_ids.shape[0])
        for step in range(max_steps):
            token_id = int(completion_ids[step].item())
            if pad_id is not None and token_id == pad_id:
                break

            step_scores = scores[step][seq_index]
            step_log_probs = torch.log_softmax(step_scores, dim=-1)
            log_probs.append(float(step_log_probs[token_id].item()))

            if eos_id is not None and token_id == eos_id:
                break

        if not log_probs:
            return 1e-6

        avg_log_prob = sum(log_probs) / len(log_probs)
        return max(1e-6, float(math.exp(avg_log_prob)))

    def _stage_output_dir(self, stage: Stage) -> str:
        base = Path(tempfile.gettempdir()) / "ttrl_or_trl" / (self._episode_key or "unknown") / stage.value
        base.mkdir(parents=True, exist_ok=True)
        return str(base)


def _build_mapping_extraction_prompt(
    description: str,
    max_numeric_features: int,
    key_param_top_k: int,
) -> str:
    return f"""
You are an optimization-parameter extraction assistant.
Extract a numeric parameter mapping from the task description.

Return ONLY valid JSON (no markdown fences).
Required JSON shape:
{{
  "instance": {{"param_name": number, ...}},
  "key_param_keys": ["param_name_1", "param_name_2", ...]
}}

Rules:
- Keep at most {max_numeric_features} numeric keys in instance.
- key_param_keys should include at most {key_param_top_k} keys.
- key_param_keys must be keys from instance.
- Prefer numbers tied to objective/constraints (capacity, demand, budget, cost, profit, bounds).
- No explanation text.

Task description:
{description}
""".strip()


def _normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, dict):
                chunks.append(str(item.get("content", "")))
            else:
                chunks.append(str(item))
        return "\n".join(chunks).strip()

    return str(value).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default




def _world_size() -> int:
    return max(1, _env_int("WORLD_SIZE", 1))


def _local_rank() -> int:
    # torchrun sets LOCAL_RANK; fallback to RANK for safety.
    return _env_int("LOCAL_RANK", _env_int("RANK", 0))

def _effective_vllm_tensor_parallel_size(
    requested_tp: int,
    use_vllm: bool,
    rank: int,
    world_size: int,
) -> int:
    if not use_vllm:
        return max(1, int(requested_tp))

    tp = max(1, int(requested_tp))
    ws = max(1, int(world_size))

    if ws % tp == 0:
        return tp

    # Choose the largest valid divisor <= requested tp; fallback to 1.
    valid = [d for d in range(1, ws + 1) if ws % d == 0 and d <= tp]
    fallback = max(valid) if valid else 1

    if rank == 0:
        print(
            "[GRPO][WARN] Invalid vLLM tensor_parallel_size for current WORLD_SIZE; "
            f"requested_tp={tp}, world_size={ws}, using_tp={fallback}. "
            "If you need tp>1, launch with torchrun so WORLD_SIZE matches."
        )

    return fallback
def _import_transformers():
    try:
        import transformers  # type: ignore

        return transformers
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "TRL backend requires a working `transformers` install. "
            "Please install or repair it with: pip install transformers"
        ) from exc


def _import_trl():
    try:
        import trl  # type: ignore

        return trl
    except ImportError as exc:
        raise RuntimeError("TRL backend requires `trl`. Install with: pip install trl datasets peft") from exc


def _import_peft():
    try:
        import peft  # type: ignore

        return peft
    except ImportError as exc:
        raise RuntimeError("TRL backend requires `peft`. Install with: pip install peft") from exc


def _import_datasets():
    try:
        import datasets  # type: ignore

        return datasets
    except ImportError as exc:
        raise RuntimeError("TRL backend requires `datasets`. Install with: pip install datasets") from exc


def _looks_like_vllm_comm_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    keys = (
        "nccl error",
        "pyncclcommunicator",
        "init_communicator",
        "socketpollconnect",
        "remote process exited",
        "vllm_client",
        "tensor parallel size",
        "must divide world size",
        "collective_rpc",
        "weight update group already initialized",
        "close_communicator first",
    )
    return any(k in text for k in keys)




