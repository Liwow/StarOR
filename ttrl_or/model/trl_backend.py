from __future__ import annotations
import contextlib
import gc
import json
import inspect
import math
import os
import tempfile
import time
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
    _warned_lora_persist_across_tasks: bool = field(init=False, default=False, repr=False)
    _warned_vllm_len_clamp: bool = field(init=False, default=False, repr=False)
    _warned_strict_group_override: bool = field(init=False, default=False, repr=False)
    _active_trainer_for_aux: Any = field(init=False, default=None, repr=False)
    def begin_episode(self, task: OptimizationTask) -> None:
        self._episode_key = task.task_id
        self._grpo_call_index = 0
        self._warned_vllm_unsupported = False
        self._force_disable_vllm_for_episode = False
        self._force_disable_vllm_for_run = False
        self._warned_vllm_disabled_for_run = False
        self._warned_lora_persist_across_tasks = False
        self._warned_vllm_len_clamp = False
        self._warned_strict_group_override = False
        trainer = self._active_trainer_for_aux
        self._active_trainer_for_aux = None
        if trainer is not None:
            self._cleanup_trainer_vllm(trainer)
        if self._model is None or self._tokenizer is None:
            self._load_fresh_episode_model()
        elif self.reset_lora_on_begin_episode:
            self._reset_lora_state()
        if self._model is not None:
            self._assert_single_lora_adapter()
            self._model.train()

    def end_episode(self) -> None:
        rank = _env_int("RANK", 0)
        self._episode_key = ""
        self._grpo_call_index = 0
        self._warned_vllm_unsupported = False
        self._force_disable_vllm_for_episode = False
        self._force_disable_vllm_for_run = False
        self._warned_vllm_disabled_for_run = False
        trainer = self._active_trainer_for_aux
        self._active_trainer_for_aux = None
        if trainer is not None:
            self._cleanup_trainer_vllm(trainer)
        if not self.reuse_base_model_across_tasks:
            self._unload_model()
            return
        if self._model is not None:
            if self.reset_lora_on_begin_episode:
                # Explicitly drop sample-specific LoRA state right after sample ends.
                self._reset_lora_state()
                if rank == 0:
                    print("[LoRA] end_episode reset complete (sample-specific LoRA dropped).")
            elif rank == 0 and not self._warned_lora_persist_across_tasks:
                print("[LoRA][WARN] reset_lora_on_begin_episode=False, LoRA state may persist across samples.")
                self._warned_lora_persist_across_tasks = True

    def _is_message_list(self, prompt: Any) -> bool:
        return isinstance(prompt, list) and all(isinstance(item, dict) and "role" in item for item in prompt)
    def _messages_to_text(self, messages: list[dict[str, Any]]) -> str:
        chunks: list[str] = []
        for message in messages:
            role = str(message.get("role", "user") or "user").strip().upper()
            content = str(message.get("content", "") or "").strip()
            if not content:
                continue
            chunks.append(f"[{role}]\n{content}")
        return "\n\n".join(chunks).strip()
    def _prompt_to_model_text(self, prompt: Any, *, add_generation_prompt: bool) -> str:
        if self._is_message_list(prompt):
            messages = list(prompt)
            chat_template = getattr(self._tokenizer, "apply_chat_template", None) if self._tokenizer is not None else None
            if callable(chat_template):
                try:
                    rendered = chat_template(messages, tokenize=False, add_generation_prompt=bool(add_generation_prompt))
                    return str(rendered or "").strip()
                except Exception:
                    pass
            return self._messages_to_text(messages)
        return _normalize_text(prompt)
    def generate(self, stage: Stage, prompt: Any, n: int) -> list[Generation]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Episode not initialized. Call begin_episode() before generate().")
        self._model.eval()
        prompt_text = self._prompt_to_model_text(prompt, add_generation_prompt=True)
        inputs = self._tokenizer(prompt_text, return_tensors="pt")
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
        prompt: Any,
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
        prompt_text = self._prompt_to_model_text(prompt, add_generation_prompt=True)
        train_dataset = datasets.Dataset.from_list([{"prompt": prompt_text}])
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
        trl_args = self._build_trl_grpo_args(config, output_dir, num_generations=k, strict_single_group=True)
        self._grpo_call_index += 1
        call_index = int(self._grpo_call_index)
        rank = _env_int("RANK", 0)
        world_size = max(1, _env_int("WORLD_SIZE", 1))
        prompt_tokens = int(self._tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1])
        mem_before = self._cuda_mem_snapshot()
        lora_adapter_count = self._lora_adapter_count()
        self._assert_single_lora_adapter()
        if rank == 0:
            print(
                f"[GRPO] start task={self._episode_key} call={call_index} "
                f"rank={rank}/{world_size} stage={stage.value} "
                f"num_generations={k} prompt_tokens={prompt_tokens} "
                f"use_vllm={bool(config.use_vllm)} vllm_mode={config.vllm_mode} "
                f"lora_adapters={lora_adapter_count}"
            )
        trainer_kwargs = self._build_grpo_trainer_kwargs(
            trl=trl,
            trl_args=trl_args,
            reward_func=reward_func,
            train_dataset=train_dataset,
        )
        train_result, used_fallback = self._train_with_optional_vllm_fallback(
            trainer_kwargs=trainer_kwargs,
            config=config,
            stage=stage,
            output_dir=output_dir,
            num_generations=k,
            on_fallback_reset=captured.clear,
            strict_single_group=True,
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
        schedule_info = self._estimate_training_schedule(
            num_samples=int(len(train_dataset)),
            per_device_train_batch_size=int(getattr(trl_args, "per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(getattr(trl_args, "gradient_accumulation_steps", 1)),
            num_train_epochs=float(getattr(trl_args, "num_train_epochs", 1.0)),
            num_generations=int(k),
            generation_batch_size=int(getattr(trl_args, "generation_batch_size", k)),
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
            "lora_adapter_count": int(lora_adapter_count),
            "cuda_mem_before": mem_before,
            "cuda_mem_after": mem_after,
            "strict_single_group": True,
            "training_schedule": schedule_info,
        }
        if "train_loss" in metrics:
            report["train_loss"] = float(metrics["train_loss"])
        if "train_runtime" in metrics:
            report["train_runtime"] = float(metrics["train_runtime"])
        if "train_steps_per_second" in metrics:
            report["train_steps_per_second"] = float(metrics["train_steps_per_second"])
        return captured, report
    def score_action_priors(self, stage: Stage, prompt: Any, candidates: list[str]) -> list[float]:
        del stage, prompt
        n = len(candidates)
        if n <= 0:
            return []
        return [1.0 / float(n)] * n

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
        prompt_text = self._prompt_to_model_text(prompt, add_generation_prompt=True)
        train_dataset = datasets.Dataset.from_list([{"prompt": prompt_text}])
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
        trl_args = self._build_trl_grpo_args(config, output_dir, num_generations=num_generations_used, strict_single_group=True)
        trainer_kwargs = self._build_grpo_trainer_kwargs(
            trl=trl,
            trl_args=trl_args,
            reward_func=reward_func,
            train_dataset=train_dataset,
        )
        train_result, used_fallback = self._train_with_optional_vllm_fallback(
            trainer_kwargs=trainer_kwargs,
            config=config,
            stage=stage,
            output_dir=output_dir,
            num_generations=num_generations_used,
            on_fallback_reset=None,
            strict_single_group=True,
        )
        metrics = dict(getattr(train_result, "metrics", {}) or {})
        schedule_info = self._estimate_training_schedule(
            num_samples=int(len(train_dataset)),
            per_device_train_batch_size=int(getattr(trl_args, "per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(getattr(trl_args, "gradient_accumulation_steps", 1)),
            num_train_epochs=float(getattr(trl_args, "num_train_epochs", 1.0)),
            num_generations=int(num_generations_used),
            generation_batch_size=int(getattr(trl_args, "generation_batch_size", num_generations_used)),
        )
        report: dict[str, Any] = {
            "updated": True,
            "stage": stage.value,
            "num_updates": 1,
            "num_samples": len(samples),
            "backend": "trl",
            "num_generations": num_generations_used,
            "fallback_disable_vllm": used_fallback,
            "group_mode": "manual_reward_binding",
            "strict_single_group": True,
            "training_schedule": schedule_info,
        }
        if "train_loss" in metrics:
            report["train_loss"] = float(metrics["train_loss"])
        if "train_runtime" in metrics:
            report["train_runtime"] = float(metrics["train_runtime"])
        if "train_steps_per_second" in metrics:
            report["train_steps_per_second"] = float(metrics["train_steps_per_second"])
        return report
    def _build_grpo_trainer_kwargs(
        self,
        *,
        trl,
        trl_args,
        reward_func,
        train_dataset,
    ) -> dict[str, Any]:
        trainer_kwargs: dict[str, Any] = {
            "model": self._model,
            "args": trl_args,
            "reward_funcs": [reward_func],
            "train_dataset": train_dataset,
        }
        trainer_sig = inspect.signature(trl.GRPOTrainer.__init__)
        if "processing_class" in trainer_sig.parameters:
            trainer_kwargs["processing_class"] = self._tokenizer
        elif "tokenizer" in trainer_sig.parameters:
            trainer_kwargs["tokenizer"] = self._tokenizer
        return trainer_kwargs
    @staticmethod
    def _estimate_training_schedule(
        *,
        num_samples: int,
        per_device_train_batch_size: int,
        gradient_accumulation_steps: int,
        num_train_epochs: float,
        num_generations: int,
        generation_batch_size: int,
    ) -> dict[str, Any]:
        samples = max(1, int(num_samples))
        bs = max(1, int(per_device_train_batch_size))
        accum = max(1, int(gradient_accumulation_steps))
        epochs = max(0.0, float(num_train_epochs))
        k = max(1, int(num_generations))
        gen_bs = max(k, int(generation_batch_size))
        dataloader_steps_per_epoch = max(1, math.ceil(samples / bs))
        optimizer_steps_per_epoch = max(1, math.ceil(dataloader_steps_per_epoch / accum))
        total_optimizer_steps_est = max(1, math.ceil(optimizer_steps_per_epoch * max(epochs, 1e-9)))
        prompts_per_step = max(1, gen_bs // k)
        return {
            "num_samples": samples,
            "per_device_train_batch_size": bs,
            "gradient_accumulation_steps": accum,
            "num_train_epochs": epochs,
            "num_generations": k,
            "generation_batch_size": gen_bs,
            "prompts_per_step": int(prompts_per_step),
            "rollouts_per_step": int(gen_bs),
            "dataloader_steps_per_epoch": int(dataloader_steps_per_epoch),
            "optimizer_steps_per_epoch_est": int(optimizer_steps_per_epoch),
            "optimizer_steps_total_est": int(total_optimizer_steps_est),
        }
    
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
        prompt: Any,
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        prefer_vllm: bool = False,
        vllm_mode: str = "",
    ) -> str | None:
        outputs = self.generate_auxiliary_texts(
            [prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prefer_vllm=prefer_vllm,
            vllm_mode=vllm_mode,
        )
        return outputs[0] if outputs else None
    def generate_auxiliary_texts(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        prefer_vllm: bool = False,
        vllm_mode: str = "",
    ) -> list[str | None]:
        if not prompts:
            return []
        if bool(prefer_vllm):
            mode = str(vllm_mode or "").lower()
            if mode == "colocate":
                out = self._generate_auxiliary_vllm_colocate_batch(
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                out = self._generate_auxiliary_vllm_server_batch(
                    prompts=prompts,
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
            return [None for _ in prompts]
        self._model.eval()
        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True)
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
        prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
        texts: list[str | None] = []
        for idx, prompt_len in enumerate(prompt_lens):
            completion_ids = output[idx][int(prompt_len):]
            completion = self._tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            texts.append(completion or None)
        return texts
    def _generate_auxiliary_vllm_colocate_batch(
        self,
        *,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[str | None]:
        if not prompts:
            return []
        trainer = self._active_trainer_for_aux
        if trainer is None:
            return []
        tokenizer = self._ensure_tokenizer_for_auxiliary_vllm()
        if tokenizer is None:
            return []
        vgen = getattr(trainer, "vllm_generation", None)
        client = getattr(vgen, "vllm_client", None) if vgen is not None else None
        response = None
        try:
            if client is not None and hasattr(client, "generate"):
                response = client.generate(
                    prompts=list(prompts),
                    n=1,
                    temperature=float(max(0.0, temperature)),
                    top_p=float(top_p),
                    max_tokens=int(max_new_tokens),
                    logprobs=0,
                )
            elif vgen is not None and hasattr(vgen, "generate"):
                response = vgen.generate(
                    prompts=list(prompts),
                    n=1,
                    temperature=float(max(0.0, temperature)),
                    top_p=float(top_p),
                    max_tokens=int(max_new_tokens),
                    logprobs=0,
                )
        except Exception:
            return []
        completion_ids = response.get("completion_ids") if isinstance(response, dict) else None
        if not isinstance(completion_ids, list) or not completion_ids:
            return []
        texts: list[str | None] = []
        for item in completion_ids[: len(prompts)]:
            if not isinstance(item, list) or not item:
                texts.append(None)
                continue
            try:
                text = tokenizer.decode(item, skip_special_tokens=True)
            except Exception:
                texts.append(None)
                continue
            texts.append(str(text or "").strip() or None)
        while len(texts) < len(prompts):
            texts.append(None)
        return texts
    def _generate_auxiliary_vllm_server(
        self,
        *,
        prompt: Any,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        vllm_mode: str,
    ) -> str | None:
        outputs = self._generate_auxiliary_vllm_server_batch(
            prompts=[prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            vllm_mode=vllm_mode,
        )
        return outputs[0] if outputs else None
    def _generate_auxiliary_vllm_server_batch(
        self,
        *,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        vllm_mode: str,
    ) -> list[str | None]:
        # Prefer external TRL vLLM server only for server mode.
        if str(vllm_mode or "").lower() != "server":
            return []
        if not prompts:
            return []
        try:
            trl = _import_trl()
        except Exception:
            return []
        base_url = (os.environ.get("VLLM_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")
        client_cls = _resolve_trl_vllm_client(trl)
        if client_cls is None:
            return []
        tokenizer = self._ensure_tokenizer_for_auxiliary_vllm()
        if tokenizer is None:
            return []
        try:
            client = client_cls(base_url=base_url)
        except Exception:
            return []
        try:
            response = client.generate(
                prompts=list(prompts),
                n=1,
                temperature=float(max(0.0, temperature)),
                top_p=float(top_p),
                max_tokens=int(max_new_tokens),
                logprobs=0,
            )
        except Exception:
            return []
        completion_ids = response.get("completion_ids") if isinstance(response, dict) else None
        if not isinstance(completion_ids, list) or not completion_ids:
            return []
        texts: list[str | None] = []
        for item in completion_ids[: len(prompts)]:
            if not isinstance(item, list) or not item:
                texts.append(None)
                continue
            try:
                text = tokenizer.decode(item, skip_special_tokens=True)
            except Exception:
                texts.append(None)
                continue
            texts.append(str(text or "").strip() or None)
        while len(texts) < len(prompts):
            texts.append(None)
        return texts
    def _ensure_tokenizer_for_auxiliary_vllm(self):
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            transformers = _import_transformers()
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.trust_remote_code,
            )
        except Exception:
            return None
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self._tokenizer = tokenizer
        return tokenizer
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
        strict_single_group: bool = False,
    ) -> tuple[Any, bool]:
        trl = _import_trl()
        rank = _env_int("RANK", 0)
        trainer = None
        trl_args = trainer_kwargs["args"]
        try:
            if bool(getattr(trl_args, "use_vllm", False)):
                self._vllm_server_maintenance(config=config, before_train=True, after_train=False)
            if rank == 0:
                print("[GRPO] creating fresh trainer for this iteration.")
            trainer = trl.GRPOTrainer(**trainer_kwargs)
            self._active_trainer_for_aux = trainer
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
                    f"stage={stage.value} reason={type(exc).__name__}: {exc}. "
                    "This fallback will use local HF generation on the trainer GPU and may be much slower / use more memory."
                )
            if callable(on_fallback_reset):
                on_fallback_reset()
            self._force_disable_vllm_for_episode = True
            self._force_disable_vllm_for_run = True
            self._warned_vllm_disabled_for_run = False
            self._active_trainer_for_aux = None
            if trainer is not None:
                self._cleanup_trainer_vllm(trainer)
                trainer = None
            fallback_args = self._build_trl_grpo_args(
                config,
                output_dir,
                num_generations=num_generations,
                force_use_vllm=False,
                strict_single_group=strict_single_group,
            )
            fallback_kwargs = dict(trainer_kwargs)
            fallback_kwargs["args"] = fallback_args
            if rank == 0:
                print("[GRPO] creating fresh fallback trainer with use_vllm=False.")
            trainer = trl.GRPOTrainer(**fallback_kwargs)
            self._active_trainer_for_aux = trainer
            train_result = trainer.train()
            return train_result, True
        finally:
            self._active_trainer_for_aux = None
            if trainer is not None:
                self._cleanup_trainer_vllm(trainer)    @staticmethod
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
    @staticmethod
    def _vllm_server_post(endpoint: str, payload: dict[str, Any] | None = None, timeout_sec: float = 8.0) -> bool:
        base_url = (os.environ.get("VLLM_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")
        url = f"{base_url}{endpoint}"
        data = json.dumps(payload or {}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=float(timeout_sec)) as resp:  # nosec B310
                return 200 <= int(getattr(resp, "status", 200)) < 300
        except Exception:
            return False
    def _vllm_server_maintenance(
        self,
        config: GRPOConfig,
        *,
        before_train: bool,
        after_train: bool,
    ) -> None:
        del after_train
        if not bool(config.use_vllm):
            return
        if str(config.vllm_mode or "").lower() != "server":
            return
        rank = _env_int("RANK", 0)
        # Keep only the minimal server-side cleanup that is required for repeated
        # TRL GRPOTrainer construction in server mode. Prefix-cache reset stays disabled
        # to avoid the extra generation latency we observed.
        if before_train and bool(config.vllm_close_communicator_after_update):
            ok = False
            for attempt in range(3):
                ok = self._vllm_server_post("/close_communicator/")
                if ok:
                    # Give the server a brief moment to tear down the previous NCCL communicator
                    # before TRL constructs a fresh VLLMGeneration communicator.
                    time.sleep(0.5)
                    break
                time.sleep(0.5)
            if rank == 0:
                if ok:
                    print("[vLLM] close_communicator before trainer init.")
                else:
                    print("[vLLM][WARN] close_communicator before trainer init did not return success; continuing anyway.")
        return

    def _build_trl_grpo_args(
        self,
        config: GRPOConfig,
        output_dir: str,
        num_generations: int | None = None,
        force_use_vllm: bool | None = None,
        strict_single_group: bool = False,
    ):
        trl = _import_trl()
        used_num_generations = int(num_generations if num_generations is not None else config.num_generations)
        used_per_device_train_batch_size = int(config.per_device_train_batch_size)
        if strict_single_group:
            # Keep one selected node => one prompt group => one GRPO group semantics.
            # This avoids hidden prompt replication inside TRL when generation_batch_size > num_generations.
            used_per_device_train_batch_size = 1
            used_generation_batch_size = max(1, used_num_generations)
            if not self._warned_strict_group_override and (
                int(config.per_device_train_batch_size) != used_per_device_train_batch_size
                or int(config.generation_batch_size) > 0
            ):
                print(
                    "[GRPO][INFO] strict_single_group=True: forcing per_device_train_batch_size=1 "
                    f"and generation_batch_size=num_generations={used_generation_batch_size}."
                )
                self._warned_strict_group_override = True
        else:
            used_generation_batch_size = self._resolve_generation_batch_size(
                configured_generation_batch_size=int(config.generation_batch_size),
                per_device_train_batch_size=used_per_device_train_batch_size,
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
        # Official TRL colocate mode supports distributed launch where each rank
        # both trains locally and participates in the colocated vLLM engine.
        # Keep use_vllm enabled here and let TRL/vLLM manage the colocated workers.
        effective_vllm_tp = _effective_vllm_tensor_parallel_size(
            requested_tp=int(config.vllm_tensor_parallel_size),
            use_vllm=effective_use_vllm,
            rank=rank,
            world_size=world_size,
        )
        max_prompt_len = int(config.max_prompt_length)
        max_completion_len = int(config.max_completion_length)
        effective_vllm_model_max_len = max(256, int(config.vllm_max_model_len))
        configured_batched_tokens = int(config.vllm_max_num_batched_tokens)
        effective_vllm_max_num_batched_tokens = configured_batched_tokens
        if effective_use_vllm:
            # Keep colocate/server defaults safe: if user does not set this explicitly,
            # align it to max_model_len so vLLM does not reject long sequences.
            if effective_vllm_max_num_batched_tokens <= 0:
                effective_vllm_max_num_batched_tokens = effective_vllm_model_max_len
            # If user explicitly set a smaller batched-token cap, shrink model len to match.
            # This avoids: max_num_batched_tokens < max_model_len.
            if effective_vllm_max_num_batched_tokens < effective_vllm_model_max_len:
                effective_vllm_model_max_len = effective_vllm_max_num_batched_tokens
            if max_completion_len >= effective_vllm_model_max_len:
                # Keep at least 128 prompt tokens budget.
                max_completion_len = max(128, effective_vllm_model_max_len - 128)
            if max_prompt_len + max_completion_len > effective_vllm_model_max_len:
                max_prompt_len = max(128, effective_vllm_model_max_len - max_completion_len)
            if rank == 0 and not self._warned_vllm_len_clamp:
                orig_prompt = int(config.max_prompt_length)
                orig_comp = int(config.max_completion_length)
                orig_model_max = int(config.vllm_max_model_len)
                if (
                    orig_prompt != max_prompt_len
                    or orig_comp != max_completion_len
                    or orig_model_max != effective_vllm_model_max_len
                    or configured_batched_tokens <= 0
                ):
                    print(
                        "[GRPO][WARN] Adjusted vLLM length settings: "
                        f"prompt {orig_prompt}->{max_prompt_len}, completion {orig_comp}->{max_completion_len}, "
                        f"vllm_max_model_len {orig_model_max}->{effective_vllm_model_max_len}, "
                        f"vllm_max_num_batched_tokens={effective_vllm_max_num_batched_tokens}."
                    )
                    self._warned_vllm_len_clamp = True
        kwargs = {
            "output_dir": output_dir,
            "learning_rate": config.learning_rate,
            "beta": config.kl_coef,
            "per_device_train_batch_size": used_per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "num_generations": used_num_generations,
            "generation_batch_size": used_generation_batch_size,
            "max_prompt_length": max_prompt_len,
            "max_completion_length": max_completion_len,
            "num_train_epochs": float(config.train_epochs),
            "max_steps": 1,
            "epsilon": float(config.clip_epsilon),
            "use_vllm": effective_use_vllm,
            "vllm_mode": config.vllm_mode,
            "vllm_enable_sleep_mode": bool(config.vllm_enable_sleep_mode),
            "vllm_gpu_memory_utilization": config.vllm_gpu_memory_utilization,
            "vllm_tensor_parallel_size": effective_vllm_tp,
            "vllm_max_model_len": effective_vllm_model_max_len,
            "report_to": [],
            "save_strategy": "no",
            "logging_steps": 1,
        }
        init_sig = inspect.signature(trl.GRPOConfig.__init__)
        if config.clip_epsilon_high is not None:
            kwargs["epsilon_high"] = float(config.clip_epsilon_high)
        if bool(getattr(config, "sync_ref_model", False)):
            if "sync_ref_model" in init_sig.parameters:
                kwargs["sync_ref_model"] = True
            sync_steps = int(getattr(config, "ref_model_sync_steps", 0) or 0)
            if sync_steps > 0:
                for key in ("ref_model_sync_steps",):
                    if key in init_sig.parameters:
                        kwargs[key] = sync_steps
                        break
            mixup_alpha = float(getattr(config, "ref_model_mixup_alpha", 0.0) or 0.0)
            if mixup_alpha > 0.0:
                for key in ("ref_model_mixup_alpha",):
                    if key in init_sig.parameters:
                        kwargs[key] = mixup_alpha
                        break
        if effective_use_vllm and "vllm_max_num_batched_tokens" in init_sig.parameters:
            kwargs["vllm_max_num_batched_tokens"] = int(effective_vllm_max_num_batched_tokens)
        elif effective_use_vllm and "max_num_batched_tokens" in init_sig.parameters:
            kwargs["max_num_batched_tokens"] = int(effective_vllm_max_num_batched_tokens)
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
            # Always bind the training process to exactly one visible GPU.
            # This prevents single-process runs from spilling onto the vLLM server GPU
            # when multiple devices are visible in the environment.
            local_rank = min(local_rank, max(0, torch.cuda.device_count() - 1))
            torch.cuda.set_device(local_rank)
            model_kwargs["device_map"] = {"": local_rank}
            if rank == 0:
                mode = "multi-process" if world_size > 1 else "single-process"
                print(
                    "[MODEL] forcing single-device load for trainer "
                    f"(mode={mode}, world_size={world_size}, local_rank={local_rank}, "
                    f"visible_cuda_devices={torch.cuda.device_count()})."
                )
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
        tokenizer.padding_side = "left"
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
    def _lora_adapter_count(self) -> int:
        if self._model is None:
            return 0
        cfg = getattr(self._model, "peft_config", None)
        if cfg is None:
            return 0
        if isinstance(cfg, dict):
            return len(cfg)
        return 1
    def _assert_single_lora_adapter(self) -> None:
        count = self._lora_adapter_count()
        if count > 1:
            raise RuntimeError(
                f"Detected {count} LoRA adapters on model; expected exactly 1 to avoid adapter stacking."
            )
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
    @staticmethod
    def _stage_tags(stage: Stage) -> list[str]:
        if stage == Stage.SCHEMA:
            return ["Type", "Sets"]
        if stage == Stage.TYPE_HINT:
            return ["Type"]
        if stage == Stage.SETS:
            return ["Sets"]
        if stage == Stage.SET_PARAM_VAR:
            return ["Parameters", "Variables"]
        if stage == Stage.PARAMETERS:
            return ["Parameters"]
        if stage == Stage.VARIABLES:
            return ["Variables"]
        if stage == Stage.OBJ_CONS:
            return ["Objective", "Constraints"]
        if stage == Stage.OBJECTIVE:
            return ["Objective"]
        if stage == Stage.CONSTRAINTS:
            return ["Constraints"]
        return ["python"]

    def _canonical_action_block(self, stage: Stage, candidate: str) -> str:
        content = str(candidate or "").strip()
        if not content:
            return ""
        tags = self._stage_tags(stage)
        lowered = content.lower()
        if any(f"<{tag.lower()}>" in lowered for tag in tags):
            return content
        if stage == Stage.CODE:
            return f"<python>\n{content}\n</python>"
        if stage in {Stage.SCHEMA, Stage.SET_PARAM_VAR, Stage.OBJ_CONS}:
            return content
        tag = tags[0] if tags else "python"
        return f"<{tag}>\n{content}\n</{tag}>"
    def _disable_adapter_context(self):
        disable_adapter = getattr(self._model, "disable_adapter", None)
        if callable(disable_adapter):
            try:
                ctx = disable_adapter()
                if ctx is not None:
                    return ctx
            except Exception:
                pass
        return contextlib.nullcontext()

    def _teacher_forced_action_scores(
        self,
        prompt_text: str,
        action_blocks: list[str],
        gamma: float = 1.0,
        disable_adapter: bool = False,
    ) -> list[float]:
        if not action_blocks:
            return []
        prefix_text = prompt_text.rstrip() + '\n'
        prefix_ids = self._tokenizer(prefix_text, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
        prefix_len_raw = int(prefix_ids.shape[0])
        max_ctx = int(getattr(getattr(self._model, 'config', None), 'max_position_embeddings', 32768) or 32768)
        full_tensors: list[torch.Tensor] = []
        prefix_lens: list[int] = []
        scores = [float('-inf')] * len(action_blocks)
        for action_block in action_blocks:
            if not action_block:
                full_tensors.append(torch.empty(0, dtype=torch.long))
                prefix_lens.append(0)
                continue
            full_ids = self._tokenizer(prefix_text + action_block, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
            prefix_len = prefix_len_raw
            if int(full_ids.shape[0]) > max_ctx:
                full_ids = full_ids[-max_ctx:]
                prefix_len = max(1, min(prefix_len, int(full_ids.shape[0]) - 1))
            full_tensors.append(full_ids)
            prefix_lens.append(prefix_len)
        valid_indices = [idx for idx, (ids, prefix_len) in enumerate(zip(full_tensors, prefix_lens, strict=False)) if int(ids.shape[0]) > prefix_len > 0]
        if not valid_indices:
            return scores
        batch_tensors = [full_tensors[idx] for idx in valid_indices]
        padded = torch.nn.utils.rnn.pad_sequence(batch_tensors, batch_first=True, padding_value=self._tokenizer.pad_token_id)
        attention_mask = (padded != self._tokenizer.pad_token_id).long()
        device = self._infer_device()
        input_ids = padded.to(device)
        attention_mask = attention_mask.to(device)
        adapter_context = self._disable_adapter_context() if disable_adapter else contextlib.nullcontext()
        with adapter_context:
            with torch.no_grad():
                logits = self._model(input_ids=input_ids, attention_mask=attention_mask).logits
                log_probs = torch.log_softmax(logits, dim=-1)
        for batch_idx, original_idx in enumerate(valid_indices):
            full_ids = batch_tensors[batch_idx]
            prefix_len = int(prefix_lens[original_idx])
            action_start = max(1, prefix_len)
            action_token_count = int(full_ids.shape[0]) - action_start
            if action_token_count <= 0:
                continue
            total_logp = 0.0
            full_ids_cpu = full_ids.detach().cpu()
            for pos in range(action_start, int(full_ids.shape[0])):
                token_id = int(full_ids_cpu[pos].item())
                total_logp += float(log_probs[batch_idx, pos - 1, token_id].item())
            scores[original_idx] = float(total_logp / (max(1, action_token_count) ** gamma))
        return scores

    def _teacher_forced_action_score(self, prompt_text: str, action_block: str, gamma: float = 1.0) -> float:
        return self._teacher_forced_action_scores(prompt_text, [action_block], gamma=gamma)[0]
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
            "If you need tp>1, launch with a distributed launcher such as accelerate so WORLD_SIZE matches."
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
def _resolve_trl_vllm_client(trl_module: Any):
    generation_mod = getattr(trl_module, "generation", None)
    if generation_mod is not None:
        client_mod = getattr(generation_mod, "vllm_client", None)
        client_cls = getattr(client_mod, "VLLMClient", None) if client_mod is not None else None
        if client_cls is not None:
            return client_cls
    try:
        from trl.generation.vllm_client import VLLMClient  # type: ignore
        return VLLMClient
    except Exception:
        return None
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














