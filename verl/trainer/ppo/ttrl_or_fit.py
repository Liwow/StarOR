from __future__ import annotations

import json
import os
import re
import uuid
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn as verl_collate_fn
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask

from verl.trainer.ttrl_or_runtime.config import PipelineConfig
from verl.trainer.ttrl_or_runtime.pipeline.ttrl_or import TTRLORRunner
from verl.trainer.ttrl_or_runtime.reward.r3_batch_planner import (
    attach_r3_plan_to_instance,
    build_r3_base_scale_prompt,
    build_r3_tests_prompt,
    build_readable_test_case,
    build_sample_r3_plan,
    format_r3_precompute_markdown,
    summarize_scale,
    summarize_test_case,
)
from verl.trainer.ttrl_or_runtime.types import Generation, OptimizationTask, Stage, TrainingSample

try:
    from omegaconf import DictConfig, ListConfig, OmegaConf
except Exception:  # pragma: no cover
    DictConfig = None  # type: ignore[assignment]
    ListConfig = None  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]


def _safe_path_component(name: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        return "task"
    safe = re.sub(r'[\\/:*?"<>|]+', "_", raw)
    safe = safe.strip(" .")
    return safe or "task"


def _model_mode_label(config: PipelineConfig) -> str:
    return "solverllm" if bool(config.mcts.solverllm_compare_mode) else "default"


def _model_identifier(config: PipelineConfig) -> str:
    raw = str(config.backend.model_name_or_path or "").strip()
    if not raw:
        return "model"
    maybe_path = Path(raw)
    candidate = maybe_path.name or raw.rstrip("/\\").split("/")[-1].split("\\")[-1]
    return _safe_path_component(candidate) or "model"


def _verl_model_log_root(config: PipelineConfig) -> Path:
    base = Path(config.log_dir)
    folder = f"model_{_model_mode_label(config)}_{_model_identifier(config)}"
    return base / folder


def _json_safe(value: Any) -> Any:
    if OmegaConf is not None and DictConfig is not None and ListConfig is not None:
        if isinstance(value, (DictConfig, ListConfig)):
            return _json_safe(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return tensor.item()
        return tensor.tolist()
    return value


def _write_json_file(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_run_config(config: PipelineConfig, model_root: Path) -> Path:
    model_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "log_root": str(model_root.resolve()),
        "mode": _model_mode_label(config),
        "model_id": _model_identifier(config),
        "config": {
            "mcts": asdict(config.mcts),
            "reward": asdict(config.reward),
            "grpo": asdict(config.grpo),
            "dataset": asdict(config.dataset),
            "backend": asdict(config.backend),
            "save_logs": config.save_logs,
            "log_dir": config.log_dir,
        },
    }
    out_path = model_root / "run_config.json"
    return _write_json_file(out_path, payload)


def _normalize_dataset_paths(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if OmegaConf is not None and DictConfig is not None and ListConfig is not None and isinstance(value, ListConfig):
        value = list(value)
    if isinstance(value, (list, tuple)):
        items = value
    else:
        raw = str(value or "").replace("\r", "\n")
        for sep in [";", "|"]:
            raw = raw.replace(sep, "\n")
        raw = raw.replace(",", "\n")
        items = raw.split("\n")
    return tuple(str(item).strip() for item in items if str(item).strip())


def _resolve_dataset_paths_from_pipeline_config(pipeline_config: PipelineConfig) -> tuple[str, ...]:
    paths = _normalize_dataset_paths(getattr(pipeline_config.dataset, "jsonl_paths", ()))
    if paths:
        return paths
    single = str(getattr(pipeline_config.dataset, "jsonl_path", "") or "").strip()
    return (single,) if single else ()


def _sample_run_dir(log_dir: str | os.PathLike[str], sample_id: str) -> Path:
    return Path(log_dir) / _safe_path_component(sample_id)


def _sample_result_path(log_dir: str | os.PathLike[str], sample_id: str) -> Path:
    return _sample_run_dir(log_dir, sample_id) / "result.json"


def _prompt_to_messages(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, list):
        out: list[dict[str, str]] = []
        for item in prompt:
            if isinstance(item, dict):
                role = str(item.get("role", "user") or "user").strip().lower()
                content = str(item.get("content", "") or "")
                out.append({"role": role or "user", "content": content})
        if out:
            return out

    text = str(prompt or "").strip()
    if not text:
        return [{"role": "user", "content": ""}]

    pattern = re.compile(r"\[(SYSTEM|USER|ASSISTANT)\]\s*", flags=re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if not matches:
        return [{"role": "user", "content": text}]

    messages: list[dict[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        role = match.group(1).lower()
        content = text[start:end].strip()
        if content:
            messages.append({"role": role, "content": content})
    return messages or [{"role": "user", "content": text}]


def _raw_sample_to_task(sample) -> OptimizationTask:
    return OptimizationTask(
        task_id=str(sample.sample_id),
        description=str(sample.question),
        instance=deepcopy(sample.instance),
    )


def _flatten_stage_reports(stage_reports: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in stage_reports.items():
        if not isinstance(value, dict):
            continue
        reward_total = value.get("best_reward")
        if isinstance(reward_total, (int, float)):
            metrics[f"ttrl_or/{key}/best_reward"] = float(reward_total)
        stage_samples = value.get("stage_samples")
        if isinstance(stage_samples, (int, float)):
            metrics[f"ttrl_or/{key}/stage_samples"] = float(stage_samples)
    return metrics


class VerlRayPolicyBackend:
    """
    Adapter that lets the migrated TTRL-OR MCTS code use verl actor/vLLM workers.
    """

    def __init__(self, trainer, pipeline_config: PipelineConfig):
        self.trainer = trainer
        self.pipeline_config = pipeline_config
        self.tokenizer = trainer.tokenizer
        self._current_task_id = ""
        self._prior_temperature = 0.5

    def _sample_reset_enabled(self) -> bool:
        return bool(self.pipeline_config.backend.reset_lora_on_begin_episode)

    def _has_episode_lora(self) -> bool:
        model_cfg = self.trainer.config.actor_rollout_ref.model
        lora_rank = int(model_cfg.get("lora_rank", 0) or 0)
        has_adapter_path = model_cfg.get("lora_adapter_path") is not None
        return lora_rank > 0 or has_adapter_path

    def _reset_actor_episode_state(self) -> None:
        if not self._sample_reset_enabled():
            return
        if not self._has_episode_lora():
            return
        if getattr(self.trainer, "use_legacy_worker_impl", "disable") != "disable":
            print("[verl-or][WARN] sample-level LoRA reset is only implemented for the new worker path; skipping reset.")
            return
        self.trainer.actor_rollout_wg.reset_actor_for_episode()
        self.trainer.checkpoint_manager.update_weights(int(self.trainer.global_steps))

    def begin_episode(self, task: OptimizationTask) -> None:
        self._current_task_id = task.task_id

    def end_episode(self) -> None:
        try:
            self._reset_actor_episode_state()
        except Exception as exc:  # noqa: BLE001
            print(f"[verl-or][WARN] actor LoRA sample reset failed: {type(exc).__name__}: {exc}")
        self._current_task_id = ""

    def generate(self, stage: Stage, prompt: Any, n: int) -> list[Generation]:
        del stage
        messages = _prompt_to_messages(prompt)
        outputs = self._generate_messages(
            [messages for _ in range(max(1, int(n)))],
            temperature=float(self.trainer.config.actor_rollout_ref.rollout.temperature),
            top_p=float(self.trainer.config.actor_rollout_ref.rollout.get("top_p", 1.0)),
            do_sample=True,
            max_new_tokens=int(self.pipeline_config.grpo.max_completion_length),
        )
        k = max(1, len(outputs))
        return [Generation(text=str(text or ""), prior=1.0 / float(k), metadata={"backend": "verl"}) for text in outputs]

    def score_action_priors(self, stage: Stage, prompt: Any, candidates: list[str]) -> list[float]:
        del stage
        if not self.pipeline_config.mcts.enable_prior or not candidates:
            n = len(candidates)
            return [1.0 / float(max(1, n))] * n

        try:
            messages = _prompt_to_messages(prompt)
            batch, response_mask = self._build_teacher_forcing_batch(messages, candidates)
            batch.meta_info["temperature"] = float(self.trainer.config.actor_rollout_ref.rollout.temperature)
            old_log_prob, _ = self.trainer._compute_old_log_prob(batch)
            log_probs = old_log_prob.batch["old_log_probs"]
            curr_scores = self._masked_mean(log_probs, response_mask)

            ref_scores = None
            if self.trainer.use_reference_policy:
                ref_batch = self.trainer._compute_ref_log_prob(batch)
                ref_log_prob = ref_batch.batch["ref_log_prob"]
                ref_scores = self._masked_mean(ref_log_prob, response_mask)

            raw_scores = curr_scores if ref_scores is None else (curr_scores - ref_scores)
            raw_scores = raw_scores / max(1e-6, float(self._prior_temperature))
            priors = torch.softmax(raw_scores, dim=0).detach().cpu().tolist()
            return [float(x) for x in priors]
        except Exception as exc:  # noqa: BLE001
            print(f"[verl-or][WARN] prior fallback to uniform: {type(exc).__name__}: {exc}")
            n = len(candidates)
            return [1.0 / float(max(1, n))] * n

    def grpo_update(self, samples: list[TrainingSample], config, stage: Stage) -> dict[str, Any]:
        del config, stage
        if not samples:
            return {"updated": False, "backend": "verl", "num_samples": 0}
        return {"updated": False, "backend": "verl", "reason": "manual_grpo_update_not_used"}

    def grpo_rollout_group(self, stage: Stage, prompt: Any, config, reward_callback):
        messages = _prompt_to_messages(prompt)
        k = max(1, int(config.num_generations))
        rollout_temperature = float(self.trainer.config.actor_rollout_ref.rollout.temperature)
        current_step = int(self.trainer.global_steps) + 1
        self.trainer.global_steps = current_step
        prompt_batch = self._make_prompt_batch(
            [messages],
            extra_infos=[{"task_id": self._current_task_id, "stage": stage.value}],
        )
        prompt_batch.meta_info["temperature"] = rollout_temperature
        prompt_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4())], dtype=object)

        gen_batch = self.trainer._get_gen_batch(prompt_batch)
        gen_batch.meta_info["temperature"] = rollout_temperature
        gen_batch.meta_info["top_p"] = float(self.trainer.config.actor_rollout_ref.rollout.get("top_p", 1.0))
        gen_batch.meta_info["do_sample"] = True
        gen_batch.meta_info["global_steps"] = current_step
        gen_batch.meta_info["max_new_tokens"] = int(config.max_completion_length)
        gen_batch_output = gen_batch.repeat(repeat_times=k, interleave=True)
        gen_output = self._rollout_generate(gen_batch_output)

        batch = prompt_batch.repeat(repeat_times=k, interleave=True)
        batch.meta_info["temperature"] = rollout_temperature
        batch = batch.union(gen_output)
        if "response_mask" not in batch.batch.keys():
            from verl.trainer.ppo.ray_trainer import compute_response_mask

            batch.batch["response_mask"] = compute_response_mask(batch)

        completions = self._decode_responses(batch)
        batch_score = getattr(reward_callback, "batch_score", None)
        if callable(batch_score):
            rewards = [float(x) for x in list(batch_score(prompt, completions))]
        else:
            rewards = [float(reward_callback(prompt, text, ridx)) for ridx, text in enumerate(completions)]
        rewards = rewards[: len(completions)] + [0.0] * max(0, len(completions) - len(rewards))

        reward_tensor = self._terminal_reward_tensor(batch.batch["response_mask"], rewards)
        batch.batch["token_level_scores"] = reward_tensor

        old_log_prob, old_log_prob_mfu = self.trainer._compute_old_log_prob(batch)
        batch = batch.union(old_log_prob)

        if self.trainer.use_reference_policy and "ref_log_prob" not in batch.batch.keys():
            ref_log_prob = self.trainer._compute_ref_log_prob(batch)
            batch = batch.union(ref_log_prob)

        kl_metrics: dict[str, Any] = {}
        if self.trainer.config.algorithm.use_kl_in_reward:
            from verl.trainer.ppo.ray_trainer import apply_kl_penalty

            batch, kl_metrics = apply_kl_penalty(
                batch,
                kl_ctrl=self.trainer.kl_ctrl_in_reward,
                kl_penalty=self.trainer.config.algorithm.kl_penalty,
            )
        else:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

        from verl.trainer.ppo.ray_trainer import compute_advantage

        batch = compute_advantage(
            batch,
            adv_estimator=self.trainer.config.algorithm.adv_estimator,
            gamma=self.trainer.config.algorithm.gamma,
            lam=self.trainer.config.algorithm.lam,
            num_repeat=k,
            norm_adv_by_std_in_grpo=self.trainer.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=self.trainer.config.algorithm,
        )

        actor_output = self.trainer._update_actor(batch)
        actor_metrics_raw = dict(actor_output.meta_info.get("metrics", {}) or {})
        actor_metrics = reduce_metrics(actor_metrics_raw) if actor_metrics_raw else {}
        actor_metrics["perf/mfu/actor_infer"] = old_log_prob_mfu

        self.trainer.checkpoint_manager.update_weights(current_step)

        generations: list[Generation] = []
        prior = 1.0 / float(max(1, len(completions)))
        for ridx, text in enumerate(completions):
            generations.append(
                Generation(
                    text=str(text or ""),
                    prior=prior,
                    metadata={
                        "backend": "verl",
                        "rollout_index": ridx,
                        "reward_total": float(rewards[ridx]),
                    },
                )
            )

        report = {
            "updated": True,
            "backend": "verl",
            "stage": stage.value,
            "num_samples": len(generations),
            "metrics": {**actor_metrics, **kl_metrics},
        }
        return generations, report

    def prepare_task_context(self, task: OptimizationTask, dataset_config) -> dict[str, Any]:
        from verl.trainer.ttrl_or_runtime.mapping import build_mapping_extractor

        extractor = build_mapping_extractor(dataset_config.mapping_extractor)
        result = extractor.extract(task=task, dataset_config=dataset_config, backend=self)
        task.instance = result.instance
        task.perturbation_map = result.perturbation_map
        return dict(result.metadata)

    def generate_mapping_from_description(self, description: str, dataset_config) -> dict[str, Any] | str | None:
        del description, dataset_config
        return None

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
        del prefer_vllm, vllm_mode
        outputs = self.generate_auxiliary_texts(
            [str(prompt)],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
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
        del prefer_vllm, vllm_mode
        messages_batch = [_prompt_to_messages(prompt) for prompt in prompts]
        return self._generate_messages(
            messages_batch,
            temperature=float(temperature),
            top_p=float(top_p),
            do_sample=bool(float(temperature) > 0.0),
            max_new_tokens=int(max_new_tokens),
        )

    def _rollout_generate(self, gen_batch: DataProto) -> DataProto:
        rollout_name = str(self.trainer.config.actor_rollout_ref.rollout.name or "").strip().lower()
        if rollout_name == "vllm":
            # In the new worker path, vLLM rollout is served by the async server interface.
            # ServerAdapter explicitly does not support synchronous worker-group generation.
            return self.trainer.async_rollout_manager.generate_sequences(gen_batch)

        if hasattr(self.trainer.actor_rollout_wg, "generate_sequences"):
            return self.trainer.actor_rollout_wg.generate_sequences(gen_batch)

        if hasattr(self.trainer, "async_rollout_manager"):
            return self.trainer.async_rollout_manager.generate_sequences(gen_batch)

        raise AttributeError("No available generate_sequences path found for the current verl rollout backend.")
    def _generate_messages(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        temperature: float,
        top_p: float,
        do_sample: bool,
        max_new_tokens: int,
    ) -> list[str | None]:
        if not messages_batch:
            return []
        prompt_batch = self._make_prompt_batch(
            messages_batch,
            extra_infos=[{"task_id": self._current_task_id, "kind": "aux"} for _ in messages_batch],
        )
        gen_batch = self.trainer._get_gen_batch(prompt_batch)
        gen_batch.meta_info["temperature"] = float(temperature)
        gen_batch.meta_info["top_p"] = float(top_p)
        gen_batch.meta_info["do_sample"] = bool(do_sample)
        gen_batch.meta_info["global_steps"] = self.trainer.global_steps
        gen_batch.meta_info["max_new_tokens"] = int(max_new_tokens)
        gen_output = self._rollout_generate(gen_batch)
        prompt_batch = prompt_batch.union(gen_output)
        return self._decode_responses(prompt_batch)

    def _make_prompt_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        extra_infos: list[dict[str, Any]] | None = None,
    ) -> DataProto:
        rows: list[dict[str, Any]] = []
        for idx, messages in enumerate(messages_batch):
            extra = extra_infos[idx] if extra_infos and idx < len(extra_infos) else {}
            rows.append(
                {
                    "data_source": "ttrl_or",
                    "reward_model": {"style": "rule"},
                    "extra_info": {"index": idx, **(extra or {})},
                    "raw_prompt": messages,
                    "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
                    "index": idx,
                    "tools_kwargs": {},
                    "interaction_kwargs": {},
                }
            )
        batch_dict = verl_collate_fn(rows)
        return DataProto.from_single_dict(batch_dict)

    def _decode_responses(self, batch: DataProto) -> list[str]:
        outputs: list[str] = []
        responses = batch.batch["responses"].detach().cpu()
        response_mask = batch.batch.get("response_mask")
        if response_mask is not None:
            response_mask = response_mask.detach().cpu()
        for idx in range(responses.shape[0]):
            ids = responses[idx]
            if response_mask is not None:
                mask = response_mask[idx].bool()
                ids = ids[mask]
            outputs.append(self.tokenizer.decode(ids.tolist(), skip_special_tokens=True))
        return outputs

    def _terminal_reward_tensor(self, response_mask: torch.Tensor, rewards: list[float]) -> torch.Tensor:
        reward_tensor = torch.zeros_like(response_mask, dtype=torch.float32)
        for idx in range(response_mask.shape[0]):
            valid_positions = torch.nonzero(response_mask[idx], as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                continue
            reward_tensor[idx, int(valid_positions[-1].item())] = float(rewards[idx])
        return reward_tensor

    def _build_teacher_forcing_batch(
        self,
        prompt_messages: list[dict[str, str]],
        candidates: list[str],
    ) -> tuple[DataProto, torch.Tensor]:
        prompt_ids = self._encode_prompt_messages(prompt_messages)
        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0)
        if not prompt_ids:
            prompt_ids = [pad_id]
        response_id_seqs = []
        for candidate in candidates:
            response_ids = self.tokenizer(
                str(candidate or ""),
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            if not response_ids:
                response_ids = [int(self.tokenizer.eos_token_id or pad_id)]
            response_id_seqs.append(response_ids)

        max_resp = max(len(seq) for seq in response_id_seqs)
        prompt_len = len(prompt_ids)
        max_total = max(len(prompt_ids) + len(seq) for seq in response_id_seqs)
        bsz = len(response_id_seqs)
        input_ids = torch.full((bsz, max_total), pad_id, dtype=torch.long)
        prompts = torch.full((bsz, prompt_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_total), dtype=torch.long)
        responses = torch.full((bsz, max_resp), pad_id, dtype=torch.long)
        response_mask = torch.zeros((bsz, max_resp), dtype=torch.long)

        for row_idx, response_ids in enumerate(response_id_seqs):
            total_ids = prompt_ids + response_ids
            prompts[row_idx, :prompt_len] = torch.tensor(prompt_ids, dtype=torch.long)
            input_ids[row_idx, : len(total_ids)] = torch.tensor(total_ids, dtype=torch.long)
            attention_mask[row_idx, : len(total_ids)] = 1
            responses[row_idx, : len(response_ids)] = torch.tensor(response_ids, dtype=torch.long)
            response_mask[row_idx, : len(response_ids)] = 1

        position_ids = compute_position_id_with_mask(attention_mask)
        batch = DataProto.from_dict(
            tensors={
                "prompts": prompts,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
                "response_mask": response_mask,
            },
            meta_info={"temperature": float(self.trainer.config.actor_rollout_ref.rollout.temperature)},
        )
        return batch, response_mask.to(dtype=torch.float32)

    def _encode_prompt_messages(self, prompt_messages: list[dict[str, str]]) -> list[int]:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                encoded = self.tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                if isinstance(encoded, torch.Tensor):
                    return encoded.tolist()
                return list(encoded)
            except Exception:
                pass
        fallback = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in prompt_messages)
        return list(
            self.tokenizer(
                fallback,
                add_special_tokens=True,
                return_attention_mask=False,
            )["input_ids"]
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp(mask.sum(dim=-1), min=1.0)
        return (values * mask).sum(dim=-1) / denom


def build_pipeline_config_from_verl_config(config) -> PipelineConfig:
    pipe = PipelineConfig()
    section = config.algorithm.get("ttrl_or", {}) if config.algorithm is not None else {}

    for name, value in dict(section.get("mcts", {})).items():
        if hasattr(pipe.mcts, name):
            setattr(pipe.mcts, name, value)
    for name, value in dict(section.get("reward", {})).items():
        if hasattr(pipe.reward, name):
            setattr(pipe.reward, name, value)
    for name, value in dict(section.get("grpo", {})).items():
        if hasattr(pipe.grpo, name):
            setattr(pipe.grpo, name, value)
    for name, value in dict(section.get("dataset", {})).items():
        if hasattr(pipe.dataset, name):
            setattr(pipe.dataset, name, value)
    for name, value in dict(section.get("backend", {})).items():
        if hasattr(pipe.backend, name):
            setattr(pipe.backend, name, value)

    pipe.grpo.num_generations = int(config.actor_rollout_ref.rollout.n)
    pipe.grpo.group_size = int(config.actor_rollout_ref.rollout.n)
    pipe.grpo.use_vllm = str(config.actor_rollout_ref.rollout.name) == "vllm"
    pipe.grpo.vllm_mode = "colocate"
    pipe.grpo.learning_rate = float(config.actor_rollout_ref.actor.optim.lr)
    pipe.grpo.kl_coef = float(config.actor_rollout_ref.actor.get("kl_loss_coef", pipe.grpo.kl_coef))
    pipe.backend.backend = "verl"
    pipe.backend.model_name_or_path = str(config.actor_rollout_ref.model.path)
    pipe.backend.temperature = float(config.actor_rollout_ref.rollout.temperature)
    pipe.backend.top_p = float(config.actor_rollout_ref.rollout.get("top_p", 1.0))
    pipe.backend.max_new_tokens = int(config.data.max_response_length)
    pipe.backend.lora_r = int(config.actor_rollout_ref.model.get("lora_rank", 0))
    pipe.backend.lora_alpha = int(config.actor_rollout_ref.model.get("lora_alpha", 0))
    pipe.backend.reset_lora_on_begin_episode = bool(
        section.get("backend", {}).get("reset_lora_on_begin_episode", pipe.backend.reset_lora_on_begin_episode)
    )
    dataset_paths = _normalize_dataset_paths(config.data.train_files)
    pipe.dataset.jsonl_paths = dataset_paths
    pipe.dataset.jsonl_path = str(dataset_paths[0]) if dataset_paths else ""
    pipe.log_dir = str(section.get("log_dir") or os.path.join(config.trainer.default_local_dir, "ttrl_or"))
    pipe.save_logs = bool(section.get("save_logs", True))
    return pipe


def _resolve_raw_samples_for_path(trainer, pipeline_config: PipelineConfig, dataset_path: str):
    dataset = getattr(trainer, "train_dataset", None)
    dataset_paths = _resolve_dataset_paths_from_pipeline_config(pipeline_config)
    if len(dataset_paths) <= 1 and dataset is not None and hasattr(dataset, "raw_samples"):
        return list(dataset.raw_samples)

    from verl.trainer.ttrl_or_runtime.dataset.loader import load_raw_task_dataset

    return load_raw_task_dataset(
        dataset_path,
        start_index=0,
        limit=None if int(trainer.config.data.get("train_max_samples", -1)) <= 0 else int(trainer.config.data.train_max_samples),
        max_numeric_features=int(pipeline_config.dataset.max_numeric_features),
        key_param_top_k=int(pipeline_config.dataset.key_param_top_k),
    )


def _prepare_r3_for_samples(raw_samples: list, runner: TTRLORRunner, rank: int, world_size: int) -> None:
    if not raw_samples or not runner.config.reward.enable_r3_reward:
        return

    cfg = runner.config
    print(f"[verl-or][r3] rank={rank}/{world_size} start precompute for {len(raw_samples)} samples")
    prompt_jobs: list[dict[str, Any]] = []
    base_prompts: list[str] = []
    tests_prompts: list[str] = []
    base_instances: list[dict[str, Any]] = []
    for sample_idx, sample in enumerate(raw_samples):
        base_instance = deepcopy(sample.instance)
        base_instances.append(base_instance)
        base_prompt = build_r3_base_scale_prompt(
            description=sample.question,
            instance=base_instance,
        )
        tests_prompt = build_r3_tests_prompt(
            description=sample.question,
            instance=base_instance,
            num_tests=max(1, int(cfg.reward.robustness_cases)),
        )
        base_prompts.append(base_prompt)
        tests_prompts.append(tests_prompt)
        prompt_jobs.append({"sample_idx": sample_idx, "sample_id": sample.sample_id, "kind": "base", "prompt": base_prompt})
        prompt_jobs.append({"sample_idx": sample_idx, "sample_id": sample.sample_id, "kind": "tests", "prompt": tests_prompt})

    outputs = runner.backend.generate_auxiliary_texts(
        [str(job["prompt"]) for job in prompt_jobs],
        max_new_tokens=int(cfg.dataset.r3_plan_max_new_tokens),
        temperature=float(cfg.dataset.r3_plan_temperature),
        top_p=float(cfg.dataset.r3_plan_top_p),
        prefer_vllm=True,
        vllm_mode="colocate",
    )
    llm_base_texts: list[str | None] = [None for _ in raw_samples]
    llm_tests_texts: list[str | None] = [None for _ in raw_samples]
    for idx, job in enumerate(prompt_jobs):
        value = outputs[idx] if idx < len(outputs) else None
        sample_idx = int(job["sample_idx"])
        if str(job["kind"]) == "base":
            llm_base_texts[sample_idx] = value
        else:
            llm_tests_texts[sample_idx] = value

    for idx, sample in enumerate(raw_samples):
        base_instance = base_instances[idx]
        plan = build_sample_r3_plan(
            sample_id=sample.sample_id,
            description=sample.question,
            instance=base_instance,
            robustness_cases=max(1, int(cfg.reward.robustness_cases)),
            llm_base_text=llm_base_texts[idx],
            llm_tests_text=llm_tests_texts[idx],
            allow_heuristic_fallback=False,
        )
        sample.instance = attach_r3_plan_to_instance(base_instance, plan)
        gold_answer = str(getattr(sample, "answer", "") or "")
        precompute_status = "ok" if plan.test_cases else "empty"
        print(
            f"[verl-or][r3] [{idx + 1}/{len(raw_samples)}] sample_id={sample.sample_id} "
            f"gt={gold_answer} status={precompute_status} source={plan.source} tests={len(plan.test_cases)} "
            f"base_scale={plan.base_obj_bounds}"
        )

        if cfg.save_logs:
            sample_dir = Path(cfg.log_dir) / _safe_path_component(sample.sample_id)
            sample_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "sample_id": sample.sample_id,
                "gold_answer": gold_answer,
                "gt": gold_answer,
                "status": precompute_status,
                "source": plan.source,
                "analysis": plan.analysis,
                "base_obj_scale": plan.base_obj_bounds,
                "base_scale_summary": summarize_scale(plan.base_obj_bounds),
                "feature_catalog": plan.feature_catalog,
                "num_tests": len(plan.test_cases),
                "mapping": plan.mapping,
                "tests": plan.test_cases,
                "tests_summary": [summarize_test_case(case) for case in plan.test_cases],
                "tests_readable": [build_readable_test_case(case) for case in plan.test_cases],
                "llm_raw_preview": plan.llm_raw_preview,
                "llm_base_preview": plan.llm_base_preview,
                "llm_tests_preview": plan.llm_tests_preview,
                "planner_prompt_preview": {
                    "base_scale": base_prompts[idx][:1200],
                    "tests": tests_prompts[idx][:1200],
                },
                "used_vllm_priority": True,
                "vllm_mode": "colocate",
            }
            _write_json_file(sample_dir / "r3_precompute.json", payload)
            (sample_dir / "r3_precompute.md").write_text(
                format_r3_precompute_markdown(
                    sample_id=str(sample.sample_id),
                    gold_answer=gold_answer,
                    source=str(plan.source),
                    analysis=str(plan.analysis),
                    base_obj_bounds=plan.base_obj_bounds,
                    tests=plan.test_cases,
                ),
                encoding="utf-8",
            )


def run_ttrl_or_fit(trainer, logger) -> None:
    trainer.global_steps = 0
    trainer._load_checkpoint()
    trainer.checkpoint_manager.update_weights(trainer.global_steps)

    pipeline_config = build_pipeline_config_from_verl_config(trainer.config)
    model_log_root = _verl_model_log_root(pipeline_config)
    _write_run_config(pipeline_config, model_log_root)
    backend = VerlRayPolicyBackend(trainer, pipeline_config)
    runner = TTRLORRunner(backend=backend, config=pipeline_config)

    rank = int(os.environ.get("RANK", "0"))
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    dataset_paths = _resolve_dataset_paths_from_pipeline_config(pipeline_config)
    if not dataset_paths:
        print("[verl-or][dataset] no dataset paths configured, exiting.")
        return

    dataset_runs: list[dict[str, Any]] = []
    total_pending_samples = 0
    max_iterations = int(max(1, pipeline_config.mcts.max_iterations))

    for dataset_path in dataset_paths:
        raw_samples = _resolve_raw_samples_for_path(trainer, pipeline_config, dataset_path)
        dataset_name = Path(dataset_path).stem if str(dataset_path).strip() else "dataset"
        dataset_log_dir = model_log_root / _safe_path_component(dataset_name)
        ordered_samples = list(raw_samples)
        if bool(trainer.config.data.shuffle):
            seed = trainer.config.data.get("seed")
            rng = np.random.default_rng(seed if seed is not None else 0)
            rng.shuffle(ordered_samples)

        skipped_completed = 0
        if bool(pipeline_config.dataset.resume_skip_completed):
            pending_samples = []
            for sample in ordered_samples:
                result_path = _sample_result_path(dataset_log_dir, str(sample.sample_id))
                if result_path.exists():
                    skipped_completed += 1
                    continue
                pending_samples.append(sample)
            ordered_samples = pending_samples

        total_pending_samples += len(ordered_samples)
        dataset_runs.append({
            "dataset_path": str(Path(dataset_path).resolve()),
            "dataset_name": dataset_name,
            "dataset_log_dir": str(dataset_log_dir),
            "raw_samples": raw_samples,
            "ordered_samples": ordered_samples,
            "skipped_completed": skipped_completed,
        })
        print(
            f"[verl-or][dataset] rank={rank}/{world_size} total_samples={len(raw_samples)} "
            f"pending_samples={len(ordered_samples)} skipped_completed={skipped_completed} "
            f"dataset={Path(dataset_path).resolve()} log_root={dataset_log_dir.resolve()}"
        )

    if total_pending_samples <= 0:
        print("[verl-or][dataset] no pending samples found across all datasets, exiting.")
        return

    total_budget = max(1, total_pending_samples * max_iterations)
    if trainer.global_steps < 0:
        trainer.global_steps = 0
    progress_bar = tqdm(total=total_budget, initial=min(total_budget, max(0, trainer.global_steps)), desc="TTRL-OR Progress")

    for dataset_idx, dataset_run in enumerate(dataset_runs, start=1):
        ordered_samples = list(dataset_run["ordered_samples"])
        if not ordered_samples:
            continue
        pipeline_config.dataset.jsonl_path = str(dataset_run["dataset_path"])
        pipeline_config.log_dir = str(dataset_run["dataset_log_dir"])
        runner.config.log_dir = str(dataset_run["dataset_log_dir"])
        print(
            f"[verl-or][dataset-run] [{dataset_idx}/{len(dataset_runs)}] dataset={dataset_run['dataset_name']} "
            f"pending_samples={len(ordered_samples)}"
        )

        _prepare_r3_for_samples(ordered_samples, runner, rank, world_size)

        for sample_idx, sample in enumerate(ordered_samples):
            if trainer.global_steps >= total_budget:
                break

            gold_answer = str(sample.answer or "")
            print(
                f"[verl-or][sample] [{sample_idx + 1}/{len(ordered_samples)}] sample_id={sample.sample_id} "
                f"dataset={dataset_run['dataset_name']} gt={gold_answer} global_step={trainer.global_steps}"
            )
            task = _raw_sample_to_task(sample)
            step_before = int(trainer.global_steps)
            result = runner.run_task(task, human_gold_answer=gold_answer)
            steps_used = max(0, int(trainer.global_steps) - step_before)
            if steps_used > 0:
                progress_bar.update(steps_used)

            best_reward = None
            best_obj = None
            if result.best_trajectory is not None and result.best_trajectory.reward is not None:
                best_reward = float(result.best_trajectory.reward.total)
                best_obj = (result.best_trajectory.reward.metadata or {}).get("obj_answer")

            metrics = {
                "training/global_step": float(trainer.global_steps),
                "training/epoch": 0.0,
                "ttrl_or/dataset_index": float(dataset_idx - 1),
                "ttrl_or/sample_index": float(sample_idx),
                "ttrl_or/sample_steps": float(steps_used),
                "ttrl_or/max_iterations": float(max_iterations),
                "ttrl_or/sample_best_reward": float(best_reward) if isinstance(best_reward, (int, float)) else float("nan"),
            }
            if best_obj is not None:
                try:
                    metrics["ttrl_or/sample_best_obj"] = float(best_obj)
                except Exception:
                    pass
            runtime = result.stage_reports.get("runtime", {}) if isinstance(result.stage_reports, dict) else {}
            if isinstance(runtime.get("total_elapsed_sec"), (int, float)):
                metrics["ttrl_or/sample_runtime_sec"] = float(runtime["total_elapsed_sec"])
            metrics.update(_flatten_stage_reports(result.stage_reports))
            print(
                f"[verl-or][sample-done] sample_id={sample.sample_id} dataset={dataset_run['dataset_name']} gt={gold_answer} "
                f"best_reward={best_reward} best_obj={best_obj} steps={steps_used}"
            )
            logger.log(data=metrics, step=max(0, trainer.global_steps))

    progress_bar.close()





