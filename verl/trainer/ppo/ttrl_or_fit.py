from __future__ import annotations

import json
import os
import re
import time
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
from verl.trainer.ttrl_or_runtime.dataset.loader import normalize_dataset_paths
from verl.trainer.ttrl_or_runtime.mcts.tree import FourStageMCTS
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
    if bool(getattr(config.dataset, "sample_run", False)):
        folder = f"{folder}_sample"
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
    if OmegaConf is not None and DictConfig is not None and ListConfig is not None and isinstance(value, ListConfig):
        value = list(value)
    return normalize_dataset_paths(value)


def _resolve_dataset_paths_from_pipeline_config(pipeline_config: PipelineConfig) -> tuple[str, ...]:
    paths = _normalize_dataset_paths(getattr(pipeline_config.dataset, "jsonl_paths", ()))
    if paths:
        return paths
    single = str(getattr(pipeline_config.dataset, "jsonl_path", "") or "").strip()
    return (single,) if single else ()


def _sample_run_dir(log_dir: str | os.PathLike[str], sample_id: str, run_tag: str = "") -> Path:
    sample_dir = Path(log_dir) / _safe_path_component(sample_id)
    run_tag_clean = str(run_tag or "").strip()
    if run_tag_clean:
        return sample_dir / _safe_path_component(run_tag_clean)
    return sample_dir


def _sample_result_path(log_dir: str | os.PathLike[str], sample_id: str, run_tag: str = "") -> Path:
    return _sample_run_dir(log_dir, sample_id, run_tag=run_tag) / "result.json"


def _none_to_default(value: Any, default: Any) -> Any:
    return default if value is None else value


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
        mcts_cfg = self.pipeline_config.mcts
        prior_temperature = _none_to_default(getattr(mcts_cfg, "prior_temperature", None), 0.6)
        prior_tail_tokens = _none_to_default(getattr(mcts_cfg, "prior_tail_tokens", None), 0)
        prior_standardize = _none_to_default(getattr(mcts_cfg, "prior_standardize", None), True)
        prior_use_ref = _none_to_default(getattr(mcts_cfg, "prior_use_ref", None), False)
        prior_min_std = _none_to_default(getattr(mcts_cfg, "prior_min_std", None), 1e-4)

        self._prior_temperature = float(prior_temperature)
        self._prior_tail_tokens = max(0, int(prior_tail_tokens))
        self._prior_standardize = bool(prior_standardize)
        self._prior_use_ref = bool(prior_use_ref)
        self._prior_min_std = float(prior_min_std)

    def _use_ttrl_enabled(self) -> bool:
        return bool(getattr(self.pipeline_config.grpo, "use_ttrl", True))

    def _stage_update_enabled(self) -> bool:
        return bool(getattr(self.pipeline_config.grpo, "stage_update", False))

    def _filter_rollout_enabled(self) -> bool:
        return bool(getattr(self.pipeline_config.mcts, "filter_rollout", False))

    def _actor_ppo_mini_batch_size(self) -> int:
        actor_cfg = getattr(getattr(self.trainer.config, "actor_rollout_ref", None), "actor", None)
        if actor_cfg is None:
            return 1
        value = None
        if isinstance(actor_cfg, dict):
            value = actor_cfg.get("ppo_mini_batch_size", None)
        else:
            value = getattr(actor_cfg, "ppo_mini_batch_size", None)
            if value is None and hasattr(actor_cfg, "get"):
                try:
                    value = actor_cfg.get("ppo_mini_batch_size", None)
                except Exception:
                    value = None
        try:
            mini_batch_size = int(value or 1)
        except Exception:
            mini_batch_size = 1
        return max(1, mini_batch_size)

    @staticmethod
    def _stage_anchor_tags(stage: Stage) -> list[str]:
        # Use the LAST stage tag in each stage group as anchor.
        # Example: Stage2 -> Variables, Stage3 -> Constraints.
        if stage == Stage.SCHEMA:
            return ["Sets", "Set"]
        if stage == Stage.SET_PARAM_VAR:
            return ["Variables", "Variable"]
        if stage == Stage.OBJ_CONS:
            return ["Constraints", "Constraint"]
        if stage == Stage.TYPE_HINT:
            return ["Type"]
        if stage == Stage.SETS:
            return ["Sets", "Set"]
        if stage == Stage.PARAMETERS:
            return ["Parameters", "Parameter"]
        if stage == Stage.VARIABLES:
            return ["Variables", "Variable"]
        if stage == Stage.OBJECTIVE:
            return ["Objective"]
        if stage == Stage.CONSTRAINTS:
            return ["Constraints", "Constraint"]
        return []

    @staticmethod
    def _find_stage_anchor_end(text: str, stage: Stage, min_len: int = 21) -> int:
        raw = str(text or "")
        if not raw:
            return -1
        required_len = max(21, int(min_len))
        best_end = -1
        for tag in VerlRayPolicyBackend._stage_anchor_tags(stage):
            od = re.escape("<")
            cd = re.escape(">")
            open_re = re.compile(rf"{od}\s*{re.escape(tag)}\s*{cd}", flags=re.IGNORECASE)
            close_re = re.compile(rf"{od}\s*/\s*{re.escape(tag)}\s*{cd}", flags=re.IGNORECASE)
            open_matches = list(open_re.finditer(raw))
            close_matches = list(close_re.finditer(raw))
            if not open_matches or not close_matches:
                continue
            for close_match in close_matches:
                close_start = int(close_match.start())
                nearest_open = None
                for open_match in reversed(open_matches):
                    if int(open_match.end()) <= close_start:
                        nearest_open = open_match
                        break
                if nearest_open is None:
                    continue
                content = raw[int(nearest_open.end()):close_start].strip()
                if len(content) < required_len:
                    continue
                best_end = int(close_match.end())
                break
            if best_end >= 0:
                break
        return best_end

    @staticmethod
    def _extract_stage_update_text(stage: Stage, text: str) -> str:
        raw_text = str(text or "")
        cleaned = FourStageMCTS._normalize_text_block(raw_text)
        if not cleaned:
            return ""
        anchor_end = VerlRayPolicyBackend._find_stage_anchor_end(cleaned, stage=stage, min_len=21)
        if anchor_end < 0:
            return ""
        python_re = re.compile(r"<\s*python\s*>", flags=re.IGNORECASE)
        python_match = python_re.search(cleaned, pos=int(anchor_end))
        if python_match is None:
            # No python tail to remove: keep the full original completion text.
            return raw_text.strip()
        prefix = cleaned[: int(python_match.start())].strip()
        return prefix

    @staticmethod
    def _prefix_match_len(lhs: list[int], rhs: list[int]) -> int:
        n = min(len(lhs), len(rhs))
        i = 0
        while i < n and int(lhs[i]) == int(rhs[i]):
            i += 1
        return i

    def _build_stage_update_response_mask(
        self,
        *,
        stage: Stage,
        batch: DataProto,
        completions: list[str],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        base_mask = batch.batch.get("response_mask")
        responses = batch.batch.get("responses")
        if base_mask is None or responses is None:
            return (
                torch.zeros(0, dtype=torch.long),
                {
                    "enabled": True,
                    "applied": False,
                    "reason": "missing_response_mask_or_responses",
                },
            )

        base_mask_cpu = base_mask.detach().cpu()
        responses_cpu = responses.detach().cpu()
        new_mask_cpu = torch.zeros_like(base_mask_cpu)

        rows = int(base_mask_cpu.shape[0]) if base_mask_cpu.ndim >= 2 else 0
        truncated_rows = 0
        full_rows = 0
        fallback_rows = 0
        effective_tokens = 0
        total_tokens = 0

        for row_idx in range(rows):
            valid_positions = torch.nonzero(base_mask_cpu[row_idx], as_tuple=False).flatten()
            valid_len = int(valid_positions.numel())
            if valid_len <= 0:
                continue
            total_tokens += int(valid_len)

            full_text = str(completions[row_idx] if row_idx < len(completions) else "")
            selected_text = self._extract_stage_update_text(stage, full_text)

            # If stage tags are not valid/missing (e.g. <20 chars), keep full update
            # to avoid accidentally masking out all learning signal.
            if len(selected_text.strip()) < 21:
                selected_len = int(valid_len)
                full_rows += 1
                fallback_rows += 1
            elif selected_text.strip() == full_text.strip():
                selected_len = int(valid_len)
                full_rows += 1
            else:
                selected_ids = self.tokenizer(
                    str(selected_text),
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]
                response_ids = responses_cpu[row_idx][valid_positions].tolist()
                match_len = self._prefix_match_len(list(selected_ids), response_ids)
                if match_len <= 0:
                    # Robust fallback when text-token alignment is imperfect.
                    selected_len = min(int(valid_len), int(len(selected_ids)))
                    if selected_len <= 0:
                        selected_len = int(valid_len)
                    fallback_rows += 1
                else:
                    selected_len = min(int(valid_len), int(match_len))

                if selected_len < valid_len:
                    truncated_rows += 1
                else:
                    full_rows += 1

            effective_tokens += int(selected_len)
            if selected_len > 0:
                new_mask_cpu[row_idx, valid_positions[:selected_len]] = 1

        applied = bool(rows > 0 and effective_tokens > 0)
        info = {
            "enabled": True,
            "applied": bool(applied),
            "stage": str(stage.value),
            "rows": int(rows),
            "rows_truncated": int(truncated_rows),
            "rows_full": int(full_rows),
            "rows_fallback": int(fallback_rows),
            "total_tokens_before": int(total_tokens),
            "total_tokens_after": int(effective_tokens),
        }
        if total_tokens > 0:
            info["token_keep_ratio"] = float(effective_tokens) / float(total_tokens)
        else:
            info["token_keep_ratio"] = 0.0

        new_mask = new_mask_cpu.to(device=base_mask.device, dtype=base_mask.dtype)
        return new_mask, info

    def _sample_reset_enabled(self) -> bool:
        if not self._use_ttrl_enabled():
            return False
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

    def generate(self, stage: Stage, prompt: Any, n: int, *, no_lora_adapter: bool = False) -> list[Generation]:
        del stage
        messages = _prompt_to_messages(prompt)
        effective_no_lora = bool(no_lora_adapter) or (not self._use_ttrl_enabled())
        outputs = self._generate_messages(
            [messages for _ in range(max(1, int(n)))],
            temperature=float(self.trainer.config.actor_rollout_ref.rollout.temperature),
            top_p=float(self.trainer.config.actor_rollout_ref.rollout.get("top_p", 1.0)),
            do_sample=True,
            max_new_tokens=int(self.pipeline_config.grpo.max_completion_length),
            no_lora_adapter=effective_no_lora,
        )
        k = max(1, len(outputs))
        return [Generation(text=str(text or ""), prior=1.0 / float(k), metadata={"backend": "verl"}) for text in outputs]

    def score_action_priors(self, stage: Stage, prompt: Any, candidates: list[str]) -> list[float]:
        del stage
        if not self._use_ttrl_enabled():
            n = len(candidates)
            return [1.0 / float(max(1, n))] * n
        if not self.pipeline_config.mcts.enable_prior or not candidates:
            n = len(candidates)
            return [1.0 / float(max(1, n))] * n

        try:
            messages = _prompt_to_messages(prompt)
            batch, response_mask = self._build_teacher_forcing_batch(messages, candidates)
            batch.meta_info["temperature"] = float(self.trainer.config.actor_rollout_ref.rollout.temperature)
            old_log_prob, _ = self.trainer._compute_old_log_prob(batch)
            log_probs = old_log_prob.batch["old_log_probs"]
            curr_scores = self._masked_mean(log_probs, response_mask, tail_k=self._prior_tail_tokens)

            ref_scores = None
            if self._prior_use_ref and self.trainer.use_reference_policy:
                ref_batch = self.trainer._compute_ref_log_prob(batch)
                ref_log_prob = ref_batch.batch["ref_log_prob"]
                ref_scores = self._masked_mean(ref_log_prob, response_mask, tail_k=self._prior_tail_tokens)

            raw_scores = curr_scores if ref_scores is None else (curr_scores - ref_scores)
            if self._prior_standardize and int(raw_scores.numel()) > 1:
                mean = raw_scores.mean()
                std = raw_scores.std(unbiased=False)
                std_value = float(std.detach().cpu().item()) if torch.is_tensor(std) else float(std)
                if std_value >= self._prior_min_std:
                    raw_scores = (raw_scores - mean) / std
                else:
                    raw_scores = raw_scores - mean
            raw_scores = raw_scores / max(1e-6, float(self._prior_temperature))
            probs = torch.softmax(raw_scores, dim=0)
            if not bool(torch.isfinite(probs).all()):
                n = len(candidates)
                return [1.0 / float(max(1, n))] * n
            priors = probs.detach().cpu().tolist()
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
        group_t0 = time.perf_counter()
        messages = _prompt_to_messages(prompt)
        k = max(1, int(config.num_generations))
        rollout_temperature = float(self.trainer.config.actor_rollout_ref.rollout.temperature)
        use_ttrl = self._use_ttrl_enabled()
        no_lora_adapter = not use_ttrl
        current_step = int(self.trainer.global_steps) + 1
        self.trainer.global_steps = current_step
        prompt_batch = self._make_prompt_batch(
            [messages],
            extra_infos=[{"task_id": self._current_task_id, "stage": stage.value}],
            no_lora_adapter=no_lora_adapter,
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
        rollout_infer_t0 = time.perf_counter()
        gen_output = self._rollout_generate(gen_batch_output)
        rollout_infer_sec = float(time.perf_counter() - rollout_infer_t0)

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

        kept_original_indices = list(range(len(completions)))
        filter_rollout_info: dict[str, Any] = {
            "enabled": bool(self._filter_rollout_enabled()),
            "applied": False,
            "reason": "disabled",
            "num_total": int(len(completions)),
            "num_kept": int(len(completions)),
            "num_dropped": 0,
        }
        if bool(self._filter_rollout_enabled()):
            raw_mask = None
            raw_stats: dict[str, Any] = {}
            if callable(batch_score):
                raw_mask = getattr(batch_score, "_last_filter_mask", None)
                raw_stats = dict(getattr(batch_score, "_last_filter_stats", {}) or {})

            if isinstance(raw_mask, (list, tuple)) and len(raw_mask) == len(completions):
                keep_mask = [bool(x) for x in raw_mask]
                keep_indices = [idx for idx, keep in enumerate(keep_mask) if keep]
                filter_rollout_info.update(
                    {
                        "enabled": True,
                        "applied": True,
                        "reason": "batch_filter_mask",
                        "num_total": int(len(keep_mask)),
                        "num_kept": int(len(keep_indices)),
                        "num_dropped": int(len(keep_mask) - len(keep_indices)),
                        "raw_filter_stats": raw_stats,
                    }
                )
                kept_original_indices = keep_indices
                if len(keep_indices) < len(completions):
                    if keep_indices:
                        batch = batch[keep_indices]
                        completions = [completions[i] for i in keep_indices]
                        rewards = [rewards[i] for i in keep_indices]
                    else:
                        completions = []
                        rewards = []
            else:
                filter_rollout_info.update(
                    {
                        "enabled": True,
                        "applied": False,
                        "reason": "missing_or_invalid_batch_filter_mask",
                        "raw_filter_stats": raw_stats,
                    }
                )

        if len(completions) == 0:
            report = {
                "updated": False,
                "backend": "verl",
                "stage": stage.value,
                "num_samples": 0,
                "use_ttrl": bool(use_ttrl),
                "reason": "all_rollouts_filtered_or_empty",
                "stage_update": {
                    "enabled": bool(self._stage_update_enabled()),
                    "applied": False,
                    "reason": "no_completion_after_filter",
                },
                "filter_rollout": filter_rollout_info,
                "timing": {
                    "rollout_vllm_infer_sec": float(rollout_infer_sec),
                    "old_log_prob_forward_sec": 0.0,
                    "actor_update_sec": 0.0,
                    "grpo_group_total_sec": float(time.perf_counter() - group_t0),
                },
            }
            return [], report

        stage_update_info: dict[str, Any] = {
            "enabled": bool(self._stage_update_enabled()),
            "applied": False,
            "reason": "disabled",
        }
        if use_ttrl and self._stage_update_enabled():
            base_response_mask = batch.batch.get("response_mask")
            stage_mask, stage_update_info = self._build_stage_update_response_mask(
                stage=stage,
                batch=batch,
                completions=completions,
            )
            if bool(stage_update_info.get("applied", False)) and base_response_mask is not None:
                # Compose masks explicitly: effective update region = response_mask AND stage_mask.
                # This is safer than direct replacement if stage_mask construction changes in future.
                effective_mask = (
                    base_response_mask.to(dtype=torch.bool) & stage_mask.to(dtype=torch.bool)
                ).to(dtype=base_response_mask.dtype)
                batch.batch["response_mask"] = effective_mask
                stage_update_info["mask_composition"] = "response_mask_and_stage_mask"
                stage_update_info["effective_tokens_after_and"] = int(effective_mask.sum().item())
            else:
                stage_update_info.setdefault("reason", "build_mask_not_applied")
        elif not use_ttrl and self._stage_update_enabled():
            stage_update_info = {
                "enabled": True,
                "applied": False,
                "reason": "use_ttrl_false_no_grpo_update",
            }

        reward_tensor = self._terminal_reward_tensor(batch.batch["response_mask"], rewards)
        batch.batch["token_level_scores"] = reward_tensor

        if not use_ttrl:
            generations: list[Generation] = []
            prior = 1.0 / float(max(1, len(completions)))
            for ridx, text in enumerate(completions):
                orig_idx = int(kept_original_indices[ridx]) if ridx < len(kept_original_indices) else int(ridx)
                generations.append(
                    Generation(
                        text=str(text or ""),
                        prior=prior,
                        metadata={
                            "backend": "verl",
                            "rollout_index": orig_idx,
                            "reward_total": float(rewards[ridx]),
                            "use_ttrl": False,
                        },
                    )
                )
            report = {
                "updated": False,
                "backend": "verl",
                "stage": stage.value,
                "num_samples": len(generations),
                "use_ttrl": False,
                "reason": "use_ttrl_disabled_no_grpo_no_lora",
                "stage_update": stage_update_info,
                "filter_rollout": filter_rollout_info,
                "timing": {
                    "rollout_vllm_infer_sec": float(rollout_infer_sec),
                    "old_log_prob_forward_sec": 0.0,
                    "actor_update_sec": 0.0,
                    "grpo_group_total_sec": float(time.perf_counter() - group_t0),
                },
            }
            return generations, report

        old_log_prob_t0 = time.perf_counter()
        old_log_prob, old_log_prob_mfu = self.trainer._compute_old_log_prob(batch)
        old_log_prob_forward_sec = float(time.perf_counter() - old_log_prob_t0)
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

        actor_batch_align_info: dict[str, Any] = {
            "mini_batch_size": int(self._actor_ppo_mini_batch_size()),
            "before": int(len(batch)),
            "after": int(len(batch)),
            "padding_size": 0,
            "applied": False,
            "reason": "already_divisible",
        }
        mini_batch_size = int(actor_batch_align_info["mini_batch_size"])
        if mini_batch_size > 1 and int(len(batch)) > 0:
            remainder = int(len(batch)) % mini_batch_size
            if remainder != 0:
                pad_size = int(mini_batch_size - remainder)
                actor_batch_align_info.update(
                    {
                        "padding_size": int(pad_size),
                        "reason": "pad_to_mini_batch_multiple",
                    }
                )
                try:
                    # FILTER_ROLLOUT can reduce valid samples (e.g., 3 out of k=8),
                    # while actor update requires batch_size % ppo_mini_batch_size == 0.
                    # Pad by repeating the first sample to satisfy trainer constraints.
                    batch.padding(padding_size=pad_size, padding_candidate="first")
                    actor_batch_align_info["applied"] = True
                    actor_batch_align_info["after"] = int(len(batch))
                except Exception as exc:  # noqa: BLE001
                    actor_batch_align_info.update(
                        {
                            "applied": False,
                            "after": int(len(batch)),
                            "reason": f"padding_failed:{type(exc).__name__}",
                            "error": str(exc),
                        }
                    )

        if mini_batch_size > 1 and int(len(batch)) % mini_batch_size != 0:
            generations: list[Generation] = []
            prior = 1.0 / float(max(1, len(completions)))
            for ridx, text in enumerate(completions):
                orig_idx = int(kept_original_indices[ridx]) if ridx < len(kept_original_indices) else int(ridx)
                generations.append(
                    Generation(
                        text=str(text or ""),
                        prior=prior,
                        metadata={
                            "backend": "verl",
                            "rollout_index": orig_idx,
                            "reward_total": float(rewards[ridx]),
                        },
                    )
                )
            report = {
                "updated": False,
                "backend": "verl",
                "stage": stage.value,
                "num_samples": len(generations),
                "use_ttrl": True,
                "reason": "actor_batch_size_not_divisible_after_filter",
                "stage_update": stage_update_info,
                "filter_rollout": filter_rollout_info,
                "actor_batch_align": actor_batch_align_info,
                "timing": {
                    "rollout_vllm_infer_sec": float(rollout_infer_sec),
                    "old_log_prob_forward_sec": float(old_log_prob_forward_sec),
                    "actor_update_sec": 0.0,
                    "grpo_group_total_sec": float(time.perf_counter() - group_t0),
                },
            }
            return generations, report

        actor_update_t0 = time.perf_counter()
        actor_output = self.trainer._update_actor(batch)
        actor_update_sec = float(time.perf_counter() - actor_update_t0)
        actor_metrics_raw = dict(actor_output.meta_info.get("metrics", {}) or {})
        actor_metrics = reduce_metrics(actor_metrics_raw) if actor_metrics_raw else {}
        actor_metrics["perf/mfu/actor_infer"] = old_log_prob_mfu

        self.trainer.checkpoint_manager.update_weights(current_step)

        generations: list[Generation] = []
        prior = 1.0 / float(max(1, len(completions)))
        for ridx, text in enumerate(completions):
            orig_idx = int(kept_original_indices[ridx]) if ridx < len(kept_original_indices) else int(ridx)
            generations.append(
                Generation(
                    text=str(text or ""),
                    prior=prior,
                    metadata={
                        "backend": "verl",
                        "rollout_index": orig_idx,
                        "reward_total": float(rewards[ridx]),
                    },
                )
            )

        report = {
            "updated": True,
            "backend": "verl",
            "stage": stage.value,
            "num_samples": len(generations),
            "use_ttrl": True,
            "stage_update": stage_update_info,
            "filter_rollout": filter_rollout_info,
            "actor_batch_align": actor_batch_align_info,
            "metrics": {**actor_metrics, **kl_metrics},
            "timing": {
                "rollout_vllm_infer_sec": float(rollout_infer_sec),
                "old_log_prob_forward_sec": float(old_log_prob_forward_sec),
                "actor_update_sec": float(actor_update_sec),
                "grpo_group_total_sec": float(time.perf_counter() - group_t0),
            },
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
            no_lora_adapter=not self._use_ttrl_enabled(),
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
        no_lora_adapter: bool = False,
    ) -> list[str | None]:
        if not messages_batch:
            return []
        prompt_batch = self._make_prompt_batch(
            messages_batch,
            extra_infos=[{"task_id": self._current_task_id, "kind": "aux"} for _ in messages_batch],
            no_lora_adapter=bool(no_lora_adapter),
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
        no_lora_adapter: bool = False,
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
                    "no_lora_adapter": bool(no_lora_adapter),
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
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor, tail_k: int = 0) -> torch.Tensor:
        mask_tensor = mask.to(dtype=values.dtype)
        if int(tail_k) > 0:
            seq_len = int(values.shape[-1])
            start = max(0, seq_len - int(tail_k))
            values = values[:, start:]
            mask_tensor = mask_tensor[:, start:]
        denom = torch.clamp(mask_tensor.sum(dim=-1), min=1.0)
        return (values * mask_tensor).sum(dim=-1) / denom


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
            sample_dir = _sample_run_dir(
                cfg.log_dir,
                str(sample.sample_id),
                run_tag=str(getattr(cfg, "run_tag", "") or ""),
            )
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
    pipeline_config = build_pipeline_config_from_verl_config(trainer.config)
    if bool(getattr(pipeline_config.grpo, "use_ttrl", True)):
        trainer.checkpoint_manager.update_weights(trainer.global_steps)
    else:
        # In pure-MCTS mode (use_ttrl=False), skip eager rollout weight sync.
        # vLLM is initialized from model path and does not require per-step policy syncing.
        print("[verl-or][info] use_ttrl=false: skip initial checkpoint_manager.update_weights")
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
    sample_run_enabled = bool(getattr(pipeline_config.dataset, "sample_run", False))
    sample_size = max(1, int(getattr(pipeline_config.dataset, "sample_size", 100) or 100))
    sample_seed = int(getattr(pipeline_config.dataset, "sample_seed", 0) or 0)

    skipped_dataset_load_errors = 0
    for dataset_path in dataset_paths:
        dataset_path_str = str(dataset_path or "").strip()
        dataset_name = Path(dataset_path_str).stem if dataset_path_str else "dataset"
        try:
            raw_samples = _resolve_raw_samples_for_path(trainer, pipeline_config, dataset_path_str)
        except Exception as exc:  # noqa: BLE001
            skipped_dataset_load_errors += 1
            print(
                f"[verl-or][dataset][skip] failed to load dataset={dataset_path_str or dataset_path!r}; "
                f"error={exc.__class__.__name__}: {exc}"
            )
            continue
        dataset_log_dir = model_log_root / _safe_path_component(dataset_name)
        ordered_samples = list(raw_samples)

        sampled_count = len(ordered_samples)
        sampled_seed = sample_seed if sample_run_enabled else None
        if sample_run_enabled and ordered_samples:
            rng_sample = np.random.default_rng(sampled_seed)
            if len(ordered_samples) > sample_size:
                sampled_indices = rng_sample.choice(len(ordered_samples), size=sample_size, replace=False).tolist()
                ordered_samples = [ordered_samples[int(i)] for i in sampled_indices]
            sampled_count = len(ordered_samples)

        if bool(trainer.config.data.shuffle):
            seed = trainer.config.data.get("seed")
            rng = np.random.default_rng(seed if seed is not None else 0)
            rng.shuffle(ordered_samples)

        run_tag = f"run_{sampled_seed}" if sample_run_enabled and sampled_seed is not None else ""
        skipped_completed = 0
        if bool(pipeline_config.dataset.resume_skip_completed):
            pending_samples = []
            for sample in ordered_samples:
                result_path = _sample_result_path(dataset_log_dir, str(sample.sample_id), run_tag=run_tag)
                if result_path.exists():
                    skipped_completed += 1
                    continue
                pending_samples.append(sample)
            ordered_samples = pending_samples

        total_pending_samples += len(ordered_samples)
        dataset_runs.append({
            "dataset_path": str(Path(dataset_path_str).resolve()),
            "dataset_name": dataset_name,
            "dataset_log_dir": str(dataset_log_dir),
            "raw_samples": raw_samples,
            "ordered_samples": ordered_samples,
            "skipped_completed": skipped_completed,
            "sample_run": bool(sample_run_enabled),
            "sample_seed": sampled_seed,
            "sample_size": int(sample_size),
            "sampled_count": int(sampled_count),
            "run_tag": str(run_tag),
        })
        print(
            f"[verl-or][dataset] rank={rank}/{world_size} total_samples={len(raw_samples)} "
            f"pending_samples={len(ordered_samples)} skipped_completed={skipped_completed} "
            f"dataset={Path(dataset_path_str).resolve()} log_root={dataset_log_dir.resolve()} "
            f"sample_run={sample_run_enabled} sampled_count={sampled_count} sample_seed={sampled_seed} run_tag={run_tag or '-'}"
        )

    if not dataset_runs:
        print(
            f"[verl-or][dataset] no valid datasets loaded (paths={len(dataset_paths)}, "
            f"load_errors={skipped_dataset_load_errors}), exiting."
        )
        return

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
        pipeline_config.run_tag = str(dataset_run.get("run_tag", "") or "")
        runner.config.log_dir = str(dataset_run["dataset_log_dir"])
        runner.config.run_tag = str(dataset_run.get("run_tag", "") or "")
        print(
            f"[verl-or][dataset-run] [{dataset_idx}/{len(dataset_runs)}] dataset={dataset_run['dataset_name']} "
            f"pending_samples={len(ordered_samples)} run_tag={dataset_run.get('run_tag') or '-'} "
            f"sample_run={dataset_run.get('sample_run')} sampled_count={dataset_run.get('sampled_count')}"
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





