from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ttrl_or.config import DatasetConfig, GRPOConfig
from ttrl_or.mapping import build_mapping_extractor
from ttrl_or.types import Generation, OptimizationTask, Stage, TrainingSample


class PolicyBackend(ABC):
    """Policy backend interface with an ephemeral LoRA lifecycle per task instance."""

    @abstractmethod
    def begin_episode(self, task: OptimizationTask) -> None:
        """Initialize temporary trainable adapters for one task instance."""

    @abstractmethod
    def end_episode(self) -> None:
        """Drop temporary adapters after one task instance is complete."""

    @abstractmethod
    def generate(self, stage: Stage, prompt: Any, n: int) -> list[Generation]:
        """Generate n candidates for a specific stage with prior probabilities."""

    def score_action_priors(self, stage: Stage, prompt: Any, candidates: list[str]) -> list[float]:
        """
        Optional hook: score current-stage child actions under the latest policy state.
        Returns normalized priors aligned with ``candidates``.
        """
        n = len(candidates)
        if n <= 0:
            return []
        return [1.0 / float(n)] * n

    @abstractmethod
    def grpo_update(self, samples: list[TrainingSample], config: GRPOConfig, stage: Stage) -> dict[str, Any]:
        """Apply one GRPO-style update step using stage samples."""
    def grpo_rollout_group(
        self,
        stage: Stage,
        prompt: Any,
        config: GRPOConfig,
        reward_callback,
    ) -> tuple[list[Generation], dict[str, Any]]:
        """
        Optional unified hook for internal rollout + policy update on one prompt group.
        Default fallback uses plain generation with no training update.
        """
        n = max(1, int(config.num_generations))
        generations = self.generate(stage, prompt, n)

        batch_score = getattr(reward_callback, "batch_score", None)
        if callable(batch_score):
            rewards = list(batch_score(prompt, [gen.text for gen in generations]))
            if len(rewards) != len(generations):
                rewards = rewards[: len(generations)] + [0.0] * max(0, len(generations) - len(rewards))
            for ridx, gen in enumerate(generations):
                gen.metadata["reward_total"] = float(rewards[ridx])
                gen.metadata["rollout_index"] = ridx
        else:
            for ridx, gen in enumerate(generations):
                gen.metadata["reward_total"] = float(reward_callback(prompt, gen.text, ridx))
                gen.metadata["rollout_index"] = ridx

        return generations, {
            "updated": False,
            "stage": stage.value,
            "num_samples": len(generations),
            "backend": type(self).__name__,
            "reason": "fallback_generation_no_trainer_update",
        }

    def prepare_task_context(self, task: OptimizationTask, dataset_config: DatasetConfig) -> dict[str, Any]:
        """
        Pre-stage hook: build mapping/perturbation context via a pluggable extractor.
        """
        extractor = build_mapping_extractor(dataset_config.mapping_extractor)
        result = extractor.extract(task=task, dataset_config=dataset_config, backend=self)

        task.instance = result.instance
        task.perturbation_map = result.perturbation_map
        return dict(result.metadata)

    def generate_mapping_from_description(
        self,
        description: str,
        dataset_config: DatasetConfig,
    ) -> dict[str, Any] | str | None:
        """
        Optional hook for LLM-based mapping extraction.
        Return a dict or JSON text, or None to fallback to rule extractor.
        """
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
        """
        Optional hook for non-policy auxiliary generation (e.g., dataset-level R3 prior planning).
        If prefer_vllm=True, backend should try vLLM path first when available.
        """
        return None

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
        """Batch auxiliary generation fallback: run one-by-one unless backend overrides it."""
        return [
            self.generate_auxiliary_text(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                prefer_vllm=prefer_vllm,
                vllm_mode=vllm_mode,
            )
            for prompt in prompts
        ]

    def generate_test_instances(self, task: OptimizationTask, k: int) -> list[dict[str, Any]]:
        """Optional: model-authored robustness tests (r3)."""
        return []


