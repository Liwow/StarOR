from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ttrl_or.config import DatasetConfig, GRPOConfig
from ttrl_or.dataset import build_instance_from_question
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
    def generate(self, stage: Stage, prompt: str, n: int) -> list[Generation]:
        """Generate n candidates for a specific stage with prior probabilities."""

    @abstractmethod
    def grpo_update(self, samples: list[TrainingSample], config: GRPOConfig, stage: Stage) -> dict[str, Any]:
        """Apply one GRPO-style update step using stage samples."""

    def prepare_task_context(self, task: OptimizationTask, dataset_config: DatasetConfig) -> dict[str, Any]:
        """
        Pre-stage hook: derive numeric mapping and perturbation map from description
        when explicit instance is not provided.
        """
        used_description_extraction = False
        if not task.instance:
            task.instance = build_instance_from_question(
                task.description,
                max_numeric_features=dataset_config.max_numeric_features,
                key_param_top_k=dataset_config.key_param_top_k,
            )
            used_description_extraction = True

        from ttrl_or.reward.perturbation import build_perturbation_map

        task.perturbation_map = build_perturbation_map(task.instance)
        return {
            "used_description_extraction": used_description_extraction,
            "num_instance_keys": len(task.instance),
            "num_numeric_keys": int(task.perturbation_map.get("num_numeric_keys", 0)),
            "num_focus_keys": int(task.perturbation_map.get("num_focus_keys", 0)),
            "focus_keys": list(task.perturbation_map.get("focus_keys", []))[:16],
        }

    def generate_test_instances(self, task: OptimizationTask, k: int) -> list[dict[str, Any]]:
        """Optional: model-authored robustness tests (r3)."""
        return []
