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
    def generate(self, stage: Stage, prompt: str, n: int) -> list[Generation]:
        """Generate n candidates for a specific stage with prior probabilities."""

    @abstractmethod
    def grpo_update(self, samples: list[TrainingSample], config: GRPOConfig, stage: Stage) -> dict[str, Any]:
        """Apply one GRPO-style update step using stage samples."""

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

    def generate_test_instances(self, task: OptimizationTask, k: int) -> list[dict[str, Any]]:
        """Optional: model-authored robustness tests (r3)."""
        return []
