from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ttrl_or.config import GRPOConfig
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

    def generate_test_instances(self, task: OptimizationTask, k: int) -> list[dict[str, Any]]:
        """Optional: model-authored robustness tests (r3)."""
        return []
