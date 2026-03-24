from __future__ import annotations

from abc import ABC, abstractmethod

from ttrl_or.types import RewardBreakdown, Trajectory


class RewardCalculator(ABC):
    @abstractmethod
    def provisional_reward(self, trajectory: Trajectory, explored: list[Trajectory]) -> RewardBreakdown:
        ...

    @abstractmethod
    def finalize_group(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        ...
