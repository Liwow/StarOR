from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from verl.trainer.ttrl_or_runtime.types import Stage, Trajectory


@dataclass(slots=True)
class SearchNode:
    stage: Stage | None
    text: str
    prior: float
    parent: SearchNode | None = None
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    children: list[SearchNode] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    prompt: str = ""

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def add_child(self, child: SearchNode) -> None:
        self.children.append(child)

    def update(self, reward: float) -> None:
        self.visits += 1
        self.value_sum += reward

    def to_partial_trajectory(self) -> Trajectory:
        cur: SearchNode | None = self
        outputs: dict[Stage, str] = {}
        priors: dict[Stage, float] = {}
        while cur is not None:
            if cur.stage is not None:
                outputs[cur.stage] = cur.text
                priors[cur.stage] = cur.prior
            cur = cur.parent
        return Trajectory(
            trajectory_id=self.node_id,
            outputs=outputs,
            priors=priors,
            metadata={"source_node": self.node_id},
        )

