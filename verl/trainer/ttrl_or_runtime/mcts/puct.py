from __future__ import annotations

import math

from verl.trainer.ttrl_or_runtime.mcts.node import SearchNode


class PUCTSelector:
    def __init__(self, c_puct: float = 1.4) -> None:
        self.c_puct = c_puct

    def score(self, parent: SearchNode, child: SearchNode) -> float:
        parent_visits = max(1, parent.visits)
        prior = max(1e-6, child.prior)
        exploration = self.c_puct * prior * math.sqrt(parent_visits) / (1 + child.visits)
        return child.q_value + exploration

    def select(self, parent: SearchNode) -> SearchNode:
        if not parent.children:
            raise ValueError("Cannot select from an empty child list.")
        return max(parent.children, key=lambda child: self.score(parent, child))

