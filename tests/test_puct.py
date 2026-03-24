from ttrl_or.mcts.node import SearchNode
from ttrl_or.mcts.puct import PUCTSelector
from ttrl_or.types import Stage


def test_puct_prefers_better_ucb_score():
    parent = SearchNode(stage=None, text="root", prior=1.0)
    parent.visits = 10

    child_a = SearchNode(stage=Stage.SCHEMA, text="a", prior=0.2, parent=parent)
    child_a.visits = 10
    child_a.value_sum = 8.0  # q=0.8

    child_b = SearchNode(stage=Stage.SCHEMA, text="b", prior=0.9, parent=parent)
    child_b.visits = 1
    child_b.value_sum = 0.4  # q=0.4, but high exploration bonus

    parent.children = [child_a, child_b]

    selector = PUCTSelector(c_puct=1.4)
    picked = selector.select(parent)
    assert picked is child_b
