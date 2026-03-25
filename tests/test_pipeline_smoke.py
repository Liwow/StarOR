import json

from ttrl_or.config import PipelineConfig
from ttrl_or.model import MockPolicyBackend
from ttrl_or.pipeline import TTRLORRunner
from ttrl_or.types import Stage


def test_pipeline_smoke_runs_end_to_end():
    config = PipelineConfig()
    config.group_size = 2
    config.mcts.expand_per_node = 1
    config.mcts.simulations_per_node = 1
    config.mcts.rollout_k = 1
    config.mcts.max_nodes_per_stage = 2
    config.save_logs = False

    backend = MockPolicyBackend(seed=11)
    runner = TTRLORRunner(backend=backend, config=config)

    result = runner.run_from_text(
        description="Choose products under capacity to maximize total value.",
        instance={"cost": 10, "value": 15, "capacity": 7},
        task_id="smoke-1",
    )

    assert result.best_trajectory is not None
    assert len(result.trajectories) > 0
    assert Stage.SCHEMA.value in result.stage_reports
    assert Stage.SET_PARAM_VAR.value in result.stage_reports
    assert Stage.OBJ_CONS.value in result.stage_reports
    assert Stage.CODE.value in result.stage_reports
    assert "final_selection" in result.stage_reports
    assert "final_group" not in result.stage_reports


def test_pipeline_can_run_with_description_only_input():
    config = PipelineConfig()
    config.group_size = 2
    config.mcts.expand_per_node = 1
    config.mcts.simulations_per_node = 1
    config.mcts.rollout_k = 1
    config.mcts.max_nodes_per_stage = 2
    config.save_logs = False
    config.reward.enable_perturb_reward = True

    backend = MockPolicyBackend(seed=13)
    runner = TTRLORRunner(backend=backend, config=config)

    result = runner.run_from_text(
        description=(
            "A factory has budget 1000 and capacity 500. "
            "Minimize cost with at least 200 units demand."
        ),
        task_id="smoke-desc-only",
    )

    assert result.best_trajectory is not None
    assert result.trace is not None
    assert result.trace.task_context.get("used_description_extraction") is True
    assert result.trace.task_context.get("num_numeric_keys", 0) >= 1


class _MockLLMMappingBackend(MockPolicyBackend):
    def generate_mapping_from_description(self, description, dataset_config):
        return json.dumps(
            {
                "instance": {"budget": 1000, "capacity": 500, "demand": 200},
                "key_param_keys": ["budget", "capacity"],
            },
            ensure_ascii=False,
        )


def test_pipeline_llm_mapping_extractor_uses_backend_hook():
    config = PipelineConfig()
    config.group_size = 2
    config.mcts.expand_per_node = 1
    config.mcts.simulations_per_node = 1
    config.mcts.rollout_k = 1
    config.mcts.max_nodes_per_stage = 2
    config.save_logs = False
    config.reward.enable_perturb_reward = True
    config.dataset.mapping_extractor = "llm"

    backend = _MockLLMMappingBackend(seed=21)
    runner = TTRLORRunner(backend=backend, config=config)

    result = runner.run_from_text(
        description="Factory planning with budget and demand constraints.",
        task_id="smoke-llm-mapping",
    )

    assert result.best_trajectory is not None
    assert result.trace is not None
    assert result.trace.task_context.get("extractor") == "llm"
    assert result.trace.task_context.get("llm_used") is True
    assert "budget" in result.trace.instance
    assert "capacity" in result.trace.instance
