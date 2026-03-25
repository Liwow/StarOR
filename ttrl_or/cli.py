from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttrl_or.config import PipelineConfig
from ttrl_or.dataset import load_jsonl_dataset
from ttrl_or.model import MockPolicyBackend, TRLPolicyBackend, TRL_IMPORT_ERROR
from ttrl_or.pipeline import TTRLORRunner


def _load_text(args: argparse.Namespace) -> str:
    if args.task_text:
        return args.task_text
    if args.task_file:
        return Path(args.task_file).read_text(encoding="utf-8-sig")
    raise ValueError("Please provide --task-text or --task-file.")


def _load_instance(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _build_backend(args: argparse.Namespace):
    if args.backend == "mock":
        return MockPolicyBackend(seed=args.seed)

    if TRLPolicyBackend is None:
        detail = f" Import error: {TRL_IMPORT_ERROR}" if TRL_IMPORT_ERROR else ""
        raise RuntimeError(
            "TRL backend is unavailable. Install optional deps first: pip install trl datasets peft transformers"
            + detail
        )

    return TRLPolicyBackend(
        model_name_or_path=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TTRL-OR pipeline on one optimization task or a raw JSONL dataset.")
    parser.add_argument("--task-text", type=str, default="")
    parser.add_argument("--task-file", type=str, default="")
    parser.add_argument("--instance-json", type=str, default="")
    parser.add_argument("--task-id", type=str, default="")

    parser.add_argument("--dataset-jsonl", type=str, default="")
    parser.add_argument("--dataset-start-index", type=int, default=0)
    parser.add_argument("--dataset-limit", type=int, default=0)
    parser.add_argument("--dataset-max-numeric-features", type=int, default=256)
    parser.add_argument("--dataset-key-param-top-k", type=int, default=16)

    parser.add_argument("--backend", type=str, choices=["mock", "trl"], default="mock")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--expand-per-node", type=int, default=3)
    parser.add_argument("--simulations-per-node", type=int, default=4)
    parser.add_argument("--rollout-k", type=int, default=2)
    parser.add_argument("--max-nodes-per-stage", type=int, default=12)

    parser.add_argument("--consensus-window", type=int, default=64)
    parser.add_argument("--robustness-cases", type=int, default=3)
    parser.add_argument("--disable-perturb-reward", action="store_true")

    parser.add_argument("--grpo-lr", type=float, default=3e-5)
    parser.add_argument("--grpo-clip", type=float, default=0.2)
    parser.add_argument("--grpo-kl", type=float, default=0.0)
    parser.add_argument("--grpo-batch-size", type=int, default=1)
    parser.add_argument("--grpo-grad-accum", type=int, default=1)
    parser.add_argument("--grpo-num-generations", type=int, default=2)
    parser.add_argument("--grpo-max-prompt-len", type=int, default=1024)
    parser.add_argument("--grpo-max-completion-len", type=int, default=256)
    parser.add_argument("--grpo-max-steps", type=int, default=1)

    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--no-save-logs", action="store_true")

    parser.add_argument("--out", type=str, default="")
    return parser


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig()
    config.group_size = args.group_size
    config.mcts.expand_per_node = args.expand_per_node
    config.mcts.simulations_per_node = args.simulations_per_node
    config.mcts.rollout_k = args.rollout_k
    config.mcts.max_nodes_per_stage = args.max_nodes_per_stage

    config.reward.local_consensus_window = args.consensus_window
    config.reward.robustness_cases = args.robustness_cases
    config.reward.enable_perturb_reward = not args.disable_perturb_reward

    config.grpo.learning_rate = args.grpo_lr
    config.grpo.clip_range = args.grpo_clip
    config.grpo.kl_coef = args.grpo_kl
    config.grpo.per_device_train_batch_size = args.grpo_batch_size
    config.grpo.gradient_accumulation_steps = args.grpo_grad_accum
    config.grpo.num_generations = args.grpo_num_generations
    config.grpo.max_prompt_length = args.grpo_max_prompt_len
    config.grpo.max_completion_length = args.grpo_max_completion_len
    config.grpo.max_steps = args.grpo_max_steps

    config.dataset.jsonl_path = args.dataset_jsonl
    config.dataset.start_index = args.dataset_start_index
    config.dataset.limit = args.dataset_limit
    config.dataset.max_numeric_features = args.dataset_max_numeric_features
    config.dataset.key_param_top_k = args.dataset_key_param_top_k

    config.log_dir = args.log_dir
    config.save_logs = not args.no_save_logs
    return config


def _run_single(args: argparse.Namespace, runner: TTRLORRunner) -> dict:
    task_text = _load_text(args)
    instance = _load_instance(args.instance_json) if args.instance_json else None
    result = runner.run_from_text(task_text, instance=instance, task_id=args.task_id or None)

    output = {
        "mode": "single",
        "task_id": result.task_id,
        "backend": args.backend,
        "stage_reports": result.stage_reports,
        "best_reward": result.best_trajectory.reward.total if result.best_trajectory and result.best_trajectory.reward else None,
        "best_reward_components": (
            {
                "r1": result.best_trajectory.reward.r1,
                "r2": result.best_trajectory.reward.r2,
                "r3": result.best_trajectory.reward.r3,
                "total": result.best_trajectory.reward.total,
            }
            if result.best_trajectory and result.best_trajectory.reward
            else None
        ),
        "num_trajectories": len(result.trajectories),
        "artifacts": (result.trace.artifacts if result.trace else {}),
    }

    if result.best_trajectory:
        print("\n# Best code\n")
        print(result.best_trajectory.code)

    return output


def _run_dataset(args: argparse.Namespace, runner: TTRLORRunner) -> dict:
    dataset_cfg = runner.config.dataset
    dataset_path = dataset_cfg.jsonl_path or args.dataset_jsonl
    if not dataset_path:
        raise ValueError("Dataset mode requires --dataset-jsonl.")

    all_samples = load_jsonl_dataset(dataset_path)
    start = max(0, dataset_cfg.start_index)
    if dataset_cfg.limit > 0:
        end = start + dataset_cfg.limit
        samples = all_samples[start:end]
    else:
        samples = all_samples[start:]

    runs: list[dict] = []
    for idx, sample in enumerate(samples, start=1):
        result = runner.run_from_text(
            description=sample.question,
            instance=None,
            task_id=sample.sample_id,
        )

        best_reward = result.best_trajectory.reward.total if result.best_trajectory and result.best_trajectory.reward else None
        runs.append(
            {
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "param_mode": sample.param_mode,
                "reference_answer": sample.answer,
                "best_reward": best_reward,
                "best_reward_components": (
                    {
                        "r1": result.best_trajectory.reward.r1,
                        "r2": result.best_trajectory.reward.r2,
                        "r3": result.best_trajectory.reward.r3,
                        "total": result.best_trajectory.reward.total,
                    }
                    if result.best_trajectory and result.best_trajectory.reward
                    else None
                ),
                "artifacts": (result.trace.artifacts if result.trace else {}),
            }
        )
        print(f"[{idx}/{len(samples)}] {sample.sample_id} best_reward={best_reward}")

    return {
        "mode": "dataset",
        "dataset_jsonl": str(Path(dataset_path).resolve()),
        "backend": args.backend,
        "num_samples": len(samples),
        "runs": runs,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = _build_config(args)
    backend = _build_backend(args)
    runner = TTRLORRunner(backend=backend, config=config)

    if config.dataset.jsonl_path:
        output = _run_dataset(args, runner)
    else:
        output = _run_single(args, runner)

    if args.out:
        Path(args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))

    return 0
