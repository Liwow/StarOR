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


def _resolve_model_name_or_path(value: str) -> str:
    maybe_path = Path(value).expanduser()
    if maybe_path.exists():
        return str(maybe_path.resolve())
    return value


def _build_backend(config: PipelineConfig):
    backend_cfg = config.backend

    if backend_cfg.backend == "mock":
        return MockPolicyBackend(seed=backend_cfg.seed)

    if TRLPolicyBackend is None:
        detail = f" Import error: {TRL_IMPORT_ERROR}" if TRL_IMPORT_ERROR else ""
        raise RuntimeError(
            "TRL backend is unavailable. Install optional deps first: pip install trl datasets peft transformers"
            + detail
        )

    return TRLPolicyBackend(
        model_name_or_path=backend_cfg.model_name_or_path,
        temperature=backend_cfg.temperature,
        top_p=backend_cfg.top_p,
        max_new_tokens=backend_cfg.max_new_tokens,
        torch_dtype=backend_cfg.torch_dtype,
        trust_remote_code=backend_cfg.trust_remote_code,
        lora_r=backend_cfg.lora_r,
        lora_alpha=backend_cfg.lora_alpha,
        lora_dropout=backend_cfg.lora_dropout,
    )


def build_parser() -> argparse.ArgumentParser:
    defaults = PipelineConfig()

    parser = argparse.ArgumentParser(
        description="Run TTRL-OR pipeline on one optimization task or a raw JSONL dataset."
    )
    parser.add_argument("--task-text", type=str, default="")
    parser.add_argument("--task-file", type=str, default="")
    parser.add_argument("--instance-json", type=str, default="")
    parser.add_argument("--task-id", type=str, default="")

    parser.add_argument("--dataset-jsonl", type=str, default=defaults.dataset.jsonl_path)
    parser.add_argument("--dataset-start-index", type=int, default=defaults.dataset.start_index)
    parser.add_argument("--dataset-limit", type=int, default=defaults.dataset.limit)
    parser.add_argument("--dataset-max-numeric-features", type=int, default=defaults.dataset.max_numeric_features)
    parser.add_argument("--dataset-key-param-top-k", type=int, default=defaults.dataset.key_param_top_k)
    parser.add_argument(
        "--mapping-extractor",
        type=str,
        choices=["rule", "llm"],
        default=defaults.dataset.mapping_extractor,
    )
    parser.add_argument("--mapping-llm-max-new-tokens", type=int, default=defaults.dataset.mapping_llm_max_new_tokens)
    parser.add_argument("--mapping-llm-temperature", type=float, default=defaults.dataset.mapping_llm_temperature)
    parser.add_argument("--mapping-llm-top-p", type=float, default=defaults.dataset.mapping_llm_top_p)

    parser.add_argument("--backend", type=str, choices=["mock", "trl"], default=defaults.backend.backend)
    parser.add_argument("--model-name", type=str, default=defaults.backend.model_name_or_path)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--seed", type=int, default=defaults.backend.seed)

    parser.add_argument("--temperature", type=float, default=defaults.backend.temperature)
    parser.add_argument("--top-p", type=float, default=defaults.backend.top_p)
    parser.add_argument("--max-new-tokens", type=int, default=defaults.backend.max_new_tokens)
    parser.add_argument("--torch-dtype", type=str, default=defaults.backend.torch_dtype)
    parser.add_argument("--trust-remote-code", action="store_true", default=defaults.backend.trust_remote_code)

    parser.add_argument("--lora-r", type=int, default=defaults.backend.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=defaults.backend.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=defaults.backend.lora_dropout)

    parser.add_argument("--max-iterations", type=int, default=defaults.mcts.max_iterations)
    parser.add_argument("--c-puct", type=float, default=defaults.mcts.c_puct)
    parser.add_argument("--mcts-stop-on-reward-one", action="store_true", default=defaults.mcts.stop_on_reward_one)

    parser.add_argument("--robustness-cases", type=int, default=defaults.reward.robustness_cases)
    parser.add_argument("--code-timeout-sec", type=int, default=defaults.reward.code_timeout_sec)
    parser.add_argument("--global-consensus-min-pool", type=int, default=defaults.reward.global_consensus_min_pool)
    parser.add_argument("--global-consensus-rel-tol", type=float, default=defaults.reward.global_consensus_rel_tol)

    r3_group = parser.add_mutually_exclusive_group()
    r3_group.add_argument("--enable-r3-reward", dest="enable_r3_reward", action="store_true")
    r3_group.add_argument("--disable-r3-reward", dest="enable_r3_reward", action="store_false")
    parser.set_defaults(enable_r3_reward=defaults.reward.enable_r3_reward)

    parser.add_argument("--grpo-lr", type=float, default=defaults.grpo.learning_rate)
    parser.add_argument("--grpo-kl", type=float, default=defaults.grpo.kl_coef)
    parser.add_argument("--grpo-batch-size", type=int, default=defaults.grpo.per_device_train_batch_size)
    parser.add_argument("--grpo-grad-accum", type=int, default=defaults.grpo.gradient_accumulation_steps)
    parser.add_argument("--grpo-num-generations", type=int, default=defaults.grpo.num_generations)
    parser.add_argument("--grpo-generation-batch-size", type=int, default=defaults.grpo.generation_batch_size)
    parser.add_argument("--grpo-max-prompt-len", type=int, default=defaults.grpo.max_prompt_length)
    parser.add_argument("--grpo-max-completion-len", type=int, default=defaults.grpo.max_completion_length)

    parser.add_argument("--grpo-use-vllm", action="store_true", default=defaults.grpo.use_vllm)
    parser.add_argument("--grpo-vllm-mode", type=str, default=defaults.grpo.vllm_mode)
    parser.add_argument(
        "--grpo-vllm-gpu-memory-utilization",
        type=float,
        default=defaults.grpo.vllm_gpu_memory_utilization,
    )
    parser.add_argument("--grpo-vllm-tensor-parallel-size", type=int, default=defaults.grpo.vllm_tensor_parallel_size)
    parser.add_argument("--grpo-vllm-max-model-len", type=int, default=defaults.grpo.vllm_max_model_len)

    parser.add_argument("--log-dir", type=str, default=defaults.log_dir)
    parser.add_argument("--no-save-logs", action="store_true", default=not defaults.save_logs)

    parser.add_argument("--out", type=str, default="")
    return parser


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig()

    config.log_dir = args.log_dir
    config.save_logs = not args.no_save_logs

    config.mcts.max_iterations = args.max_iterations
    config.mcts.c_puct = args.c_puct
    config.mcts.stop_on_reward_one = args.mcts_stop_on_reward_one

    config.reward.robustness_cases = args.robustness_cases
    config.reward.code_timeout_sec = args.code_timeout_sec
    config.reward.global_consensus_min_pool = args.global_consensus_min_pool
    config.reward.global_consensus_rel_tol = args.global_consensus_rel_tol
    config.reward.enable_r3_reward = args.enable_r3_reward

    config.grpo.learning_rate = args.grpo_lr
    config.grpo.kl_coef = args.grpo_kl
    config.grpo.per_device_train_batch_size = args.grpo_batch_size
    config.grpo.gradient_accumulation_steps = args.grpo_grad_accum
    config.grpo.num_generations = args.grpo_num_generations
    config.grpo.generation_batch_size = args.grpo_generation_batch_size
    config.grpo.max_prompt_length = args.grpo_max_prompt_len
    config.grpo.max_completion_length = args.grpo_max_completion_len
    config.grpo.use_vllm = args.grpo_use_vllm
    config.grpo.vllm_mode = args.grpo_vllm_mode
    config.grpo.vllm_gpu_memory_utilization = args.grpo_vllm_gpu_memory_utilization
    config.grpo.vllm_tensor_parallel_size = args.grpo_vllm_tensor_parallel_size
    config.grpo.vllm_max_model_len = args.grpo_vllm_max_model_len

    config.dataset.jsonl_path = args.dataset_jsonl
    config.dataset.start_index = args.dataset_start_index
    config.dataset.limit = args.dataset_limit
    config.dataset.max_numeric_features = args.dataset_max_numeric_features
    config.dataset.key_param_top_k = args.dataset_key_param_top_k
    config.dataset.mapping_extractor = args.mapping_extractor
    config.dataset.mapping_llm_max_new_tokens = args.mapping_llm_max_new_tokens
    config.dataset.mapping_llm_temperature = args.mapping_llm_temperature
    config.dataset.mapping_llm_top_p = args.mapping_llm_top_p

    model_value = args.model_path if args.model_path else args.model_name
    config.backend.backend = args.backend
    config.backend.model_name_or_path = _resolve_model_name_or_path(model_value)
    config.backend.seed = args.seed
    config.backend.temperature = args.temperature
    config.backend.top_p = args.top_p
    config.backend.max_new_tokens = args.max_new_tokens
    config.backend.torch_dtype = args.torch_dtype
    config.backend.trust_remote_code = args.trust_remote_code
    config.backend.lora_r = args.lora_r
    config.backend.lora_alpha = args.lora_alpha
    config.backend.lora_dropout = args.lora_dropout

    return config


def _run_single(args: argparse.Namespace, runner: TTRLORRunner) -> dict:
    task_text = _load_text(args)
    instance = _load_instance(args.instance_json) if args.instance_json else None
    result = runner.run_from_text(task_text, instance=instance, task_id=args.task_id or None)

    output = {
        "mode": "single",
        "task_id": result.task_id,
        "backend": runner.config.backend.backend,
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
        "task_context": (result.trace.task_context if result.trace else {}),
        "perturbation_map": (result.trace.perturbation_map if result.trace else {}),
    }

    if result.best_trajectory:
        print("\n# Best code\n")
        print(result.best_trajectory.code)

    return output


def _run_dataset(args: argparse.Namespace, runner: TTRLORRunner) -> dict:
    dataset_cfg = runner.config.dataset
    dataset_path = dataset_cfg.jsonl_path
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
                "task_context": (result.trace.task_context if result.trace else {}),
            }
        )
        print(f"[{idx}/{len(samples)}] {sample.sample_id} best_reward={best_reward}")

    return {
        "mode": "dataset",
        "dataset_jsonl": str(Path(dataset_path).resolve()),
        "backend": runner.config.backend.backend,
        "num_samples": len(samples),
        "runs": runs,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = _build_config(args)
    backend = _build_backend(config)
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
