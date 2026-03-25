from __future__ import annotations

import gc
import inspect
import math
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ttrl_or.config import GRPOConfig
from ttrl_or.model.backend import PolicyBackend
from ttrl_or.types import Generation, OptimizationTask, Stage, TrainingSample


@dataclass(slots=True)
class TRLPolicyBackend(PolicyBackend):
    """
    Real training backend using Hugging Face + TRL GRPOTrainer.

    Notes:
    - Requires optional deps: trl, peft, datasets, transformers.
    - Uses temporary LoRA adapters for each task episode.
    """

    model_name_or_path: str
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 256
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    _tokenizer: Any = field(init=False, default=None, repr=False)
    _model: Any = field(init=False, default=None, repr=False)
    _episode_key: str = field(init=False, default="", repr=False)

    def begin_episode(self, task: OptimizationTask) -> None:
        self._episode_key = task.task_id
        self._load_fresh_episode_model()

    def end_episode(self) -> None:
        self._episode_key = ""
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, stage: Stage, prompt: str, n: int) -> list[Generation]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Episode not initialized. Call begin_episode() before generate().")

        self._model.eval()
        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = self._infer_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=n,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        seqs = output.sequences
        scores = output.scores

        generations: list[Generation] = []
        for idx in range(seqs.shape[0]):
            full_ids = seqs[idx]
            comp_ids = full_ids[prompt_len:]
            completion = self._tokenizer.decode(comp_ids, skip_special_tokens=True)
            prior = self._sequence_prior(idx, comp_ids, scores)
            generations.append(
                Generation(
                    text=completion,
                    prior=prior,
                    metadata={"backend": "trl", "stage": stage.value},
                )
            )
        return generations

    def grpo_update(self, samples: list[TrainingSample], config: GRPOConfig, stage: Stage) -> dict[str, Any]:
        if not samples:
            return {"updated": False, "stage": stage.value, "num_samples": 0, "backend": "trl"}

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Episode not initialized. Call begin_episode() before grpo_update().")

        trl = _import_trl()
        datasets = _import_datasets()

        prompt_rows = [{"prompt": sample.prompt} for sample in samples]
        train_dataset = datasets.Dataset.from_list(prompt_rows)

        reward_lookup_exact, reward_lookup_prompt = _build_reward_lookup(samples)

        def reward_func(prompts, completions, **kwargs):
            rewards: list[float] = []
            for prompt_text, completion_text in zip(prompts, completions, strict=False):
                p = _normalize_text(prompt_text)
                c = _normalize_text(completion_text)
                exact_key = (p, c)
                if exact_key in reward_lookup_exact:
                    rewards.append(reward_lookup_exact[exact_key])
                else:
                    rewards.append(reward_lookup_prompt.get(p, 0.0))
            return rewards

        output_dir = self._stage_output_dir(stage)
        trl_args = self._build_trl_grpo_args(config, output_dir)

        trainer_kwargs = {
            "model": self._model,
            "args": trl_args,
            "reward_funcs": reward_func,
            "train_dataset": train_dataset,
        }
        trainer_sig = inspect.signature(trl.GRPOTrainer.__init__)
        if "processing_class" in trainer_sig.parameters:
            trainer_kwargs["processing_class"] = self._tokenizer
        elif "tokenizer" in trainer_sig.parameters:
            trainer_kwargs["tokenizer"] = self._tokenizer

        trainer = trl.GRPOTrainer(**trainer_kwargs)
        train_result = trainer.train()
        metrics = dict(getattr(train_result, "metrics", {}) or {})

        report: dict[str, Any] = {
            "updated": True,
            "stage": stage.value,
            "num_samples": len(samples),
            "backend": "trl",
        }
        if "train_loss" in metrics:
            report["train_loss"] = float(metrics["train_loss"])
        if "train_runtime" in metrics:
            report["train_runtime"] = float(metrics["train_runtime"])
        if "train_steps_per_second" in metrics:
            report["train_steps_per_second"] = float(metrics["train_steps_per_second"])
        return report

    def generate_test_instances(self, task: OptimizationTask, k: int) -> list[dict[str, Any]]:
        from ttrl_or.reward.perturbation import generate_perturbed_instances_from_map

        tests = generate_perturbed_instances_from_map(task.instance, task.perturbation_map, k)
        if tests:
            return tests

        fallback: list[dict[str, Any]] = []
        for i in range(k):
            case: dict[str, Any] = {}
            scale = 1.0 + 0.1 * ((i % 3) - 1)
            for key, value in task.instance.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    case[key] = round(float(value) * scale, 6)
                else:
                    case[key] = value
            fallback.append(case)
        return fallback

    def _build_trl_grpo_args(self, config: GRPOConfig, output_dir: str):
        trl = _import_trl()

        kwargs = {
            "output_dir": output_dir,
            "learning_rate": config.learning_rate,
            "beta": config.kl_coef,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "num_generations": config.num_generations,
            "max_prompt_length": config.max_prompt_length,
            "max_completion_length": config.max_completion_length,
            "max_steps": config.max_steps,
            "report_to": [],
            "save_strategy": "no",
            "logging_steps": 1,
        }

        init_sig = inspect.signature(trl.GRPOConfig.__init__)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in init_sig.parameters}
        return trl.GRPOConfig(**filtered_kwargs)

    def _load_fresh_episode_model(self) -> None:
        transformers = _import_transformers()
        peft = _import_peft()

        dtype = self._resolve_torch_dtype()
        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": self.trust_remote_code,
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"

        base_model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **model_kwargs)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name_or_path, trust_remote_code=self.trust_remote_code
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        lora_config = peft.LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=list(self.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = peft.get_peft_model(base_model, lora_config)
        model.train()

        self._model = model
        self._tokenizer = tokenizer

    def _resolve_torch_dtype(self):
        if self.torch_dtype == "auto":
            return "auto"
        if not hasattr(torch, self.torch_dtype):
            raise ValueError(f"Unknown torch dtype: {self.torch_dtype}")
        return getattr(torch, self.torch_dtype)

    def _infer_device(self) -> torch.device:
        if hasattr(self._model, "device"):
            return self._model.device
        return next(self._model.parameters()).device

    def _sequence_prior(self, seq_index: int, completion_ids: torch.Tensor, scores: list[torch.Tensor]) -> float:
        if not scores:
            return 1e-6

        log_probs: list[float] = []
        eos_id = self._tokenizer.eos_token_id
        pad_id = self._tokenizer.pad_token_id

        max_steps = min(len(scores), completion_ids.shape[0])
        for step in range(max_steps):
            token_id = int(completion_ids[step].item())
            if pad_id is not None and token_id == pad_id:
                break

            step_scores = scores[step][seq_index]
            step_log_probs = torch.log_softmax(step_scores, dim=-1)
            log_probs.append(float(step_log_probs[token_id].item()))

            if eos_id is not None and token_id == eos_id:
                break

        if not log_probs:
            return 1e-6

        avg_log_prob = sum(log_probs) / len(log_probs)
        return max(1e-6, float(math.exp(avg_log_prob)))

    def _stage_output_dir(self, stage: Stage) -> str:
        base = Path(tempfile.gettempdir()) / "ttrl_or_trl" / (self._episode_key or "unknown") / stage.value
        base.mkdir(parents=True, exist_ok=True)
        return str(base)


def _normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, dict):
                content = item.get("content", "")
                chunks.append(str(content))
            else:
                chunks.append(str(item))
        return "\n".join(chunks).strip()

    return str(value).strip()


def _build_reward_lookup(samples: list[TrainingSample]) -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    exact: dict[tuple[str, str], float] = {}
    prompt_bucket: dict[str, list[float]] = defaultdict(list)

    for sample in samples:
        p = _normalize_text(sample.prompt)
        c = _normalize_text(sample.completion)
        exact[(p, c)] = float(sample.reward)
        prompt_bucket[p].append(float(sample.reward))

    prompt_mean = {k: sum(v) / len(v) for k, v in prompt_bucket.items()}
    return exact, prompt_mean


def _import_transformers():
    try:
        import transformers  # type: ignore

        return transformers
    except Exception as exc:
        raise RuntimeError(
            "TRL backend requires a working `transformers` install. "
            "Please install or repair it with: pip install transformers"
        ) from exc


def _import_trl():
    try:
        import trl  # type: ignore

        return trl
    except ImportError as exc:
        raise RuntimeError(
            "TRL backend requires `trl`. Install with: pip install trl datasets peft"
        ) from exc


def _import_peft():
    try:
        import peft  # type: ignore

        return peft
    except ImportError as exc:
        raise RuntimeError("TRL backend requires `peft`. Install with: pip install peft") from exc


def _import_datasets():
    try:
        import datasets  # type: ignore

        return datasets
    except ImportError as exc:
        raise RuntimeError("TRL backend requires `datasets`. Install with: pip install datasets") from exc


