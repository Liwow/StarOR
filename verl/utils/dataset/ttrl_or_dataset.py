from __future__ import annotations

from copy import deepcopy
from typing import Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.trainer.ttrl_or_runtime.dataset.loader import RawTaskSample, load_raw_task_dataset


class TTRLORDataset(Dataset):
    """
    verl-compatible wrapper for raw TTRL-OR JSONL samples.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ) -> None:
        del tokenizer, processor

        if isinstance(data_files, (list, tuple)):
            if not data_files:
                raise ValueError("TTRLORDataset requires at least one data file.")
            data_file = str(data_files[0])
        else:
            data_file = str(data_files)

        dataset_limit = int(max_samples) if int(max_samples) > 0 else None
        self.raw_samples: list[RawTaskSample] = load_raw_task_dataset(
            data_file,
            start_index=0,
            limit=dataset_limit,
            max_numeric_features=int(config.get("ttrl_or_max_numeric_features", 256)),
            key_param_top_k=int(config.get("ttrl_or_key_param_top_k", 16)),
        )
        self.data_file = data_file
        self.config = config

    def __len__(self) -> int:
        return len(self.raw_samples)

    def __getitem__(self, item: int) -> dict:
        sample = self.raw_samples[item]
        raw_payload = deepcopy(sample.raw)
        if isinstance(raw_payload, dict):
            raw_payload.pop("answer", None)
            raw_payload.pop("en_answer", None)
        extra_info = {
            "index": int(item),
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "question": sample.question,
            "param_mode": sample.param_mode,
            "tables": deepcopy(sample.tables),
            "inline_numbers": deepcopy(sample.inline_numbers),
            "instance": deepcopy(sample.instance),
            "raw": raw_payload,
        }
        return {
            "data_source": sample.dataset,
            "reward_model": {"style": "rule"},
            "extra_info": extra_info,
            "raw_prompt": [{"role": "user", "content": sample.question}],
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "index": int(item),
            "tools_kwargs": {},
            "interaction_kwargs": {},
        }
