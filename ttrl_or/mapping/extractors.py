from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from ttrl_or.config import DatasetConfig
from ttrl_or.dataset import build_instance_from_question
from ttrl_or.types import OptimizationTask


class MappingLLMBackend(Protocol):
    def generate_mapping_from_description(
        self,
        description: str,
        dataset_config: DatasetConfig,
    ) -> dict[str, Any] | str | None:
        ...


@dataclass(slots=True)
class MappingExtractionResult:
    instance: dict[str, Any]
    perturbation_map: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class MappingExtractor(Protocol):
    name: str

    def extract(
        self,
        task: OptimizationTask,
        dataset_config: DatasetConfig,
        backend: MappingLLMBackend | None = None,
    ) -> MappingExtractionResult:
        ...


@dataclass(slots=True)
class RuleMappingExtractor:
    name: str = "rule"

    def extract(
        self,
        task: OptimizationTask,
        dataset_config: DatasetConfig,
        backend: MappingLLMBackend | None = None,
    ) -> MappingExtractionResult:
        used_description_extraction = False
        if task.instance:
            instance = dict(task.instance)
        else:
            instance = build_instance_from_question(
                task.description,
                max_numeric_features=dataset_config.max_numeric_features,
                key_param_top_k=dataset_config.key_param_top_k,
            )
            used_description_extraction = True

        from ttrl_or.reward.perturbation import build_perturbation_map

        perturbation_map = build_perturbation_map(instance)
        metadata = {
            "extractor": self.name,
            "used_description_extraction": used_description_extraction,
            "num_instance_keys": len(instance),
            "num_numeric_keys": int(perturbation_map.get("num_numeric_keys", 0)),
            "num_focus_keys": int(perturbation_map.get("num_focus_keys", 0)),
            "focus_keys": list(perturbation_map.get("focus_keys", []))[:16],
        }
        return MappingExtractionResult(instance=instance, perturbation_map=perturbation_map, metadata=metadata)


@dataclass(slots=True)
class LLMMappingExtractor:
    name: str = "llm"
    fallback: RuleMappingExtractor = field(default_factory=RuleMappingExtractor)

    def extract(
        self,
        task: OptimizationTask,
        dataset_config: DatasetConfig,
        backend: MappingLLMBackend | None = None,
    ) -> MappingExtractionResult:
        if task.instance:
            fallback_result = self.fallback.extract(task, dataset_config, backend)
            fallback_result.metadata["extractor"] = self.name
            fallback_result.metadata["llm_used"] = False
            fallback_result.metadata["llm_reason"] = "explicit_instance_provided"
            return fallback_result

        if backend is None or not hasattr(backend, "generate_mapping_from_description"):
            fallback_result = self.fallback.extract(task, dataset_config, backend)
            fallback_result.metadata["extractor"] = self.name
            fallback_result.metadata["llm_used"] = False
            fallback_result.metadata["llm_reason"] = "backend_missing_llm_hook"
            return fallback_result

        raw = backend.generate_mapping_from_description(task.description, dataset_config)
        parsed = _parse_llm_mapping_output(raw)
        if parsed is None:
            fallback_result = self.fallback.extract(task, dataset_config, backend)
            fallback_result.metadata["extractor"] = self.name
            fallback_result.metadata["llm_used"] = False
            fallback_result.metadata["llm_reason"] = "llm_parse_failed_or_empty"
            fallback_result.metadata["llm_raw_preview"] = _preview_text(raw)
            return fallback_result

        instance = _normalize_instance_from_llm(parsed, dataset_config)
        if not instance:
            fallback_result = self.fallback.extract(task, dataset_config, backend)
            fallback_result.metadata["extractor"] = self.name
            fallback_result.metadata["llm_used"] = False
            fallback_result.metadata["llm_reason"] = "llm_instance_empty_after_normalize"
            fallback_result.metadata["llm_raw_preview"] = _preview_text(raw)
            return fallback_result

        from ttrl_or.reward.perturbation import build_perturbation_map

        perturbation_map = build_perturbation_map(instance)
        metadata = {
            "extractor": self.name,
            "llm_used": True,
            "llm_raw_preview": _preview_text(raw),
            "used_description_extraction": True,
            "num_instance_keys": len(instance),
            "num_numeric_keys": int(perturbation_map.get("num_numeric_keys", 0)),
            "num_focus_keys": int(perturbation_map.get("num_focus_keys", 0)),
            "focus_keys": list(perturbation_map.get("focus_keys", []))[:16],
        }
        return MappingExtractionResult(instance=instance, perturbation_map=perturbation_map, metadata=metadata)


def build_mapping_extractor(name: str) -> MappingExtractor:
    lowered = (name or "rule").strip().lower()
    if lowered == "llm":
        return LLMMappingExtractor()
    return RuleMappingExtractor()


def _parse_llm_mapping_output(raw: dict[str, Any] | str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw

    text = str(raw).strip()
    if not text:
        return None

    if text.startswith("```"):
        text = _strip_markdown_fence(text)

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) <= 2:
        return stripped.strip("`")

    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _normalize_instance_from_llm(payload: dict[str, Any], dataset_config: DatasetConfig) -> dict[str, Any]:
    if isinstance(payload.get("instance"), dict):
        source = payload.get("instance", {})
    else:
        source = payload

    numeric_pairs: list[tuple[str, int | float]] = []
    for key, value in source.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            numeric_pairs.append((str(key), value))

    if not numeric_pairs:
        return {}

    limit = max(1, int(dataset_config.max_numeric_features))
    numeric_pairs = numeric_pairs[:limit]

    instance: dict[str, Any] = {k: v for k, v in numeric_pairs}

    hinted_keys: list[str] = []
    for field_name in ("key_param_keys", "focus_keys", "important_keys"):
        candidate = payload.get(field_name)
        if isinstance(candidate, list):
            hinted_keys = [str(x) for x in candidate if isinstance(x, str)]
            if hinted_keys:
                break

    valid_hints = [k for k in hinted_keys if k in instance]
    if not valid_hints:
        top_k = max(1, int(dataset_config.key_param_top_k))
        valid_hints = [k for k, _ in numeric_pairs[:top_k]]

    instance["__param_mode__"] = "llm"
    instance["__num_numeric_features__"] = len(numeric_pairs)
    instance["__key_param_keys__"] = valid_hints[: max(1, int(dataset_config.key_param_top_k))]
    instance["__feature_meta__"] = [
        {
            "key": key,
            "source": "llm",
            "score": 100,
            "snippet": "llm_extracted",
        }
        for key, _ in numeric_pairs[:64]
    ]
    return instance


def _preview_text(raw: dict[str, Any] | str | None, max_len: int = 280) -> str:
    if raw is None:
        return ""
    text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."

