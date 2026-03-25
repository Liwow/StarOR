from __future__ import annotations

from ttrl_or.cli import _build_backend, _build_config, build_parser
from ttrl_or.config import PipelineConfig
from ttrl_or.model import MockPolicyBackend


def test_build_config_uses_local_model_path(tmp_path):
    model_dir = tmp_path / "local_model"
    model_dir.mkdir()

    parser = build_parser()
    args = parser.parse_args(["--backend", "trl", "--model-path", str(model_dir)])
    config = _build_config(args)

    assert config.backend.backend == "trl"
    assert config.backend.model_name_or_path == str(model_dir.resolve())


def test_build_config_keeps_hf_model_id():
    parser = build_parser()
    args = parser.parse_args(["--backend", "trl", "--model-name", "Qwen/Qwen2.5-1.5B-Instruct"])
    config = _build_config(args)

    assert config.backend.backend == "trl"
    assert config.backend.model_name_or_path == "Qwen/Qwen2.5-1.5B-Instruct"


def test_build_backend_reads_backend_config():
    config = PipelineConfig()
    config.backend.backend = "mock"
    config.backend.seed = 123

    backend = _build_backend(config)

    assert isinstance(backend, MockPolicyBackend)
    assert backend.seed == 123