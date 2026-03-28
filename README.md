# TTRL-OR

A modular prototype for **test-time reinforcement learning** on optimization modeling tasks.

Input: natural language optimization task description.

Pipeline:
1. `schema` (now outputs schema + skill + cautions)
2. `set_param_var`
3. `obj_cons`
4. `code`

Each stage uses MCTS with PUCT and rollout-to-code evaluation. Inside a stage, each selected node triggers one internal TRL-GRPO rollout group (`num_generations = k`), which simultaneously updates LoRA and expands `k` children for that stage. At the end of stage 4, final candidates are reranked by finalized reward to choose the answer, then LoRA is dropped before the next task.

Provisional consensus (`r1`) now uses a **global consensus pool** per task instance: when pool size is small it uses order-of-magnitude matching, and once pool is large enough it switches to majority voting with relative tolerance.

## Core Design Goals

- Lightweight and swappable training backend (`mock` or `trl`).
- Reward function is easy to replace (`ttrl_or/reward/default_reward.py`).
- MCTS selection/expansion is modular (`ttrl_or/mcts/puct.py`, `ttrl_or/mcts/tree.py`).
- Prompt templates are isolated (`ttrl_or/prompts/templates.py`).

## Reward Definition

For a group of `g` trajectories:

- `r1`: majority-vote consensus reward (`1` if trajectory matches consensus, else `0`)
- `r2`: execution reward (`1` if code runs, else `0`) when `r1 = 0`
- `r3`: robustness reward (`1` if model-generated perturbed tests pass, else `0`) when `r1 = 1`

Combination:

- if `r1 == 1`: `reward = r1 * 0.9 + r3 * 0.1`
- else: `reward = r2 * 0.2`

## Environment Setup

Recommended (Python 3.10+):

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[hf,dev]"
```

If your pip still fails editable install, use fallback:

```bash
pip install -r requirements-hf.txt
pip install -e .
```

Notes:

- `setup.py` is included for compatibility with older tooling.
- `pyproject.toml` is the primary package configuration.

## Data Reading (Raw JSONL First)

You can now run directly on the original raw files under `data/*.jsonl` (no unified conversion required).

Supported raw styles in one loader:

- parameters embedded in free text
- parameters represented in markdown tables

The training pipeline now uses **description-only input** by default.
The backend pre-extracts a numeric parameter map and perturbation map *before stage-1 generation*.

The extracted map is built from:

- numeric mentions from the question text
- numeric cells from markdown tables
- `__key_param_keys__` (heuristic key parameters from objective/constraint context)

Python API:

```python
from ttrl_or.dataset import load_raw_task_dataset
samples = load_raw_task_dataset("data/NL4OPT.jsonl", limit=10)
```

Optional: keep unified normalization tools for offline preprocessing:

```bash
python tools/normalize_data.py --input data/IndustryOR_fixedV2.jsonl --output data/unified/IndustryOR_fixedV2.unified.jsonl
python tools/normalize_all_data.py --input-dir data --output-dir data/unified
```

## Quick Start

Single-task mode (description only, mock backend):

```bash
python -m ttrl_or --backend mock --task-file examples/task.txt
```

Optional: still supported for ablation/debug to provide explicit numeric instance:

```bash
python -m ttrl_or --backend mock --task-file examples/task.txt --instance-json examples/instance.json
```

Dataset mode on raw JSONL (recommended for your current workflow):

```bash
python -m ttrl_or --backend mock --dataset-jsonl data/NL4OPT.jsonl --dataset-limit 20 --out outputs/nl4opt_mock.json
```

TRL backend (real GRPO updates via `trl`):

```bash
python -m ttrl_or --backend trl --model-name model/Qwen/Qwen3-4B-Instruct-2507 --dataset-jsonl data/NL4OPT.jsonl --dataset-limit 20
```

TRL backend with a local downloaded model directory:

```bash
python -m ttrl_or --backend trl --model-path E:/models/Qwen2.5-1.5B-Instruct --dataset-jsonl data/NL4OPT.jsonl --dataset-limit 20
```

TRL + vLLM generation backend (optional, requires compatible `trl` + `vllm` environment):

```bash
python -m ttrl_or --backend trl --model-name model/Qwen/Qwen3-4B-Instruct-2507 --dataset-jsonl data/NL4OPT.jsonl --dataset-limit 20 --grpo-use-vllm --grpo-vllm-mode server --grpo-vllm-gpu-memory-utilization 0.85
```

## Scripted Launch (Single or Multi GPU)

All launch scripts are under `scripts/`.

1. Edit common parameters in `scripts/run.sh`

- `CUDA_VISIBLE_DEVICES`
- `NPROC_PER_NODE` (training process count)
  - single-card: `NPROC_PER_NODE=1`
  - multi-card: `NPROC_PER_NODE=2` (or 4)
- `MODEL_NAME_OR_PATH`
- `DATASET_JSONL`, `DATASET_LIMIT`
- `USE_VLLM`, `VLLM_MODE`

`run.sh` now auto-selects launcher:
- `NPROC_PER_NODE=1` -> `python -m ttrl_or ...`
- `NPROC_PER_NODE>1` -> `torchrun --nproc_per_node=N -m ttrl_or ...`

2. (Optional) Terminal A: start vLLM service for server mode

```bash
chmod +x scripts/*.sh
scripts/start_vllm_server.sh
```

Important: for `--grpo-vllm-mode server`, do NOT start plain `vllm.entrypoints.openai.api_server` directly. Use `trl vllm-serve` (our script does this by default), otherwise TRL can fail on `init_communicator` with 404.

3. Terminal B: run training

```bash
scripts/run.sh
```

4. If you do not want vLLM

Set `USE_VLLM=false` in `scripts/run.sh`, then run the same script.

Note: in `scripts/run.sh`, if `NPROC_PER_NODE>1` and `VLLM_MODE=colocate`, the script will auto-disable `USE_VLLM` to avoid duplicated GPU memory usage.

Useful knobs:

- MCTS: `--max-iterations`, `--c-puct`, `--mcts-stop-on-reward-one`
- Reward: `--global-consensus-rel-tol`, `--robustness-cases`, `--code-timeout-sec`, `--enable-r3-reward`, `--disable-r3-reward`
  - `r3` perturbation now uses backend pre-extracted mapping (`focus_keys` + value map).
- Dataset loader: `--dataset-start-index`, `--dataset-limit`, `--dataset-max-numeric-features`, `--dataset-key-param-top-k`
  - Mapping extractor plugin: `--mapping-extractor rule|llm`
  - LLM extractor knobs: `--mapping-llm-max-new-tokens`, `--mapping-llm-temperature`, `--mapping-llm-top-p`
- GRPO: `--grpo-lr`, `--grpo-batch-size`, `--grpo-grad-accum`, `--grpo-num-generations`, `--grpo-generation-batch-size`, `--grpo-max-completion-len`, `--grpo-use-vllm`, `--grpo-vllm-mode`, `--grpo-vllm-gpu-memory-utilization`, `--grpo-vllm-tensor-parallel-size`, `--grpo-vllm-max-model-len`
- Logging: `--log-dir` and `--no-save-logs`
- Backend/model: `--backend`, `--model-name`, `--model-path`, `--seed`, `--torch-dtype`, `--trust-remote-code`
- Generation: `--temperature`, `--top-p`, `--max-new-tokens`

## Configuration Reference

All defaults are defined in `ttrl_or/config.py`.

### MCTSConfig

- `max_iterations` (Key): max number of MCTS iterations. One iteration = global-leaf select -> stage expansion by GRPO internal rollout group -> backprop update.
  - Search stops early if selected leaf is expanded to `code` stage, or if `stop_on_reward_one=True` and a rollout reward reaches `1.0`.
- `c_puct` (Key): PUCT exploration coefficient.
  - Higher => more prior-driven exploration.
  - Lower => more exploitation of current Q.
- `stop_on_reward_one` (Key): if enabled, once any rollout reward reaches `1.0`, MCTS stops immediately for this task instance.
  - Useful for aggressive latency reduction when a perfect candidate appears early.
  - Disable it when you want fuller exploration for stability.
- `num_generations` (Key, under `GRPOConfig`): internal GRPO rollout group size per selected node (`k`). Must be >= 2 for TRL GRPO.
  - Larger => lower reward variance but higher per-iteration cost.
### RewardConfig

- `code_timeout_sec` (Key): timeout per code execution.
  - Prevents bad code from blocking training.
- `robustness_cases` (Key): number of perturbed tests for `r3`.
  - Larger => stronger robustness filter.
  - Increases execution cost.
- `enable_r3_reward` (Key): switch to enable/disable `r3` robustness testing.
  - `True`: run key-parameter perturbation tests for robustness reward.
  - `False`: skip perturbation tests and treat `r3` as passed (`1.0`).

### GRPOConfig

- `learning_rate` (Key): optimizer learning rate for GRPO update.
  - Primary stability knob.
- `kl_coef`: KL penalty weight.
  - Higher keeps policy closer to reference behavior.
- `per_device_train_batch_size` (Key): per-device batch size for GRPO.
- `gradient_accumulation_steps` (Key): accumulation factor.
  - Effective batch is `batch_size * grad_accum`.
- `num_generations` (Key): generations per selected-node rollout group. This directly controls how many stage children are expanded from one selected node.
- `generation_batch_size` (Key): TRL generation batch size. `0` means auto, and runtime will round it up to a multiple of `num_generations` to avoid GRPOConfig divisibility errors.
- `max_prompt_length`: truncation limit for prompt tokens.
- `max_completion_length`: truncation limit for completion tokens.

- `use_vllm` (Key): whether TRL GRPO uses vLLM as generation backend.
  - `False`: default HF generation path.
  - `True`: pass vLLM options into TRL (effective only if your TRL version supports them).
- `vllm_mode`: vLLM run mode passed to TRL (commonly `server` or `colocate`, depending on TRL version).
- `vllm_gpu_memory_utilization` (Key): target fraction of GPU memory reserved by vLLM KV/cache scheduler.
- `vllm_tensor_parallel_size` (Key): TP shard count for vLLM inference workers.
- `vllm_max_model_len` (Key): max sequence length used by vLLM engine.

### BackendConfig

- `backend`: backend selector (`mock` or `trl`).
- `model_name_or_path` (Key): model identifier used by TRL backend.
  - Supports Hugging Face repo id (for example `Qwen/Qwen2.5-1.5B-Instruct`).
  - Supports local downloaded directory path (for example `E:/models/Qwen2.5-1.5B-Instruct`).
- `seed`: random seed used by mock backend sampling.
- `temperature`, `top_p`, `max_new_tokens`: generation knobs for backend sampling.
- `torch_dtype`: dtype passed into model loading (`auto`, `float16`, `bfloat16`, etc.).
- `trust_remote_code`: whether to allow remote model code when loading.
- `lora_r`, `lora_alpha`, `lora_dropout`: LoRA adapter hyperparameters in TRL backend.

### PipelineConfig
- `save_logs`: whether to write per-task artifacts.
- `log_dir`: output directory root for logs (default `logs/`).

### DatasetConfig

- `jsonl_path`: input raw dataset path for dataset mode.
- `start_index`: start offset for dataset slicing.
- `limit`: max number of samples to run (`0` means no limit).
- `max_numeric_features`: cap on numeric features extracted into the mapping instance.
- `key_param_top_k`: top-k key parameters used as perturbation focus candidates.
- `mapping_extractor`: mapping strategy plugin (`rule` or `llm`).
- `mapping_llm_max_new_tokens`: completion length for LLM extractor output.
- `mapping_llm_temperature`: temperature for LLM extractor generation.
- `mapping_llm_top_p`: top-p for LLM extractor generation.

## Recommended Starting Ranges

- `max_iterations`: `8` to `64`
- `num_generations`: `2` to `6`
- `c_puct`: `1.0` to `2.0`
- `learning_rate`: `1e-5` to `1e-4`

## Logs and Artifacts

By default each instance writes logs under `logs/<task_id>/`:

- `run_summary.json`: task info, config, stage reports, final selection, best trajectory summary
- `runtime_summary.json`: sample-level runtime summary (overall seconds, iter count, per-iter time/reward)
- `runtime_summary.md`: readable runtime report (total time, iter count, per-iter time/reward table)
- `mcts_iterations.json`: full per-iteration structured logs (for scripts/analysis)
- `mcts_iterations.md`: human-readable per-iteration report (selection, PUCT candidates, best rollout, prompt/answer, timing)
- `stage_events.json`: detailed per-stage/per-expansion events
- `stage_events.md`: human-readable stage summary
- `mcts_stats.json`: concise MCTS node expansion/reuse and rollout counts per stage
- `final_trajectories.json`: selected final trajectories with reward and code
- `best_code.py`: final selected code

Disable all saving with `--no-save-logs`.

## Tests

```bash
pytest -q
```

## Project Layout

- `ttrl_or/pipeline/ttrl_or.py`: end-to-end runner (`MCTS -> online per-group GRPO updates across 4 stages -> reward-based final selection`)
- `ttrl_or/mcts/`: tree search and PUCT selection
- `ttrl_or/reward/`: execution + consensus + robustness reward
- `ttrl_or/model/backend.py`: backend interface
- `ttrl_or/model/mock_backend.py`: mock backend for local pipeline checks
- `ttrl_or/model/trl_backend.py`: TRL + PEFT LoRA backend for real GRPO updates
- `ttrl_or/prompts/`: prompt templates and builder
- `ttrl_or/dataset/`: raw dataset loading, instance construction, and optional unified normalization
- `ttrl_or/mapping/`: pluggable mapping extractors (`rule` / `llm`)
- `scripts/start_vllm_server.sh`: vLLM OpenAI-compatible service launcher (4-GPU defaults)
- `scripts/run.sh`: one-command TTRL-OR + TRL runner (edit parameters directly in file)

## Notes

- `MockPolicyBackend` intentionally does not train.
- `TRLPolicyBackend` creates temporary LoRA adapters per task instance and drops them at episode end.
- If `trl/peft/datasets` are missing, `--backend trl` will raise a clear install error.
