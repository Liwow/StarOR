# TTRL-OR

A modular prototype for **test-time reinforcement learning** on optimization modeling tasks.

Input: natural language optimization task description.

Pipeline:
1. `schema`
2. `set_param_var`
3. `obj_cons`
4. `code`

Each stage uses MCTS with PUCT and rollout-to-code evaluation. After each stage, one GRPO update is applied to the temporary LoRA state. At the end of stage 4, final candidates are reranked by finalized reward to choose the answer, then LoRA is dropped before the next task.

Provisional consensus (`r1`) is computed with a **stage-local sliding window**, so each stage only votes against recent rollouts from that same stage.

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
python -m ttrl_or --backend trl --model-name Qwen/Qwen2.5-1.5B-Instruct --dataset-jsonl data/NL4OPT.jsonl --dataset-limit 20
```

Useful knobs:

- MCTS: `--group-size`, `--expand-per-node`, `--simulations-per-node`, `--rollout-k`, `--max-nodes-per-stage`
- Reward: `--consensus-window`, `--robustness-cases`, `--disable-perturb-reward`
  - `r3` perturbation now uses backend pre-extracted mapping (`focus_keys` + value map).
- Dataset loader: `--dataset-start-index`, `--dataset-limit`, `--dataset-max-numeric-features`, `--dataset-key-param-top-k`
- GRPO: `--grpo-lr`, `--grpo-batch-size`, `--grpo-grad-accum`, `--grpo-num-generations`, `--grpo-max-steps`
- Logging: `--log-dir` and `--no-save-logs`
- Generation: `--temperature`, `--top-p`, `--max-new-tokens`

## Configuration Reference

All defaults are defined in `ttrl_or/config.py`.

### MCTSConfig

- `expand_per_node` (Key): max number of children expanded under one parent at one stage.
  - Larger value => broader exploration.
  - Too large can increase cost quickly.
- `simulations_per_node` (Key): how many select/expand simulations run per parent node per stage.
  - Larger value => more stable Q estimates.
  - Runtime increases roughly linearly.
- `max_nodes_per_stage` (Key): frontier cap after each stage.
  - Controls beam width between stages.
  - Too small may prune good branches too early.
- `c_puct` (Key): PUCT exploration coefficient.
  - Higher => more prior-driven exploration.
  - Lower => more exploitation of current Q.
- `rollout_k` (Key): rollout count for each selected child during simulation.
  - Larger => lower reward variance.
  - Most expensive parameter in MCTS inner loop.

### RewardConfig

- `code_timeout_sec` (Key): timeout per code execution.
  - Prevents bad code from blocking training.
- `robustness_cases` (Key): number of perturbed tests for `r3`.
  - Larger => stronger robustness filter.
  - Increases execution cost.
- `local_consensus_window` (Key): stage-local sliding window size for provisional `r1` consensus.
  - `<= 0` means use all explored rollouts in current stage.
  - Smaller window tracks recent behavior; larger window stabilizes vote.
- `enable_perturb_reward` (Key): switch to enable/disable perturbation-based `r3`.
  - `True`: run key-parameter perturbation tests for robustness reward.
  - `False`: skip perturbation tests and treat `r3` as passed (`1.0`).

### GRPOConfig

- `learning_rate` (Key): optimizer learning rate for GRPO update.
  - Primary stability knob.
- `clip_range`: policy ratio clip range.
  - Larger can update faster but less stable.
- `kl_coef`: KL penalty weight.
  - Higher keeps policy closer to reference behavior.
- `per_device_train_batch_size` (Key): per-device batch size for GRPO.
- `gradient_accumulation_steps` (Key): accumulation factor.
  - Effective batch is `batch_size * grad_accum`.
- `num_generations` (Key): generations per prompt used by GRPO trainer.
- `max_prompt_length`: truncation limit for prompt tokens.
- `max_completion_length`: truncation limit for completion tokens.
- `max_steps` (Key): GRPO optimizer steps per stage update call.

### PipelineConfig

- `group_size` (Key): number of stage-4 candidates used in final reward reranking.
- `save_logs`: whether to write per-task artifacts.
- `log_dir`: output directory root for logs (default `logs/`).

## Recommended Starting Ranges

- `expand_per_node`: `2` to `5`
- `simulations_per_node`: `2` to `6`
- `rollout_k`: `1` to `4`
- `max_nodes_per_stage`: `8` to `32`
- `c_puct`: `1.0` to `2.0`
- `local_consensus_window`: `16` to `128`
- `group_size`: `4` to `16`
- `learning_rate`: `1e-5` to `1e-4`

## Logs and Artifacts

By default each instance writes logs under `logs/<task_id>/`:

- `run_summary.json`: task info, config, stage reports, final selection, best trajectory summary
- `stage_events.jsonl`: detailed per-stage/per-expansion events
  - prior probability
  - rollout reward details (`r1/r2/r3/total`)
  - Q/visit before and after updates
  - GRPO report per stage
- `mcts_stats.json`: concise MCTS node expansion/reuse and rollout counts per stage
- `final_trajectories.json`: selected final trajectories with reward and code
- `best_code.py`: final selected code

Disable all saving with `--no-save-logs`.

## Tests

```bash
pytest -q
```

## Project Layout

- `ttrl_or/pipeline/ttrl_or.py`: end-to-end runner (`MCTS -> 4 stage GRPO updates -> reward-based final selection`)
- `ttrl_or/mcts/`: tree search and PUCT selection
- `ttrl_or/reward/`: execution + consensus + robustness reward
- `ttrl_or/model/backend.py`: backend interface
- `ttrl_or/model/mock_backend.py`: mock backend for local pipeline checks
- `ttrl_or/model/trl_backend.py`: TRL + PEFT LoRA backend for real GRPO updates
- `ttrl_or/prompts/`: prompt templates and builder
- `ttrl_or/dataset/`: raw dataset loading, instance construction, and optional unified normalization

## Notes

- `MockPolicyBackend` intentionally does not train.
- `TRLPolicyBackend` creates temporary LoRA adapters per task instance and drops them at episode end.
- If `trl/peft/datasets` are missing, `--backend trl` will raise a clear install error.


