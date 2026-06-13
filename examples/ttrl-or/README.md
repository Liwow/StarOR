# StarOR Examples

`run_staror.sh` is the canonical launch script. Its search, adaptation, reward,
and model hyperparameters match the paper configuration.

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3-4B-Instruct-2507 \
TRAIN_FILE="$PWD/data/IndustryOR_fixedV2.jsonl" \
bash examples/ttrl-or/run_staror.sh
```

The following environment variables can be overridden without editing the
script:

- `CUDA_VISIBLE_DEVICES`
- `DATA_ROOT`
- `TRAIN_FILE`
- `MODEL_NAME_OR_PATH`
- `OUTPUT_DIR`

Additional Hydra overrides can be appended to the command. For example:

```bash
bash examples/ttrl-or/run_staror.sh trainer.logger='["console","wandb"]'
```

`run_ttrl_or.sh` is retained as a compatibility wrapper and forwards all
arguments to `run_staror.sh`.
