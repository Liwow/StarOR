# TTRL-OR

This repository now keeps only the verl-based implementation.

The abandoned standalone TRL implementation has been removed. Runtime code for the current path is under:

```text
verl/trainer/ttrl_or_runtime/
verl/trainer/ppo/ttrl_or_fit.py
verl/utils/dataset/ttrl_or_dataset.py
```

## Install

```bash
pip install -e ".[hf,dev]"
```

For older tooling:

```bash
pip install -r requirements-hf.txt
pip install -e .
```

## Data Tools

The offline data preparation helpers still work, but they now import from the verl runtime package:

```bash
python tools/normalize_data.py --input data/IndustryOR_fixedV2.jsonl --output data/unified/IndustryOR_fixedV2.unified.jsonl
python tools/normalize_all_data.py --input-dir data --output-dir data/unified
```

## Notes

- Packaging only discovers `verl` packages.
- Legacy standalone entry points and TRL backend scripts have been removed.
- Tests that targeted the removed standalone package have been removed with it.
