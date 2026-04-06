# TTRL-OR Examples

This directory contains example launch scripts for the migrated TTRL-OR flow on top of `verl`.

Current example:

- `run_qwen2_5_7b_ttrl_or_vllm_lora.sh`

Notes:

- This path uses the custom `algorithm.ttrl_or.enable=True` fit branch.
- Rollout generation still uses verl native `vllm` rollout.
- Actor updates still use verl native GRPO/PPO update functions.
- Sample-local LoRA reset is enabled via:
  - `algorithm.ttrl_or.backend.reset_lora_on_begin_episode=true`
- LoRA settings are aligned with verl's LoRA guidance:
  - `actor_rollout_ref.rollout.load_format=safetensors`
  - `actor_rollout_ref.model.target_modules=all-linear`
  - `actor_rollout_ref.rollout.layered_summon=True`
  - `trainer.use_legacy_worker_impl=disable`
