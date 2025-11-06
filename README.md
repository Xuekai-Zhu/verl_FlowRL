# Branch: dev-fp16

## Precision Configuration

| Component | Precision |
|-----------|-----------|
| **Training (Actor)** | float16 |
| **Training (Ref)** | float16 |
| **Inference (vLLM Rollout)** | float16 |
| **Gradient Scaler** | ShardedGradScaler (enabled) |

## Modified Files

1. `verl/trainer/config/actor/actor.yaml` - Added `dtype: float16`
2. `verl/trainer/config/ref/ref.yaml` - Added `dtype: float16`
3. `verl/trainer/config/rollout/rollout.yaml` - Changed to `dtype: float16`
4. `verl/workers/actor/dp_actor.py` - Added ShardedGradScaler for FP16
5. `recipe/flowrl/flowrl_fsdp_worker.py` - Made dtype configurable
6. `recipe/flowrl/flowrl_actor.py` - Use config dtype in autocast

## Source

FP16 implementation based on: https://github.com/sail-sg/Precision-RL/blob/main/verl_fp16.patch

## Alternative Branch

- **dev-bf16-train-fp16-infer**: Uses bfloat16 for training (more stable for RL)
