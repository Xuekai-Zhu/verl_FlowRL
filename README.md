# Branch: dev-fp16

## Configuration
- **Training dtype:** float16
- **Inference dtype:** float16
- **Gradient Scaler:** ShardedGradScaler enabled

## Features
- Pure FP16 training and inference
- Faster training compared to bfloat16
- Lower memory usage
- Compatible with FlashAttention

## Key Files
- `verl/trainer/config/actor/actor.yaml`: dtype = float16
- `verl/trainer/config/ref/ref.yaml`: dtype = float16
- `verl/trainer/config/rollout/rollout.yaml`: dtype = float16
- `recipe/flowrl/flowrl_fsdp_worker.py`: Default dtype = float16

## Alternative Branch
- **dev-bf16-train-fp16-infer**: Uses bfloat16 for training (more stable for RL)
