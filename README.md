# Branch: dev-fp16

## Precision Configuration

| Component | Precision |
|-----------|-----------|
| **Training (Actor)** | float16 |
| **Training (Ref)** | float16 |
| **Inference (vLLM Rollout)** | float16 |
| **Gradient Scaler** | ShardedGradScaler (enabled) |

## Modified Files

### 1. `verl/trainer/config/actor/actor.yaml`
```diff
+ dtype: float16
```

### 2. `verl/trainer/config/ref/ref.yaml`
```diff
+ dtype: float16
```

### 3. `verl/trainer/config/rollout/rollout.yaml`
```diff
- dtype: bfloat16
+ dtype: float16
```

### 4. `verl/workers/actor/dp_actor.py`
```diff
+ # Add ShardedGradScaler for fp16
+ if self.config.dtype == "float16":
+     from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
+     self.scaler = ShardedGradScaler(growth_interval=400)
+ else:
+     self.scaler = None

+ # Use scaler in backward
+ if self.scaler is not None:
+     self.scaler.unscale_(self.actor_optimizer)

+ # Use scaler in optimizer step
+ if self.scaler is not None:
+     self.scaler.step(self.actor_optimizer)
+     self.scaler.update()
```

### 5. `recipe/flowrl/flowrl_fsdp_worker.py`
```diff
- param_dtype = torch.bfloat16
+ param_dtype = PrecisionType.to_dtype(self.config.actor.get("dtype", "float16"))

+ vllm_dtype = PrecisionType.to_dtype(self.config.rollout.dtype)
- torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
+ torch_dtype = torch.float32 if self._is_actor else vllm_dtype
```

### 6. `recipe/flowrl/flowrl_actor.py`
```diff
- with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
+ from verl.utils.torch_dtypes import PrecisionType
+ torch_dtype = PrecisionType.to_dtype(self.config.dtype)
+ with torch.autocast(device_type=self.device_name, dtype=torch_dtype):

- loss.backward()
+ if self.scaler is not None:
+     self.scaler.scale(loss).backward()
+ else:
+     loss.backward()
```

## Source

FP16 implementation based on: https://github.com/sail-sg/Precision-RL/blob/main/verl_fp16.patch

## Alternative Branch

- **dev-bf16-train-fp16-infer**: Uses bfloat16 for training (more stable for RL)
