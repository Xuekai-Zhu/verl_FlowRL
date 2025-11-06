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

---

## Changes from `dev` Branch

### Configuration Files

#### 1. `verl/trainer/config/actor/actor.yaml`
```diff
+ dtype: float16
```

#### 2. `verl/trainer/config/ref/ref.yaml`
```diff
+ dtype: float16
```

#### 3. `verl/trainer/config/rollout/rollout.yaml`
```diff
- dtype: bfloat16
+ dtype: float16
```

### Code Changes

#### 4. `verl/workers/actor/dp_actor.py`
Added FP16 gradient scaler support:
```python
# Initialize ShardedGradScaler for fp16
if self.config.dtype == "float16":
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
    self.scaler = ShardedGradScaler(growth_interval=400)

# Use scaler in backward pass
if self.scaler is not None:
    self.scaler.unscale_(self.actor_optimizer)

# Use scaler in optimizer step
if self.scaler is not None:
    self.scaler.step(self.actor_optimizer)
    self.scaler.update()
```

#### 5. `recipe/flowrl/flowrl_fsdp_worker.py`
```diff
# Default training dtype changed
- param_dtype = torch.bfloat16
+ param_dtype = PrecisionType.to_dtype(self.config.actor.get("dtype", "float16"))

# vLLM dtype uses config
+ vllm_dtype = PrecisionType.to_dtype(self.config.rollout.dtype)
- torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
+ torch_dtype = torch.float32 if self._is_actor else vllm_dtype
```

#### 6. `recipe/flowrl/flowrl_actor.py`
```diff
# Use dynamic dtype from config
- with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
+ from verl.utils.torch_dtypes import PrecisionType
+ torch_dtype = PrecisionType.to_dtype(self.config.dtype)
+ with torch.autocast(device_type=self.device_name, dtype=torch_dtype):

# Use scaler in backward pass
- loss.backward()
+ if self.scaler is not None:
+     self.scaler.scale(loss).backward()
+ else:
+     loss.backward()
```

### Summary of Changes
- Changed all dtype configurations from `bfloat16` to `float16`
- Added `ShardedGradScaler` for FP16 training stability
- Made dtype configurable instead of hardcoded
- Updated autocast to use config dtype
- Modified optimizer step to use gradient scaler when FP16 is enabled
