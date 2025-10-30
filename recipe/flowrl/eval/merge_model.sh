#!/bin/bash
set -x

# Configuration
BACKEND="fsdp"
LOCAL_DIR="/mnt/petrelfs/linzhouhan/xuekaizhu/verl_FlowRL/outputs/ckpts/FlowRL_Scaling/test-debug/global_step_10/actor"
TARGET_DIR="/mnt/petrelfs/linzhouhan/xuekaizhu/verl_FlowRL/outputs/merged_models/test-debug-step10"

# Convert FSDP checkpoint to HuggingFace format
python -m verl.model_merger merge \
    --backend $BACKEND \
    --local_dir $LOCAL_DIR \
    --target_dir $TARGET_DIR
