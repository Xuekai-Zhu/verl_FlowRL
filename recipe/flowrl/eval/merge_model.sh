#!/bin/bash

# Simple script to merge FSDP checkpoint to HuggingFace format

python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir /mnt/petrelfs/linzhouhan/xuekaizhu/verl_FlowRL/outputs/ckpts/FlowRL_Scaling/test-debug/global_step_10/actor \
    --target_dir /mnt/petrelfs/linzhouhan/xuekaizhu/verl_FlowRL/outputs/merged_models/test-debug-step10
