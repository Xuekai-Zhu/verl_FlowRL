#!/bin/bash
set -x

# ============================================
# Configuration - Edit these variables
# ============================================
BACKEND="${BACKEND:-fsdp}"
# Your checkpoint from training
LOCAL_DIR="${LOCAL_DIR:-/mnt/shared-storage-user/llmit/user/xuekaizhu/verl_FlowRL/work_dirs/FlowRL_Scaling/FlowRL-Qwen2.5-7B-token-level-1221/20251220_162109/ckpts/global_step_90/actor}"
# Where to save the merged HuggingFace model
TARGET_DIR="${TARGET_DIR:-/mnt/shared-storage-user/llmit/user/xuekaizhu/verl_FlowRL/work_dirs/FlowRL_Scaling/FlowRL-Qwen2.5-7B-token-level-1221/20251220_162109/huggingface/global_step_90}"
# ============================================

# Change to working directory
cd /mnt/shared-storage-user/llmit/user/xuekaizhu/verl_FlowRL

# Activate conda environment
source /mnt/shared-storage-user/llmit/user/chengguangran/miniconda3/etc/profile.d/conda.sh
conda activate verl

echo '========================================'
echo 'Merging FSDP checkpoint to HuggingFace format'
echo 'This script is for models WITHOUT proj_z'
echo '(CISPO checkpoints, FlowRL no-log-z checkpoints)'
echo '========================================'
echo "Backend: ${BACKEND}"
echo "Source: ${LOCAL_DIR}"
echo "Target: ${TARGET_DIR}"
echo '========================================'

# Convert FSDP checkpoint to HuggingFace format
python -m verl.model_merger merge \
    --backend ${BACKEND} \
    --local_dir ${LOCAL_DIR} \
    --target_dir ${TARGET_DIR}

echo ''
echo '========================================'
echo 'Merge complete!'
echo "Merged model saved to: ${TARGET_DIR}"
echo '========================================'
