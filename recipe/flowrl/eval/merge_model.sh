#!/bin/bash
set -x

# ============================================
# Configuration - Edit these variables
# ============================================
BACKEND="${BACKEND:-fsdp}"
LOCAL_DIR="${LOCAL_DIR:-/mnt/shared-storage-user/llmit/user/xuekaizhu/verl_FlowRL/work_dirs/FlowRL_Scaling/FlowRL-Qwen2.5-0.5B-DAPO-Math-prompt-modified-reward/20251103_090837/ckpts/global_step_20/actor}"
TARGET_DIR="${TARGET_DIR:-/mnt/shared-storage-user/llmit/user/xuekaizhu/verl_FlowRL/work_dirs/FlowRL_Scaling/FlowRL-Qwen2.5-0.5B-DAPO-Math-prompt-modified-reward/20251103_090837/huggingface/global_step_20}"
# ============================================

TARGET_DIR_WITH_PROJ_Z="${TARGET_DIR}_with_proj_z"

# Change to working directory
cd /mnt/shared-storage-user/llmit/user/xuekaizhu/verl_FlowRL

# Activate conda environment
source /mnt/shared-storage-user/llmit/user/chengguangran/miniconda3/etc/profile.d/conda.sh
conda activate verl

echo '========================================'
echo 'Step 1: Merging FSDP checkpoint to HuggingFace format'
echo "Backend: ${BACKEND}"
echo "Source: ${LOCAL_DIR}"
echo "Target (with proj_z): ${TARGET_DIR_WITH_PROJ_Z}"
echo '========================================'

# Convert FSDP checkpoint to HuggingFace format
python -m verl.model_merger merge \
    --backend ${BACKEND} \
    --local_dir ${LOCAL_DIR} \
    --target_dir ${TARGET_DIR_WITH_PROJ_Z}

echo ''
echo '========================================'
echo 'Step 2: Removing proj_z parameters'
echo "Input: ${TARGET_DIR_WITH_PROJ_Z}"
echo "Output (clean): ${TARGET_DIR}"
echo '========================================'

# Remove proj_z parameters
python recipe/flowrl/intern-s2/eval/remove_proj_z.py \
    ${TARGET_DIR_WITH_PROJ_Z} \
    ${TARGET_DIR}

echo ''
echo '========================================'
echo 'Merge complete!'
echo "Clean model (no proj_z): ${TARGET_DIR}"
echo "Original model (with proj_z): ${TARGET_DIR_WITH_PROJ_Z}"
echo '========================================'
