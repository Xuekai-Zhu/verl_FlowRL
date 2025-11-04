#!/bin/bash
# Run parameter check locally (no rlaunch, just direct python)

cd /mnt/shared-storage-user/llmit/user/xuekaizhu/verl_FlowRL
source /mnt/shared-storage-user/llmit/user/chengguangran/miniconda3/etc/profile.d/conda.sh
conda activate verl

# Check your trained checkpoint for project_z parameter
python recipe/flowrl/intern-s2/eval/check_model_parameters.py \
  /mnt/shared-storage-user/llmit/user/xuekaizhu/verl_FlowRL/work_dirs/FlowRL_Scaling/FlowRL-Qwen2.5-0.5B-DAPO-Math-prompt-modified-reward/20251103_090837/huggingface/global_step_20 \
  project_z
