#!/bin/bash
# Test model generation with vLLM

MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/llmit/user/xuekaizhu/verl_FlowRL/work_dirs/FlowRL_Scaling/FlowRL-Qwen2.5-0.5B-DAPO-Math-prompt-modified-reward/20251103_090837/huggingface/global_step_20}"

cd /mnt/shared-storage-user/llmit/user/xuekaizhu/verl_FlowRL
source /mnt/shared-storage-user/llmit/user/chengguangran/miniconda3/etc/profile.d/conda.sh
conda activate verl

# Disable vLLM V1 engine
export VLLM_USE_V1=0

echo "Testing model generation..."
echo "Model: ${MODEL_PATH}"
echo ""

python recipe/flowrl/intern-s2/eval/vllm/test_model_generation.py "${MODEL_PATH}"
