#!/bin/bash
# Check model parameters using rlaunch

MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/llmit/user/xuekaizhu/verl_FlowRL/work_dirs/FlowRL_Scaling/FlowRL-Qwen2.5-0.5B-DAPO-Math-prompt-modified-reward/20251103_090837/huggingface/global_step_20}"
SEARCH_TERM="${SEARCH_TERM:-project_z}"

rlaunch --cpu=8 --memory=16000 --charged-group=llmit_gpu --private-machine=no \
--mount=gpfs://gpfs1/llmit:/mnt/shared-storage-user/llmit \
--mount=gpfs://gpfs1/large-model-center-share-weights:/mnt/shared-storage-user/large-model-center-share-weights \
--mount=gpfs://gpfs1/llmrazor-share:/mnt/shared-storage-user/llmrazor-share \
--image=registry.h.pjlab.org.cn/ailab-puyu-puyu_gpu/xtuner:pt28_20250911_6652194 \
-- bash -c "
cd /mnt/shared-storage-user/llmit/user/xuekaizhu/verl_FlowRL && \
source /mnt/shared-storage-user/llmit/user/chengguangran/miniconda3/etc/profile.d/conda.sh && \
conda activate verl && \
python recipe/flowrl/intern-s2/eval/check_model_parameters.py '${MODEL_PATH}' '${SEARCH_TERM}'
"
