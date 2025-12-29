#!/bin/bash
# Launch merge model job with rlaunch
# Edit configuration in merge_model.sh

rlaunch --gpu=2 \
--cpu=64 \
--memory=160000 \
--charged-group=llmit_gpu \
--private-machine=yes \
--mount=gpfs://gpfs1/llmit:/mnt/shared-storage-user/llmit \
--mount=gpfs://gpfs1/large-model-center-share-weights:/mnt/shared-storage-user/large-model-center-share-weights \
--mount=gpfs://gpfs1/llmrazor-share:/mnt/shared-storage-user/llmrazor-share \
--image=registry.h.pjlab.org.cn/ailab-puyu-puyu_gpu/xtuner:pt28_20250911_6652194 \
-- bash 
