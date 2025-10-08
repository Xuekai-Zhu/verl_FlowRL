#!/usr/bin/env bash
set -xeuo pipefail

unset ROCR_VISIBLE_DEVICES
project_name='FlowRL'
exp_name='test-debug'

# Algorithm settings
adv_estimator=grpo

# KL settings (ref policy needed for FlowRL, but KL penalty disabled)
use_kl_in_reward=True  # Enable ref policy for ref_log_prob (needed for FlowRL TB loss)
kl_coef=0.0  # But set KL coefficient to 0 (no KL penalty in reward)
use_kl_loss=False
kl_loss_coef=0.0

# FlowRL trajectory balance coefficient
tb_coef=15.0

# TIS - Truncated Importance Sampling
tis_imp_ratio_cap=2.0

# DAPO Dual-clip parameters
clip_ratio_low=0.2
clip_ratio_high=0.28

# FlowRL Loss Variant Selection
# Options: "vanilla" (no TIS/clip), "clip_only" (clip IS only), "tis_clip" (both TIS + clip)
loss_variant="tis_clip"

# Sequence lengths (REDUCED FOR DEBUGGING)
max_prompt_length=256  # Reduced from 1024 for debugging
max_response_length=512  # Reduced from 5120 for debugging

# Overlong buffer for very long responses (DISABLED FOR DEBUGGING)
enable_overlong_buffer=False  # Disabled for debugging
# overlong_buffer_len=$((1024 * 4))
# overlong_penalty_factor=1.0

# Loss aggregation
loss_agg_mode="token-mean"

# Filter groups - dynamic sampling
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10

# Batch sizes (REDUCED FOR DEBUGGING)
train_prompt_bsz=8  # Reduced for debugging
gen_prompt_bsz=16   # Reduced for debugging
n_resp_per_prompt=4 # Reduced for debugging
train_prompt_mini_bsz=4  # Reduced for debugging (micro batch size)

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}

# Paths
MODEL_PATH=${MODEL_PATH:-"${WORKING_DIR}/downloads/models/Qwen/Qwen2.5-1.5B"}
CKPTS_DIR=${CKPTS_DIR:-"${WORKING_DIR}/outputs/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${WORKING_DIR}/downloads/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${WORKING_DIR}/downloads/data/aime-2024.parquet"}

# Sampling
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
n_gpus=8
sp_size=1
use_dynamic_bsz=True  # DISABLED FOR DEBUGGING - easier to track batch sizes
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=False
gen_tp=1
# Truncated Importance Sampling (TIS) -> https://fengyao.notion.site/off-policy-rl

# Please note that server mode(agent loop) hasn't return rollout_log_probs for now.
# so currently, server mode is not supported for TIS.

# To turn on TIS, you need to set the following parameters. Note 2.0 is a hyper-parameter and can be tuned.
#   actor_rollout_ref.actor.tis_imp_ratio_cap=2.0
#   actor_rollout_ref.rollout.calculate_log_probs=True

    
python3 -m recipe.flowrl.main_flowrl \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    +actor_rollout_ref.actor.loss_variant=${loss_variant} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.tis_imp_ratio_cap=${tis_imp_ratio_cap} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${n_gpus} \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=10 \
    trainer.save_freq=50 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto

# todo: algorithm.tb_coef=${tb_coef} \
# +actor_rollout_ref.actor.proj_layer=3 \
#     reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
#    reward_model.overlong_buffer.len=${overlong_buffer_len} \
#    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \