# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simplified FlowRL Actor - just overrides the loss computation.

This is a minimal wrapper around veRL's DataParallelPPOActor that replaces
the PPO loss with FlowRL trajectory balance loss.
"""

import os
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.utils.py_functional import append_to_dict

# Import FlowRL objective functions
from recipe.flowrl.flowrl_objectives import (
    compute_flowrl,
    compute_flowrl_with_old_policy,
    compute_flowrl_no_log_z,
    compute_flowrl_old_policy_no_log_z,
)


class FlowRLActor(DataParallelPPOActor):
    """
    FlowRL Actor - replaces PPO loss with FlowRL trajectory balance loss.

    This is a minimal subclass that only overrides the loss computation part
    of update_policy(). Everything else (forward pass, optimization, etc.) is
    inherited from DataParallelPPOActor.
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # FlowRL hyperparameters
        self.flowrl_beta_coef = 15.0  # β coefficient for reward scaling

    def update_policy(self, data):
        """
        Override update_policy to use FlowRL loss instead of PPO loss.

        This method is nearly identical to DataParallelPPOActor.update_policy,
        but replaces the policy loss computation with FlowRL objectives.
        """
        # Import needed here to avoid circular imports
        from verl import DataProto
        from verl.utils.device import get_device_id
        from verl.utils.profiler import GPUMemoryLogger
        import logging
        logger = logging.getLogger(__file__)

        # Make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    from verl.utils.seqlen_balancing import prepare_dynamic_batch
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    # Only get ref_log_prob if use_kl_loss is True
                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                    else:
                        ref_log_prob = None

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # Forward pass
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    if on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]

                    # ==== FlowRL: Use FlowRL objectives instead of PPO loss ====
                    flowrl_objective = os.getenv('FLOWRL_OBJECTIVE', 'vanilla')

                    if flowrl_objective == 'vanilla':
                        policy_loss, flowrl_metrics = compute_flowrl(
                            log_prob=log_prob,
                            ref_log_prob=ref_log_prob,
                            old_log_prob=old_log_prob,
                            reward=advantages,
                            response_mask=response_mask,
                            beta_coef=self.flowrl_beta_coef,
                        )
                    elif flowrl_objective == 'old_policy':
                        policy_loss, flowrl_metrics = compute_flowrl_with_old_policy(
                            log_prob=log_prob,
                            ref_log_prob=ref_log_prob,
                            old_log_prob=old_log_prob,
                            reward=advantages,
                            response_mask=response_mask,
                            beta_coef=self.flowrl_beta_coef,
                        )
                    elif flowrl_objective == 'no_log_z':
                        policy_loss, flowrl_metrics = compute_flowrl_no_log_z(
                            log_prob=log_prob,
                            ref_log_prob=ref_log_prob,
                            old_log_prob=old_log_prob,
                            reward=advantages,
                            response_mask=response_mask,
                            beta_coef=self.flowrl_beta_coef,
                        )
                    elif flowrl_objective == 'old_policy_no_log_z':
                        policy_loss, flowrl_metrics = compute_flowrl_old_policy_no_log_z(
                            log_prob=log_prob,
                            ref_log_prob=ref_log_prob,
                            old_log_prob=old_log_prob,
                            reward=advantages,
                            response_mask=response_mask,
                            beta_coef=self.flowrl_beta_coef,
                        )
                    else:
                        raise ValueError(f"Unknown FLOWRL_OBJECTIVE: {flowrl_objective}. "
                                       f"Supported values: 'vanilla', 'old_policy', 'no_log_z', 'old_policy_no_log_z'")

                    loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics.update(flowrl_metrics)
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)

        self.actor_optimizer.zero_grad()
        return metrics
