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
FlowRL objective functions for trajectory balance learning.

This module contains pure objective functions that can be used with standard veRL actors.
No custom actor classes needed - just use these functions in place of PPO loss.
"""

import torch
import verl.utils.torch_functional as verl_F


def compute_flowrl(
    log_prob,
    ref_log_prob,
    old_log_prob,
    reward,
    response_mask,
    beta_coef=15.0,
    clip_ratio=None,
    rollout_log_probs=None
):
    """
    FlowRL objective with importance weight clipping (max=10), using reference policy.

    Args:
        log_prob: Current policy log probabilities (bs, response_len)
        ref_log_prob: Reference policy log probabilities (bs, response_len)
        old_log_prob: Old policy log probabilities (bs, response_len)
        reward: Rewards (bs, response_len)
        response_mask: Mask for valid tokens (bs, response_len)
        beta_coef: β coefficient for reward scaling (default: 15.0)
        clip_ratio: Unused (kept for API compatibility)
        rollout_log_probs: Unused (kept for API compatibility)

    Returns:
        avg_loss: Scalar loss value
        loss_term_dict: Dictionary of metrics
    """
    # Average token log-probs & rewards over valid positions
    avg_log_prob = verl_F.masked_mean(log_prob, response_mask, axis=1)
    avg_ref_log_prob = verl_F.masked_mean(ref_log_prob, response_mask, axis=1)
    seq_log_reward = verl_F.masked_mean(reward, response_mask, axis=1)

    # FlowRL residual: logpf - β*R - logpref
    delta = avg_log_prob - beta_coef * seq_log_reward - avg_ref_log_prob

    # Importance ratio from current vs old policy
    log_w = verl_F.masked_sum(log_prob - old_log_prob, response_mask, axis=1)
    imp_w_raw = torch.exp(log_w).detach()
    imp_w = torch.clamp(imp_w_raw, max=10)

    # Loss: weighted squared residual with clipped importance weights
    weighted_losses = imp_w * (delta ** 2)
    avg_loss = torch.mean(weighted_losses)

    # KL divergences
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    approx_kl_ref = log_prob - ref_log_prob
    ref_kl = verl_F.masked_mean(-approx_kl_ref, response_mask)

    # Metrics
    loss_term_dict = {
        "actor/log_prob": verl_F.masked_mean(log_prob, response_mask).detach().item(),
        "actor/old_log_prob": verl_F.masked_mean(old_log_prob, response_mask).detach().item(),
        "actor/ref_log_prob": verl_F.masked_mean(ref_log_prob, response_mask).detach().item(),
        "actor/log_reward": verl_F.masked_mean(reward, response_mask).detach().item(),
        "actor/final_loss": avg_loss.detach().item(),
        "actor/importance_weight": imp_w.mean().detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/ref_kl": ref_kl.detach().item(),
    }

    return avg_loss, loss_term_dict


def compute_flowrl_with_old_policy(
    log_prob,
    ref_log_prob,
    old_log_prob,
    reward,
    response_mask,
    beta_coef=15.0,
    clip_ratio=None,
    rollout_log_probs=None
):
    """
    FlowRL objective using OLD POLICY instead of reference policy.

    Same as compute_flowrl but with ref_log_prob replaced by old_log_prob in the residual.

    Args:
        log_prob: Current policy log probabilities (bs, response_len)
        ref_log_prob: Reference policy log probabilities (bs, response_len) - used for KL metrics only
        old_log_prob: Old policy log probabilities (bs, response_len)
        reward: Rewards (bs, response_len)
        response_mask: Mask for valid tokens (bs, response_len)
        beta_coef: β coefficient for reward scaling (default: 15.0)
        clip_ratio: Unused (kept for API compatibility)
        rollout_log_probs: Unused (kept for API compatibility)

    Returns:
        avg_loss: Scalar loss value
        loss_term_dict: Dictionary of metrics
    """
    # Average token log-probs & rewards over valid positions
    avg_log_prob = verl_F.masked_mean(log_prob, response_mask, axis=1)
    avg_old_log_prob = verl_F.masked_mean(old_log_prob, response_mask, axis=1)
    seq_log_reward = verl_F.masked_mean(reward, response_mask, axis=1)

    # FlowRL residual: logpf - β*R - log_old (using old instead of ref)
    delta = avg_log_prob - beta_coef * seq_log_reward - avg_old_log_prob

    # Importance ratio from current vs old policy
    log_w = verl_F.masked_sum(log_prob - old_log_prob, response_mask, axis=1)
    imp_w_raw = torch.exp(log_w).detach()
    imp_w = torch.clamp(imp_w_raw, max=10)

    # Loss: weighted squared residual with clipped importance weights
    weighted_losses = imp_w * (delta ** 2)
    avg_loss = torch.mean(weighted_losses)

    # KL divergences
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Metrics
    loss_term_dict = {
        "actor/log_prob": verl_F.masked_mean(log_prob, response_mask).detach().item(),
        "actor/old_log_prob": verl_F.masked_mean(old_log_prob, response_mask).detach().item(),
        "actor/log_reward": verl_F.masked_mean(reward, response_mask).detach().item(),
        "actor/final_loss": avg_loss.detach().item(),
        "actor/importance_weight": imp_w.mean().detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
    }

    # Only compute ref_kl if ref_log_prob is provided
    if ref_log_prob is not None:
        approx_kl_ref = log_prob - ref_log_prob
        ref_kl = verl_F.masked_mean(-approx_kl_ref, response_mask)
        loss_term_dict["actor/ref_log_prob"] = verl_F.masked_mean(ref_log_prob, response_mask).detach().item()
        loss_term_dict["actor/ref_kl"] = ref_kl.detach().item()

    return avg_loss, loss_term_dict


def compute_flowrl_no_log_z(
    log_prob,
    ref_log_prob,
    old_log_prob,
    reward,
    response_mask,
    beta_coef=15.0,
    clip_ratio=None,
    rollout_log_probs=None
):
    """
    FlowRL objective WITHOUT log Z (ablation study).

    This is actually the same as compute_flowrl since we removed log_z everywhere.
    Kept for backward compatibility with existing configs.

    Args:
        log_prob: Current policy log probabilities (bs, response_len)
        ref_log_prob: Reference policy log probabilities (bs, response_len)
        old_log_prob: Old policy log probabilities (bs, response_len)
        reward: Rewards (bs, response_len)
        response_mask: Mask for valid tokens (bs, response_len)
        beta_coef: β coefficient for reward scaling (default: 15.0)
        clip_ratio: Unused (kept for API compatibility)
        rollout_log_probs: Unused (kept for API compatibility)

    Returns:
        avg_loss: Scalar loss value
        loss_term_dict: Dictionary of metrics
    """
    # Average token log-probs & rewards over valid positions
    avg_log_prob = verl_F.masked_mean(log_prob, response_mask, axis=1)
    avg_ref_log_prob = verl_F.masked_mean(ref_log_prob, response_mask, axis=1)
    seq_log_reward = verl_F.masked_mean(reward, response_mask, axis=1)

    # FlowRL residual WITHOUT log_z: logpf - β*R - logpref
    delta = avg_log_prob - beta_coef * seq_log_reward - avg_ref_log_prob

    # Importance ratio from current vs old policy
    log_w = verl_F.masked_sum(log_prob - old_log_prob, response_mask, axis=1)
    imp_w_raw = torch.exp(log_w).detach()
    imp_w = torch.clamp(imp_w_raw, max=10)

    # Loss: weighted squared residual with clipped importance weights
    weighted_losses = imp_w * (delta ** 2)
    avg_loss = torch.mean(weighted_losses)

    # KL divergences
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    approx_kl_ref = log_prob - ref_log_prob
    ref_kl = verl_F.masked_mean(-approx_kl_ref, response_mask)

    # Metrics (no log_z in this version)
    loss_term_dict = {
        "actor/log_prob": verl_F.masked_mean(log_prob, response_mask).detach().item(),
        "actor/old_log_prob": verl_F.masked_mean(old_log_prob, response_mask).detach().item(),
        "actor/ref_log_prob": verl_F.masked_mean(ref_log_prob, response_mask).detach().item(),
        "actor/log_reward": verl_F.masked_mean(reward, response_mask).detach().item(),
        "actor/final_loss": avg_loss.detach().item(),
        "actor/importance_weight": imp_w.mean().detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/ref_kl": ref_kl.detach().item(),
    }

    return avg_loss, loss_term_dict


def compute_flowrl_old_policy_no_log_z(
    log_prob,
    ref_log_prob,
    old_log_prob,
    reward,
    response_mask,
    beta_coef=15.0,
    clip_ratio=None,
    rollout_log_probs=None
):
    """
    FlowRL objective using OLD POLICY and WITHOUT log Z (combined ablation).

    Combines two modifications:
    1. Uses old policy instead of reference policy
    2. Removes the partition function estimation from the objective

    Args:
        log_prob: Current policy log probabilities (bs, response_len)
        ref_log_prob: Reference policy log probabilities (bs, response_len) - used for KL metrics only
        old_log_prob: Old policy log probabilities (bs, response_len)
        reward: Rewards (bs, response_len)
        response_mask: Mask for valid tokens (bs, response_len)
        beta_coef: β coefficient for reward scaling (default: 15.0)
        clip_ratio: Unused (kept for API compatibility)
        rollout_log_probs: Unused (kept for API compatibility)

    Returns:
        avg_loss: Scalar loss value
        loss_term_dict: Dictionary of metrics
    """
    # Average token log-probs & rewards over valid positions
    avg_log_prob = verl_F.masked_mean(log_prob, response_mask, axis=1)
    avg_old_log_prob = verl_F.masked_mean(old_log_prob, response_mask, axis=1)
    seq_log_reward = verl_F.masked_mean(reward, response_mask, axis=1)

    # FlowRL residual WITHOUT log_z and using old policy: logpf - β*R - log_old
    delta = avg_log_prob - beta_coef * seq_log_reward - avg_old_log_prob

    # Importance ratio from current vs old policy
    log_w = verl_F.masked_sum(log_prob - old_log_prob, response_mask, axis=1)
    imp_w_raw = torch.exp(log_w).detach()
    imp_w = torch.clamp(imp_w_raw, max=10)

    # Loss: weighted squared residual with clipped importance weights
    weighted_losses = imp_w * (delta ** 2)
    avg_loss = torch.mean(weighted_losses)

    # KL divergences
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Metrics (no log_z in this version)
    loss_term_dict = {
        "actor/log_prob": verl_F.masked_mean(log_prob, response_mask).detach().item(),
        "actor/old_log_prob": verl_F.masked_mean(old_log_prob, response_mask).detach().item(),
        "actor/log_reward": verl_F.masked_mean(reward, response_mask).detach().item(),
        "actor/final_loss": avg_loss.detach().item(),
        "actor/importance_weight": imp_w.mean().detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
    }

    # Only compute ref_kl if ref_log_prob is provided
    if ref_log_prob is not None:
        approx_kl_ref = log_prob - ref_log_prob
        ref_kl = verl_F.masked_mean(-approx_kl_ref, response_mask)
        loss_term_dict["actor/ref_log_prob"] = verl_F.masked_mean(ref_log_prob, response_mask).detach().item()
        loss_term_dict["actor/ref_kl"] = ref_kl.detach().item()

    return avg_loss, loss_term_dict
