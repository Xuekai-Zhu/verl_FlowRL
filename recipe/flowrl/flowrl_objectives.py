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

All functions follow token-level computation style consistent with veRL core algorithms.
"""

import torch
import verl.utils.torch_functional as verl_F


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str = "token-mean"):
    """
    Aggregate token-level losses to scalar loss.

    Args:
        loss_mat: Token-level loss matrix, shape: (bs, response_length)
        loss_mask: Binary mask for valid tokens, shape: (bs, response_length)
        loss_agg_mode: Aggregation mode
            - "token-mean": Average over all valid tokens
            - "seq-mean-token-sum": Sum tokens per sequence, then mean over sequences
            - "seq-mean-token-mean": Mean tokens per sequence, then mean over sequences

    Returns:
        Scalar loss value
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.mean(seq_losses)
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = verl_F.masked_mean(loss_mat, loss_mask, axis=-1)
        loss = torch.mean(seq_losses)
    else:
        raise ValueError(f"Unknown loss_agg_mode: {loss_agg_mode}")
    return loss


def compute_flowrl(
    log_prob=None,
    ref_log_prob=None,
    old_log_prob=None,
    reward=None,
    response_mask=None,
    beta_coef=15.0,
    clip_ratio=None,
    rollout_log_probs=None,
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
    log_prob=None,
    old_log_prob=None,
    reward=None,
    response_mask=None,
    beta_coef=15.0,
    clip_ratio=None,
    rollout_log_probs=None,
):
    """
    FlowRL objective using OLD POLICY instead of reference policy.

    Same as compute_flowrl but with ref_log_prob replaced by old_log_prob in the residual.

    Args:
        log_prob: Current policy log probabilities (bs, response_len)
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

    # KL divergences (only old policy KL, no ref policy)
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

    return avg_loss, loss_term_dict


def compute_flowrl_no_log_z(
    log_prob=None,
    ref_log_prob=None,
    old_log_prob=None,
    reward=None,
    response_mask=None,
    beta_coef=15.0,
    clip_ratio=None,
    rollout_log_probs=None,
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
    log_prob=None,
    old_log_prob=None,
    reward=None,
    response_mask=None,
    beta_coef=15.0,
    clip_ratio=None,
    rollout_log_probs=None,
):
    """
    FlowRL objective using OLD POLICY and WITHOUT log Z (combined ablation).

    Combines two modifications:
    1. Uses old policy instead of reference policy
    2. Removes the partition function estimation from the objective

    Args:
        log_prob: Current policy log probabilities (bs, response_len)
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
    # delta = avg_log_prob - beta_coef * seq_log_reward

    # Importance ratio from current vs old policy
    log_w = verl_F.masked_sum(log_prob - old_log_prob, response_mask, axis=1)
    imp_w_raw = torch.exp(log_w).detach()
    imp_w = torch.clamp(imp_w_raw, max=10)

    # Loss: weighted squared residual with clipped importance weights
    weighted_losses = imp_w * (delta ** 2)
    avg_loss = torch.mean(weighted_losses)

    # KL divergences (only old policy KL, no ref policy)
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Metrics (no log_z, no ref_log_prob in this version)
    loss_term_dict = {
        "actor/log_prob": verl_F.masked_mean(log_prob, response_mask).detach().item(),
        "actor/old_log_prob": verl_F.masked_mean(old_log_prob, response_mask).detach().item(),
        "actor/log_reward": verl_F.masked_mean(reward, response_mask).detach().item(),
        "actor/final_loss": avg_loss.detach().item(),
        "actor/importance_weight": imp_w.mean().detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
    }

    return avg_loss, loss_term_dict


def compute_flowrl_old_policy_no_log_z_token_level(
    log_prob=None,
    old_log_prob=None,
    reward=None,
    response_mask=None,
    beta_coef: float = 15.0,
    loss_agg_mode: str = "token-mean",
    imp_weight_clip_low: float = 0.2,
    imp_weight_clip_high: float = 0.28,
):
    # Token-level FlowRL residual WITHOUT log_z and using old policy:
    # residual_t = log_prob_t - β*reward_t - old_log_prob_t
    # Shape: (bs, response_length)
    token_residual = log_prob - beta_coef * reward - old_log_prob

    # Compute TOKEN-LEVEL importance weights (following CISPO pattern)
    # ratio_t = exp(log_prob_t - old_log_prob_t), shape: (bs, response_length)
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)  # Numerical stability
    ratio = torch.exp(negative_approx_kl)

    # Detach and clip importance weights around 1.0 (CISPO-style)
    ratio = ratio.detach()
    imp_w_token = torch.clamp(
        ratio,
        min=1.0 - imp_weight_clip_low,
        max=1.0 + imp_weight_clip_high
    )  # Shape: (bs, response_length), e.g., [0.8, 1.28]

    # Token-level squared residuals, shape: (bs, response_length)
    token_sq_residuals = token_residual ** 2

    # Weighted token-level losses (fully token-level), shape: (bs, response_length)
    weighted_token_losses = imp_w_token * token_sq_residuals

    # Aggregate to scalar loss using configurable mode
    avg_loss = agg_loss(weighted_token_losses, response_mask, loss_agg_mode)

    # KL divergence (token-level, then aggregated)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Clipping statistics
    clipped_high = (ratio > 1.0 + imp_weight_clip_high).float()
    clipped_low = (ratio < 1.0 - imp_weight_clip_low).float()
    clip_frac_high = verl_F.masked_mean(clipped_high, response_mask)
    clip_frac_low = verl_F.masked_mean(clipped_low, response_mask)
    clip_frac_total = verl_F.masked_mean(clipped_high + clipped_low, response_mask)

    # Metrics
    residual_var = verl_F.masked_var(token_residual, response_mask)
    residual_std = torch.sqrt(residual_var + 1e-8)

    loss_term_dict = {
        "actor/log_prob": verl_F.masked_mean(log_prob, response_mask).detach().item(),
        "actor/old_log_prob": verl_F.masked_mean(old_log_prob, response_mask).detach().item(),
        "actor/log_reward": verl_F.masked_mean(reward, response_mask).detach().item(),
        "actor/final_loss": avg_loss.detach().item(),
        "actor/importance_weight": verl_F.masked_mean(imp_w_token, response_mask).detach().item(),
        "actor/imp_weight_raw": verl_F.masked_mean(ratio, response_mask).detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/residual_mean": verl_F.masked_mean(token_residual, response_mask).detach().item(),
        "actor/residual_std": residual_std.detach().item(),
        "actor/clip_frac_high": clip_frac_high.detach().item(),
        "actor/clip_frac_low": clip_frac_low.detach().item(),
        "actor/clip_frac_total": clip_frac_total.detach().item(),
    }

    return avg_loss, loss_term_dict
