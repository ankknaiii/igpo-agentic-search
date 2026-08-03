"""GRPO / IGPO clipped surrogate objectives (paper Eq. 1 / Eq. 8)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PolicyLossOutput:
    """Aggregated statistics for one surrogate-loss evaluation."""

    loss: torch.Tensor
    pg_loss: torch.Tensor
    kl_loss: torch.Tensor
    clipfrac: torch.Tensor
    approx_kl: torch.Tensor
    mean_ratio: torch.Tensor


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute the mean of ``x`` over positions where ``mask`` is positive."""
    denom = mask.sum().clamp_min(1.0)
    return (x * mask).sum() / denom


def compute_token_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token log-probabilities for next-token prediction.

    Args:
        logits: Tensor of shape ``(B, L, V)``.
        labels: Tensor of shape ``(B, L)``.

    Returns:
        Tensor of shape ``(B, L)`` containing token log-probabilities.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def clipped_surrogate_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    clip_eps: float = 0.2,
    kl_coef: float = 0.001,
    ref_logprobs: torch.Tensor | None = None,
) -> PolicyLossOutput:
    """Evaluate the token-level clipped PPO/GRPO surrogate objective.

    Args:
        logprobs: Current-policy log-probabilities, shape ``(B, T)``.
        old_logprobs: Sampling-policy log-probabilities, shape ``(B, T)``.
        advantages: Token-level advantages, shape ``(B, T)``.
        mask: Decision-token mask, shape ``(B, T)``.
        clip_eps: PPO clipping threshold.
        kl_coef: Coefficient of the KL regularizer.
        ref_logprobs: Optional reference-policy log-probabilities.

    Returns:
        ``PolicyLossOutput`` containing loss terms and diagnostics.
    """
    ratio = torch.exp(logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    pg = -torch.min(unclipped, clipped)
    pg_loss = masked_mean(pg, mask)

    clipfrac = masked_mean((torch.abs(ratio - 1.0) > clip_eps).float(), mask)
    approx_kl = masked_mean(old_logprobs - logprobs, mask)
    mean_ratio = masked_mean(ratio, mask)

    if ref_logprobs is not None and kl_coef > 0:
        log_ratio = ref_logprobs - logprobs
        kl = (torch.exp(log_ratio) - 1.0) - log_ratio
        kl_loss = kl_coef * masked_mean(kl, mask)
    else:
        kl_loss = torch.zeros((), device=logprobs.device, dtype=logprobs.dtype)

    loss = pg_loss + kl_loss
    return PolicyLossOutput(
        loss=loss,
        pg_loss=pg_loss.detach(),
        kl_loss=kl_loss.detach() if torch.is_tensor(kl_loss) else kl_loss,
        clipfrac=clipfrac.detach(),
        approx_kl=approx_kl.detach(),
        mean_ratio=mean_ratio.detach(),
    )


def broadcast_turn_advantages_to_tokens(
    turn_advantages: list[float],
    turn_token_spans: list[tuple[int, int]],
    seq_len: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Broadcast each turn advantage onto its decision-token span ``[start, end)``."""
    adv = torch.zeros(seq_len, device=device, dtype=dtype)
    for a, (start, end) in zip(turn_advantages, turn_token_spans):
        if end > start:
            adv[start:end] = float(a)
    return adv
