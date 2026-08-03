"""Intrinsic information-gain process rewards (paper Eq. 3–4).

At turn ``t``, the ground-truth answer is teacher-forced under the current
context. The immediate process reward is the adjacent-turn change in the
policy's probability (or mean log-probability) of that answer.

References:
    Wang et al., IGPO, ICLR 2026, arXiv:2510.14967.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn.functional as F


InfoGainType = Literal["prob_diff", "log_prob_diff"]

GT_PREFIX = "\nNow there's enough information to answer\n</think>\n<answer>\n"
GT_SUFFIX = "\n</answer>"


@dataclass
class BeliefState:
    """Belief about the ground-truth answer under a given context."""

    mean_log_prob: float
    value: float
    token_log_probs: list[float]


def wrap_ground_truth(answer: str) -> str:
    """Wrap a ground-truth answer using the official IGPO schema."""
    return f"{GT_PREFIX}{answer}{GT_SUFFIX}"


@torch.no_grad()
def teacher_force_answer_logprobs(
    model,
    tokenizer,
    context_ids: torch.Tensor,
    answer_text: str,
    *,
    device: torch.device | None = None,
) -> BeliefState:
    """Compute per-token log-probabilities of the ground-truth answer.

    Context is forwarded once with KV cache retained; answer tokens are then
    scored under that cache to avoid redundant full-sequence recomputation.

    Args:
        model: Causal language model.
        tokenizer: Matching tokenizer.
        context_ids: One-dimensional context token ids.
        answer_text: Raw ground-truth answer text.
        device: Target device.

    Returns:
        ``BeliefState`` containing mean log-probability and token-level scores.
    """
    device = device or context_ids.device
    wrapped = wrap_ground_truth(answer_text)
    answer_ids = tokenizer(wrapped, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].to(device)
    answer_ids = answer_ids.squeeze(0)
    if answer_ids.numel() == 0:
        return BeliefState(mean_log_prob=0.0, value=1.0, token_log_probs=[])

    ctx = context_ids.to(device).unsqueeze(0)
    ctx_attn = torch.ones_like(ctx)
    ctx_out = model(input_ids=ctx, attention_mask=ctx_attn, use_cache=True)
    past = ctx_out.past_key_values

    # Logit at the last context position predicts the first answer token.
    first_logits = ctx_out.logits[:, -1:, :]
    ans = answer_ids.unsqueeze(0)
    ans_attn = torch.ones_like(ans)
    ans_out = model(
        input_ids=ans,
        attention_mask=torch.cat([ctx_attn, ans_attn], dim=1),
        past_key_values=past,
        use_cache=False,
    )
    # Remaining answer tokens are predicted by answer-position logits.
    rest_logits = ans_out.logits[:, :-1, :] if ans.shape[1] > 1 else ans_out.logits[:, :0, :]
    pred_logits = torch.cat([first_logits, rest_logits], dim=1)
    if pred_logits.shape[1] != ans.shape[1]:
        # Fallback to a single full forward if cache shapes diverge across backends.
        full = torch.cat([context_ids.to(device), answer_ids], dim=0).unsqueeze(0)
        outputs = model(input_ids=full, attention_mask=torch.ones_like(full))
        ctx_len = context_ids.numel()
        pred_logits = outputs.logits[:, ctx_len - 1 : ctx_len - 1 + answer_ids.numel(), :]

    log_probs = F.log_softmax(pred_logits, dim=-1)
    token_lp = log_probs.gather(2, ans.unsqueeze(-1)).squeeze(-1).squeeze(0)
    mean_lp = float(token_lp.mean().item())
    return BeliefState(
        mean_log_prob=mean_lp,
        value=math.exp(mean_lp),
        token_log_probs=token_lp.detach().cpu().tolist(),
    )


def belief_value(state: BeliefState, info_gain_type: InfoGainType) -> float:
    """Map a belief state to a scalar used by adjacent-turn differencing."""
    if info_gain_type == "log_prob_diff":
        return state.mean_log_prob
    return state.value


def compute_info_gain_rewards(
    beliefs: Sequence[BeliefState],
    *,
    info_gain_type: InfoGainType = "prob_diff",
) -> list[float]:
    """Convert per-turn belief states into adjacent-turn information-gain rewards."""
    if len(beliefs) < 2:
        return []

    rewards: list[float] = []
    prev = belief_value(beliefs[0], info_gain_type)
    for state in beliefs[1:]:
        cur = belief_value(state, info_gain_type)
        ig = cur - prev
        if math.isnan(ig) or math.isinf(ig):
            rewards.append(0.0)
        else:
            rewards.append(float(ig))
        prev = cur
    return rewards


@torch.no_grad()
def compute_trajectory_info_gains(
    model,
    tokenizer,
    context_id_list: Sequence[torch.Tensor],
    ground_truth: str,
    *,
    info_gain_type: InfoGainType = "prob_diff",
    device: torch.device | None = None,
) -> tuple[list[float], list[BeliefState]]:
    """Compute information-gain rewards for one rollout given cumulative contexts."""
    beliefs = [
        teacher_force_answer_logprobs(
            model, tokenizer, ctx, ground_truth, device=device
        )
        for ctx in context_id_list
    ]
    return compute_info_gain_rewards(beliefs, info_gain_type=info_gain_type), beliefs
