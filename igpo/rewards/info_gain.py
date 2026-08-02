"""Intrinsic information-gain process rewards (paper Eq. 3–4).

At turn t we teacher-force the ground-truth answer under the current context
and measure how much the policy's belief in the correct answer increased:

    π_θ(a | q, o_≤t) = exp( mean_j log π_θ(a_j | q, o_≤t, a_<j) )
    r_t = π_θ(a | q, o_≤t) - π_θ(a | q, o_≤t-1)     # prob_diff (paper default)

Alternatively with log_prob_diff:
    r_t = mean(log π_t) - mean(log π_{t-1})

The IG reward is stop-gradient: it is used as a scalar advantage signal only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn.functional as F


InfoGainType = Literal["prob_diff", "log_prob_diff"]


# Matches official IGPO GT wrapping for teacher forcing.
GT_PREFIX = "\nNow there's enough information to answer\n</think>\n<answer>\n"
GT_SUFFIX = "\n</answer>"


@dataclass
class BeliefState:
    """Belief about the ground-truth answer under a given context."""

    mean_log_prob: float
    value: float  # probability or log-prob depending on info_gain_type
    token_log_probs: list[float]


def wrap_ground_truth(answer: str) -> str:
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
    """Compute per-token log-probs of ground-truth answer under teacher forcing.

    Args:
        model: Causal LM (peft or transformers).
        tokenizer: Matching tokenizer.
        context_ids: 1D LongTensor of context token ids (prompt + prior turns).
        answer_text: Raw ground-truth answer (will be schema-wrapped).
    """
    device = device or context_ids.device
    wrapped = wrap_ground_truth(answer_text)
    answer_ids = tokenizer(wrapped, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].to(device)
    answer_ids = answer_ids.squeeze(0)
    if answer_ids.numel() == 0:
        return BeliefState(mean_log_prob=0.0, value=1.0, token_log_probs=[])

    full = torch.cat([context_ids.to(device), answer_ids], dim=0).unsqueeze(0)
    attn = torch.ones_like(full)
    outputs = model(input_ids=full, attention_mask=attn)
    logits = outputs.logits  # (1, L, V)

    # Next-token prediction: logit at position k predicts token k+1.
    ctx_len = context_ids.numel()
    # Logits that predict answer tokens start at ctx_len - 1.
    pred_logits = logits[0, ctx_len - 1 : ctx_len - 1 + answer_ids.numel(), :]
    log_probs = F.log_softmax(pred_logits, dim=-1)
    token_lp = log_probs.gather(1, answer_ids.unsqueeze(1)).squeeze(1)
    mean_lp = float(token_lp.mean().item())
    return BeliefState(
        mean_log_prob=mean_lp,
        value=math.exp(mean_lp),
        token_log_probs=token_lp.detach().cpu().tolist(),
    )


def belief_value(state: BeliefState, info_gain_type: InfoGainType) -> float:
    if info_gain_type == "log_prob_diff":
        return state.mean_log_prob
    return state.value


def compute_info_gain_rewards(
    beliefs: Sequence[BeliefState],
    *,
    info_gain_type: InfoGainType = "prob_diff",
) -> list[float]:
    """Convert per-turn belief states into adjacent-turn IG rewards.

    beliefs[0] is Turn-0 belief (prompt only, before any interaction).
    beliefs[t] for t>=1 is belief after turn t's tool interaction.
    Returns rewards for turns 1..T-1 (length = len(beliefs) - 1).
    """
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
    """Compute IG rewards for one rollout given cumulative contexts.

    context_id_list[0] = prompt-only context (Turn 0)
    context_id_list[t] = context after turn t interaction for t>=1
    """
    beliefs = [
        teacher_force_answer_logprobs(
            model, tokenizer, ctx, ground_truth, device=device
        )
        for ctx in context_id_list
    ]
    return compute_info_gain_rewards(beliefs, info_gain_type=info_gain_type), beliefs
