"""Turn-level group-relative advantage for IGPO (paper Eq. 5–7).

Pipeline:
  1. Collect all turn rewards in a GRPO group: IG turns + final outcome turn
  2. Z-normalize (joint or separate for IG vs outcome)
  3. Discounted cumulative advantage: Ã_t = Σ_{k=t}^T γ^{k-t} A_k
  4. Broadcast Ã_t to all decision tokens of turn t
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


NormMode = Literal["joint", "separate"]


@dataclass
class TurnRewardTrajectory:
    """One rollout's dense reward vector."""

    info_gains: list[float]  # length T-1 (may be empty for single-turn)
    outcome: float  # final-turn reward (F1 or format penalty)
    group_id: int = 0

    @property
    def rewards(self) -> list[float]:
        return list(self.info_gains) + [self.outcome]

    @property
    def num_turns(self) -> int:
        return len(self.rewards)


def _zscore(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if values.size == 0:
        return values
    mean = values.mean()
    std = values.std()
    if std < eps:
        return values - mean  # zero when all equal → advantage collapse for that set
    return (values - mean) / (std + eps)


def normalize_group_rewards(
    trajectories: Sequence[TurnRewardTrajectory],
    *,
    norm_mode: NormMode = "separate",
    eps: float = 1e-6,
) -> list[list[float]]:
    """Group-wise z-normalization of turn rewards.

    separate: normalize IG rewards and outcome rewards independently (official default in train.sh)
    joint: normalize all turn rewards together (paper Eq. 6)
    """
    if not trajectories:
        return []

    groups = sorted({t.group_id for t in trajectories})
    out: list[list[float] | None] = [None] * len(trajectories)

    for gid in groups:
        idxs = [i for i, t in enumerate(trajectories) if t.group_id == gid]
        group_trajs = [trajectories[i] for i in idxs]

        if norm_mode == "joint":
            pool = np.array([r for t in group_trajs for r in t.rewards], dtype=np.float64)
            normed = _zscore(pool, eps=eps)
            cursor = 0
            for local_i, i in enumerate(idxs):
                n = group_trajs[local_i].num_turns
                out[i] = normed[cursor : cursor + n].tolist()
                cursor += n
            continue

        # separate
        ig_pool = np.array(
            [r for t in group_trajs for r in t.info_gains], dtype=np.float64
        )
        outcome_pool = np.array([t.outcome for t in group_trajs], dtype=np.float64)
        ig_normed = _zscore(ig_pool, eps=eps)
        outcome_normed = _zscore(outcome_pool, eps=eps)

        ig_cursor = 0
        for local_i, i in enumerate(idxs):
            t = group_trajs[local_i]
            n_ig = len(t.info_gains)
            piece = ig_normed[ig_cursor : ig_cursor + n_ig].tolist() if n_ig else []
            ig_cursor += n_ig
            out[i] = piece + [float(outcome_normed[local_i])]

    return [x if x is not None else [] for x in out]


def discounted_turn_advantages(
    normalized_rewards: Sequence[float],
    *,
    gamma: float = 1.0,
) -> list[float]:
    """Ã_t = Σ_{k=t}^T γ^{k-t} A_k  (paper Eq. 7)."""
    n = len(normalized_rewards)
    if n == 0:
        return []
    adv = [0.0] * n
    running = 0.0
    for t in range(n - 1, -1, -1):
        running = float(normalized_rewards[t]) + gamma * running
        adv[t] = running
    return adv


def compute_igpo_advantages(
    trajectories: Sequence[TurnRewardTrajectory],
    *,
    gamma: float = 1.0,
    norm_mode: NormMode = "separate",
    eps: float = 1e-6,
) -> list[list[float]]:
    """Full IGPO advantage pipeline for a batch of grouped rollouts."""
    normalized = normalize_group_rewards(trajectories, norm_mode=norm_mode, eps=eps)
    return [discounted_turn_advantages(r, gamma=gamma) for r in normalized]


def grpo_outcome_advantages(
    outcomes: Sequence[float],
    group_ids: Sequence[int],
    *,
    eps: float = 1e-6,
) -> list[float]:
    """Classic GRPO: one scalar advantage per rollout from outcome reward only."""
    outcomes_arr = np.asarray(outcomes, dtype=np.float64)
    group_ids_arr = np.asarray(group_ids)
    adv = np.zeros_like(outcomes_arr)
    for gid in np.unique(group_ids_arr):
        mask = group_ids_arr == gid
        adv[mask] = _zscore(outcomes_arr[mask], eps=eps)
    return adv.tolist()


def advantage_collapse_rate(
    outcomes: Sequence[float],
    group_ids: Sequence[int],
    *,
    eps: float = 1e-8,
) -> float:
    """Fraction of groups whose outcome rewards have near-zero variance.

    This is the metric behind paper Figure 1 (zero-advantage groups).
    """
    outcomes_arr = np.asarray(outcomes, dtype=np.float64)
    group_ids_arr = np.asarray(group_ids)
    groups = np.unique(group_ids_arr)
    if len(groups) == 0:
        return 0.0
    collapsed = 0
    for gid in groups:
        vals = outcomes_arr[group_ids_arr == gid]
        if vals.std() < eps:
            collapsed += 1
    return collapsed / len(groups)
