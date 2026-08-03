"""Unit tests for IGPO advantage math and advantage-collapse diagnostics."""

from __future__ import annotations

from igpo.advantage.turn_level import (
    TurnRewardTrajectory,
    advantage_collapse_rate,
    compute_igpo_advantages,
    discounted_turn_advantages,
    grpo_outcome_advantages,
    normalize_group_rewards,
)


def test_discounted_advantages_gamma_one():
    assert discounted_turn_advantages([0.1, 0.2, 1.0], gamma=1.0) == [1.3, 1.2, 1.0]


def test_default_gamma_preserves_temporal_weighting():
    got = discounted_turn_advantages([1.0, 0.0, 0.0], gamma=0.95)
    assert abs(got[0] - 1.0) < 1e-8
    assert abs(got[1]) < 1e-8


def test_discounted_advantages_gamma_half():
    got = discounted_turn_advantages([1.0, 1.0, 1.0], gamma=0.5)
    assert abs(got[2] - 1.0) < 1e-8
    assert abs(got[1] - 1.5) < 1e-8
    assert abs(got[0] - 1.75) < 1e-8


def test_grpo_collapse_all_wrong():
    outcomes = [0.0, 0.0, 0.0, 0.0]
    group_ids = [0, 0, 0, 0]
    adv = grpo_outcome_advantages(outcomes, group_ids)
    assert all(abs(a) < 1e-8 for a in adv)
    assert advantage_collapse_rate(outcomes, group_ids) == 1.0


def test_grpo_collapse_all_correct():
    outcomes = [1.0, 1.0, 1.0, 1.0]
    group_ids = [0, 0, 0, 0]
    adv = grpo_outcome_advantages(outcomes, group_ids)
    assert all(abs(a) < 1e-8 for a in adv)


def test_igpo_nonzero_when_outcome_collapsed():
    # All outcomes identical (collapse for GRPO), but IG differs across turns/rollouts.
    trajs = [
        TurnRewardTrajectory(info_gains=[0.10, -0.02], outcome=0.0, group_id=0),
        TurnRewardTrajectory(info_gains=[0.01, 0.20], outcome=0.0, group_id=0),
        TurnRewardTrajectory(info_gains=[-0.05, 0.00], outcome=0.0, group_id=0),
        TurnRewardTrajectory(info_gains=[0.08, 0.03], outcome=0.0, group_id=0),
    ]
    assert advantage_collapse_rate([0, 0, 0, 0], [0, 0, 0, 0]) == 1.0
    advs = compute_igpo_advantages(trajs, gamma=1.0, norm_mode="separate")
    # At least some turn advantages should be non-zero thanks to IG variance.
    flat = [a for row in advs for a in row]
    assert any(abs(x) > 1e-6 for x in flat)


def test_separate_vs_joint_normalization_shapes():
    trajs = [
        TurnRewardTrajectory(info_gains=[0.2], outcome=1.0, group_id=0),
        TurnRewardTrajectory(info_gains=[-0.1], outcome=0.0, group_id=0),
    ]
    sep = normalize_group_rewards(trajs, norm_mode="separate")
    joint = normalize_group_rewards(trajs, norm_mode="joint")
    assert len(sep) == 2 and len(joint) == 2
    assert len(sep[0]) == 2 and len(joint[0]) == 2
