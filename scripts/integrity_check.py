#!/usr/bin/env python3
"""Integrity checks for advantage collapse diagnostics and offline retrieval."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run IGPO integrity checks.")
    parser.add_argument("--train-steps", type=int, default=0, help="Optional lightweight training steps.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    from igpo.advantage.turn_level import (
        TurnRewardTrajectory,
        advantage_collapse_rate,
        compute_igpo_advantages,
        grpo_outcome_advantages,
    )

    outcomes = [0.0, 0.0, 0.0, 0.0]
    assert advantage_collapse_rate(outcomes, [0, 0, 0, 0]) == 1.0
    assert all(abs(a) < 1e-8 for a in grpo_outcome_advantages(outcomes, [0, 0, 0, 0]))

    trajs = [
        TurnRewardTrajectory([0.2, 0.0], 0.0, 0),
        TurnRewardTrajectory([0.0, 0.3], 0.0, 0),
        TurnRewardTrajectory([-0.1, 0.1], 0.0, 0),
        TurnRewardTrajectory([0.05, -0.05], 0.0, 0),
    ]
    advs = compute_igpo_advantages(trajs, gamma=0.95, norm_mode="separate")
    assert any(abs(x) > 1e-6 for row in advs for x in row)
    print("[ok] advantage collapse diagnostics and IGPO dense signal")

    from igpo.env.mock_kb import DOCUMENTS, search_kb

    assert len(DOCUMENTS) >= 50
    hits = search_kb("capital of France Paris", noise=False)
    assert hits and "Paris" in hits[0]["title"]
    print("[ok] offline retrieval corpus")

    from igpo.rewards.f1 import word_f1

    # pred=[the,the,paris], gold=[the,paris] => precision=2/3, recall=1, F1=0.8
    assert abs(word_f1("the the paris", "the paris") - 0.8) < 1e-6
    print("[ok] SQuAD-style word F1")

    if args.train_steps <= 0:
        print("[done] integrity check")
        return 0

    from igpo.train.trainer import TrainConfig, run_training

    cfg = TrainConfig(
        model_name=args.model,
        algo="igpo",
        max_steps=args.train_steps,
        prompts_per_step=1,
        group_size=2,
        max_turns=2,
        max_new_tokens=64,
        ppo_epochs=2,
        output_dir="./outputs/integrity",
        eval_every=0,
    )
    hist = run_training(cfg)
    print("[ok] lightweight training validation:", hist[-1] if hist else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
