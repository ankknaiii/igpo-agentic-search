#!/usr/bin/env python3
"""CPU/GPU smoke test: algorithm units + optional 1-step tiny train."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=0, help="If >0, run tiny training")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    # 1) Pure unit-level checks without model download.
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
    advs = compute_igpo_advantages(trajs, gamma=1.0, norm_mode="separate")
    assert any(abs(x) > 1e-6 for row in advs for x in row)
    print("[ok] advantage collapse + IGPO dense signal")

    from igpo.env.mock_kb import search_kb

    hits = search_kb("capital of France Paris")
    assert hits and "Paris" in hits[0]["title"]
    print("[ok] mock search KB")

    if args.train_steps <= 0:
        print("[done] smoke (no training)")
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
        output_dir="./outputs/smoke",
    )
    hist = run_training(cfg)
    print("[ok] train steps:", len(hist), "last=", hist[-1] if hist else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
