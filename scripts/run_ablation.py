#!/usr/bin/env python3
"""Run GRPO vs IGPO ablations over multiple random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from igpo.train.trainer import TrainConfig, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IGPO/GRPO ablation runner.")
    parser.add_argument("--algo", choices=["grpo", "igpo"], required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--output_dir", type=str, default="./ablation_results")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    for seed in args.seed:
        out = root / f"{args.algo}_seed{seed}"
        cfg = TrainConfig(
            model_name=args.model_name,
            algo=args.algo,
            seed=seed,
            max_steps=args.max_steps,
            output_dir=str(out),
            eval_every=max(1, args.max_steps // 2),
        )
        history = run_training(cfg)
        summary = {
            "algo": args.algo,
            "seed": seed,
            "final": history[-1].__dict__ if history else {},
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[ok] {args.algo} seed={seed} -> {out}")


if __name__ == "__main__":
    main()
