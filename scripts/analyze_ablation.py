#!/usr/bin/env python3
"""Aggregate ablation outputs and compute Welch t-tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze IGPO/GRPO ablation results.")
    parser.add_argument("--input_dir", type=str, default="./ablation_results")
    parser.add_argument("--metric", type=str, default="collapse_rate")
    return parser.parse_args()


def collect(root: Path, algo: str, metric: str) -> list[float]:
    values = []
    for path in sorted(root.glob(f"{algo}_seed*/summary.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        final = obj.get("final") or {}
        if metric in final:
            values.append(float(final[metric]))
        else:
            # Prefer held-out metrics when present in eval files.
            eval_path = path.parent / "eval_metrics.jsonl"
            if eval_path.exists():
                lines = [json.loads(x) for x in eval_path.read_text(encoding="utf-8").splitlines() if x.strip()]
                if lines and metric in lines[-1]:
                    values.append(float(lines[-1][metric]))
    return values


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir)
    igpo = collect(root, "igpo", args.metric)
    grpo = collect(root, "grpo", args.metric)
    report = {
        "metric": args.metric,
        "igpo": {
            "n": len(igpo),
            "mean": float(np.mean(igpo)) if igpo else None,
            "std": float(np.std(igpo, ddof=1)) if len(igpo) > 1 else 0.0,
            "values": igpo,
        },
        "grpo": {
            "n": len(grpo),
            "mean": float(np.mean(grpo)) if grpo else None,
            "std": float(np.std(grpo, ddof=1)) if len(grpo) > 1 else 0.0,
            "values": grpo,
        },
    }
    if len(igpo) >= 2 and len(grpo) >= 2:
        t_stat, p_value = stats.ttest_ind(igpo, grpo, equal_var=False)
        report["welch_ttest"] = {"t": float(t_stat), "p": float(p_value)}
    out = root / "analysis.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
