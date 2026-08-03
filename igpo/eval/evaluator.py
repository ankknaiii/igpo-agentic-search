"""Held-out evaluation for IGPO / GRPO agentic search policies."""

from __future__ import annotations

from typing import Any

import torch

from igpo.advantage.turn_level import advantage_collapse_rate
from igpo.agent.rollout import SearchRolloutEngine
from igpo.train.data import default_eval_data_path, load_qa_jsonl


class IGPOEvaluator:
    """Evaluate a policy on a held-out offline QA split."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        eval_data_path: str = "",
        max_turns: int = 3,
        max_new_tokens: int = 128,
        info_gain_type: str = "prob_diff",
        device: torch.device | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        path = eval_data_path or str(default_eval_data_path())
        self.dataset = load_qa_jsonl(path)
        self.device = device or next(model.parameters()).device
        self.engine = SearchRolloutEngine(
            model,
            tokenizer,
            max_turns=max_turns,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            info_gain_type=info_gain_type,
            device=self.device,
        )

    @torch.no_grad()
    def evaluate(self, n_samples: int = 100) -> dict[str, Any]:
        """Evaluate greedy rollouts on up to ``n_samples`` held-out examples.

        Args:
            n_samples: Maximum number of evaluation samples.

        Returns:
            Dictionary with ``mean_f1``, ``mean_em``, ``collapse_rate``, ``mean_abs_ig``.
        """
        samples = self.dataset[: min(n_samples, len(self.dataset))]
        if not samples:
            return {
                "mean_f1": 0.0,
                "mean_em": 0.0,
                "collapse_rate": 0.0,
                "mean_abs_ig": 0.0,
                "n_samples": 0,
            }

        self.model.eval()
        f1s, ems, outcomes, igs = [], [], [], []
        for sample in samples:
            r = self.engine.rollout(sample["question"], sample["answer"])
            f1s.append(r.outcome["f1"])
            ems.append(r.outcome["em"])
            outcomes.append(r.outcome["reward"])
            igs.extend(r.info_gains)

        # Single-sample groups: collapse rate is defined over identical outcome groups.
        group_ids = list(range(len(outcomes)))
        return {
            "mean_f1": float(sum(f1s) / len(f1s)),
            "mean_em": float(sum(ems) / len(ems)),
            "collapse_rate": float(advantage_collapse_rate(outcomes, group_ids)),
            "mean_abs_ig": float(sum(abs(x) for x in igs) / max(1, len(igs))),
            "n_samples": len(samples),
        }
