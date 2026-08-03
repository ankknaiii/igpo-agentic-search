"""Dataset loading utilities for offline QA and hub-backed benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_qa_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL QA dataset.

    Args:
        path: Path to a JSONL file with ``question`` and ``answer`` fields.

    Returns:
        List of sample dictionaries.
    """
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def default_data_path() -> Path:
    """Return the default offline training dataset path."""
    return Path(__file__).resolve().parents[1] / "data" / "qa_offline.jsonl"


def default_eval_data_path() -> Path:
    """Return the default held-out evaluation dataset path."""
    return Path(__file__).resolve().parents[1] / "data" / "qa_eval.jsonl"


def load_qa_from_hub(
    dataset_name: str = "hotpot_qa",
    split: str = "validation",
    *,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load a standard QA benchmark from ModelScope Hub with HuggingFace fallback.

    Args:
        dataset_name: Dataset identifier.
        split: Dataset split name.
        max_samples: Optional maximum number of retained rows.

    Returns:
        List of normalized samples with ``question`` and ``answer`` keys.
    """
    rows: list[dict[str, Any]] = []
    try:
        from modelscope.msdatasets import MsDataset

        ds = MsDataset.load(dataset_name, split=split)
        raw = [row for row in ds]
    except Exception:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split=split)
        raw = [dict(row) for row in ds]

    for row in raw:
        q = row.get("question") or row.get("query") or ""
        a = row.get("answer") or row.get("answers") or ""
        if isinstance(a, list):
            a = a[0] if a else ""
        if isinstance(a, dict):
            texts = a.get("text") or []
            a = texts[0] if texts else ""
        if not q or not a:
            continue
        rows.append({"question": str(q), "answer": str(a), **{k: v for k, v in row.items() if k not in {"question", "answer"}}})
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows
