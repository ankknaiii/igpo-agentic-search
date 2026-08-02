"""Dataset helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_qa_jsonl(path: str | Path) -> list[dict[str, Any]]:
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
    return Path(__file__).resolve().parents[1] / "data" / "qa_mini.jsonl"
