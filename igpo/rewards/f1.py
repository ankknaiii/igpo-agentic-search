"""Word-level F1 and format checks used by IGPO outcome rewards."""

from __future__ import annotations

import re
import string
from typing import Iterable


_TAGS = ("think", "tool_call", "answer", "search")


def preprocess_text(text: str) -> str:
    text = text.lower()
    for punct in string.punctuation:
        text = text.replace(punct, " ")
    return re.sub(r"\s+", " ", text).strip()


def extract_answer(solution_str: str) -> str | None:
    match = re.search(r"<answer>(.*?)</answer>", solution_str, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def check_tags_balance(solution_str: str) -> bool:
    for tag in _TAGS:
        start, end = f"<{tag}>", f"</{tag}>"
        if solution_str.count(start) != solution_str.count(end):
            return False
        pos = -1
        while True:
            start_pos = solution_str.find(start, pos + 1)
            if start_pos < 0:
                break
            end_pos = solution_str.find(end, start_pos)
            if end_pos < 0:
                return False
            pos = end_pos
    return True


def word_f1(pred: str, gold: str) -> float:
    pred_tokens = set(preprocess_text(pred).split())
    gold_tokens = set(preprocess_text(gold).split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_outcome_reward(
    solution_str: str,
    ground_truth: str | Iterable[str],
    *,
    format_penalty: float = -2.0,
    require_format: bool = True,
) -> dict:
    """Outcome reward used at the answer turn (paper Eq. 2).

    Returns dict with keys: reward, f1, em, format_ok, answer.
    """
    if isinstance(ground_truth, str):
        golds = [g for g in ground_truth.split("<|answer_split|>") if g.strip()]
    else:
        golds = list(ground_truth)

    format_ok = check_tags_balance(solution_str) if require_format else True
    answer = extract_answer(solution_str)

    if require_format and (not format_ok or answer is None):
        return {
            "reward": float(format_penalty),
            "f1": 0.0,
            "em": 0.0,
            "format_ok": False,
            "answer": answer or "",
        }

    answer = answer or ""
    f1 = max((word_f1(answer, g) for g in golds), default=0.0)
    em = float(
        any(preprocess_text(answer) == preprocess_text(g) for g in golds)
    )
    return {
        "reward": float(f1),
        "f1": float(f1),
        "em": em,
        "format_ok": True,
        "answer": answer,
    }
