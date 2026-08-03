"""Word-level F1 and structural format checks for outcome rewards.

References:
    Wang et al., IGPO, ICLR 2026, Eq. 2.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable


_TAG_RE = re.compile(r"</?(think|answer|search|tool_response|tool_call)>", re.IGNORECASE)


def preprocess_text(text: str) -> str:
    """Normalize text for token-level F1 comparison."""
    text = text.lower()
    for punct in string.punctuation:
        text = text.replace(punct, " ")
    return re.sub(r"\s+", " ", text).strip()


def extract_answer(solution_str: str) -> str | None:
    """Extract content inside the first ``<answer>`` span."""
    match = re.search(r"<answer>(.*?)</answer>", solution_str, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def check_tags_balance(solution_str: str) -> bool:
    """Validate XML-style nesting of structured tags via a stack."""
    stack: list[str] = []
    for m in _TAG_RE.finditer(solution_str):
        tag = m.group(1).lower()
        is_close = m.group(0).startswith("</")
        if is_close:
            if not stack or stack[-1] != tag:
                return False
            stack.pop()
        else:
            stack.append(tag)
    return len(stack) == 0


def word_f1(pred: str, gold: str) -> float:
    """Compute SQuAD-style word-level F1 with token multiplicity."""
    pred_tokens = preprocess_text(pred).split()
    gold_tokens = preprocess_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_outcome_reward(
    solution_str: str,
    ground_truth: str | Iterable[str],
    *,
    format_penalty: float = -2.0,
    require_format: bool = True,
) -> dict:
    """Compute the outcome reward used at the answer turn (Eq. 2).

    Args:
        solution_str: Assistant text to be scored.
        ground_truth: Reference answer string or list of alternatives.
        format_penalty: Penalty assigned when structural constraints are violated.
        require_format: Whether structural validity is enforced.

    Returns:
        Dictionary with keys ``reward``, ``f1``, ``em``, ``format_ok``, ``answer``.
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
    em = float(any(preprocess_text(answer) == preprocess_text(g) for g in golds))
    return {
        "reward": float(f1),
        "f1": float(f1),
        "em": em,
        "format_ok": True,
        "answer": answer,
    }
