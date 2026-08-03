"""Multi-turn agentic search rollout with information-gain belief tracking.

Sampling token ids are retained without decode–encode round-trips so that
importance-sampling ratios remain well-defined under GRPO/IGPO updates.

References:
    Wang et al., IGPO, ICLR 2026, arXiv:2510.14967.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import torch

from igpo.agent.prompts import build_system_prompt, build_user_prompt
from igpo.env.mock_kb import format_search_results, search_kb
from igpo.rewards.f1 import compute_outcome_reward
from igpo.rewards.info_gain import (
    BeliefState,
    belief_value,
    teacher_force_answer_logprobs,
)


_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", flags=re.DOTALL | re.IGNORECASE
)
_ANSWER_RE = re.compile(r"<answer>.*?</answer>", flags=re.DOTALL | re.IGNORECASE)


@dataclass
class TurnRecord:
    """Single interaction turn within a rollout."""

    role: str  # assistant | tool
    text: str
    token_ids: list[int] = field(default_factory=list)
    # Decision tokens receive gradient; tool responses are masked.
    is_decision: bool = True


@dataclass
class RolloutResult:
    """Complete multi-turn trajectory with process and outcome rewards."""

    question: str
    ground_truth: str
    turns: list[TurnRecord]
    info_gains: list[float]
    beliefs: list[BeliefState]
    outcome: dict[str, Any]
    num_search_turns: int
    finished_with_answer: bool
    prompt_ids: list[int]
    context_checkpoints: list[list[int]] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(t.text for t in self.turns)

    @property
    def assistant_text(self) -> str:
        return "\n".join(t.text for t in self.turns if t.role == "assistant")


def parse_tool_query(text: str) -> str | None:
    """Extract the search query from a structured tool call.

    Compatibility with the shorthand ``<search>query</search>`` format is retained.
    """
    match = _TOOL_CALL_RE.search(text)
    if not match:
        m2 = re.search(r"<search>(.*?)</search>", text, flags=re.DOTALL | re.IGNORECASE)
        return m2.group(1).strip() if m2 else None
    raw = match.group(1)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # 兜底处理：提取 query 字符串。
        m = re.search(r'"query"\s*:\s*"([^"]+)"', raw)
        return m.group(1).strip() if m else None
    args = obj.get("arguments", obj)
    query = args.get("query") if isinstance(args, dict) else None
    if isinstance(query, list):
        query = " ".join(str(x) for x in query)
    return str(query).strip() if query else None


def has_final_answer(text: str) -> bool:
    """Return whether the assistant turn contains a final answer without a tool call."""
    return bool(_ANSWER_RE.search(text)) and parse_tool_query(text) is None


class SearchRolloutEngine:
    """Generate multi-turn search trajectories and intrinsic IG rewards."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        max_turns: int = 4,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.95,
        info_gain_type: str = "prob_diff",
        search_top_k: int = 3,
        device: torch.device | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.info_gain_type = info_gain_type
        self.search_top_k = search_top_k
        self.device = device or next(model.parameters()).device

    def _messages_to_ids(self, messages: list[dict]) -> torch.Tensor:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Align with the official IGPO prompt convention by forcing a think prefix.
        text = text + "<think>"
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ][0]
        return ids.to(self.device)

    @torch.no_grad()
    def _generate_assistant(self, messages: list[dict]) -> tuple[str, list[int]]:
        """Sample an assistant continuation and return (text, sampled_token_ids).

        Token ids are taken directly from ``generate`` outputs. Decode–encode
        round-trips are prohibited to preserve importance-sampling validity.
        """
        input_ids = self._messages_to_ids(messages).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        gen = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-5),
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = gen[0, input_ids.shape[1] :]
        token_ids = new_tokens.detach().cpu().tolist()
        continuation = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        if not continuation.lstrip().startswith("<think>") and not continuation.lstrip().startswith(
            "</think>"
        ):
            text = "<think>" + continuation
        else:
            text = (
                continuation
                if continuation.lstrip().startswith("<think>")
                else "<think>" + continuation
            )
        return text.strip(), token_ids

    @torch.no_grad()
    def rollout(self, question: str, ground_truth: str) -> RolloutResult:
        """Roll out one multi-turn trajectory and compute IG / outcome rewards."""
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(question)},
        ]
        turns: list[TurnRecord] = []
        beliefs: list[BeliefState] = []
        info_gains: list[float] = []
        checkpoints: list[list[int]] = []

        prompt_ids = self._messages_to_ids(messages)
        checkpoints.append(prompt_ids.detach().cpu().tolist())
        b0 = teacher_force_answer_logprobs(
            self.model, self.tokenizer, prompt_ids, ground_truth, device=self.device
        )
        beliefs.append(b0)
        prev_value = belief_value(b0, self.info_gain_type)  # type: ignore[arg-type]

        finished = False
        search_turns = 0

        for _step in range(self.max_turns):
            assistant_text, asst_ids = self._generate_assistant(messages)
            turns.append(
                TurnRecord(
                    role="assistant",
                    text=assistant_text,
                    token_ids=asst_ids,
                    is_decision=True,
                )
            )
            messages.append({"role": "assistant", "content": assistant_text})

            if has_final_answer(assistant_text):
                finished = True
                break

            query = parse_tool_query(assistant_text)
            if query is None:
                break

            results = search_kb(query, top_k=self.search_top_k, noise=True)
            tool_text = (
                f"<tool_response>\n{format_search_results(results)}\n</tool_response>"
            )
            tool_ids = self.tokenizer(tool_text, add_special_tokens=False).input_ids
            turns.append(
                TurnRecord(
                    role="tool",
                    text=tool_text,
                    token_ids=tool_ids,
                    is_decision=False,
                )
            )
            messages.append({"role": "user", "content": tool_text})
            search_turns += 1

            ctx_ids = self._messages_to_ids(messages)
            checkpoints.append(ctx_ids.detach().cpu().tolist())
            bt = teacher_force_answer_logprobs(
                self.model, self.tokenizer, ctx_ids, ground_truth, device=self.device
            )
            beliefs.append(bt)
            cur_value = belief_value(bt, self.info_gain_type)  # type: ignore[arg-type]
            ig = float(cur_value - prev_value)
            if ig != ig or ig in (float("inf"), float("-inf")):
                ig = 0.0
            info_gains.append(ig)
            prev_value = cur_value

        last_assistant_text = ""
        for turn in reversed(turns):
            if turn.role == "assistant":
                last_assistant_text = turn.text
                break
        outcome = compute_outcome_reward(last_assistant_text, ground_truth)

        return RolloutResult(
            question=question,
            ground_truth=ground_truth,
            turns=turns,
            info_gains=info_gains,
            beliefs=beliefs,
            outcome=outcome,
            num_search_turns=search_turns,
            finished_with_answer=finished,
            prompt_ids=prompt_ids.detach().cpu().tolist(),
            context_checkpoints=checkpoints,
        )
