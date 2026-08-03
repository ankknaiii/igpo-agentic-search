from igpo.agent.rollout import has_final_answer, parse_tool_query
from igpo.rewards.f1 import check_tags_balance, compute_outcome_reward, word_f1


def test_word_f1_exact():
    assert word_f1("Paris", "Paris") == 1.0


def test_word_f1_with_multiplicity():
    # Counter-based F1: pred has duplicate "the".
    score = word_f1("the the paris", "the paris")
    assert abs(score - (2 * (2 / 3) * (2 / 2) / ((2 / 3) + (2 / 2)))) < 1e-6


def test_outcome_format_penalty():
    out = compute_outcome_reward("just text", "Paris")
    assert out["reward"] == -2.0
    assert out["format_ok"] is False


def test_outcome_f1():
    text = "<think>ok</think><answer>Paris</answer>"
    out = compute_outcome_reward(text, "Paris")
    assert out["f1"] == 1.0
    assert out["reward"] == 1.0


def test_tag_stack_balance():
    assert check_tags_balance("<think>x</think><answer>y</answer>")
    assert not check_tags_balance("<think>x<answer>y</answer>")


def test_parse_tool_and_answer():
    tool = '<think>x</think><tool_call>{"name":"web_search","arguments":{"query":"Paris capital"}}</tool_call>'
    assert parse_tool_query(tool) == "Paris capital"
    assert has_final_answer(tool) is False
    ans = "<think>x</think><answer>Paris</answer>"
    assert has_final_answer(ans) is True


def test_outcome_uses_last_assistant_semantics():
    # Assistant answer followed by a tool turn text should not be scored via tool text.
    from igpo.agent.rollout import TurnRecord
    from igpo.rewards.f1 import compute_outcome_reward as cor

    turns = [
        TurnRecord(role="assistant", text="<think>x</think><answer>Paris</answer>", is_decision=True),
        TurnRecord(role="tool", text="<tool_response>noise</tool_response>", is_decision=False),
    ]
    last_assistant = ""
    for turn in reversed(turns):
        if turn.role == "assistant":
            last_assistant = turn.text
            break
    out = cor(last_assistant, "Paris")
    assert out["reward"] == 1.0
