from igpo.agent.rollout import has_final_answer, parse_tool_query
from igpo.rewards.f1 import compute_outcome_reward, word_f1


def test_word_f1_exact():
    assert word_f1("Paris", "Paris") == 1.0


def test_outcome_format_penalty():
    out = compute_outcome_reward("just text", "Paris")
    assert out["reward"] == -2.0
    assert out["format_ok"] is False


def test_outcome_f1():
    text = "<think>ok</think><answer>Paris</answer>"
    out = compute_outcome_reward(text, "Paris")
    assert out["f1"] == 1.0
    assert out["reward"] == 1.0


def test_parse_tool_and_answer():
    tool = '<think>x</think><tool_call>{"name":"web_search","arguments":{"query":"Paris capital"}}</tool_call>'
    assert parse_tool_query(tool) == "Paris capital"
    assert has_final_answer(tool) is False
    ans = "<think>x</think><answer>Paris</answer>"
    assert has_final_answer(ans) is True
