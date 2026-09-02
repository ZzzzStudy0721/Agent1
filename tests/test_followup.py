"""Follow-up chain test (WBS 4.4): short answers trigger follow-ups, max 2 rounds.

Usage: python tests/test_followup.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))

import agent

SHORT = "用了 YOLOv8n。"


def invoke_with(graph, history, idx, followup_count):
    state = {
        "messages": history,
        "question_index": idx,
        "followup_count": followup_count,
        "followup_verdict": "",
        "output": "",
    }
    return graph.invoke(state)


def test_short_answer_triggers_followup():
    graph = agent.build_graph()
    history = [("user", SHORT)]
    result = invoke_with(graph, history, idx=0, followup_count=0)
    print(f"[short answer] output: {result['output'][:100]!r}")
    assert "追问" in result["output"], "short answer should trigger a follow-up"
    assert result["question_index"] == 0, "follow-up should not advance the question"


def test_followup_capped_at_two_rounds():
    graph = agent.build_graph()
    history = [("user", SHORT)]
    result = invoke_with(graph, history, idx=0, followup_count=2)
    print(f"[cap at 2] output: {result['output'][:100]!r}")
    assert "追问" not in result["output"], "follow-up must be capped at 2 rounds"
    assert result["question_index"] == 1, "after cap, review should advance the question"


if __name__ == "__main__":
    test_short_answer_triggers_followup()
    test_followup_capped_at_two_rounds()
    print("\nFollow-up test passed")
