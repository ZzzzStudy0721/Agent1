"""Decision routing guardrail tests (WBS 4.4, v2): pure-function checks need no
LLM calls; one live round checks that a decision keeps the interview open.

Usage: python tests/test_followup.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))

import agent

TOPICS = ["话题A：说明A", "话题B：说明B", "话题C：说明C"]


def test_valid_topic_index():
    # invalid index falls back to the first uncovered topic
    assert agent._valid_topic_index(99, TOPICS, []) == 1
    assert agent._valid_topic_index(1, TOPICS, [1]) == 2
    assert agent._valid_topic_index(3, TOPICS, [1, 2]) == 3
    # all covered -> 0
    assert agent._valid_topic_index(1, TOPICS, [1, 2, 3]) == 0


def test_sanitize_decision():
    # switch_topic with no uncovered topic left -> end
    d = agent._sanitize_decision(agent.Decision(action="switch_topic"), TOPICS, [1, 2, 3])
    assert d.action == "end"
    # switch_topic pointing at a covered topic -> first uncovered one
    d = agent._sanitize_decision(
        agent.Decision(action="switch_topic", topic_index=1), TOPICS, [1]
    )
    assert d.topic_index == 2
    # followup / end pass through untouched
    d = agent._sanitize_decision(agent.Decision(action="followup"), TOPICS, [])
    assert d.action == "followup"
    d = agent._sanitize_decision(agent.Decision(action="end"), TOPICS, [])
    assert d.action == "end"


def test_route_start_guardrails():
    base = agent.init_state(TOPICS)
    # round cap forces wrap before any LLM call
    s = {**base, "round_count": agent.MAX_ROUNDS}
    assert agent.route_start(s) == "wrap"
    # all topics covered forces wrap
    s = {**base, "covered_topics": [1, 2, 3]}
    assert agent.route_start(s) == "wrap"
    # otherwise go to the decision node
    assert agent.route_start(base) == "decision"


def test_route_after_decision():
    base = agent.init_state(TOPICS)
    # end decision wraps
    s = {**base, "decision": {"action": "end"}}
    assert agent.route_after_decision(s) == "wrap"
    # otherwise route to interviewer
    s = {**base, "decision": {"action": "followup"}}
    assert agent.route_after_decision(s) == "interviewer"
    s = {**base, "decision": {"action": "switch_topic", "topic_index": 2}}
    assert agent.route_after_decision(s) == "interviewer"


def test_live_short_answer_round():
    # one live round: short answer, graph must return one question and stay open
    graph = agent.build_graph()
    state = agent.init_state(TOPICS)
    state["messages"] = [("user", "用了 YOLOv8n。")]
    result = graph.invoke(state)
    print(f"[live round] output: {result['output'][:100]!r}")
    assert result["output"], "empty output"
    assert result["round_count"] == 1
    assert not result["finished"], "interview should not end after one round"
    assert result["messages"][-1][0] == "assistant"


if __name__ == "__main__":
    test_valid_topic_index()
    test_sanitize_decision()
    test_route_start_guardrails()
    test_route_after_decision()
    test_live_short_answer_round()
    print("\nFollow-up / routing test passed")
