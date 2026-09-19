"""Loop-graph (LangGraph Harness) tests: interrupt-driven interview loop with a
stubbed LLM — no live API calls, no live marker needed.

Covers: greeting + interrupt on first run, resume runs one interviewer turn,
an end decision wraps the interview, and a finished thread gets the closing
note instead of a fresh greeting.
"""
import os
import sys
from types import SimpleNamespace

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))

import agent  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402
import pytest  # noqa: E402

TOPICS = ["话题A：说明A", "话题B：说明B", "话题C：说明C"]


class FakeLLM:
    """Stub chat model: structured output returns a scripted Decision, plain
    invoke returns fixed text, tool calling returns no tool calls."""

    def __init__(self, decision=None, text="追问：请补充量化数据。"):
        self._decision = decision or agent.Decision(action="followup")
        self._text = text

    def with_structured_output(self, schema):
        return _Structured(self._decision)

    def invoke(self, *args, **kwargs):
        return SimpleNamespace(content=self._text, tool_calls=[])

    def bind_tools(self, tools):
        return self


class _Structured:
    def __init__(self, decision):
        self._decision = decision

    def invoke(self, *args, **kwargs):
        return self._decision


@pytest.fixture
def loop_graph(monkeypatch):
    """Loop graph with topic extraction and the chat model stubbed out."""
    monkeypatch.setattr(agent, "prepare_topics", lambda n=8: TOPICS)
    monkeypatch.setattr(agent, "verify_answer", lambda answer: "无冲突")
    monkeypatch.setattr(
        agent.models, "get_chat_model", lambda temperature=0: FakeLLM()
    )
    return agent.build_interview_graph(MemorySaver())


def _start(graph, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(agent.init_state(), config)
    return config, result


def test_first_run_greets_and_interrupts(loop_graph):
    config, result = _start(loop_graph, "t-greet")
    assert "__interrupt__" in result, "first run must pause for the answer"
    assert result["topics"] == TOPICS
    assert result["messages"][-1].type == "ai"
    assert result["messages"][-1].content == agent.GREETING


def test_resume_runs_one_interviewer_turn(loop_graph):
    config, _ = _start(loop_graph, "t-turn")
    result = loop_graph.invoke(Command(resume="我用 YOLOv8n。"), config)
    assert "__interrupt__" in result, "open interview keeps waiting for answers"
    assert not result["finished"]
    assert result["round_count"] == 1
    assert result["messages"][-2].type == "human"
    assert result["messages"][-2].content == "我用 YOLOv8n。"
    assert result["messages"][-1].type == "ai"
    assert result["messages"][-1].content == "追问：请补充量化数据。"


def test_sparse_input_run(loop_graph):
    """A Harness run may start with an empty input dict: the graph fills every
    missing field itself instead of raising KeyError (regression)."""
    config = {"configurable": {"thread_id": "t-sparse"}}
    result = loop_graph.invoke({}, config)
    assert "__interrupt__" in result, "first run must pause for the answer"
    assert result["topics"] == TOPICS
    assert result["finished"] is False
    assert result["round_count"] == 0
    assert result["messages"][-1].content == agent.GREETING


def test_message_dict_from_harness(loop_graph):
    """Messages arriving as plain dicts (what the Harness UI sends) are
    normalized, and a run that already carries the candidate's message answers
    it directly instead of greeting into the void."""
    config = {"configurable": {"thread_id": "t-dict"}}
    result = loop_graph.invoke(
        {"messages": [{"role": "user", "content": "我是应届生。"}]}, config
    )
    assert "__interrupt__" in result
    assert result["round_count"] == 1, "the candidate's message must be answered"
    assert result["messages"][0].type == "human"
    assert result["messages"][0].content == "我是应届生。"
    assert result["messages"][-1].type == "ai"


def test_end_decision_wraps_interview(monkeypatch):
    monkeypatch.setattr(agent, "prepare_topics", lambda n=8: TOPICS)
    monkeypatch.setattr(agent, "verify_answer", lambda answer: "无冲突")
    monkeypatch.setattr(
        agent.models,
        "get_chat_model",
        lambda temperature=0: FakeLLM(
            decision=agent.Decision(action="end"), text="总结：表现不错。"
        ),
    )
    graph = agent.build_interview_graph(MemorySaver())
    config, _ = _start(graph, "t-wrap")
    result = graph.invoke(Command(resume="介绍完毕。"), config)
    assert "__interrupt__" not in result, "wrap must end the graph"
    assert result["finished"]
    assert result["messages"][-1].type == "ai"
    assert "面试结束" in result["messages"][-1].content


def test_finished_thread_gets_closing_note(monkeypatch):
    monkeypatch.setattr(agent, "prepare_topics", lambda n=8: TOPICS)
    monkeypatch.setattr(agent, "verify_answer", lambda answer: "无冲突")
    monkeypatch.setattr(
        agent.models,
        "get_chat_model",
        lambda temperature=0: FakeLLM(decision=agent.Decision(action="end")),
    )
    graph = agent.build_interview_graph(MemorySaver())
    config, _ = _start(graph, "t-farewell")
    result = graph.invoke(Command(resume="介绍完毕。"), config)
    assert result["finished"]
    # a new message on the finished thread reruns the graph: prepare sees
    # finished=True and answers with the closing note instead of a greeting
    result2 = graph.invoke({"messages": [HumanMessage(content="再问一下？")]}, config)
    assert "__interrupt__" not in result2
    assert result2["messages"][-1].content == agent.FINISHED_NOTE
