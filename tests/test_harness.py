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
    """Stub chat model: structured output returns a scripted Decision — or a
    scripted InterviewScore when wrap asks for the review scores — plain invoke
    returns fixed text, tool calling returns no tool calls."""

    def __init__(self, decision=None, text="追问：请补充量化数据。", score=None):
        self._decision = decision or agent.Decision(action="followup")
        self._text = text
        # every structured call's `method` kwarg, for the regression test below
        self.methods: list = []
        self._score = score or agent.InterviewScore(
            technical_depth=7,
            communication=6,
            project_experience=8,
            job_fit=7,
            star_completeness=5,
            summary="整体表现稳健，项目细节讲得清楚。",
            strength="项目经历描述具体。",
            improvement="缺少量化数据。",
        )

    def with_structured_output(self, schema, **kwargs):
        self.methods.append(kwargs.get("method"))
        if schema is agent.InterviewScore:
            return _Structured(self._score)
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


def test_wrap_records_structured_score(monkeypatch):
    """The review must write a chartable score dict, not just prose."""
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
    config, _ = _start(graph, "t-score")
    result = graph.invoke(Command(resume="介绍完毕。"), config)

    assert result["score"]["technical_depth"] == 7
    assert result["score"]["star_completeness"] == 5
    assert set(result["score"]) >= {key for key, _ in agent.SCORE_DIMENSIONS}
    content = result["messages"][-1].content
    assert "本场评分" in content, "scores must show up in the chat transcript"


def test_scoring_failure_falls_back_to_text(monkeypatch):
    """A broken scoring call must not block the interview from wrapping up."""
    monkeypatch.setattr(agent, "prepare_topics", lambda n=8: TOPICS)
    monkeypatch.setattr(agent, "verify_answer", lambda answer: "无冲突")
    llm = FakeLLM(decision=agent.Decision(action="end"), text="总结：表现不错。")

    def boom(schema, **kwargs):
        if schema is agent.InterviewScore:
            raise RuntimeError("scoring backend down")
        return _Structured(llm._decision)

    monkeypatch.setattr(llm, "with_structured_output", boom)
    monkeypatch.setattr(agent.models, "get_chat_model", lambda temperature=0: llm)
    graph = agent.build_interview_graph(MemorySaver())
    config, _ = _start(graph, "t-score-fail")
    result = graph.invoke(Command(resume="介绍完毕。"), config)

    assert result["finished"], "wrap must still finish"
    assert result["score"] == {}
    assert "总结：表现不错。" in result["messages"][-1].content


def test_structured_calls_pin_function_calling(monkeypatch):
    """Regression: DeepSeek's OpenAI-compatible endpoint ignores the json_schema
    response format that `with_structured_output` picks by default — it answers
    in prose, pydantic fails, and decision routing silently degrades to
    `switch_topic` (no follow-ups at all). Every structured call must pin the
    method explicitly; this test fails if a new call site forgets."""
    monkeypatch.setattr(agent, "prepare_topics", lambda n=8: TOPICS)
    monkeypatch.setattr(agent, "verify_answer", lambda answer: "无冲突")
    llm = FakeLLM(decision=agent.Decision(action="end"), text="总结：表现不错。")
    monkeypatch.setattr(agent.models, "get_chat_model", lambda temperature=0: llm)
    graph = agent.build_interview_graph(MemorySaver())
    config, _ = _start(graph, "t-method")
    graph.invoke(Command(resume="介绍完毕。"), config)

    assert llm.methods, "no structured call reached the model"
    assert set(llm.methods) == {agent.models.STRUCTURED_METHOD}


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
