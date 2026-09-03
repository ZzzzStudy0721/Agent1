"""Interviewer agent (M3, v2): free-form interview driven by a loop state machine.

The graph is a DAG invoked once per CLI/UI round; the conversational loop lives in
the Python driver (run_interview / ui.advance). Inside the graph, an LLM decision
node (structured output) routes to the interviewer node (ask a follow-up or switch
topic) or the wrap node (debrief + evidence verification + report export).
Programmatic guardrails (topic coverage, round cap) keep the free-form
conversation from drifting (WBS 4.2/4.4).

Graph: START -> guardrails -> (decision -> (interviewer | wrap) | wrap) -> END
"""
import operator
import os
import re
import sys
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

import app
import models
import retrieval

DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

MAX_ROUNDS = 20  # hard cap on interviewer turns (guardrail)
MAX_FOLLOWUPS = 3  # hard cap on follow-up rounds per topic (guardrail)

GREETING = (
    "你好，我是你的 AI 面试官。今天围绕你的简历和岗位 JD 自由提问，没有固定题单。"
    "请先简单介绍一下你自己，我会根据你的回答深入追问。"
)

# Fallback topic anchors when data/ is empty or topic extraction fails.
DEFAULT_TOPICS = [
    "毕设项目深挖：车辆检测与计数的技术细节",
    "模型选型：为什么选 YOLOv8n、对比实验怎么做的",
    "工程落地：系统架构、性能与部署方案",
    "AI 协作经验：如何用 AI 辅助开发、Vibe Coding 流程",
    "项目难点：遇到的最大挑战与解决思路",
    "技术视野：对 RAG、Agent 等新技术的理解",
    "学习能力：转方向与快速上手新技术的经历",
    "职业规划：求职方向与个人定位",
]


class Decision(BaseModel):
    """Structured output of the decision node."""

    action: Literal["followup", "switch_topic", "end"]
    topic_index: int = 0  # 1-based index into the topic list; only for switch_topic


class InterviewState(TypedDict):
    messages: Annotated[list, operator.add]  # full ("role", text) history
    topics: list  # topic anchors, e.g. "毕设深挖：车辆检测与计数的技术细节"
    covered_topics: list  # 1-based indexes of finished topics (program-maintained)
    current_topic: int  # 1-based index of the active topic, 0 = none yet
    topic_round_count: int  # follow-up rounds on the current topic (program-maintained)
    round_count: int  # interviewer turns so far
    decision: dict  # {"action": ..., "topic_index": ...}, set by decision node
    output: Annotated[str, operator.add]  # text added this round
    finished: bool


@tool
def export_report(filename: str, content: str) -> str:
    """Export the interview wrap-up report to a markdown file under reports/.

    Args:
        filename: report file name, e.g. interview_report.md
        content: full markdown content of the report
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"报告已保存到 {path}"


_checker = None  # lazy-loaded retrieval pipeline for evidence verification


def _get_checker():
    """Lazily build the (vectorstore, bm25, reranker) pipeline once per process."""
    global _checker
    if _checker is not None:
        return _checker
    embeddings = HuggingFaceEmbeddings(model_name=app.EMBED_MODEL)
    chunks = app.load_and_chunk()
    try:
        store = Chroma(
            embedding_function=embeddings,
            persist_directory=app.DB_DIR,
            collection_name=app.COLLECTION,
        )
    except Exception:
        store = app.build_vectorstore(chunks, embeddings, collection_name=app.COLLECTION)
    _checker = (store, retrieval.BM25Index(chunks), retrieval.Reranker())
    return _checker


def verify_answer(answer: str) -> str:
    """Fact-check the answer against resume/thesis (WBS 4.5, degraded version).

    Returns conflict lines or "无冲突".
    """
    store, bm25, reranker = _get_checker()
    v = store.similarity_search(answer, k=retrieval.CANDIDATE_K)
    b = bm25.search(answer, top_k=retrieval.CANDIDATE_K)
    fused = retrieval.rrf_fusion(v, b, top_k=retrieval.CANDIDATE_K)
    ranked = reranker.rerank_with_scores(answer, fused)
    evidence = "\n\n".join(
        f"[{i}] {c.page_content}" for i, (c, _) in enumerate(ranked[:8], 1)
    )
    llm = models.get_chat_model(temperature=0)
    prompt = (
        "你是事实核对员。对比候选人回答与简历/论文原文，只找与原文明确相反的硬事实"
        "（数字、指标、技术选型、模块名称）。\n"
        "判定规则（严格遵守）：\n"
        "- 原文未提及的内容不算矛盾（宁可漏报，不可错报）\n"
        "- 只有回答与原文在数字、指标、技术选型上明确冲突时才算矛盾\n"
        "输出格式（严格遵守）：\n"
        "- 若无矛盾，只输出一行：无冲突\n"
        "- 若有矛盾，先输出一行：发现矛盾，然后每条一行：⚠️ 回答称X，但原文是Y\n"
        "不要列出与原文一致的项，不要解释。\n\n"
        f"候选人回答：{answer}\n\n简历/论文原文片段：\n{evidence}"
    )
    return llm.invoke(prompt).content.strip()


# ---------------- knowledge base loading & topic extraction ----------------

def load_jd(data_dir: str = DATA_DIR) -> str | None:
    """Read the first file whose name contains 'JD' from the knowledge base."""
    if not os.path.isdir(data_dir):
        return None
    for name in sorted(os.listdir(data_dir)):
        if "JD" in name and name.endswith((".md", ".txt")):
            with open(os.path.join(data_dir, name), encoding="utf-8") as f:
                return f.read()
    return None


def load_resume(data_dir: str = DATA_DIR, max_chars: int = 8000) -> str:
    """Read non-JD knowledge-base files (resume / thesis / project docs)."""
    if not os.path.isdir(data_dir):
        return ""
    parts = []
    for name in sorted(os.listdir(data_dir)):
        if "JD" not in name and name.endswith((".md", ".txt")):
            with open(os.path.join(data_dir, name), encoding="utf-8") as f:
                parts.append(f.read())
    return "\n\n".join(parts)[:max_chars]


def parse_topics(text: str) -> list:
    """Parse LLM output lines of the form 'name：description' into topic anchors."""
    topics = []
    for line in text.splitlines():
        line = line.strip().strip("*").strip()  # tolerate markdown bold
        line = re.sub(r"^\d+[.、]\s*", "", line)
        if "：" not in line and ":" not in line:
            continue
        name, desc = re.split(r"[:：]", line, maxsplit=1)
        name, desc = name.strip(), desc.strip().strip("*").strip()
        if 0 < len(name) <= 12 and desc:
            topics.append(f"{name}：{desc}")
    return topics


def generate_topics(jd_text: str, resume_text: str, n: int = 8) -> list:
    """Extract n interview topic anchors from JD + resume (WBS 4.2, v2)."""
    llm = models.get_chat_model(temperature=0.3)
    prompt = (
        "你是资深面试官。根据岗位 JD 和候选人简历/项目经历，抽取面试话题锚点。\n"
        "要求：\n"
        f"1. 共 {n} 个话题，JD 技术要求与候选人项目经历都要覆盖\n"
        "2. 每个话题一行，格式：名称：一句话说明（例如：模型选型：为什么最终选 YOLOv8n）\n"
        "3. 名称简短（2-8 字），说明不超过 30 字，话题之间不重叠\n\n"
        f"JD：\n{jd_text or '（无）'}\n\n"
        f"候选人简历/项目经历：\n{resume_text or '（无）'}"
    )
    text = llm.invoke(prompt).content
    topics = parse_topics(text)
    # de-duplicate topic names: LLMs sometimes parrot the format template
    seen = set()
    deduped = []
    for t in topics:
        name = t.split("：", 1)[0]
        if name not in seen:
            seen.add(name)
            deduped.append(t)
    topics = deduped
    if len(topics) < n:
        topics += [t for t in DEFAULT_TOPICS if t not in topics][: n - len(topics)]
    return topics[:n]


def prepare_topics(n: int = 8) -> list:
    """Build topic anchors from data/ if available, else the fallback list."""
    jd_text = load_jd()
    resume_text = load_resume()
    if jd_text or resume_text:
        return generate_topics(jd_text or "", resume_text or "", n)
    return DEFAULT_TOPICS[:n]


# ---------------- decision routing helpers (pure, unit-testable) ----------------

def _valid_topic_index(idx: int, topics: list, covered: list) -> int:
    """Return idx if it is a valid uncovered 1-based topic index, else the first
    uncovered one (0 if all topics are covered)."""
    if 1 <= idx <= len(topics) and idx not in covered:
        return idx
    for i in range(1, len(topics) + 1):
        if i not in covered:
            return i
    return 0


def _sanitize_decision(
    d: Decision, topics: list, covered: list, current_topic: int = 0, topic_rounds: int = 0
) -> Decision:
    """Programmatic guardrails on the LLM decision:
    - a followup request past MAX_FOLLOWUPS rounds on the current topic is
      forced to switch to an uncovered topic (or end when none is left);
    - a switch_topic request must point at an uncovered topic, otherwise end.
    """
    if d.action == "followup" and topic_rounds >= MAX_FOLLOWUPS:
        # 当前话题已追问到上限：即使 LLM 想继续深挖，也强制切走。
        # current_topic is treated as covered here so the forced switch never
        # lands back on the topic we are trying to leave.
        available = covered + ([current_topic] if current_topic else [])
        idx = _valid_topic_index(0, topics, available)
        if idx == 0:
            return Decision(action="end")
        return Decision(action="switch_topic", topic_index=idx)
    if d.action == "switch_topic":
        idx = _valid_topic_index(d.topic_index, topics, covered)
        if idx == 0:
            return Decision(action="end")
        return Decision(action="switch_topic", topic_index=idx)
    return d


def _history_text(messages: list, last_n: int | None = None) -> str:
    """Format the ("role", text) history for prompts."""
    msgs = messages[-last_n:] if last_n else messages
    return "\n".join(f"{'候选人' if r == 'user' else '面试官'}: {t}" for r, t in msgs)


def route_after_decision(state: InterviewState) -> str:
    """Route the LLM decision: end -> wrap, otherwise speak."""
    d = state.get("decision") or {}
    return "wrap" if d.get("action") == "end" else "interviewer"


def route_start(state: InterviewState) -> str:
    """Hard wrap guardrails, checked before the LLM decision node runs."""
    if state["round_count"] >= MAX_ROUNDS:
        return "wrap"
    if len(state["covered_topics"]) >= len(state["topics"]):
        return "wrap"
    return "decision"


# ---------------- graph nodes ----------------

def decision_node(state: InterviewState) -> dict:
    """Decide the interviewer's next move (followup / switch_topic / end)."""
    topics = state["topics"]
    covered = state["covered_topics"]
    llm = models.get_chat_model(temperature=0).with_structured_output(
        Decision
    )
    topic_list = "\n".join(f"{i}. {t}" for i, t in enumerate(topics, 1))
    prompt = (
        "你是面试导演，决定面试官下一步动作。\n"
        "动作选项：\n"
        "- followup：候选人刚回答的点值得深挖（回答过短、缺量化数据、回避核心、"
        "或提到值得追问的技术细节）\n"
        "- switch_topic：当前话题已聊透，切换到另一个未覆盖的话题\n"
        "- end：所有话题基本覆盖、对话已充分，可以结束面试\n\n"
        f"话题清单（编号. 话题名：考察说明）：\n{topic_list}\n\n"
        f"已覆盖话题编号：{covered or '无'}\n"
        f"当前话题编号：{state['current_topic'] or '无'}\n"
        f"当前话题已追问轮数：{state['topic_round_count']}（上限 {MAX_FOLLOWUPS} 轮，"
        "达到后必须 switch_topic）\n\n"
        "规则：\n"
        "1. 回答过短（少于 30 字）或缺量化数据时优先 followup\n"
        "2. 同一话题最多追问 3 轮，之后必须换话题\n"
        "3. 优先挑选未覆盖的话题；topic_index 用清单里的编号\n"
        "4. 话题已基本覆盖时选 end\n\n"
        f"对话历史：\n{_history_text(state['messages'])}\n\n"
        "现在决定下一步。"
    )
    try:
        d = llm.invoke(prompt)
        if not isinstance(d, Decision):
            d = None
    except Exception:
        d = None
    if d is None:
        d = Decision(action="switch_topic")
    d = _sanitize_decision(
        d, topics, covered,
        current_topic=state["current_topic"],
        topic_rounds=state["topic_round_count"],
    )
    return {"decision": d.model_dump()}


def _ask_followup(topic_text: str, history: str) -> str:
    """Generate one deep-dive follow-up question on the current topic."""
    llm = models.get_chat_model(temperature=0.3)
    prompt = (
        "你是面试官，正在深挖当前话题。基于对话历史生成一个深入追问。\n"
        "要求：只问一个点；聚焦量化数据、技术取舍或困难反思；不重复已问过的内容；\n"
        "不要输出任何前缀或解释，直接输出问题。\n\n"
        f"当前话题：{topic_text}\n\n"
        f"最近对话：\n{history}\n\n追问："
    )
    return llm.invoke(prompt).content.strip()


def _ask_topic_question(topic_text: str, history: str) -> str:
    """Open a new topic with a natural transition question."""
    llm = models.get_chat_model(temperature=0.3)
    prompt = (
        "你是面试官。把话题切换到新方向并提一个开场问题。\n"
        "要求：可以用一句过渡（如「我们换个话题」），问题要具体、可展开；\n"
        "不要输出任何前缀或解释，直接输出你要对候选人说的话。\n\n"
        f"新话题：{topic_text}\n\n"
        f"最近对话：\n{history}\n\n你要说的话："
    )
    return llm.invoke(prompt).content.strip()


def interviewer_node(state: InterviewState) -> dict:
    """Speak as the interviewer: ask a follow-up or open a new topic."""
    d = state["decision"]
    topics = state["topics"]
    covered = state["covered_topics"]
    history = _history_text(state["messages"], last_n=6)
    if d["action"] == "switch_topic":
        idx = _valid_topic_index(d.get("topic_index", 0), topics, covered)
        if state["current_topic"] and state["current_topic"] not in covered:
            covered = covered + [state["current_topic"]]
        current_topic = idx
        topic_rounds = 0  # fresh topic: reset the follow-up counter
        text = _ask_topic_question(topics[idx - 1], history)
    else:  # followup
        current_topic = state["current_topic"]
        topic_rounds = state["topic_round_count"] + 1
        topic_text = topics[current_topic - 1] if current_topic else "自由话题"
        text = _ask_followup(topic_text, history)
    return {
        "messages": [("assistant", text)],
        "output": text,
        "current_topic": current_topic,
        "topic_round_count": topic_rounds,
        "covered_topics": covered,
        "round_count": state["round_count"] + 1,
    }


def wrap_node(state: InterviewState) -> dict:
    """Debrief: summary + evidence verification + report export via tool calling."""
    llm = models.get_chat_model(temperature=0.1)
    transcript = _history_text(state["messages"])
    prompt = (
        "你是面试复盘助手。基于整场面试记录，输出：\n"
        "1. 整体表现总结（2-3 句）\n2. 最突出的 1 个优点\n3. 最需改进的 1 个问题\n\n"
        f"面试记录：\n{transcript}"
    )
    summary = llm.invoke(prompt).content
    # Evidence verification runs once at the end (WBS 4.5). Only answers that
    # carry hard numbers/metrics are worth checking; keep the batch small so
    # the checker does not confuse "absent from the source" with "contradicts".
    numeric_answers = [
        t
        for r, t in state["messages"]
        if r == "user" and re.search(r"\d+(\.\d+)?\s*%?|\d+\s*FPS|mAP", t)
    ]
    verification = verify_answer(" ".join(numeric_answers))
    if verification and "无冲突" not in verification:
        summary += f"\n\n{verification}"
    # Tool calling: let the LLM decide to export the report via the tool
    llm_with_tools = llm.bind_tools([export_report])
    tool_msg = llm_with_tools.invoke(
        f"请调用 export_report 工具，把下面的复盘报告保存为 interview_report.md：\n\n{summary}"
    )
    export_note = ""
    for call in tool_msg.tool_calls:
        result = export_report.invoke(call["args"])
        export_note = f"\n\n📄 {result}"
    return {
        "output": "\n" + summary + export_note + "\n\n面试结束，感谢作答！",
        "finished": True,
    }


# ---------------- graph & drivers ----------------

def build_graph():
    graph = StateGraph(InterviewState)
    graph.add_node("decision", decision_node)
    graph.add_node("interviewer", interviewer_node)
    graph.add_node("wrap", wrap_node)
    graph.add_conditional_edges(
        START, route_start, {"decision": "decision", "wrap": "wrap"}
    )
    graph.add_conditional_edges(
        "decision",
        route_after_decision,
        {"interviewer": "interviewer", "wrap": "wrap"},
    )
    graph.add_edge("interviewer", END)
    graph.add_edge("wrap", END)
    return graph.compile()


def init_state(topics: list | None = None) -> InterviewState:
    """Fresh interview state. topics default to the fallback list."""
    return {
        "messages": [],
        "topics": topics or DEFAULT_TOPICS,
        "covered_topics": [],
        "current_topic": 0,
        "topic_round_count": 0,
        "round_count": 0,
        "decision": {},
        "output": "",
        "finished": False,
    }


def run_interview():
    topics = prepare_topics()
    print(f"[i] 话题锚点已就绪（{len(topics)} 个）")
    graph = build_graph()
    state = init_state(topics)
    print("\n面试官：" + GREETING)
    while True:
        answer = input("\n你的回答：").strip()
        if answer.lower() in ("quit", "exit", "q"):
            print("面试提前结束。")
            break
        state["messages"] = state["messages"] + [("user", answer)]
        state["output"] = ""
        state["decision"] = {}
        result = graph.invoke(state)
        state = dict(result)
        if state["output"]:
            print("\n面试官：" + state["output"])
        if state["finished"]:
            break


if __name__ == "__main__":
    run_interview()
