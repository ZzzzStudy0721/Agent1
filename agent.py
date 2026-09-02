"""Interviewer agent (M3): LangGraph state machine driving the interview flow.

Graph: START -> review -> route -> (ask | wrap) -> END
Each CLI round invokes the graph once: the user's previous answer comes in via
state["messages"], the review node scores it, then the graph asks the next
question or wraps up. JD-based question generation lands in WBS 4.2.
"""
import operator
import os
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, StateGraph

DEMO_QUESTIONS = [
    "你毕设中对比了哪些目标检测模型？为什么最终选择 YOLOv8n？",
    "车辆计数采用了什么算法？有哪些防误计机制？",
    "你平时如何与 AI 协作完成开发？",
]

GREETING = (
    "你好，我是你的 AI 面试官。今天围绕你的简历和项目经历提问，"
    "请尽量用具体数据和技术细节回答。让我们开始。"
)

MAX_FOLLOWUP = 2  # follow-up rounds per question (WBS 4.4 degraded version)


class InterviewState(TypedDict):
    messages: Annotated[list, operator.add]  # user answers accumulate across rounds
    question_index: int
    followup_count: int  # how many follow-ups already asked for current question
    followup_verdict: str  # "followup" | "review", set by followup_check for routing
    # str + operator.add concatenates, so review text and question text both
    # survive the round instead of the later node overwriting the earlier one
    output: Annotated[str, operator.add]


def followup_check(state: InterviewState) -> dict:
    """Decide: ask a follow-up (short/vague/evasive answer, max 2 rounds) or review."""
    idx = state["question_index"]
    if idx >= len(DEMO_QUESTIONS) or not state["messages"]:
        return {"followup_verdict": "review"}
    if state.get("followup_count", 0) >= MAX_FOLLOWUP:
        return {"followup_verdict": "review"}
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
    answer = state["messages"][-1]
    prompt = (
        "你正在面试候选人。判断以下回答是否需要追问。\n"
        "需要追问的情况（满足其一即可）：\n"
        "1. 回答过短（少于 30 字）\n"
        "2. 缺少量化细节或具体数据\n"
        "3. 回避了问题核心\n\n"
        f"问题：{DEMO_QUESTIONS[idx]}\n回答：{answer}\n\n"
        "只输出一个字：追 或 过"
    )
    verdict = llm.invoke(prompt).content.strip()
    return {"followup_verdict": "followup" if verdict.startswith("追") else "review"}


def followup_node(state: InterviewState) -> dict:
    """Generate one deep-dive follow-up question on the weak spot of the answer."""
    idx = state["question_index"]
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.1)
    prompt = (
        "候选人的回答不够具体。基于问题和回答，生成一个深入追问，"
        "只问一个点，不要重复原问题：\n\n"
        f"问题：{DEMO_QUESTIONS[idx]}\n回答：{state['messages'][-1]}\n\n追问："
    )
    followup = llm.invoke(prompt).content.strip()
    return {
        "output": f"\n追问：{followup}",
        "followup_count": state.get("followup_count", 0) + 1,
    }


def review_node(state: InterviewState) -> dict:
    """Review the previous answer on 3 dimensions, or greet on the first round."""
    idx = state["question_index"]
    if not state["messages"]:
        return {"output": GREETING}
    answer = state["messages"][-1]
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.1)
    prompt = (
        "你是面试点评官。从三个维度点评候选人回答：\n"
        "1. STAR 完整性（背景/任务/行动/结果是否完整）\n"
        "2. 技术深度（是否有具体技术细节和数据）\n"
        "3. 表达清晰度（结构是否清楚）\n\n"
        f"候选人回答：{answer}\n\n"
        "给出 3-5 句点评，并指出一个最值得改进的点。"
    )
    review = llm.invoke(prompt).content
    return {"output": review, "question_index": idx + 1, "followup_count": 0}


def route_after_review(state: InterviewState) -> str:
    if state["question_index"] >= len(DEMO_QUESTIONS):
        return "wrap"
    return "ask"


def ask_node(state: InterviewState) -> dict:
    idx = state["question_index"]
    return {"output": f"\n第 {idx + 1} 题：{DEMO_QUESTIONS[idx]}"}


def wrap_node(state: InterviewState) -> dict:
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.1)
    transcript = "\n".join(
        f"Q{i + 1}: {DEMO_QUESTIONS[i]}\n"
        f"A: {state['messages'][i] if i < len(state['messages']) else '(未回答)'}"
        for i in range(len(DEMO_QUESTIONS))
    )
    prompt = (
        "你是面试复盘助手。基于整场面试记录，输出：\n"
        "1. 整体表现总结（2-3 句）\n2. 最突出的 1 个优点\n3. 最需改进的 1 个问题\n\n"
        f"面试记录：\n{transcript}"
    )
    return {"output": "\n" + llm.invoke(prompt).content + "\n\n面试结束，感谢作答！"}


def build_graph():
    graph = StateGraph(InterviewState)
    graph.add_node("followup_check", followup_check)
    graph.add_node("followup", followup_node)
    graph.add_node("review", review_node)
    graph.add_node("ask", ask_node)
    graph.add_node("wrap", wrap_node)
    graph.add_edge(START, "followup_check")
    graph.add_conditional_edges(
        "followup_check",
        lambda s: s["followup_verdict"],
        {"followup": "followup", "review": "review"},
    )
    graph.add_edge("followup", END)
    graph.add_conditional_edges(
        "review", route_after_review, {"ask": "ask", "wrap": "wrap"}
    )
    graph.add_edge("ask", END)
    graph.add_edge("wrap", END)
    return graph.compile()


def run_interview():
    graph = build_graph()
    history = []
    state = {
        "messages": history,
        "question_index": 0,
        "followup_count": 0,
        "followup_verdict": "",
        "output": "",
    }
    while True:
        result = graph.invoke(state)
        print(result["output"])
        if result["question_index"] >= len(DEMO_QUESTIONS):
            break
        answer = input("\n你的回答：").strip()
        if answer.lower() in ("quit", "exit", "q"):
            break
        history.append(("user", answer))
        state = {
            "messages": history,
            "question_index": result["question_index"],
            "followup_count": result.get("followup_count", 0),
            "followup_verdict": "",
            "output": "",
        }


if __name__ == "__main__":
    run_interview()
