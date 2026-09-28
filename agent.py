"""面试官 agent（M3, v2）：由循环状态机驱动的自由形式面试。

两个图共用同一批节点：

- build_turn_graph()：把面试官的一轮对话建成一个 DAG —— guardrails -> LLM 决策
  （结构化输出）-> interviewer 节点（追问或切换话题）| wrap 节点
  （复盘 + 证据核验 + 报告导出）。供逐轮驱动器使用。
- build_interview_graph()：完整的对话循环。START -> 准备话题
  -> greet -> interrupt() 等待候选人回答 -> 一轮对话（上面那个 DAG
  作为子图）-> 回到 interrupt()，直到 wrap 结束面试。
  这是 LangGraph Harness 的入口（`langgraph dev`）：thread 状态会被
  checkpoint，因此调试 UI 可以跨多次运行驱动整场面试。

程序化 guardrails（话题覆盖、轮数上限）能防止自由形式的
对话跑偏（WBS 4.2/4.4）。
"""

import logging
import operator
import os
import re
import sys
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import app
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
import models
from pydantic import BaseModel, Field
import retrieval

DATA_DIR = os.path.join(BASE_DIR, 'data')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# 由 UI 上传流程写入的专用面试输入。当它们存在时，
# 话题锚点只从中提取，而不再来自整个 data/ 目录，
# 这样刚上传的简历就能驱动面试，
# 知识库的其余部分则继续为证据核验服务。
UPLOADED_RESUME = 'uploaded_resume.md'
UPLOADED_JD = 'uploaded_JD.md'

logger = logging.getLogger(__name__)

MAX_ROUNDS = 20  # 面试官轮数的硬上限（guardrail）
MAX_FOLLOWUPS = 3  # 每个话题追问轮数的硬上限（guardrail）

GREETING = (
    '你好，我是你的 AI 面试官。今天围绕你的简历和岗位 JD 自由提问，没有固定题单。'
    '请先简单介绍一下你自己，我会根据你的回答深入追问。'
)

# 当用户持续向一个已结束的面试 thread 发消息时展示。
FINISHED_NOTE = '本次面试已结束，复盘报告见上方。如需重新开始，请在调试界面新建一个 Thread。'

# 当 data/ 为空或话题抽取失败时的兜底话题锚点。
DEFAULT_TOPICS = [
    '毕设项目深挖：车辆检测与计数的技术细节',
    '模型选型：为什么选 YOLOv8n、对比实验怎么做的',
    '工程落地：系统架构、性能与部署方案',
    'AI 协作经验：如何用 AI 辅助开发、Vibe Coding 流程',
    '项目难点：遇到的最大挑战与解决思路',
    '技术视野：对 RAG、Agent 等新技术的理解',
    '学习能力：转方向与快速上手新技术的经历',
    '职业规划：求职方向与个人定位',
]


class Decision(BaseModel):
    """decision 节点的结构化输出。"""

    action: Literal['followup', 'switch_topic', 'end']
    topic_index: int = 0  # 话题列表中的 1-based 编号；仅用于 switch_topic


class InterviewScore(BaseModel):
    """wrap 节点的结构化评分：五个维度各 1-10 分，外加文字点评。

    Kept separate from the free-text summary so the UI can chart the scores
    (radar / trend) instead of parsing them back out of prose.
    """

    technical_depth: int = Field(ge=1, le=10, description='技术深度：对技术细节和原理的掌握')
    communication: int = Field(ge=1, le=10, description='表达逻辑：回答的条理性与清晰度')
    project_experience: int = Field(ge=1, le=10, description='项目经验：经历的丰富度与真实度')
    job_fit: int = Field(ge=1, le=10, description='岗位匹配：与目标岗位要求的契合度')
    star_completeness: int = Field(ge=1, le=10, description='回答完整度：是否讲清情境-任务-行动-结果')
    summary: str = Field(description='整体表现总结，2-3 句')
    strength: str = Field(description='最突出的 1 个优点')
    improvement: str = Field(description='最需改进的 1 个问题')


SCORE_DIMENSIONS = [
    ('technical_depth', '技术深度'),
    ('communication', '表达逻辑'),
    ('project_experience', '项目经验'),
    ('job_fit', '岗位匹配'),
    ('star_completeness', '回答完整度'),
]


class InterviewState(TypedDict):
    # add_messages（而非普通 append）：把 Harness UI 发来的 dict
    # 在进入状态时规范化为 HumanMessage/AIMessage 对象。
    messages: Annotated[list, add_messages]
    topics: list  # 话题锚点，例如 "毕设深挖：车辆检测与计数的技术细节"
    covered_topics: list  # 已完成话题的 1-based 编号（由程序维护）
    current_topic: int  # 当前话题的 1-based 编号，0 = 还没有
    topic_round_count: int  # 当前话题已追问的轮数（由程序维护）
    round_count: int  # 到目前为止的面试官轮数
    decision: dict  # {"action": ..., "topic_index": ...}，由 decision 节点写入
    output: Annotated[str, operator.add]  # 本轮新增的文本
    finished: bool
    score: dict  # 复盘的 InterviewScore.model_dump()，未结束/评分失败时为空


@tool
def export_report(filename: str, content: str) -> str:
    """把面试复盘报告导出为 reports/ 下的 markdown 文件。

    Args:
        filename: 报告文件名，例如 interview_report.md
        content: 报告的完整 markdown 内容
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return f'报告已保存到 {path}'


_checker = None  # 用于证据核验的检索流水线，懒加载


def reset_checker() -> None:
    """丢弃缓存的流水线 —— 在向量索引重建之后调用。

    缓存的 Chroma 句柄指向的 collection 会被重建操作删除，
    不丢弃它的话，证据核验会静默地查询一个过期/已失效的 collection。
    """
    global _checker
    _checker = None


def _get_checker():
    """每个进程懒加载构建一次 (vectorstore, bm25, reranker) 流水线。"""
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
    """把回答与简历/论文做事实核对（WBS 4.5，降级版）。

    返回冲突说明行，或 "无冲突"。
    """
    store, bm25, reranker = _get_checker()
    v = store.similarity_search(answer, k=retrieval.CANDIDATE_K)
    b = bm25.search(answer, top_k=retrieval.CANDIDATE_K)
    fused = retrieval.rrf_fusion(v, b, top_k=retrieval.CANDIDATE_K)
    ranked = reranker.rerank_with_scores(answer, fused)
    evidence = '\n\n'.join(f'[{i}] {c.page_content}' for i, (c, _) in enumerate(ranked[:8], 1))
    llm = models.get_chat_model(temperature=0)
    prompt = (
        '你是事实核对员。对比候选人回答与简历/论文原文，只找与原文明确相反的硬事实'
        '（数字、指标、技术选型、模块名称）。\n'
        '判定规则（严格遵守）：\n'
        '- 原文未提及的内容不算矛盾（宁可漏报，不可错报）\n'
        '- 只有回答与原文在数字、指标、技术选型上明确冲突时才算矛盾\n'
        '输出格式（严格遵守）：\n'
        '- 若无矛盾，只输出一行：无冲突\n'
        '- 若有矛盾，先输出一行：发现矛盾，然后每条一行：⚠️ 回答称X，但原文是Y\n'
        '不要列出与原文一致的项，不要解释。\n\n'
        f'候选人回答：{answer}\n\n简历/论文原文片段：\n{evidence}'
    )
    return llm.invoke(prompt, config={'callbacks': models.get_callbacks()}).content.strip()


# ---------------- 知识库加载与话题抽取 ----------------


def save_uploaded_doc(text: str, kind: str, data_dir: str = DATA_DIR) -> str:
    """把上传的简历 / JD 持久化为专用的面试输入。

    `kind` 为 'resume' 或 'jd'。以 .md 存储，这样现有的加载器和
    向量索引无需额外改动就能读取。返回写入的路径。
    """
    if kind not in ('resume', 'jd'):
        raise ValueError(f'unknown kind: {kind}')
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, UPLOADED_RESUME if kind == 'resume' else UPLOADED_JD)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return path


def load_jd(data_dir: str = DATA_DIR) -> str | None:
    """若上传的 JD 存在则读取它，否则读取 data/ 中第一个文件名含 *JD* 的文件。"""
    uploaded = os.path.join(data_dir, UPLOADED_JD)
    if os.path.isfile(uploaded):
        with open(uploaded, encoding='utf-8') as f:
            return f.read()
    if not os.path.isdir(data_dir):
        return None
    for name in sorted(os.listdir(data_dir)):
        if 'JD' in name and name.endswith(('.md', '.txt')):
            with open(os.path.join(data_dir, name), encoding='utf-8') as f:
                return f.read()
    return None


def load_resume(data_dir: str = DATA_DIR, max_chars: int = 8000) -> str:
    """若上传的简历存在则读取它，否则读取 data/ 中所有非 JD 文件。"""
    uploaded = os.path.join(data_dir, UPLOADED_RESUME)
    if os.path.isfile(uploaded):
        with open(uploaded, encoding='utf-8') as f:
            return f.read()[:max_chars]
    if not os.path.isdir(data_dir):
        return ''
    parts = []
    for name in sorted(os.listdir(data_dir)):
        if 'JD' not in name and name.endswith(('.md', '.txt')):
            with open(os.path.join(data_dir, name), encoding='utf-8') as f:
                parts.append(f.read())
    return '\n\n'.join(parts)[:max_chars]


def parse_topics(text: str) -> list:
    """把 '名称：说明' 形式的 LLM 输出行解析成话题锚点。"""
    topics = []
    for line in text.splitlines():
        line = line.strip().strip('*').strip()  # 容忍 markdown 加粗符号
        line = re.sub(r'^\d+[.、]\s*', '', line)
        if '：' not in line and ':' not in line:
            continue
        name, desc = re.split(r'[:：]', line, maxsplit=1)
        name, desc = name.strip(), desc.strip().strip('*').strip()
        if 0 < len(name) <= 12 and desc:
            topics.append(f'{name}：{desc}')
    return topics


def generate_topics(jd_text: str, resume_text: str, n: int = 8) -> list:
    """从 JD + 简历中抽取 n 个面试话题锚点（WBS 4.2, v2）。"""
    llm = models.get_chat_model(temperature=0.3)
    prompt = (
        '你是资深面试官。根据岗位 JD 和候选人简历/项目经历，抽取面试话题锚点。\n'
        '要求：\n'
        f'1. 共 {n} 个话题，JD 技术要求与候选人项目经历都要覆盖\n'
        '2. 每个话题一行，格式：名称：一句话说明（例如：模型选型：为什么最终选 YOLOv8n）\n'
        '3. 名称简短（2-8 字），说明不超过 30 字，话题之间不重叠\n\n'
        f'JD：\n{jd_text or "（无）"}\n\n'
        f'候选人简历/项目经历：\n{resume_text or "（无）"}'
    )
    text = llm.invoke(prompt, config={'callbacks': models.get_callbacks()}).content
    topics = parse_topics(text)
    # 对话题名去重：LLM 有时会照抄格式模板
    seen = set()
    deduped = []
    for t in topics:
        name = t.split('：', 1)[0]
        if name not in seen:
            seen.add(name)
            deduped.append(t)
    topics = deduped
    if len(topics) < n:
        topics += [t for t in DEFAULT_TOPICS if t not in topics][: n - len(topics)]
    return topics[:n]


def prepare_topics(n: int = 8) -> list:
    """若 data/ 可用则从中构建话题锚点，否则使用兜底列表。"""
    jd_text = load_jd()
    resume_text = load_resume()
    if jd_text or resume_text:
        try:
            return generate_topics(jd_text or '', resume_text or '', n)
        except Exception as e:
            logger.warning('topic extraction failed (%s), falling back to defaults', e)
    return DEFAULT_TOPICS[:n]


# ---------------- 决策路由辅助函数（纯函数，可单元测试） ----------------


def _valid_topic_index(idx: int, topics: list, covered: list) -> int:
    """若 idx 是有效的、未覆盖的 1-based 话题编号则返回它，否则返回第一个
    未覆盖的编号（所有话题都已覆盖时返回 0）。"""
    if 1 <= idx <= len(topics) and idx not in covered:
        return idx
    for i in range(1, len(topics) + 1):
        if i not in covered:
            return i
    return 0


def _sanitize_decision(
    d: Decision, topics: list, covered: list, current_topic: int = 0, topic_rounds: int = 0
) -> Decision:
    """对 LLM 决策施加程序化 guardrails：
    - 当前话题的追问已达 MAX_FOLLOWUPS 轮后，followup 请求会被强制切换到一个
      未覆盖的话题（没有剩余话题时则结束）；
    - switch_topic 请求必须指向一个未覆盖的话题，否则结束。
    """
    if d.action == 'followup' and topic_rounds >= MAX_FOLLOWUPS:
        # 当前话题已追问到上限：即使 LLM 想继续深挖，也强制切走。
        # 这里把 current_topic 视为已覆盖，这样强制切换永远不会又落回
        # 我们正想离开的那个话题。
        available = covered + ([current_topic] if current_topic else [])
        idx = _valid_topic_index(0, topics, available)
        if idx == 0:
            return Decision(action='end')
        return Decision(action='switch_topic', topic_index=idx)
    if d.action == 'switch_topic':
        idx = _valid_topic_index(d.topic_index, topics, covered)
        if idx == 0:
            return Decision(action='end')
        return Decision(action='switch_topic', topic_index=idx)
    return d


def _history_text(messages: list, last_n: int | None = None) -> str:
    """格式化对话历史（LangChain messages），供 prompt 使用。"""
    msgs = messages[-last_n:] if last_n else messages
    return '\n'.join(f'{"候选人" if m.type == "human" else "面试官"}: {m.content}' for m in msgs)


def route_after_decision(state: InterviewState) -> str:
    """路由 LLM 决策：end -> wrap，否则由面试官发言。"""
    d = state.get('decision') or {}
    return 'wrap' if d.get('action') == 'end' else 'interviewer'


def route_start(state: InterviewState) -> str:
    """硬性的 wrap guardrails，在 LLM decision 节点运行前检查。"""
    if state['round_count'] >= MAX_ROUNDS:
        return 'wrap'
    if len(state['covered_topics']) >= len(state['topics']):
        return 'wrap'
    return 'decision'


# ---------------- 图节点 ----------------


def decision_node(state: InterviewState) -> dict:
    """决定面试官的下一步动作（followup / switch_topic / end）。"""
    topics = state['topics']
    covered = state['covered_topics']
    llm = models.get_chat_model(temperature=0).with_structured_output(
        Decision, method=models.STRUCTURED_METHOD
    )
    topic_list = '\n'.join(f'{i}. {t}' for i, t in enumerate(topics, 1))
    prompt = (
        '你是面试导演，决定面试官下一步动作。\n'
        '动作选项：\n'
        '- followup：候选人刚回答的点值得深挖（回答过短、缺量化数据、回避核心、'
        '或提到值得追问的技术细节）\n'
        '- switch_topic：当前话题已聊透，切换到另一个未覆盖的话题\n'
        '- end：所有话题基本覆盖、对话已充分，可以结束面试\n\n'
        f'话题清单（编号. 话题名：考察说明）：\n{topic_list}\n\n'
        f'已覆盖话题编号：{covered or "无"}\n'
        f'当前话题编号：{state["current_topic"] or "无"}\n'
        f'当前话题已追问轮数：{state["topic_round_count"]}（上限 {MAX_FOLLOWUPS} 轮，'
        '达到后必须 switch_topic）\n\n'
        '规则：\n'
        '1. 回答过短（少于 30 字）或缺量化数据时优先 followup\n'
        '2. 同一话题最多追问 3 轮，之后必须换话题\n'
        '3. 优先挑选未覆盖的话题；topic_index 用清单里的编号\n'
        '4. 话题已基本覆盖时选 end\n\n'
        f'对话历史：\n{_history_text(state["messages"])}\n\n'
        '现在决定下一步。'
    )
    try:
        d = llm.invoke(prompt, config={'callbacks': models.get_callbacks()})
        if not isinstance(d, Decision):
            d = None
    except Exception:
        d = None
    if d is None:
        d = Decision(action='switch_topic')
    d = _sanitize_decision(
        d,
        topics,
        covered,
        current_topic=state['current_topic'],
        topic_rounds=state['topic_round_count'],
    )
    logger.info(
        'decision: %s topic=%d round=%d covered=%s',
        d.action,
        d.topic_index,
        state['round_count'],
        covered,
    )
    return {'decision': d.model_dump()}


def _ask_followup(topic_text: str, history: str) -> str:
    """针对当前话题生成一个深入追问。"""
    llm = models.get_chat_model(temperature=0.3)
    prompt = (
        '你是面试官，正在深挖当前话题。基于对话历史生成一个深入追问。\n'
        '要求：只问一个点；聚焦量化数据、技术取舍或困难反思；不重复已问过的内容；\n'
        '不要输出任何前缀或解释，直接输出问题。\n\n'
        f'当前话题：{topic_text}\n\n'
        f'最近对话：\n{history}\n\n追问：'
    )
    return llm.invoke(prompt, config={'callbacks': models.get_callbacks()}).content.strip()


def _ask_topic_question(topic_text: str, history: str) -> str:
    """用一句自然的过渡提问开启一个新话题。"""
    llm = models.get_chat_model(temperature=0.3)
    prompt = (
        '你是面试官。把话题切换到新方向并提一个开场问题。\n'
        '要求：可以用一句过渡（如「我们换个话题」），问题要具体、可展开；\n'
        '不要输出任何前缀或解释，直接输出你要对候选人说的话。\n\n'
        f'新话题：{topic_text}\n\n'
        f'最近对话：\n{history}\n\n你要说的话：'
    )
    return llm.invoke(prompt, config={'callbacks': models.get_callbacks()}).content.strip()


def interviewer_node(state: InterviewState) -> dict:
    """以面试官身份发言：追问或开启一个新话题。"""
    d = state['decision']
    topics = state['topics']
    covered = state['covered_topics']
    history = _history_text(state['messages'], last_n=6)
    if d['action'] == 'switch_topic':
        idx = _valid_topic_index(d.get('topic_index', 0), topics, covered)
        if state['current_topic'] and state['current_topic'] not in covered:
            covered = covered + [state['current_topic']]
        current_topic = idx
        topic_rounds = 0  # 新话题：重置追问计数器
        text = _ask_topic_question(topics[idx - 1], history)
    else:  # 追问
        current_topic = state['current_topic']
        topic_rounds = state['topic_round_count'] + 1
        topic_text = topics[current_topic - 1] if current_topic else '自由话题'
        text = _ask_followup(topic_text, history)
    return {
        'messages': [AIMessage(content=text)],
        'output': text,
        'current_topic': current_topic,
        'topic_round_count': topic_rounds,
        'covered_topics': covered,
        'round_count': state['round_count'] + 1,
    }


def _format_score(score: InterviewScore) -> str:
    """把结构化评分渲染成聊天区展示的文本。"""
    lines = '\n'.join(
        f'- {label}：{getattr(score, key)}/10' for key, label in SCORE_DIMENSIONS
    )
    return (
        f'📊 本场评分\n{lines}\n\n'
        f'{score.summary}\n\n'
        f'✅ 最突出的优点：{score.strength}\n'
        f'⚠️ 最需改进的问题：{score.improvement}'
    )


def wrap_node(state: InterviewState) -> dict:
    """复盘：结构化评分 + 证据核验 + 通过 tool calling 导出报告。"""
    llm = models.get_chat_model(temperature=0.1)
    transcript = _history_text(state['messages'])
    prompt = (
        '你是面试复盘助手。基于整场面试记录，给出结构化评分与点评。\n'
        '每个维度 1-10 分（10 分为满分），标准要严格：回答空洞、缺细节缺数据的，'
        '该维度不超过 5 分。\n'
        '点评必须具体：指出是哪一轮回答、哪个技术点，不要写套话。\n\n'
        f'面试记录：\n{transcript}'
    )
    score_data: dict = {}
    try:
        score = llm.with_structured_output(
            InterviewScore, method=models.STRUCTURED_METHOD
        ).invoke(prompt, config={'callbacks': models.get_callbacks()})
        if not isinstance(score, InterviewScore):
            raise TypeError(f'unexpected structured output: {type(score).__name__}')
        summary = _format_score(score)
        score_data = score.model_dump()
    except Exception as e:
        # 结构化输出失败不能挡住面试收尾：退回自由文本总结。
        logger.warning('structured scoring failed (%s), falling back to text summary', e)
        summary = llm.invoke(prompt, config={'callbacks': models.get_callbacks()}).content
    # 证据核验只在最后统一跑一次（WBS 4.5）。只有带硬数字/指标的
    # 答案才值得核验；把待核验的批次控制得小一些，避免核验器
    # 把「原文中未提及」误判为「与原文矛盾」。
    numeric_answers = [
        m.content
        for m in state['messages']
        if m.type == 'human' and re.search(r'\d+(\.\d+)?\s*%?|\d+\s*FPS|mAP', m.content)
    ]
    verification = ''
    try:
        verification = verify_answer(' '.join(numeric_answers))
    except Exception as e:
        logger.warning('evidence verification failed (%s), skipping', e)
    if verification and '无冲突' not in verification:
        summary += f'\n\n{verification}'
    # Tool calling：让 LLM 决定通过工具导出报告
    llm_with_tools = llm.bind_tools([export_report])
    tool_msg = llm_with_tools.invoke(
        f'请调用 export_report 工具，把下面的复盘报告保存为 interview_report.md：\n\n{summary}',
        config={'callbacks': models.get_callbacks()},
    )
    export_note = ''
    for call in tool_msg.tool_calls:
        result = export_report.invoke(call['args'])
        export_note = f'\n\n📄 {result}'
    text = '\n' + summary + export_note + '\n\n面试结束，感谢作答！'
    return {
        'messages': [AIMessage(content=text)],
        'output': text,
        'finished': True,
        'score': score_data,
    }


# ---------------- 图与驱动器 ----------------


def build_turn_graph():
    """把面试官的一轮对话建成 DAG（即原先的 build_graph，语义不变）：

    START -> guardrails -> (decision -> (interviewer | wrap) | wrap) -> END

    逐轮驱动器（评测脚本、单轮测试）每收到一个回答就调用它一次；
    build_interview_graph() 把它作为完整循环中的轮次子图嵌入。
    """
    graph = StateGraph(InterviewState)
    graph.add_node('decision', decision_node)
    graph.add_node('interviewer', interviewer_node)
    graph.add_node('wrap', wrap_node)
    graph.add_conditional_edges(START, route_start, {'decision': 'decision', 'wrap': 'wrap'})
    graph.add_conditional_edges(
        'decision',
        route_after_decision,
        {'interviewer': 'interviewer', 'wrap': 'wrap'},
    )
    graph.add_edge('interviewer', END)
    graph.add_edge('wrap', END)
    return graph.compile()


def init_state(topics: list | None = None) -> InterviewState:
    """全新的面试状态。topics 为空 = 由 prepare_node 懒加载抽取。"""
    return {
        'messages': [],
        'topics': topics or [],
        'covered_topics': [],
        'current_topic': 0,
        'topic_round_count': 0,
        'round_count': 0,
        'decision': {},
        'output': '',
        'finished': False,
        'score': {},
    }


# ---------------- 循环图（LangGraph Harness 入口） ----------------


def prepare_node(state: InterviewState) -> dict:
    """补齐默认值，然后懒加载抽取话题（每个 thread 一次）。

    Harness 的运行可能从稀疏状态启动（空输入，或只有用户的第一条消息），
    因此所有缺失字段都会在任何代码读取它之前补齐。
    已结束的面试给出结束提醒，而不是再次问候。
    """
    updates = {k: v for k, v in init_state().items() if state.get(k) is None}
    if state.get('finished'):
        updates.update(messages=[AIMessage(content=FINISHED_NOTE)], output=FINISHED_NOTE)
        return updates
    if not state.get('topics'):
        updates['topics'] = prepare_topics()
        logger.info('topic anchors ready: %d', len(updates['topics']))
    return updates


def greet_node(state: InterviewState) -> dict:
    """用问候语开场，然后暂停等待第一个回答。"""
    return {'messages': [AIMessage(content=GREETING)], 'output': GREETING}


def wait_human_node(state: InterviewState) -> dict:
    """暂停等待候选人回答；Harness UI（或 CLI 驱动器）用回答文本
    恢复图的执行。"""
    answer = interrupt('等待候选人回答')
    return {'messages': [HumanMessage(content=answer)]}


def route_prepare(state: InterviewState) -> str:
    """prepare 之后：已结束的面试直接结束而不再次问候；
    输入中已经带有候选人消息的运行会直接回答该消息，
    而不是对着空气问候。"""
    if state['finished']:
        return 'end'
    if any(m.type == 'human' for m in state['messages']):
        return 'turn'
    return 'greet'


def route_after_answer(state: InterviewState) -> str:
    """候选人回答之后：跑一轮面试官对话
    （其 guardrails 可能直接 wrap 整场面试）。"""
    return 'end' if state['finished'] else 'turn'


def route_after_turn(state: InterviewState) -> str:
    """面试官一轮结束之后：面试仍在进行时
    等待下一个回答。"""
    return 'end' if state['finished'] else 'wait_human'


def _last_ai_text(result: dict) -> str:
    """图结果中最后一条面试官消息的文本。"""
    for m in reversed(result.get('messages', [])):
        if m.type == 'ai':
            return m.content
    return ''


def build_interview_graph(checkpointer=None):
    """供 LangGraph Harness 使用的完整面试循环：

    START -> prepare（抽取话题 | 已结束时道别）-> greet -> wait_human
    wait_human ->（interrupt，用回答 resume）-> turn DAG -> wait_human ... 直到
    wrap 置 finished -> END。

    该循环在轮次之间 interrupt，因此需要 checkpointer 才能 resume：
    Harness 服务器注入它自己的（那里传 checkpointer=None），而
    本地驱动器（CLI / Streamlit / 测试）传入 MemorySaver()。
    """
    graph = StateGraph(InterviewState)
    graph.add_node('prepare', prepare_node)
    graph.add_node('greet', greet_node)
    graph.add_node('wait_human', wait_human_node)
    graph.add_node('turn', build_turn_graph())
    graph.add_edge(START, 'prepare')
    graph.add_conditional_edges(
        'prepare', route_prepare, {'greet': 'greet', 'turn': 'turn', 'end': END}
    )
    graph.add_edge('greet', 'wait_human')
    graph.add_conditional_edges('wait_human', route_after_answer, {'turn': 'turn', 'end': END})
    graph.add_conditional_edges('turn', route_after_turn, {'wait_human': 'wait_human', 'end': END})
    return graph.compile(checkpointer=checkpointer)


def run_interview():
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s'
    )
    graph = build_interview_graph(MemorySaver())
    config: RunnableConfig = {'configurable': {'thread_id': 'cli'}}
    try:
        result = graph.invoke(init_state(), config)
    except Exception as e:
        logger.error('graph invoke failed: %s', e)
        print('[!] 面试官启动失败（LLM 调用异常），请稍后重试。')
        return
    print('\n面试官：' + _last_ai_text(result))
    while True:
        answer = input('\n你的回答：').strip()
        if answer.lower() in ('quit', 'exit', 'q'):
            print('面试提前结束。')
            break
        try:
            result = graph.invoke(Command(resume=answer), config)
        except Exception as e:
            logger.error('graph invoke failed: %s', e)
            print('\n[!] 面试官暂时开小差了（LLM 调用失败），请重新回答一次。')
            continue
        print('\n面试官：' + _last_ai_text(result))
        if '__interrupt__' not in result:
            break


if __name__ == '__main__':
    run_interview()
