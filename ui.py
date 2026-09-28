"""Streamlit UI（WBS 5.1）：知识库问答 Tab + 模拟面试 Tab。

分两个终端启动：
  1. uvicorn api:fastapi_app --host 127.0.0.1 --port 8000
  2. streamlit run ui.py
"""
import json
import os
import uuid
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
import requests
import streamlit as st

# 与 agent 不同，history 只依赖标准库，可以在顶部导入：
# 历史记录区随页面一起渲染，不会拖慢首屏。
from history import SCORE_DIMENSIONS, load_history  # noqa: E402

# 注意：`agent` 在下面的面试相关函数内部延迟导入。
# 若在这里导入，页面加载时就会跑 torch / sentence-transformers，
# 把整个脚本阻塞 1-2 分钟——在那之前页面看起来像是坏掉了。

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="面试官模拟 RAG Agent", layout="centered")
st.title("🎯 面试官模拟 RAG Agent")


def _iter_sse(resp):
    """从 SSE 响应体中逐块产出文本片段。"""
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        d = json.loads(line[6:])
        if "chunk" in d:
            yield d["chunk"]
        elif "answer" in d:
            yield d["answer"]
        elif "error" in d:
            yield f"⚠️ {d['error']}"


def stream_qa(question):
    """向 /chat 发 POST 请求并渲染 SSE 流；返回拼接好的完整回答。"""
    try:
        with requests.post(
            f"{API}/chat", json={"question": question}, timeout=180, stream=True
        ) as resp:
            if resp.status_code != 200:
                return f"⚠️ {resp.json().get('detail', '接口返回异常')}"
            return st.write_stream(_iter_sse(resp))
    except Exception as e:
        return f"⚠️ 无法连接后端（先启动 uvicorn？）：{e}"

page = st.sidebar.radio("模式切换", ["💬 知识库问答", "🎤 模拟面试"], key="page")

# ---------------- 知识库问答页（经由 FastAPI） ----------------
if page == "💬 知识库问答":
    st.caption("回答带原文引用；检索不到相关内容会明确拒答，绝不编造。")
    with st.sidebar:
        st.header("📚 知识库上传")
        st.caption("支持 md / txt / pdf（扫描版 PDF 无法提取文字）；上传后自动重建索引，约 1 分钟。")
        uploaded = st.file_uploader("选择文件", type=["md", "txt", "pdf"])
        if uploaded is not None:
            with st.spinner("上传并重建索引中…"):
                resp = requests.post(
                    f"{API}/upload",
                    files={"file": (uploaded.name, uploaded.getvalue())},
                    timeout=300,
                )
            if resp.ok:
                data = resp.json()
                st.success(f"已保存 {data.get('saved')}（{data.get('chunks')} 个片段）")
            else:
                st.error("上传失败")

    if "qa_messages" not in st.session_state:
        st.session_state.qa_messages = []
    for m in st.session_state.qa_messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
    question = st.chat_input("问点什么，例如：车辆计数误差控制在多少？（输入「开始面试」切换面试模式）", key="qa_input")
    if question:
        if "开始面试" in question:
            # 统一入口：切到面试模式，面试官提问，用户作答
            st.session_state.page = "🎤 模拟面试"
            st.session_state.start_interview = True
            st.rerun()
        st.session_state.qa_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("检索中…"):
                answer = stream_qa(question)
        st.session_state.qa_messages.append({"role": "assistant", "content": answer})

# ---------------- 模拟面试页（本地 LangGraph agent） ----------------
elif page == "🎤 模拟面试":
    st.caption(
        "AI 面试官：基于 JD+简历抽取话题锚点，自由对话 + 即兴追问，"
        "话题覆盖与轮数护栏防止跑偏；结束后统一复盘 + 证据核实 + 报告导出。"
        "首次启动需加载模型，请耐心等待。"
    )

    def start_interview():
        # 延迟导入：agent 会拉起 torch + sentence-transformers（常驻内存约 1.6 GB，
        # 首次加载要 1-2 分钟）。放在 spinner 内部执行，页面本身就能秒开，
        # 同时也告诉用户这个按钮为什么慢。
        import agent

        graph = agent.build_interview_graph(MemorySaver())
        config: Any = {"configurable": {"thread_id": uuid.uuid4().hex}}
        with st.spinner("正在加载模型并抽取话题锚点（首次约 1-2 分钟）…"):
            result = graph.invoke(agent.init_state(), config)
        st.session_state.interview = {
            "graph": graph,
            "config": config,
            "state": dict(result),
            "log": [("assistant", agent._last_ai_text(result))],
            "finished": False,
        }

    def advance(answer):
        import agent

        it = st.session_state.interview
        it["log"].append(("user", answer))
        result = it["graph"].invoke(Command(resume=answer), it["config"])
        it["state"] = dict(result)
        it["finished"] = result["finished"]
        it["log"].append(("assistant", agent._last_ai_text(result)))

    def _sync_uploads() -> bool:
        """把编辑过的预览内容写入 data/，仅在内容有变化时才重建索引。

        会与磁盘上已有的内容做比对，所以连点两次「开始」
        也不会重复重建索引。
        """
        import agent

        changed = False
        for kind, key in (("resume", "resume_preview"), ("jd", "jd_preview")):
            text = (st.session_state.get(key) or "").strip()
            if not text:
                continue
            name = agent.UPLOADED_RESUME if kind == "resume" else agent.UPLOADED_JD
            path = os.path.join(agent.DATA_DIR, name)
            current = ""
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    current = f.read().strip()
            if text != current:
                agent.save_uploaded_doc(text, kind)
                changed = True
        if changed:
            import app

            agent.reset_checker()
            with st.spinner("正在重建检索索引（约半分钟）…"):
                app.rebuild_index()
        return changed

    it = st.session_state.get("interview")

    # ---------- 可选：上传本次面试要用的简历 / 岗位 JD ----------
    with st.expander("📄 上传简历 / 岗位 JD（可选）", expanded=not it):
        st.caption(
            "支持 PDF、Word(.docx)、md、txt。提取出的文字先显示在下面，确认无误再点开始按钮；"
            "不传则沿用 data/ 目录里已有的资料。"
        )
        col_resume, col_jd = st.columns(2)
        with col_resume:
            resume_up = st.file_uploader(
                "简历", type=["pdf", "docx", "md", "txt"], key="up_resume"
            )
        with col_jd:
            jd_up = st.file_uploader(
                "岗位 JD 文件（可选）", type=["pdf", "docx", "md", "txt"], key="up_jd"
            )

        if resume_up is not None and st.session_state.get("_seen_resume") != resume_up.name:
            import app

            st.session_state["_seen_resume"] = resume_up.name
            try:
                st.session_state.resume_preview = app.extract_text(
                    resume_up.getvalue(), resume_up.name
                )
            except ValueError as e:
                st.session_state.resume_preview = ""
                st.error(f"简历解析失败：{e}")

        if jd_up is not None and st.session_state.get("_seen_jd") != jd_up.name:
            import app

            st.session_state["_seen_jd"] = jd_up.name
            try:
                st.session_state.jd_preview = app.extract_text(jd_up.getvalue(), jd_up.name)
            except ValueError as e:
                st.error(f"JD 解析失败：{e}")

        if st.session_state.get("_seen_resume"):
            st.text_area("简历文字（提取结果，可直接修改）", key="resume_preview", height=220)
        st.text_area(
            "岗位 JD（可直接粘贴文字，也可上传文件；留空表示不用 JD）",
            key="jd_preview",
            height=120,
        )

    if st.button("开始 / 重新开始面试") or st.session_state.pop("start_interview", False):
        _sync_uploads()
        start_interview()

    # ---------- 历史面试记录：分数表 + 单场详情回看 ----------
    with st.expander("📊 历史面试记录"):
        entries = load_history()
        if not entries:
            st.caption("还没有记录。完成一场面试后，分数和点评会自动存到这里。")
        else:
            st.dataframe(
                [
                    {
                        "时间": e.get("time", ""),
                        "岗位": e.get("job", ""),
                        "轮数": e.get("rounds", 0),
                        **{label: e.get(key, "") for key, label in SCORE_DIMENSIONS},
                    }
                    for e in entries
                ],
                hide_index=True,
            )
            pick = st.selectbox(
                "查看某一场的详情",
                range(len(entries)),
                index=len(entries) - 1,
                format_func=lambda i: (
                    f"{entries[i].get('time', '未知时间')}"
                    f"（{entries[i].get('rounds', 0)} 轮）"
                ),
            )
            entry = entries[pick]
            st.markdown(
                "\n".join(
                    f"- {label}：{entry.get(key, '?')}/10"
                    for key, label in SCORE_DIMENSIONS
                )
            )
            if entry.get("summary"):
                st.write(entry["summary"])
            if entry.get("strength"):
                st.success(f"✅ 最突出的优点：{entry['strength']}")
            if entry.get("improvement"):
                st.warning(f"⚠️ 最需改进的问题：{entry['improvement']}")
            if entry.get("topics"):
                st.caption("本场话题：" + "；".join(entry["topics"]))
            if entry.get("report"):
                st.caption(f"📄 完整报告：reports/{entry['report']}")

    it = st.session_state.get("interview")  # start_interview() 可能刚把它设置好
    if it:
        for role, content in it["log"]:
            with st.chat_message(role):
                st.write(content)
        if it["finished"]:
            st.success("面试结束，复盘报告已导出到 reports/ 目录")
        else:
            answer = st.chat_input("你的回答…（输入 quit 退出）", key="iv_input")
            if answer and answer.lower() not in ("quit", "exit", "q"):
                with st.chat_message("user"):
                    st.write(answer)
                with st.chat_message("assistant"):
                    with st.spinner("面试官思考中…"):
                        advance(answer)
                st.rerun()
