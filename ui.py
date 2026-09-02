"""Streamlit UI (WBS 5.1): Q&A chat tab + mock interview tab.

Run in two terminals:
  1. uvicorn api:fastapi_app --host 127.0.0.1 --port 8000
  2. streamlit run ui.py
"""
import requests
import streamlit as st

import agent

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="面试官模拟 RAG Agent", layout="centered")
st.title("🎯 面试官模拟 RAG Agent")

tab_qa, tab_iv = st.tabs(["💬 知识库问答", "🎤 模拟面试"])

# ---------------- Q&A tab (through FastAPI) ----------------
with tab_qa:
    st.caption("回答带原文引用；检索不到相关内容会明确拒答，绝不编造。")
    with st.sidebar:
        st.header("📚 知识库上传")
        st.caption("支持 md / txt；上传后自动重建索引，约 1 分钟。")
        uploaded = st.file_uploader("选择文件", type=["md", "txt"])
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
    question = st.chat_input("问点什么，例如：车辆计数误差控制在多少？", key="qa_input")
    if question:
        st.session_state.qa_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("检索中…"):
                try:
                    resp = requests.post(f"{API}/chat", json={"question": question}, timeout=180)
                    answer = resp.json().get("answer", "接口返回异常")
                except Exception as e:
                    answer = f"⚠️ 无法连接后端（先启动 uvicorn？）：{e}"
            st.write(answer)
        st.session_state.qa_messages.append({"role": "assistant", "content": answer})

# ---------------- Mock interview tab (local LangGraph agent) ----------------
with tab_iv:
    st.caption(
        "AI 面试官：JD 定制出题 → 追问（最多 2 轮）→ 三维度点评 → "
        "证据核实（与简历矛盾点出）→ 复盘报告导出。首次启动需加载模型，请耐心等待。"
    )

    def start_interview():
        jd_text = agent.load_jd()
        if jd_text:
            with st.spinner("基于 JD 生成面试题中…"):
                agent.DEMO_QUESTIONS[:] = agent.generate_questions(jd_text)
        graph = agent.build_graph()
        st.session_state.interview = {
            "graph": graph,
            "history": [],
            "state": {
                "messages": [],
                "question_index": 0,
                "followup_count": 0,
                "followup_verdict": "",
                "output": "",
            },
            "log": [],
            "finished": False,
        }
        # first invoke: greeting + question 1
        with st.spinner("面试官就位中…"):
            result = graph.invoke(st.session_state.interview["state"])
        st.session_state.interview["log"].append(("assistant", result["output"]))
        st.session_state.interview["state"] = dict(result)

    def advance(answer):
        it = st.session_state.interview
        it["history"].append(("user", answer))
        it["state"]["messages"] = it["history"]
        it["state"]["output"] = ""
        result = it["graph"].invoke(it["state"])
        it["log"].append(("assistant", result["output"]))
        it["state"] = dict(result)
        it["finished"] = result["question_index"] >= len(agent.DEMO_QUESTIONS)

    if st.button("开始 / 重新开始面试"):
        start_interview()

    it = st.session_state.get("interview")
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
                    with st.spinner("面试官思考中（检索 + 点评 + 证据核实，约 1 分钟）…"):
                        advance(answer)
                st.rerun()
