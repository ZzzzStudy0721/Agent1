"""Streamlit UI (WBS 5.1): chat with the RAG agent through the FastAPI backend.

Run in two terminals:
  1. uvicorn api:fastapi_app --host 127.0.0.1 --port 8000
  2. streamlit run ui.py
"""
import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="面试官模拟 RAG Agent", layout="centered")
st.title("🎯 面试官模拟 RAG Agent")
st.caption("知识库问答模式：回答带原文引用；检索不到相关内容会明确拒答，绝不编造。")

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

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

question = st.chat_input("问点什么，例如：车辆计数误差控制在多少？")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
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
    st.session_state.messages.append({"role": "assistant", "content": answer})
