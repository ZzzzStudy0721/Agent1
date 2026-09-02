"""Guardrail test (WBS 3.4): in-scope questions answered with citations, out-of-scope refused.

Usage: python tests/test_guardrail.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))

from langchain_chroma import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings

import app
import retrieval


def build_pipeline():
    chunks = app.load_and_chunk()
    embeddings = HuggingFaceEmbeddings(model_name=app.EMBED_MODEL)
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=app.DB_DIR,
        collection_name=app.COLLECTION,
    )
    bm25 = retrieval.BM25Index(chunks)
    reranker = retrieval.Reranker()
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.1)
    return vectorstore, bm25, reranker, llm


def test_in_scope_answered():
    vectorstore, bm25, reranker, llm = build_pipeline()
    ans = app.answer_question("车辆计数误差控制在多少？", vectorstore, bm25, reranker, llm)
    print(f"[in-scope] {ans[:80]}...")
    assert ans and app.REFUSAL not in ans, "in-scope question should be answered"


def test_out_of_scope_refused():
    vectorstore, bm25, reranker, llm = build_pipeline()
    ans = app.answer_question("我养了几只猫？", vectorstore, bm25, reranker, llm)
    print(f"[out-of-scope] {ans}")
    assert ans == app.REFUSAL, "out-of-scope question should be refused"


if __name__ == "__main__":
    test_in_scope_answered()
    test_out_of_scope_refused()
    print("\nGuardrail test passed")
