"""Guardrail test (WBS 3.4): in-scope questions answered with citations, out-of-scope refused.

Usage: python tests/test_guardrail.py
"""
import json
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


IN_SCOPE = [
    "车辆计数误差控制在多少？",
    "简历中提到了模型对比，具体有哪些？",
    "毕设里对比了哪些模型？",
    "我毕设里对比了哪些模型",
]


def test_in_scope_answered():
    vectorstore, bm25, reranker, llm = build_pipeline()
    for q in IN_SCOPE:
        ans = app.answer_question(q, vectorstore, bm25, reranker, llm)
        print(f"[in-scope] {q}\n  -> {ans[:60]}...")
        assert ans and app.REFUSAL not in ans, f"in-scope question refused: {q}"


def test_out_of_scope_refused():
    vectorstore, bm25, reranker, llm = build_pipeline()
    ans = app.answer_question("我养了几只猫？", vectorstore, bm25, reranker, llm)
    print(f"[out-of-scope] {ans}")
    assert ans == app.REFUSAL, "out-of-scope question should be refused"


def test_eval_set_not_refused():
    """Every question in the eval set must pass the vector gate (no false refusal)."""
    vectorstore, bm25, reranker, llm = build_pipeline()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.json"), encoding="utf-8") as f:
        questions = json.load(f)
    refused = []
    for q in questions:
        top_vec = vectorstore.similarity_search_with_relevance_scores(q["question"], k=1)
        if not top_vec or top_vec[0][1] < app.VECTOR_THRESHOLD:
            refused.append((q["question"], top_vec[0][1] if top_vec else None))
    print(f"[eval-set gate] {len(questions) - len(refused)}/{len(questions)} pass vector gate")
    for q, s in refused:
        print(f"  refused: {q} (score={s:.3f})")
    assert not refused, f"{len(refused)} eval questions falsely refused"


if __name__ == "__main__":
    test_in_scope_answered()
    test_out_of_scope_refused()
    test_eval_set_not_refused()
    print("\nGuardrail test passed")
