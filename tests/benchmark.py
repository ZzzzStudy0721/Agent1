"""Latency benchmark: per-stage timing for the Q&A pipeline.

Non-functional requirement (plan 1.4): retrieval + generation < 10s.
Usage: python tests/benchmark.py
"""
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))

import api
import app
import retrieval

QUESTIONS = [
    "车辆计数误差控制在多少？",
    "毕设中对比了哪些目标检测模型？",
    "我毕设里对比了哪些模型",
]

store, bm25, reranker, llm = api.get_pipeline()

print(f"{'question':34s} {'gate':>7s} {'vec':>7s} {'bm25':>7s} {'rrf':>7s} {'rerank':>8s} {'llm':>7s} {'total':>7s}")
for q in QUESTIONS:
    t0 = time.perf_counter()
    top_vec = store.similarity_search_with_relevance_scores(q, k=1)
    t1 = time.perf_counter()
    v = store.similarity_search(q, k=retrieval.CANDIDATE_K)
    t2 = time.perf_counter()
    b = bm25.search(q, top_k=retrieval.CANDIDATE_K)
    t3 = time.perf_counter()
    fused = retrieval.rrf_fusion(v, b, top_k=retrieval.CANDIDATE_K)
    t4 = time.perf_counter()
    ranked = reranker.rerank_with_scores(q, fused)
    t5 = time.perf_counter()
    docs = [c for c, _ in ranked[:4]]
    if top_vec and top_vec[0][1] >= app.VECTOR_THRESHOLD:
        app.generate(q, docs, llm)
    t6 = time.perf_counter()
    print(
        f"{q[:32]:34s} {t1-t0:6.2f}s {t2-t1:6.2f}s {t3-t2:6.2f}s "
        f"{t4-t3:6.2f}s {t5-t4:7.2f}s {t6-t5:6.2f}s {t6-t0:6.2f}s"
    )
