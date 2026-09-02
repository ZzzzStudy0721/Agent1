"""Diagnose: why some questions get refused. Print fusion candidates + rerank scores.

Usage: python tests/diagnose.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))

import retrieval
import eval_retrieval as er

QUESTIONS = [
    "简历中提到了模型对比，具体有哪些？",
    "毕设里对比了哪些模型？",
    "我毕设里对比了哪些模型",
    "车辆计数误差控制在多少？",  # control: works in tests
    "我养了几只猫？",  # irrelevant controls
    "今天晚饭吃什么？",
    "NBA 湖人队最近战绩怎么样？",
    "长安大学的校训是什么？",
]

chunks, store = er.rebuild()
retrievers = er.make_retrievers(chunks, store)
reranker = retrieval.Reranker()

for q in QUESTIONS:
    # Signal 1: vector similarity (Chinese embedding, robust to colloquial wording)
    v_scored = store.similarity_search_with_relevance_scores(q, k=3)
    # Signal 2: BM25 keyword score
    bm25_max = max(retrieval.BM25Index(chunks).bm25.get_scores(retrieval.tokenize(q))) if chunks else 0
    # Signal 3: rerank score (current guardrail)
    cands = retrievers["rrf"](q, retrieval.CANDIDATE_K)
    ranked = reranker.rerank_with_scores(q, cands)
    print(f"\nQ: {q}")
    print(f"  vector top-1 = {v_scored[0][1]:.3f}   bm25 max = {bm25_max:.3f}   rerank top-1 = {ranked[0][1]:+.3f}")
    print(f"    vector top-3: {[round(s, 3) for _, s in v_scored]}")
