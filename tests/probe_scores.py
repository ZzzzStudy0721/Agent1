"""Probe reranker scores for relevant vs irrelevant questions, to pick a refusal threshold.

Usage: python tests/probe_scores.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))

import retrieval
import eval_retrieval as er

RELEVANT = [
    "毕设中对比了哪些目标检测模型？",
    "车辆计数误差控制在多少？",
    "Flask 用电监控系统拆分为哪几个模块？",
    "系统整体架构分哪三层？",
]
IRRELEVANT = [
    "我养了几只猫？",
    "今天晚饭吃什么？",
    "NBA 湖人队最近战绩怎么样？",
    "长安大学的校训是什么？",
]

chunks, store = er.rebuild()
retrievers = er.make_retrievers(chunks, store)
reranker = retrieval.Reranker()

print("=== relevant questions (top-1 rerank score) ===")
for q in RELEVANT:
    cands = retrievers["rrf"](q, retrieval.CANDIDATE_K)
    ranked = reranker.rerank_with_scores(q, cands)
    print(f"  {ranked[0][1]:+.3f}  {q}")

print("\n=== irrelevant questions (top-1 rerank score) ===")
for q in IRRELEVANT:
    cands = retrievers["rrf"](q, retrieval.CANDIDATE_K)
    ranked = reranker.rerank_with_scores(q, cands)
    print(f"  {ranked[0][1]:+.3f}  {q}")
