"""混合检索（WBS 3.1/3.2）：BM25 关键词召回 + 向量语义召回，再用 RRF 融合。

做成独立模块，方便被 app.py、评测脚本以及后续的 LangGraph agent 复用。
"""
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

RRF_K = 60
CANDIDATE_K = 16  # 每个召回器在融合前返回的候选数量
# 2026-09-04 实测过的权衡：20->重排 4.0s，16->3.2s；每一步都检查过召回情况
RERANK_MODEL = "BAAI/bge-reranker-base"


def tokenize(text: str) -> list:
    """用 jieba 做中文分词（BM25 基于 token 计算）。"""
    return list(jieba.cut(text))


class BM25Index:
    """基于文档 chunk 的内存 BM25 关键词索引。"""

    def __init__(self, chunks: list):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])

    def search(self, query: str, top_k: int = CANDIDATE_K) -> list:
        """基于关键词词频的召回，按 BM25 分数排序。"""
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:top_k]]


def rrf_fusion(vector_results: list, bm25_results: list, k: int = RRF_K, top_k: int = 4) -> list:
    """RRF（Reciprocal Rank Fusion，倒数排名融合）：score(d) = sum(1 / (k + rank_i(d)))。

    在任一列表中排名靠前的 chunk 都能胜出；在两个列表中都靠前的胜出最多。
    """
    scores = {}
    chunk_by_id = {}
    for rank, c in enumerate(vector_results):
        chunk_by_id[id(c)] = c
        scores[id(c)] = scores.get(id(c), 0.0) + 1.0 / (k + rank + 1)
    for rank, c in enumerate(bm25_results):
        chunk_by_id[id(c)] = c
        scores[id(c)] = scores.get(id(c), 0.0) + 1.0 / (k + rank + 1)
    ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [chunk_by_id[cid] for cid in ranked]


class Reranker:
    """CrossEncoder 重排：给每个 (query, chunk) 对打分，再排序。"""

    def __init__(self, model_name: str = RERANK_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank_with_scores(self, query: str, chunks: list) -> list:
        """返回按分数降序排列的 [(chunk, score)]。"""
        pairs = [(query, c.page_content) for c in chunks]
        scores = self.model.predict(pairs)
        return sorted(zip(chunks, scores, strict=True), key=lambda x: x[1], reverse=True)

    def rerank(self, query: str, chunks: list, top_k: int = 4) -> list:
        return [c for c, _ in self.rerank_with_scores(query, chunks)[:top_k]]
