"""Hybrid retrieval (WBS 3.1/3.2): BM25 keyword recall + vector semantic recall, fused by RRF.

Standalone module so it can be reused by app.py, eval scripts and later the LangGraph agent.
"""
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

RRF_K = 60
CANDIDATE_K = 16  # each recaller returns this many candidates before fusion
# trade-off tested 2026-09-04: 20->4.0s rerank, 16->3.2s; recall checked per step
RERANK_MODEL = "BAAI/bge-reranker-base"


def tokenize(text: str) -> list:
    """Chinese tokenization with jieba (BM25 operates on tokens)."""
    return list(jieba.cut(text))


class BM25Index:
    """In-memory BM25 keyword index over document chunks."""

    def __init__(self, chunks: list):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])

    def search(self, query: str, top_k: int = CANDIDATE_K) -> list:
        """Keyword-frequency based recall, sorted by BM25 score."""
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:top_k]]


def rrf_fusion(vector_results: list, bm25_results: list, k: int = RRF_K, top_k: int = 4) -> list:
    """Reciprocal Rank Fusion: score(d) = sum(1 / (k + rank_i(d))).

    A chunk ranked high in either list wins; high in both wins most.
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
    """CrossEncoder reranking: score each (query, chunk) pair, then sort."""

    def __init__(self, model_name: str = RERANK_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank_with_scores(self, query: str, chunks: list) -> list:
        """Return [(chunk, score)] sorted by score desc."""
        pairs = [(query, c.page_content) for c in chunks]
        scores = self.model.predict(pairs)
        return sorted(zip(chunks, scores, strict=True), key=lambda x: x[1], reverse=True)

    def rerank(self, query: str, chunks: list, top_k: int = 4) -> list:
        return [c for c, _ in self.rerank_with_scores(query, chunks)[:top_k]]
