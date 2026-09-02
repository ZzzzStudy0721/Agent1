"""Retrieval evaluation on self-built test set (WBS 3.5).

Usage: python tests/eval_retrieval.py [vector|bm25|rrf]   (default: run all three)
"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import app
import retrieval

EVAL_SET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.json")
COLLECTION = "kb_full"
TOP_K = 5


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=app.EMBED_MODEL)


def rebuild():
    """Drop stale kb_full collection and rebuild; return chunks and vectorstore."""
    embeddings = get_embeddings()
    existing = Chroma(
        embedding_function=embeddings,
        persist_directory=app.DB_DIR,
        collection_name=COLLECTION,
    )
    try:
        existing.delete_collection()
    except Exception:
        pass  # collection does not exist yet
    chunks = app.load_and_chunk()
    store = app.build_vectorstore(chunks, embeddings, collection_name=COLLECTION)
    return chunks, store


def evaluate(name, retriever_fn, questions, top_k=TOP_K):
    """Recall@5 = fraction of answer keywords covered by top-k chunks.
    MRR = 1 / rank of first chunk that hits any keyword (0 if none)."""
    recall_sum = mrr_sum = 0.0
    for q in questions:
        chunks = retriever_fn(q["question"], top_k)
        hits = set()
        first_rank = 0
        for i, c in enumerate(chunks, 1):
            text = c.page_content.lower()
            for kw in q["keywords"]:
                if kw.lower() in text:
                    hits.add(kw)
                    if first_rank == 0:
                        first_rank = i
        recall = len(hits) / len(q["keywords"])
        mrr = 1.0 / first_rank if first_rank else 0.0
        recall_sum += recall
        mrr_sum += mrr
    n = len(questions)
    recall_avg = recall_sum / n
    mrr_avg = mrr_sum / n
    print(f"  {name:8s} Recall@5 = {recall_avg:.3f}   MRR = {mrr_avg:.3f}")
    return recall_avg, mrr_avg


def make_retrievers(chunks, store):
    bm25 = retrieval.BM25Index(chunks)

    def vector(q, k):
        return store.similarity_search(q, k=k)

    def bm25_only(q, k):
        return bm25.search(q, top_k=k)

    def rrf(q, k):
        v = store.similarity_search(q, k=retrieval.CANDIDATE_K)
        b = bm25.search(q, top_k=retrieval.CANDIDATE_K)
        return retrieval.rrf_fusion(v, b, top_k=k)

    return {"vector": vector, "bm25": bm25_only, "rrf": rrf}


if __name__ == "__main__":
    with open(EVAL_SET, encoding="utf-8") as f:
        questions = json.load(f)
    chunks, store = rebuild()
    print(f"[i] vectorstore rebuilt, {len(chunks)} chunks, {len(questions)} questions\n")
    retrievers = make_retrievers(chunks, store)
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    modes = [mode] if mode != "all" else list(retrievers)
    for m in modes:
        evaluate(m, retrievers[m], questions)
