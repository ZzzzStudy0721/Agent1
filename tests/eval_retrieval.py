"""Retrieval evaluation on self-built test set (WBS 3.5).

Usage: python tests/eval_retrieval.py [mode]
- vector: pure vector baseline (MVP)
- more modes (bm25 / rrf / rerank) added step by step in M2
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

EVAL_SET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.json")
COLLECTION = "kb_full"
TOP_K = 5


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=app.EMBED_MODEL)


def rebuild_vectorstore():
    """Drop stale kb_full collection and rebuild so paper content is indexed."""
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
    return app.build_vectorstore(chunks, embeddings, collection_name=COLLECTION)


def retrieve(vectorstore, query: str, top_k: int = TOP_K):
    """Pure vector retrieval — the ablation baseline."""
    return vectorstore.similarity_search(query, k=top_k)


def evaluate(vectorstore, questions, top_k=TOP_K):
    """Recall@5 = fraction of answer keywords covered by top-k chunks.
    MRR = 1 / rank of first chunk that hits any keyword (0 if none)."""
    recall_sum = mrr_sum = 0.0
    for q in questions:
        chunks = retrieve(vectorstore, q["question"], top_k)
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
        missing = [k for k in q["keywords"] if k not in hits]
        print(f"  recall={recall:.2f} mrr={mrr:.3f}  miss={missing}  | {q['question'][:36]}")
    n = len(questions)
    print(f"\nRecall@5 = {recall_sum / n:.3f}   MRR = {mrr_sum / n:.3f}   ({n} questions)")


if __name__ == "__main__":
    with open(EVAL_SET, encoding="utf-8") as f:
        questions = json.load(f)
    store = rebuild_vectorstore()
    print(f"[i] vectorstore rebuilt, collection={COLLECTION}, questions={len(questions)}\n")
    evaluate(store, questions)
