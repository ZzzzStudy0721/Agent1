"""Interview mock RAG agent — MVP (M1): load -> chunk -> embed -> retrieve -> answer with citations.

Standalone functions (load_and_chunk / retrieve / generate) so that M3 can
mount them as LangGraph nodes without rewriting.
"""
import os
import sys

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Must run before HF imports so HF_ENDPOINT (mirror) takes effect
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Fix mojibake on Windows GBK terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(  # type: ignore[attr-defined]
        encoding="utf-8", errors="replace"
    )

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import retrieval

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 51  # ~10% of chunk size
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
TOP_K = 4
COLLECTION = "kb_full"
# Guardrail: refuse when the best rerank score is below this.
# Probed 2026-09-02: relevant 0.876~0.999, irrelevant 0.000~0.028.
RERANK_THRESHOLD = 0.5
REFUSAL = "知识库中没有相关信息，无法回答。"


def load_and_chunk(data_dir: str = DATA_DIR) -> list:
    """Load all md/txt files under data/ and split into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for name in sorted(os.listdir(data_dir)):
        if name.endswith((".md", ".txt")):
            with open(os.path.join(data_dir, name), encoding="utf-8") as f:
                doc = Document(page_content=f.read(), metadata={"source": name})
            chunks.extend(splitter.split_documents([doc]))
    return chunks


def build_vectorstore(chunks: list, embeddings, collection_name: str = "langchain") -> Chroma:
    """Embed chunks with local bge-small-zh and persist to ChromaDB."""
    return Chroma.from_documents(
        chunks, embeddings, persist_directory=DB_DIR, collection_name=collection_name
    )


def retrieve(query: str, vectorstore: Chroma, top_k: int = TOP_K) -> list:
    """Pure vector retrieval — this is also the ablation baseline for M2."""
    return vectorstore.similarity_search(query, k=top_k)


def generate(query: str, context_docs: list, llm) -> str:
    """Generate an answer with inline citations; refuse when context is irrelevant."""
    numbered = "\n\n".join(
        f"[{i}] {d.page_content}" for i, d in enumerate(context_docs, 1)
    )
    system = (
        "你是面试辅导助手。只根据参考资料回答问题,引用处标注编号如[1]。"
        "如果参考资料与问题不相关,直接回答:\"知识库中没有相关信息,无法回答。\""
        "禁止编造资料中没有的内容。"
    )
    messages = [
        ("system", system),
        ("human", f"问题: {query}\n\n参考资料:\n{numbered}"),
    ]
    return llm.invoke(messages).content


def answer_question(query, vectorstore, bm25, reranker, llm, top_k=TOP_K):
    """Hybrid retrieval + rerank with refusal guardrail (M2 pipeline)."""
    v = vectorstore.similarity_search(query, k=retrieval.CANDIDATE_K)
    b = bm25.search(query, top_k=retrieval.CANDIDATE_K)
    fused = retrieval.rrf_fusion(v, b, top_k=retrieval.CANDIDATE_K)
    ranked = reranker.rerank_with_scores(query, fused)
    if not ranked or ranked[0][1] < RERANK_THRESHOLD:
        return REFUSAL
    docs = [c for c, _ in ranked[:top_k]]
    return generate(query, docs, llm)


def main():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.1)

    chunks = load_and_chunk()
    if not chunks:
        print("[!] No md/txt files found under data/. Put your resume/project docs there first.")
        return
    bm25 = retrieval.BM25Index(chunks)

    if os.path.isdir(DB_DIR):
        vectorstore = Chroma(
            embedding_function=embeddings, persist_directory=DB_DIR, collection_name=COLLECTION
        )
        print(f"[i] Loaded existing vector store ({DB_DIR})")
    else:
        vectorstore = build_vectorstore(chunks, embeddings, collection_name=COLLECTION)
        print(f"[i] {len(chunks)} chunks indexed, vector store persisted to {DB_DIR}")

    reranker = retrieval.Reranker()
    print("Ask questions about your resume/projects (input 'quit' to exit).")
    while True:
        query = input("\nQ: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        answer = answer_question(query, vectorstore, bm25, reranker, llm)
        print(f"\nA: {answer}")


if __name__ == "__main__":
    main()
