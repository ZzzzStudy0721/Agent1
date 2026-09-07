"""Interview mock RAG agent — MVP (M1): load -> chunk -> embed -> retrieve -> answer with citations.

Standalone functions (load_and_chunk / retrieve / generate) so that M3 can
mount them as LangGraph nodes without rewriting.
"""

import logging
import os
import sys
import time

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Must run before HF imports so HF_ENDPOINT (mirror) takes effect
load_dotenv(os.path.join(BASE_DIR, '.env'))

# Fix mojibake on Windows GBK terminals
if sys.platform == 'win32':
    sys.stdout.reconfigure(  # type: ignore[attr-defined]
        encoding='utf-8', errors='replace'
    )

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import models
import retrieval

EMBED_MODEL = 'BAAI/bge-small-zh-v1.5'
CHUNK_SIZE = 512
CHUNK_OVERLAP = 51  # ~10% of chunk size
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_DIR = os.path.join(BASE_DIR, 'chroma_db')
TOP_K = 4
COLLECTION = 'kb_full'
# Guardrail: refuse when the best vector similarity is below this.
# Probed 2026-09-02 (8 samples): relevant 0.283~0.656, irrelevant 0.011~0.127.
# Rerank score was NOT used: English-pretrained bge-reranker-base fails on
# colloquial Chinese questions (relevant as low as 0.019 vs irrelevant 0.028).
VECTOR_THRESHOLD = 0.2
REFUSAL = '知识库中没有相关信息，无法回答。'

logger = logging.getLogger(__name__)


def load_and_chunk(data_dir: str = DATA_DIR) -> list:
    """Load all md/txt/pdf files under data/ and split into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if name.endswith(('.md', '.txt')):
            with open(path, encoding='utf-8') as f:
                doc = Document(page_content=f.read(), metadata={'source': name})
        elif name.endswith('.pdf'):
            from pypdf import PdfReader

            reader = PdfReader(path)
            text = '\n'.join((page.extract_text() or '') for page in reader.pages)
            if not text.strip():
                logger.warning(f'{name}: no text extracted (scanned PDF?), skipped')
                continue
            doc = Document(page_content=text, metadata={'source': name})
        else:
            continue
        chunks.extend(splitter.split_documents([doc]))
    return chunks


def build_vectorstore(chunks: list, embeddings, collection_name: str = 'langchain') -> Chroma:
    """Embed chunks with local bge-small-zh and persist to ChromaDB."""
    return Chroma.from_documents(
        chunks, embeddings, persist_directory=DB_DIR, collection_name=collection_name
    )


def retrieve(query: str, vectorstore: Chroma, top_k: int = TOP_K) -> list:
    """Pure vector retrieval — this is also the ablation baseline for M2."""
    return vectorstore.similarity_search(query, k=top_k)


def generate(query: str, context_docs: list, llm) -> str:
    """Generate an answer with inline citations; refuse when context is irrelevant."""
    numbered = '\n\n'.join(f'[{i}] {d.page_content}' for i, d in enumerate(context_docs, 1))
    system = (
        '你是面试辅导助手。只根据参考资料回答问题,引用处标注编号如[1]。'
        '回答简洁,直给结论和关键数据,控制在 150 字以内。'
        '如果参考资料与问题不相关,直接回答:"知识库中没有相关信息,无法回答。"'
        '禁止编造资料中没有的内容。'
    )
    messages = [
        ('system', system),
        ('human', f'问题: {query}\n\n参考资料:\n{numbered}'),
    ]
    # max_tokens caps long-winded answers: LLM generation is the biggest latency
    # slice (6-7.5s), and short answers are also better for interview demos
    start = time.perf_counter()
    answer = llm.invoke(messages, max_tokens=350).content
    logger.info("LLM answer for %r: %.2fs", query[:30], time.perf_counter() - start)
    return answer


def answer_question(query, vectorstore, bm25, reranker, llm, top_k=TOP_K):
    """Hybrid retrieval + rerank with refusal guardrail (M2 pipeline).

    Refusal gate uses the Chinese embedding's similarity score, which stays
    robust on colloquial questions (unlike the English-pretrained reranker).
    """
    start = time.perf_counter()
    top_vec = vectorstore.similarity_search_with_relevance_scores(query, k=1)
    if not top_vec or top_vec[0][1] < VECTOR_THRESHOLD:
        logger.info(
            "refused %r (gate %.3f)", query[:30], top_vec[0][1] if top_vec else -1
        )
        return REFUSAL
    v = vectorstore.similarity_search(query, k=retrieval.CANDIDATE_K)
    b = bm25.search(query, top_k=retrieval.CANDIDATE_K)
    fused = retrieval.rrf_fusion(v, b, top_k=retrieval.CANDIDATE_K)
    ranked = reranker.rerank_with_scores(query, fused)
    docs = [c for c, _ in ranked[:top_k]]
    logger.info(
        "hybrid retrieval for %r: %.2fs (gate %.3f)",
        query[:30], time.perf_counter() - start, top_vec[0][1],
    )
    return generate(query, docs, llm)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    llm = models.get_chat_model(temperature=0.1)

    chunks = load_and_chunk()
    if not chunks:
        print('[!] No md/txt files found under data/. Put your resume/project docs there first.')
        return
    bm25 = retrieval.BM25Index(chunks)

    if os.path.isdir(DB_DIR):
        vectorstore = Chroma(
            embedding_function=embeddings, persist_directory=DB_DIR, collection_name=COLLECTION
        )
        logger.info(f'Loaded existing vector store ({DB_DIR})')
    else:
        vectorstore = build_vectorstore(chunks, embeddings, collection_name=COLLECTION)
        logger.info(f'{len(chunks)} chunks indexed, vector store persisted to {DB_DIR}')

    reranker = retrieval.Reranker()
    print("Ask questions about your resume/projects (input 'quit' to exit).")
    print('输入「开始面试」切换到面试官模式（AI 提问，你回答）。')
    while True:
        query = input('\n请输入: ').strip()
        if query.lower() in ('quit', 'exit', 'q'):
            break
        if not query:
            continue
        if '开始面试' in query:
            import agent  # lazy import to avoid circular dependency

            agent.run_interview()
            continue
        try:
            answer = answer_question(query, vectorstore, bm25, reranker, llm)
        except Exception as e:
            logger.error('LLM call failed: %s', e)
            answer = '[!] 回答生成失败（LLM 调用异常），请稍后重试'
        print(f'\nA: {answer}')


if __name__ == '__main__':
    main()
