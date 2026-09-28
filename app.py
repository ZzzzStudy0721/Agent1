"""面试模拟 RAG agent —— MVP（M1）：load -> chunk -> embed -> retrieve -> 带引用回答。

各函数（load_and_chunk / retrieve / generate）都保持独立，这样 M3 可以
直接把它们挂成 LangGraph 节点，无需重写。
"""

import io
import logging
import os
import sys
import time

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 必须在导入 HF 相关库之前执行，HF_ENDPOINT（镜像源）才会生效
load_dotenv(os.path.join(BASE_DIR, '.env'))

# 修复 Windows GBK 终端下的乱码
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
CHUNK_OVERLAP = 51  # 约为 chunk size 的 10%
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_DIR = os.path.join(BASE_DIR, 'chroma_db')
TOP_K = 4
COLLECTION = 'kb_full'
# 护栏：最高向量相似度低于该值时就拒答。
# 2026-09-02 实测（8 个样本）：相关的 0.283~0.656，不相关的 0.011~0.127。
# 没有采用 rerank 分数：英文预训练的 bge-reranker-base 在中文口语化
# 提问上会失效（相关的低至 0.019，不相关的却有 0.028）。
VECTOR_THRESHOLD = 0.2
REFUSAL = '知识库中没有相关信息，无法回答。'

logger = logging.getLogger(__name__)


def load_and_chunk(data_dir: str = DATA_DIR) -> list:
    """加载 data/ 下所有 md/txt/pdf 文件，并切分成带重叠的 chunk。"""
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


def extract_text(source, filename: str) -> str:
    """从上传的文件（.md / .txt / .pdf / .docx）中提取纯文本。

    `source` 是文件系统路径或原始字节。Word 表格会被摊平成用竖线分隔的
    行：简历通常是用表格排版的，只取 `document.paragraphs` 会悄悄丢掉
    大部分内容。
    提取不出任何可读内容时（例如扫描版 PDF）抛出 ValueError，
    这样调用方可以提示用户，而不是去索引一个空文档。
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext in ('.md', '.txt'):
        if isinstance(source, bytes):
            raw = source
        else:
            with open(source, 'rb') as f:
                raw = f.read()
        text = raw.decode('utf-8', errors='replace')
    elif ext == '.pdf':
        from pypdf import PdfReader

        stream = io.BytesIO(source) if isinstance(source, bytes) else source
        text = '\n'.join((page.extract_text() or '') for page in PdfReader(stream).pages)
    elif ext == '.docx':
        import docx

        stream = io.BytesIO(source) if isinstance(source, bytes) else source
        document = docx.Document(stream)
        parts = [p.text for p in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                parts.append(' | '.join(cell.text.strip() for cell in row.cells))
        text = '\n'.join(parts)
    else:
        raise ValueError(f'不支持的文件格式 {ext}（支持 md / txt / pdf / docx）')
    text = text.strip()
    if not text:
        raise ValueError('没有提取到文字（可能是扫描版 PDF 或纯图片文档）')
    return text


def rebuild_index() -> int:
    """基于 data/ 重建持久化的向量库；返回 chunk 数量。

    在上传文件之后调用，这样新文档既能被问答流程检索到，
    也能被面试证据检查检索到。
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    chunks = load_and_chunk()
    existing = Chroma(
        embedding_function=embeddings,
        persist_directory=DB_DIR,
        collection_name=COLLECTION,
    )
    try:
        existing.delete_collection()
    except Exception:
        pass
    build_vectorstore(chunks, embeddings, collection_name=COLLECTION)
    return len(chunks)


def build_vectorstore(chunks: list, embeddings, collection_name: str = 'langchain') -> Chroma:
    """用本地 bge-small-zh 对 chunk 做 embedding，并持久化到 ChromaDB。"""
    return Chroma.from_documents(
        chunks, embeddings, persist_directory=DB_DIR, collection_name=collection_name
    )


def retrieve(query: str, vectorstore: Chroma, top_k: int = TOP_K) -> list:
    """纯向量检索 —— 这也是 M2 的消融基线。"""
    return vectorstore.similarity_search(query, k=top_k)


def _build_messages(query: str, context_docs: list) -> list:
    """构建生成回答所用的 prompt messages（流式与非流式共用）。"""
    numbered = '\n\n'.join(f'[{i}] {d.page_content}' for i, d in enumerate(context_docs, 1))
    system = (
        '你是面试辅导助手。只根据参考资料回答问题,引用处标注编号如[1]。'
        '回答简洁,直给结论和关键数据,控制在 150 字以内。'
        '如果参考资料与问题不相关,直接回答:"知识库中没有相关信息,无法回答。"'
        '禁止编造资料中没有的内容。'
    )
    return [
        ('system', system),
        ('human', f'问题: {query}\n\n参考资料:\n{numbered}'),
    ]


def generate(query: str, context_docs: list, llm) -> str:
    """生成带行内引用的回答（非流式）。"""
    # max_tokens 用来限制啰嗦的回答：LLM 生成是耗时最大的一段
    # （6-7.5s），而且简短的答案在做面试演示时效果也更好
    start = time.perf_counter()
    answer = llm.invoke(
        _build_messages(query, context_docs),
        max_tokens=350,
        config={"callbacks": models.get_callbacks()},
    ).content
    logger.info("LLM answer for %r: %.2fs", query[:30], time.perf_counter() - start)
    return answer


def stream_answer(query: str, context_docs: list, llm):
    """一边生成一边产出回答片段（方便 SSE 推送）。"""
    for chunk in llm.stream(
        _build_messages(query, context_docs),
        max_tokens=350,
        config={"callbacks": models.get_callbacks()},
    ):
        if chunk.content:
            yield chunk.content


def retrieve_or_refuse(query, vectorstore, bm25, reranker, top_k=TOP_K):
    """执行拒答判断 + 混合检索；返回 (refused: bool, docs: list)。

    拒答判断用的是中文 embedding 的相似度分数，在口语化提问上依然稳定
    （不像英文预训练的 reranker）。
    """
    start = time.perf_counter()
    top_vec = vectorstore.similarity_search_with_relevance_scores(query, k=1)
    if not top_vec or top_vec[0][1] < VECTOR_THRESHOLD:
        logger.info(
            "refused %r (gate %.3f)", query[:30], top_vec[0][1] if top_vec else -1
        )
        return True, []
    v = vectorstore.similarity_search(query, k=retrieval.CANDIDATE_K)
    b = bm25.search(query, top_k=retrieval.CANDIDATE_K)
    fused = retrieval.rrf_fusion(v, b, top_k=retrieval.CANDIDATE_K)
    ranked = reranker.rerank_with_scores(query, fused)
    docs = [c for c, _ in ranked[:top_k]]
    logger.info(
        "hybrid retrieval for %r: %.2fs (gate %.3f)",
        query[:30], time.perf_counter() - start, top_vec[0][1],
    )
    return False, docs


def answer_question(query, vectorstore, bm25, reranker, llm, top_k=TOP_K):
    """带拒答护栏的混合检索 + 重排（非流式，M2 流程）。"""
    refused, docs = retrieve_or_refuse(query, vectorstore, bm25, reranker, top_k)
    if refused:
        return REFUSAL
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
            import agent  # 延迟导入，避免循环依赖

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
