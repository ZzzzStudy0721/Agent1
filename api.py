"""FastAPI service layer (WBS 5.1): /chat Q&A endpoint + /upload knowledge endpoint.

Run: uvicorn api:fastapi_app --host 127.0.0.1 --port 8000
"""
import asyncio
import json
import logging
import os
import sys

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel

import app
import models
import retrieval

fastapi_app = FastAPI(title="Interview Mock RAG Agent API", version="0.1.0")

logger = logging.getLogger(__name__)

_pipeline = None  # lazy-loaded (vectorstore, bm25, reranker, llm)


def get_pipeline():
    """Build the Q&A pipeline once; rebuilt after /upload invalidates it."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    embeddings = HuggingFaceEmbeddings(model_name=app.EMBED_MODEL)
    chunks = app.load_and_chunk()
    try:
        store = Chroma(
            embedding_function=embeddings,
            persist_directory=app.DB_DIR,
            collection_name=app.COLLECTION,
        )
    except Exception:
        store = app.build_vectorstore(chunks, embeddings, collection_name=app.COLLECTION)
    _pipeline = (
        store,
        retrieval.BM25Index(chunks),
        retrieval.Reranker(),
        models.get_chat_model(temperature=0.1),
    )
    return _pipeline


class ChatRequest(BaseModel):
    question: str


def _sse(payload: dict) -> str:
    """Format one server-sent event."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@fastapi_app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Stream the answer over SSE.

    Pipeline init and local-model retrieval are CPU-bound, so they run in a
    thread (asyncio.to_thread) and never block the event loop; the sync LLM
    stream generator is iterated by Starlette's threadpool the same way.
    """
    try:
        store, bm25, reranker, llm = await asyncio.to_thread(get_pipeline)
    except Exception as e:
        logger.exception("pipeline init failed: %s", e)
        raise HTTPException(status_code=503, detail="知识库初始化失败，请稍后重试") from e
    try:
        refused, docs = await asyncio.to_thread(
            app.retrieve_or_refuse, req.question, store, bm25, reranker
        )
    except Exception as e:
        logger.exception("retrieval failed: %s", e)
        raise HTTPException(status_code=503, detail="检索失败，请稍后重试") from e

    def sse_answer():
        if refused:
            yield _sse({"answer": app.REFUSAL, "refused": True})
            return
        try:
            for chunk in app.stream_answer(req.question, docs, llm):
                yield _sse({"chunk": chunk})
            yield _sse({"done": True})
        except Exception as e:
            logger.exception("LLM stream failed: %s", e)
            yield _sse({"error": "LLM 调用失败，请稍后重试"})

    return StreamingResponse(sse_answer(), media_type="text/event-stream")


@fastapi_app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    """Save an md/txt/pdf file into the knowledge base and rebuild the index."""
    if not file.filename.endswith((".md", ".txt", ".pdf")):
        return {"error": "only .md / .txt / .pdf files are supported"}
    os.makedirs(app.DATA_DIR, exist_ok=True)
    path = os.path.join(app.DATA_DIR, os.path.basename(file.filename))
    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)
    # invalidate pipeline and rebuild the vector store so the new doc is searchable
    global _pipeline
    _pipeline = None
    embeddings = HuggingFaceEmbeddings(model_name=app.EMBED_MODEL)
    chunks = app.load_and_chunk()
    existing = Chroma(
        embedding_function=embeddings,
        persist_directory=app.DB_DIR,
        collection_name=app.COLLECTION,
    )
    try:
        existing.delete_collection()
    except Exception:
        pass
    app.build_vectorstore(chunks, embeddings, collection_name=app.COLLECTION)
    logger.info("uploaded %s, index rebuilt (%d chunks)", file.filename, len(chunks))
    return {"saved": file.filename, "chunks": len(chunks)}
