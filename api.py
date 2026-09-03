"""FastAPI service layer (WBS 5.1): /chat Q&A endpoint + /upload knowledge endpoint.

Run: uvicorn api:fastapi_app --host 127.0.0.1 --port 8000
"""
import os
import sys

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, File, UploadFile
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel

import app
import models
import retrieval

fastapi_app = FastAPI(title="Interview Mock RAG Agent API", version="0.1.0")

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


class ChatResponse(BaseModel):
    answer: str
    refused: bool


@fastapi_app.post("/chat")
def chat(req: ChatRequest) -> ChatResponse:
    store, bm25, reranker, llm = get_pipeline()
    answer = app.answer_question(req.question, store, bm25, reranker, llm)
    return ChatResponse(answer=answer, refused=(answer == app.REFUSAL))


@fastapi_app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    """Save an md/txt file into the knowledge base and rebuild the index."""
    if not file.filename.endswith((".md", ".txt")):
        return {"error": "only .md / .txt files are supported"}
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
    return {"saved": file.filename, "chunks": len(chunks)}
