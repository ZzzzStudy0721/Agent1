"""MVP end-to-end test: index data/, retrieve, generate answer with citations.

First run downloads the embedding model (~100MB via HF mirror, may take minutes).
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))  # before HF imports so HF_ENDPOINT takes effect

from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings

import app


def test_end_to_end():
    chunks = app.load_and_chunk()
    assert chunks, "no chunks loaded from data/"
    print(f"[1] {len(chunks)} chunks loaded")

    embeddings = HuggingFaceEmbeddings(model_name=app.EMBED_MODEL)
    vectorstore = app.build_vectorstore(chunks, embeddings)
    print("[2] vector store built")

    docs = app.retrieve("我毕设用了什么模型", vectorstore)
    assert docs, "no docs retrieved"
    print(f"[3] retrieved {len(docs)} docs, top-1: {docs[0].page_content[:60]}...")

    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.1)
    answer = app.generate("我毕设用了什么模型", docs, llm)
    assert answer, "empty answer"
    print(f"[4] answer:\n{answer}")


if __name__ == "__main__":
    test_end_to_end()
    print("\nMVP test passed")
