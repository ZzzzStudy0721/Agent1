"""FastAPI layer test: /chat answers with citations, refuses out-of-scope, /upload saves.

Usage: python tests/test_fastapi.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))

from fastapi.testclient import TestClient

import api


def test_chat_in_scope():
    client = TestClient(api.fastapi_app)
    resp = client.post("/chat", json={"question": "车辆计数误差控制在多少？"})
    assert resp.status_code == 200
    data = resp.json()
    print(f"[chat in-scope] refused={data['refused']}, answer={data['answer'][:60]}...")
    assert data["answer"] and not data["refused"]


def test_chat_refused():
    client = TestClient(api.fastapi_app)
    resp = client.post("/chat", json={"question": "我养了几只猫？"})
    data = resp.json()
    print(f"[chat refused] refused={data['refused']}, answer={data['answer']}")
    assert data["refused"] and data["answer"] == "知识库中没有相关信息，无法回答。"


def test_upload_roundtrip():
    client = TestClient(api.fastapi_app)
    resp = client.post(
        "/upload",
        files={"file": ("测试上传.md", "# 测试文档\n\n这是一个测试上传的文档，包含关键数字 12345。".encode("utf-8"))},
    )
    data = resp.json()
    print(f"[upload] {data}")
    assert "saved" in data and "chunks" in data
    # the uploaded doc should be searchable after rebuild
    resp2 = client.post("/chat", json={"question": "测试文档里的关键数字是多少？"})
    print(f"[chat after upload] {resp2.json()['answer'][:80]}")
    assert "12345" in resp2.json()["answer"], "uploaded content not searchable"
    # cleanup: remove the test doc and invalidate the pipeline so the real
    # knowledge base stays clean (index rebuilds lazily on next request)
    os.remove(os.path.join(api.app.DATA_DIR, "测试上传.md"))
    api._pipeline = None


if __name__ == "__main__":
    test_chat_in_scope()
    test_chat_refused()
    test_upload_roundtrip()
    print("\nFastAPI test passed")
