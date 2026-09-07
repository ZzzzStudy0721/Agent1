"""FastAPI layer test: /chat streams SSE with citations, refuses out-of-scope,
/upload saves and rebuilds.

Usage: python tests/test_fastapi.py
"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))

from fastapi.testclient import TestClient

import api
import pytest

pytestmark = pytest.mark.live  # needs the retrieval pipeline + LLM


def _chat(client, question):
    """POST /chat and assemble the SSE stream into (answer, refused)."""
    resp = client.post("/chat", json={"question": question})
    assert resp.status_code == 200, f"expected 200, got {resp.status_code}: {resp.text[:200]}"
    chunks = []
    refused = False
    for line in resp.iter_lines():
        if not line or not line.startswith("data: "):
            continue
        d = json.loads(line[6:])
        if "chunk" in d:
            chunks.append(d["chunk"])
        elif "answer" in d:
            return d["answer"], d.get("refused", False)
        elif "error" in d:
            return f"⚠️ {d['error']}", False
        elif "done" in d:
            break
    return "".join(chunks), refused


def test_chat_in_scope():
    client = TestClient(api.fastapi_app)
    answer, refused = _chat(client, "车辆计数误差控制在多少？")
    print(f"[chat in-scope] refused={refused}, answer={answer[:60]}...")
    assert answer and not refused


def test_chat_refused():
    client = TestClient(api.fastapi_app)
    answer, refused = _chat(client, "我养了几只猫？")
    print(f"[chat refused] refused={refused}, answer={answer}")
    assert refused and answer == "知识库中没有相关信息，无法回答。"


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
    answer, _ = _chat(client, "测试文档里的关键数字是多少？")
    print(f"[chat after upload] {answer[:80]}")
    assert "12345" in answer, "uploaded content not searchable"
    # cleanup: remove the test doc and invalidate the pipeline so the real
    # knowledge base stays clean (index rebuilds lazily on next request)
    os.remove(os.path.join(api.app.DATA_DIR, "测试上传.md"))
    api._pipeline = None


def test_upload_pdf():
    """PDF upload: text extracted, indexed, and searchable."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    pdf.cell(text="This is a project PDF document with magic number 98765 for RAG indexing.")
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_doc.pdf")
    pdf.output(pdf_path)

    client = TestClient(api.fastapi_app)
    with open(pdf_path, "rb") as f:
        resp = client.post("/upload", files={"file": ("test_doc.pdf", f.read())})
    data = resp.json()
    print(f"[upload pdf] {data}")
    assert "saved" in data, f"pdf upload failed: {data}"
    answer, _ = _chat(client, "PDF 文档里的 magic number 是多少？")
    print(f"[chat after pdf upload] {answer[:80]}")
    assert "98765" in answer, "uploaded PDF content not searchable"
    # cleanup
    os.remove(os.path.join(api.app.DATA_DIR, "test_doc.pdf"))
    os.remove(pdf_path)
    api._pipeline = None


if __name__ == "__main__":
    test_chat_in_scope()
    test_chat_refused()
    test_upload_roundtrip()
    test_upload_pdf()
    print("\nFastAPI test passed")
