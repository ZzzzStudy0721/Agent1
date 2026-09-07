"""Smoke test: verify DeepSeek API key works (WBS task 1.3).

Usage: python tests/test_api.py   (or pytest, skipped without the key)
"""
import json
import os
import urllib.request

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_key():
    with open(os.path.join(BASE_DIR, ".env"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


@pytest.mark.live
def test_api_key_works():
    key = _read_key()
    assert key, "DEEPSEEK_API_KEY not found in .env"

    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(
            {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                "max_tokens": 10,
            }
        ).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    content = data["choices"][0]["message"]["content"]
    print(f"API OK, reply: {content}")
    assert "OK" in content, f"Unexpected reply: {content}"


if __name__ == "__main__":
    test_api_key_works()
    print("\nAPI smoke test passed")
