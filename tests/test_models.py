"""Model factory test (WBS 1.4): env vars pick the backend.

Usage:
    python tests/test_models.py              # resolution checks only, no API call
    CHAT_MODEL=claude-... python tests/test_models.py   # Claude branch check

A live-call verification against a real vendor (Zhipu GLM, etc.) is a
manual step: fill CHAT_MODEL / CHAT_BASE_URL / CHAT_API_KEY in .env and
run `python -c "import models; print(models.get_chat_model().invoke('你好'))"`.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))

from langchain_anthropic import ChatAnthropic  # noqa: E402
from langchain_deepseek import ChatDeepSeek  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

import models  # noqa: E402


def _clean_env():
    for k in ("CHAT_MODEL", "CHAT_BASE_URL", "CHAT_API_KEY"):
        os.environ.pop(k, None)


def test_default_is_deepseek():
    _clean_env()
    llm = models.get_chat_model()
    assert isinstance(llm, ChatDeepSeek), f"expected ChatDeepSeek, got {type(llm).__name__}"


def test_openai_compatible_switch():
    """CHAT_BASE_URL routes any OpenAI-compatible vendor through ChatOpenAI."""
    _clean_env()
    os.environ["CHAT_MODEL"] = "glm-4.5"
    os.environ["CHAT_BASE_URL"] = "https://open.bigmodel.cn/api/paas/v4"
    os.environ["CHAT_API_KEY"] = "sk-test"
    llm = models.get_chat_model()
    assert isinstance(llm, ChatOpenAI), f"expected ChatOpenAI, got {type(llm).__name__}"
    print(f"[openai-compatible] {llm.model_name} -> {llm.openai_api_base}")
    _clean_env()


def test_claude_branch_code_ready():
    """Claude resolves through init_chat_model (code-ready, no key needed to
    construct — a live call needs ANTHROPIC_API_KEY, unavailable in CN)."""
    _clean_env()
    os.environ["CHAT_MODEL"] = "claude-haiku-4-5-20251001"
    llm = models.get_chat_model()
    assert isinstance(llm, ChatAnthropic), f"expected ChatAnthropic, got {type(llm).__name__}"
    print(f"[claude] {llm.model} instance OK (no call made)")
    _clean_env()


if __name__ == "__main__":
    # read the switch before tests run: tests clean CHAT_* env themselves
    wants_claude = os.environ.get("CHAT_MODEL", "").startswith("claude")
    test_default_is_deepseek()
    print("[deepseek] default factory resolves to ChatDeepSeek")
    test_openai_compatible_switch()
    if wants_claude:
        test_claude_branch_code_ready()
    else:
        print("[claude] skipped (set CHAT_MODEL=claude-... to check the branch)")
    print("\nModel factory test passed")
