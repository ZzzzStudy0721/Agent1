"""Model factory test (WBS 1.4): CHAT_MODEL env var picks the backend.

Usage:
    python tests/test_models.py                       # default: DeepSeek
    CHAT_MODEL=claude-haiku-4-5-20251001 python tests/test_models.py

The Claude run needs ANTHROPIC_API_KEY in .env (a Haiku call costs a
fraction of a cent). Both runs only instantiate the client — no tokens
are spent on construction.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))

from langchain_anthropic import ChatAnthropic  # noqa: E402
from langchain_deepseek import ChatDeepSeek  # noqa: E402

import models  # noqa: E402


def test_default_is_deepseek():
    os.environ.pop("CHAT_MODEL", None)
    llm = models.get_chat_model()
    assert isinstance(llm, ChatDeepSeek), f"expected ChatDeepSeek, got {type(llm).__name__}"


def test_claude_env_switch():
    os.environ["CHAT_MODEL"] = "claude-haiku-4-5-20251001"
    llm = models.get_chat_model()
    assert isinstance(llm, ChatAnthropic), f"expected ChatAnthropic, got {type(llm).__name__}"
    print(f"[claude] {llm.model} instance OK (no call made)")


if __name__ == "__main__":
    # read the switch before tests run: test_default pops CHAT_MODEL
    wants_claude = os.environ.get("CHAT_MODEL", "").startswith("claude")
    test_default_is_deepseek()
    print("[deepseek] default factory resolves to ChatDeepSeek")
    if wants_claude:
        test_claude_env_switch()
    else:
        print("[claude] skipped (set CHAT_MODEL=claude-... to verify the Anthropic backend)")
    print("\nModel factory test passed")
