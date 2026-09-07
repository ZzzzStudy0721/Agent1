"""Model factory test (WBS 1.4): env vars pick the backend.

Usage:
    python tests/test_models.py              # resolution checks only, no API call

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

from langchain_deepseek import ChatDeepSeek  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

import models  # noqa: E402


def _clean_env():
    for k in ("CHAT_MODEL", "CHAT_BASE_URL", "CHAT_API_KEY", "LANGFUSE_PUBLIC_KEY"):
        os.environ.pop(k, None)


def test_validate_settings():
    from settings import Settings

    # isolate: empty strings fall back to os.environ in pydantic-settings,
    # so remove the real keys from the environment for this test
    saved = {
        k: os.environ.pop(k, None)
        for k in ("DEEPSEEK_API_KEY", "CHAT_BASE_URL", "CHAT_API_KEY")
    }
    try:
        # no key at all -> clear error naming the missing key
        s = Settings(_env_file=None, chat_model="deepseek-chat")
        assert "DEEPSEEK_API_KEY" in models.validate_settings(s)
        # deepseek key present -> OK
        s = Settings(_env_file=None, chat_model="deepseek-chat", deepseek_api_key="sk-x")
        assert models.validate_settings(s) is None
        # OpenAI-compatible endpoint without its key -> clear error
        s = Settings(_env_file=None, chat_base_url="https://x.example")
        assert "CHAT_API_KEY" in models.validate_settings(s)
        # OpenAI-compatible endpoint with key -> OK
        s = Settings(_env_file=None, chat_base_url="https://x.example", chat_api_key="sk-y")
        assert models.validate_settings(s) is None
    finally:
        for k, v in saved.items():
            if v:
                os.environ[k] = v
    print("[validate] missing-key errors are clear and specific")


def test_callbacks_disabled_without_langfuse_key():
    _clean_env()
    os.environ["LANGFUSE_PUBLIC_KEY"] = ""  # neutralize a .env-configured key
    assert models.get_callbacks() == [], "no Langfuse key -> no callbacks"


def test_callbacks_enabled_with_langfuse_key():
    _clean_env()
    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-test"
    os.environ["LANGFUSE_SECRET_KEY"] = "sk-test"
    callbacks = models.get_callbacks()
    assert len(callbacks) == 1
    print(f"[langfuse] {type(callbacks[0]).__name__} constructed (no network call)")
    _clean_env()


def test_default_is_deepseek():
    _clean_env()
    # override whatever .env configures (GLM etc.) so the default branch runs
    os.environ["CHAT_MODEL"] = "deepseek-chat"
    os.environ["CHAT_BASE_URL"] = ""
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


if __name__ == "__main__":
    test_default_is_deepseek()
    print("[deepseek] default factory resolves to ChatDeepSeek")
    test_openai_compatible_switch()
    test_validate_settings()
    test_callbacks_disabled_without_langfuse_key()
    test_callbacks_enabled_with_langfuse_key()
    print("\nModel factory test passed")
