"""Model factory (WBS 1.4): env vars switch the chat backend.

Resolution order:
1. CHAT_BASE_URL set -> ChatOpenAI against any OpenAI-compatible vendor
                        (Zhipu GLM / Qwen / Kimi / SiliconFlow ...);
                        CHAT_API_KEY carries the credential, CHAT_MODEL the name
2. otherwise         -> langchain init_chat_model resolves deepseek-chat to
                        ChatDeepSeek (default)

Examples (.env):
    # DeepSeek (default — nothing extra to set)
    CHAT_MODEL=deepseek-chat
    # Zhipu GLM (OpenAI-compatible endpoint)
    CHAT_MODEL=glm-4.5
    CHAT_BASE_URL=https://open.bigmodel.cn/api/paas/v4
    CHAT_API_KEY=xxxx
"""
import os

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Must run before langchain imports so env vars (.env) are available
load_dotenv(os.path.join(BASE_DIR, ".env"))

from langchain.chat_models import init_chat_model  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from settings import Settings, get_settings  # noqa: E402

DEFAULT_MODEL = "deepseek-chat"


def validate_settings(s: Settings) -> str | None:
    """Return a human-readable error if the model config is unusable, else None.

    Fails fast at startup/instantiation with a clear message instead of a
    confusing auth error on the first LLM call.
    """
    if s.chat_base_url and not s.chat_api_key:
        return "CHAT_BASE_URL is set, so CHAT_API_KEY must be set too"
    if (
        not s.chat_base_url
        and s.chat_model.startswith("deepseek")
        and not s.deepseek_api_key
    ):
        return "DEEPSEEK_API_KEY is not set: add it to .env (see README section 2)"
    return None


def get_chat_model(temperature: float = 0.1) -> BaseChatModel:
    """Instantiate the chat model selected by CHAT_MODEL / CHAT_BASE_URL.

    All backends share max_retries + timeout so transient API failures retry
    silently instead of crashing the caller.
    """
    s = get_settings()
    problem = validate_settings(s)
    if problem:
        raise ValueError(problem)
    if s.chat_base_url:
        return ChatOpenAI(
            model=s.chat_model,
            base_url=s.chat_base_url,
            api_key=s.chat_api_key,
            temperature=temperature,
            max_retries=2,
            timeout=120,
        )
    return init_chat_model(
        s.chat_model, temperature=temperature, max_retries=2, timeout=120
    )


def get_callbacks() -> list:
    """Langfuse callback handlers if configured, else an empty list.

    With LANGFUSE_PUBLIC_KEY set, pass the result as
    `config={"callbacks": models.get_callbacks()}` on LLM calls to trace
    tokens/latency (one fresh handler = one trace). Zero overhead otherwise:
    the heavy langfuse import happens only when configured.
    """
    if not get_settings().langfuse_public_key:
        return []
    from langfuse.langchain import CallbackHandler  # noqa: PLC0415

    return [CallbackHandler()]
