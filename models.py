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

DEFAULT_MODEL = "deepseek-chat"


def get_chat_model(temperature: float = 0.1) -> BaseChatModel:
    """Instantiate the chat model selected by CHAT_MODEL / CHAT_BASE_URL."""
    model = os.environ.get("CHAT_MODEL", DEFAULT_MODEL)
    base_url = os.environ.get("CHAT_BASE_URL")
    if base_url:
        api_key = os.environ.get("CHAT_API_KEY")
        if not api_key:
            raise ValueError("CHAT_BASE_URL is set, so CHAT_API_KEY must be set too")
        return ChatOpenAI(
            model=model, base_url=base_url, api_key=api_key, temperature=temperature
        )
    return init_chat_model(model, temperature=temperature)
