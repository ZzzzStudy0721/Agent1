"""Model factory (WBS 1.4): one env var switches the chat backend.

    CHAT_MODEL=deepseek-chat             -> ChatDeepSeek  (default: cost, CN network)
    CHAT_MODEL=claude-haiku-4-5-20251001 -> ChatAnthropic

Built on langchain's init_chat_model, so every node in app/agent talks to
the same ChatModel interface and never touches a vendor SDK directly.
"""
import os

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Must run before langchain imports so env vars (.env) are available
load_dotenv(os.path.join(BASE_DIR, ".env"))

from langchain.chat_models import init_chat_model  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402

DEFAULT_MODEL = "deepseek-chat"


def get_chat_model(temperature: float = 0.1) -> BaseChatModel:
    """Instantiate the chat model selected by the CHAT_MODEL env var."""
    model = os.environ.get("CHAT_MODEL", DEFAULT_MODEL)
    return init_chat_model(model, temperature=temperature)
