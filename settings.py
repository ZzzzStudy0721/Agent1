"""Central configuration via pydantic-settings (WBS 1.4).

All model-layer knobs live here; env vars override .env values. Call
get_settings() instead of reading os.environ directly. Construction is cheap,
so it is intentionally not cached — tests mutate env vars between calls.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # chat backend (see models.py for resolution order)
    chat_model: str = "deepseek-chat"
    chat_base_url: str = ""  # any OpenAI-compatible endpoint (GLM / Qwen / Kimi ...)
    chat_api_key: str = ""

    # DeepSeek credential for the default branch
    deepseek_api_key: str = ""

    # Judge model for LLM-as-judge evaluation (kept separate from the
    # generation backend so scoring stays impartial / cross-vendor)
    judge_model: str = "deepseek-chat"
    judge_base_url: str = ""
    judge_api_key: str = ""

    # Langfuse tracing (optional; empty = disabled)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = ""


def get_settings() -> Settings:
    """Settings instance (env vars + .env)."""
    return Settings()
