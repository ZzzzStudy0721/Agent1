"""通过 pydantic-settings 做集中配置（WBS 1.4）。

模型层的所有可调项都放在这里；环境变量的值会覆盖 .env 里的值。请调用
get_settings()，不要直接读 os.environ。构造开销很小，
所以有意不做缓存——测试会在多次调用之间修改环境变量。
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 对话后端（解析顺序见 models.py）
    chat_model: str = "deepseek-chat"
    chat_base_url: str = ""  # 任意 OpenAI 兼容端点（GLM / Qwen / Kimi ...）
    chat_api_key: str = ""

    # 默认分支所需的 DeepSeek 凭据
    deepseek_api_key: str = ""

    # 用于 LLM-as-judge 评估的裁判模型（与生成后端分开，
    # 这样打分才能保持中立 / 跨厂商）
    judge_model: str = "deepseek-chat"
    judge_base_url: str = ""
    judge_api_key: str = ""

    # Langfuse 链路追踪（可选；留空 = 关闭）
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = ""


def get_settings() -> Settings:
    """Settings 实例（环境变量 + .env）。"""
    return Settings()
