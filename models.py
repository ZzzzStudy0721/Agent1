"""模型工厂（WBS 1.4）：通过环境变量切换对话后端。

解析顺序：
1. 设置了 CHAT_BASE_URL -> 用 ChatOpenAI 对接任意 OpenAI 兼容厂商
                        （智谱 GLM / Qwen / Kimi / SiliconFlow ...）；
                        凭据放在 CHAT_API_KEY，模型名放在 CHAT_MODEL
2. 否则                -> 由 langchain 的 init_chat_model 把 deepseek-chat 解析成
                        ChatDeepSeek（默认）

示例（.env）：
    # DeepSeek（默认——不需要额外设置）
    CHAT_MODEL=deepseek-chat
    # 智谱 GLM（OpenAI 兼容端点）
    CHAT_MODEL=glm-4.5
    CHAT_BASE_URL=https://open.bigmodel.cn/api/paas/v4
    CHAT_API_KEY=xxxx
"""
import os

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 必须在导入 langchain 之前执行，这样环境变量（.env）才可用
load_dotenv(os.path.join(BASE_DIR, ".env"))

from langchain.chat_models import init_chat_model  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from settings import Settings, get_settings  # noqa: E402

DEFAULT_MODEL = "deepseek-chat"

# 所有 with_structured_output 调用都必须显式带上这个方式。
# DeepSeek 的 OpenAI 兼容端点不支持 OpenAI 的 json_schema 响应格式（默认方式）：
# 收到请求后会照常返回自然语言，pydantic 解析直接失败，调用方只能走兜底分支。
# 2026-09-29 实测（真实 API）：function_calling 正常，json_mode 与默认方式均报
# ValidationError / OutputParserException。
STRUCTURED_METHOD = "function_calling"


def validate_settings(s: Settings) -> str | None:
    """若模型配置不可用，返回一条人可读的错误信息，否则返回 None。

    在启动/实例化阶段就快速失败并给出清晰提示，而不是等到第一次调用 LLM
    时才抛出让人摸不着头脑的鉴权错误。
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
    """实例化由 CHAT_MODEL / CHAT_BASE_URL 选定的对话模型。

    所有后端共用 max_retries + timeout，这样偶发的 API 失败会静默重试，
    而不是把调用方直接弄崩。
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


def get_judge_model(temperature: float = 0.0) -> BaseChatModel:
    """实例化用于 LLM-as-judge 评估的裁判模型。

    独立于生成后端（默认走 DeepSeek），这样打分才能
    保持跨厂商、保持中立。
    """
    s = get_settings()
    if s.judge_base_url:
        if not s.judge_api_key:
            raise ValueError("JUDGE_BASE_URL is set, so JUDGE_API_KEY must be set too")
        return ChatOpenAI(
            model=s.judge_model,
            base_url=s.judge_base_url,
            api_key=s.judge_api_key,
            temperature=temperature,
            max_retries=2,
            timeout=120,
        )
    if s.judge_model.startswith("deepseek") and not s.deepseek_api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set: the default judge needs it (see README section 2)"
        )
    return init_chat_model(
        s.judge_model, temperature=temperature, max_retries=2, timeout=120
    )


def get_callbacks() -> list:
    """若已配置则返回 Langfuse 回调处理器，否则返回空列表。

    设置好 LANGFUSE_PUBLIC_KEY 后，在 LLM 调用处把返回值作为
    `config={"callbacks": models.get_callbacks()}` 传入，即可追踪
    token/延迟（每新建一个 handler 就对应一条 trace）。未配置时零开销：
    只有配置了才会执行 langfuse 这个重量级导入。
    """
    if not get_settings().langfuse_public_key:
        return []
    from langfuse.langchain import CallbackHandler  # noqa: PLC0415

    return [CallbackHandler()]
