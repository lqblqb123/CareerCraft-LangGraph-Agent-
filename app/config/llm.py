"""LLM factory — unified interface for Qwen, OpenAI, and Claude models.

Uses LangChain's ChatOpenAI with provider-specific base URLs.
"""

from langchain_openai import ChatOpenAI

from app.config.settings import Settings


def create_llm(settings: Settings | None = None, **kwargs) -> ChatOpenAI:
    """Create a LangChain ChatOpenAI instance configured for the active provider.

    Args:
        settings: Application settings. Uses global settings if not provided.
        **kwargs: Additional kwargs passed to ChatOpenAI (e.g., temperature, max_tokens).

    Returns:
        A configured ChatOpenAI chat model instance.
    """
    if settings is None:
        from app.config.settings import get_settings

        settings = get_settings()

    model = kwargs.pop("model", settings.llm_model)
    temperature = kwargs.pop("temperature", settings.llm_temperature)
    max_tokens = kwargs.pop("max_tokens", settings.llm_max_tokens)

    base_url = settings.effective_base_url
    api_key = settings.effective_api_key

    if not api_key:
        raise ValueError(
            "No API key configured. Set DASHSCOPE_API_KEY or "
            "AI_ARCHITECT_LLM_API_KEY in your environment or .env file."
        )

    init_kwargs: dict = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "request_timeout": 120,  # 单次请求 2 分钟超时
        "max_retries": 2,        # 失败自动重试 2 次
        **kwargs,
    }

    if base_url:
        init_kwargs["base_url"] = base_url

    return ChatOpenAI(**init_kwargs)
