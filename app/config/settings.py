"""Application settings via Pydantic Settings."""

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AI_ARCHITECT_",
        case_sensitive=False,
    )

    # --- LLM Configuration ---
    llm_provider: str = "qwen"
    llm_model: str = "qwen-plus"
    llm_api_key: str = ""
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 16384

    # --- Qwen-specific ---
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY")

    # --- OpenAI-compatible ---
    openai_api_key: str = ""
    openai_base_url: str = ""

    # --- Storage ---
    db_path: str = "data/architect.db"
    output_dir: str = "output"

    @property
    def effective_api_key(self) -> str:
        """Return the API key for the configured provider."""
        if self.llm_provider == "qwen":
            return self.dashscope_api_key or self.llm_api_key
        return self.openai_api_key or self.llm_api_key

    @property
    def effective_base_url(self) -> str | None:
        """Return the base URL for the configured provider, or None for default."""
        if self.llm_provider == "qwen":
            return self.llm_base_url
        if self.openai_base_url:
            return self.openai_base_url
        return None


_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
