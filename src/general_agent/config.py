"""Typed application settings loaded from environment variables / `.env`.

Every tunable lives here so nothing is hard-coded across the codebase.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM / embeddings -------------------------------------------------
    openai_api_key: SecretStr | None = None
    # "provider:model" strings understood by langchain's init_chat_model.
    # Defaults favour minimal cost over quality: gpt-5-nano is OpenAI's cheapest model.
    # NOTE: gpt-5-nano is deprecated and shuts down 2026-12-11; switch to
    # "openai:gpt-5.6-luna" (cheapest non-deprecated) before then.
    planner_model: str = "openai:gpt-5-nano"
    executor_model: str = "openai:gpt-5-nano"
    # Reasoning models bill hidden reasoning tokens as output; "minimal" keeps that lowest.
    # Set to empty for models without a reasoning_effort parameter.
    reasoning_effort: str | None = "minimal"
    embedding_model: str = "text-embedding-3-small"
    embedding_dims: int = 1536

    # --- Storage ----------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    workspace_dir: Path = PROJECT_ROOT / "workspace"
    knowledge_dir: Path = PROJECT_ROOT / "knowledge"
    user_id: str = "default-user"

    # --- Guardrails / limits ---------------------------------------------
    max_tasks_per_plan: int = Field(default=8, ge=1)
    max_replans: int = Field(default=2, ge=0)
    executor_max_steps: int = Field(default=8, ge=2)
    recursion_limit: int = Field(default=60, ge=10)
    max_file_read_bytes: int = 200_000
    http_timeout_seconds: float = 10.0
    memory_search_limit: int = 5
    # Skip saving a new memory if an existing one is at least this similar (cosine).
    memory_dedupe_threshold: float = 0.9
    require_write_approval: bool = True

    log_level: str = "WARNING"

    @property
    def checkpoints_db(self) -> Path:
        """Short-term memory: per-thread graph checkpoints."""
        return self.data_dir / "checkpoints.db"

    @property
    def memory_db(self) -> Path:
        """Long-term memory: semantic store (memories, conversations, documents)."""
        return self.data_dir / "memory.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
