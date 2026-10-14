"""Model factories.

`init_chat_model` keeps the code provider-agnostic: swapping to another provider is a
`.env` change (e.g. PLANNER_MODEL=anthropic:claude-sonnet-5-5) plus installing its package.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, cast

from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import OpenAIEmbeddings

from general_agent.config import get_settings


def _api_key_kwargs() -> dict[str, Any]:
    key = get_settings().openai_api_key
    return {"api_key": key.get_secret_value()} if key else {}


def _chat_model(model: str) -> BaseChatModel:
    # max_retries: the SDK retries rate limits / transient 5xx with exponential backoff.
    kwargs: dict[str, Any] = {"max_retries": 3, **_api_key_kwargs()}
    return cast(BaseChatModel, init_chat_model(model, **kwargs))


@lru_cache
def get_planner_llm() -> BaseChatModel:
    """Stronger model used for planning, replanning and final synthesis."""
    return _chat_model(get_settings().planner_model)


@lru_cache
def get_executor_llm() -> BaseChatModel:
    """Cheaper/faster model used by the per-task ReAct executors."""
    return _chat_model(get_settings().executor_model)


@lru_cache
def get_embeddings() -> Embeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        dimensions=settings.embedding_dims,
        **_api_key_kwargs(),
    )
