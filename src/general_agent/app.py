"""Composition root: wires settings, models, persistence, tools and the graph together."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from langgraph.store.sqlite import SqliteStore

from general_agent.config import Settings, get_settings
from general_agent.graph.builder import AgentGraph, build_graph
from general_agent.graph.deps import AgentDeps
from general_agent.llm import get_embeddings, get_executor_llm, get_planner_llm
from general_agent.memory.persistence import open_checkpointer, open_store
from general_agent.tools import get_tools


@dataclass
class AgentSession:
    graph: AgentGraph
    store: SqliteStore
    settings: Settings


@contextmanager
def open_memory_store(settings: Settings | None = None) -> Iterator[SqliteStore]:
    settings = settings or get_settings()
    with open_store(settings.memory_db, get_embeddings(), settings.embedding_dims) as store:
        yield store


@contextmanager
def agent_session(settings: Settings | None = None) -> Iterator[AgentSession]:
    settings = settings or get_settings()
    with (
        open_memory_store(settings) as store,
        open_checkpointer(settings.checkpoints_db) as checkpointer,
    ):
        deps = AgentDeps(
            planner_llm=get_planner_llm(),
            executor_llm=get_executor_llm(),
            store=store,
            tools=get_tools(
                store, settings.user_id, require_write_approval=settings.require_write_approval
            ),
            settings=settings,
            user_id=settings.user_id,
        )
        yield AgentSession(build_graph(deps, checkpointer), store, settings)
