"""Tool registry: the single place that decides which tools the executors get."""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore

from general_agent.tools.ask_user import ask_user
from general_agent.tools.files import FileSandbox, default_sandbox, make_file_tools
from general_agent.tools.math_tools import calculator
from general_agent.tools.memory_tools import make_memory_tools
from general_agent.tools.search_tools import web_search
from general_agent.tools.time_tools import get_current_time
from general_agent.tools.weather import get_weather


def get_tools(
    store: BaseStore,
    user_id: str,
    *,
    sandbox: FileSandbox | None = None,
    require_write_approval: bool = True,
) -> list[BaseTool]:
    return [
        get_current_time,
        get_weather,
        ask_user,
        calculator,
        web_search,
        *make_file_tools(sandbox or default_sandbox(), require_approval=require_write_approval),
        *make_memory_tools(store, user_id),
    ]


def describe_tools(tools: list[BaseTool]) -> str:
    """Compact tool list for the planner prompt."""
    return "\n".join(f"- {t.name}: {(t.description or '').splitlines()[0]}" for t in tools)


__all__ = ["describe_tools", "get_tools"]
