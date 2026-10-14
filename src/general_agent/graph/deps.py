"""Dependencies injected into every node factory.

Passing these explicitly (instead of importing globals inside nodes) lets tests swap in
fake models, an in-memory store and a temp sandbox without monkeypatching.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore

from general_agent.config import Settings
from general_agent.prompts import neutralize
from general_agent.state import TaskResultDict


@dataclass(frozen=True)
class AgentDeps:
    planner_llm: BaseChatModel
    executor_llm: BaseChatModel
    store: BaseStore
    tools: list[BaseTool]
    settings: Settings
    user_id: str


# ---------------------------------------------------------------- shared helpers


def now_str() -> str:
    return datetime.now().astimezone().strftime("%A %d %B %Y, %H:%M %Z")


def format_history(messages: list[AnyMessage], limit: int = 10, max_chars: int = 1500) -> str:
    """Render the last few conversation messages as a compact transcript."""
    lines = []
    for m in messages[-limit:]:
        role = {"human": "User", "ai": "Assistant"}.get(m.type)
        if role and m.text:
            lines.append(f"{role}: {m.text[:max_chars]}")
    return neutralize("\n".join(lines)) or "(no previous messages)"


def format_results(results: list[TaskResultDict]) -> str:
    if not results:
        return "(no tasks were executed)"
    return neutralize(
        "\n\n".join(
            f"### {r['task_id']} [{r['status']}]: {r['description']}\n{r['output']}"
            for r in results
        )
    )
