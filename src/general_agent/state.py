"""Graph state and the structured-output schemas the LLM fills in.

Design note: graph *state* uses plain TypedDicts (not Pydantic models) so checkpoints
serialize with the strict, safe-types-only msgpack serializer. Pydantic models are only
used as LLM output schemas and are converted with `.model_dump()` before entering state.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- state


class TaskDict(TypedDict):
    id: str
    description: str
    depends_on: list[str]
    suggested_tools: list[str]


class TaskResultDict(TypedDict):
    task_id: str
    description: str
    status: Literal["done", "failed"]
    output: str
    tools_used: list[str]


def merge_results(
    left: dict[str, TaskResultDict] | None, right: dict[str, TaskResultDict] | None
) -> dict[str, TaskResultDict]:
    """Reducer for `results`: parallel executors each add their own key.

    Passing `None` resets the results (done at the start of every new user turn,
    because the checkpointer keeps state from previous turns in the same thread).
    """
    if right is None:
        return {}
    return {**(left or {}), **right}


def extend_or_reset(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Reducer for `clarifications`: the clarify node and parallel executors (via ask_user)
    each append their Q&A; `None` resets the list at the start of a new user turn."""
    if right is None:
        return []
    return [*(left or []), *right]


class AgentState(TypedDict, total=False):
    # Conversation history for this thread (short-term memory, persisted by the checkpointer).
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    # Per-turn working fields (reset by load_context on every new user message).
    input: str
    context: str
    clarifications: Annotated[list[str], extend_or_reset]
    plan: list[TaskDict]
    results: Annotated[dict[str, TaskResultDict], merge_results]
    replans: int
    final_answer: str
    pending_question: str


class ExecutorInput(TypedDict):
    """Payload sent (via `Send`) to one executor instance."""

    task: TaskDict
    dependency_results: list[TaskResultDict]
    context: str
    user_id: str


# ------------------------------------------------------------ LLM output schemas
# No field defaults on purpose: OpenAI strict structured outputs require every field to be
# "required", so the model always fills each one explicitly.


class Task(BaseModel):
    id: str = Field(description="Short unique id, e.g. 't1', 't2'.")
    description: str = Field(
        description="Self-contained instruction an executor can complete without seeing the "
        "original request. Include all concrete values (cities, numbers, file paths)."
    )
    depends_on: list[str] = Field(
        description="Ids of EARLIER tasks whose output this task needs. Empty = can run in "
        "parallel immediately.",
    )
    suggested_tools: list[str] = Field(description="Tool names likely needed (may be empty).")


class Plan(BaseModel):
    clarification_question: str | None = Field(
        description="Set ONLY if essential information is missing and cannot be found in the "
        "context or via tools. When set, tasks should be empty.",
    )
    tasks: list[Task] = Field(
        description="Ordered tasks. Empty if the request is small talk or fully answerable from "
        "the conversation/context alone.",
    )


class ReplanDecision(BaseModel):
    reasoning: str = Field(description="One or two sentences explaining the decision.")
    action: Literal["finish", "add_tasks"]
    new_tasks: list[Task] = Field(
        description="Additional tasks when action == 'add_tasks' (else empty). Ids must not "
        "reuse existing ids.",
    )


class MemoryExtraction(BaseModel):
    facts: list[str] = Field(
        description="Durable facts or preferences about the user worth remembering across "
        "sessions, each as a short standalone sentence in third person, e.g. "
        "'The user lives in Pune.' Empty if none.",
    )
