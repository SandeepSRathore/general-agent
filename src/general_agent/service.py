"""UI-agnostic turn runner shared by the CLI and the HTTP API.

`stream_turn` runs one step of a conversation (a new message, or a resume after an
interrupt) and yields plain-dict events. Callers decide how to render them:

    {"type": "plan",        "plan": [...]}         planner produced tasks
    {"type": "replan",      "plan": [...]}         replanner added follow-up tasks
    {"type": "task_result", "result": {...}}       one executor finished
    {"type": "token",       "text": "..."}         a streamed piece of the final answer
    {"type": "answer",      "text": "..."}         the complete final answer
    {"type": "interrupted", "interrupts": [...]}   paused: questions for the user (last event)
    {"type": "done",        "answer": "..."}       turn finished (last event)
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any, cast

from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Interrupt

from general_agent.graph.builder import AgentGraph

Event = dict[str, Any]


def new_thread_id() -> str:
    return uuid.uuid4().hex[:12]


def make_config(thread_id: str, recursion_limit: int) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}


def interrupt_to_dict(intr: Interrupt) -> dict[str, Any]:
    payload = intr.value if isinstance(intr.value, dict) else {"question": str(intr.value)}
    return {"id": intr.id, **payload}


def pending_interrupts(graph: AgentGraph, config: RunnableConfig) -> list[dict[str, Any]]:
    return [interrupt_to_dict(i) for i in graph.get_state(config).interrupts]


def message_input(text: str) -> dict[str, Any]:
    return {"messages": [HumanMessage(text)]}


def resume_input(answers: dict[str, Any]) -> Command[Any]:
    """Answers keyed by interrupt id; one resume can answer several parallel interrupts."""
    return Command(resume=answers)


def stream_turn(graph: AgentGraph, config: RunnableConfig, graph_input: Any) -> Iterator[Event]:
    answer = ""
    for item in graph.stream(graph_input, config, stream_mode=["updates", "messages"]):
        # With several stream modes, each item is a (mode, chunk) tuple.
        mode, chunk = cast(tuple[str, Any], item)
        if mode == "messages":
            message, meta = chunk
            # Only the final answer's tokens; planner/executor internals stay hidden.
            is_answer = meta.get("langgraph_node") == "synthesizer"
            if is_answer and isinstance(message, AIMessageChunk) and message.text:
                yield {"type": "token", "text": message.text}
            continue

        for node, update in chunk.items():
            if not isinstance(update, dict):
                continue
            # A planner update that sets pending_question routes to clarify: no plan yet.
            if node == "planner" and "plan" in update and not update.get("pending_question"):
                yield {"type": "plan", "plan": update["plan"]}
            elif node == "executor":
                for result in update.get("results", {}).values():
                    yield {"type": "task_result", "result": result}
            elif node == "replanner" and "plan" in update:
                yield {"type": "replan", "plan": update["plan"]}
            elif node == "synthesizer":
                answer = update.get("final_answer", "")
                yield {"type": "answer", "text": answer}

    if interrupts := pending_interrupts(graph, config):
        yield {"type": "interrupted", "interrupts": interrupts}
    else:
        yield {"type": "done", "answer": answer}
