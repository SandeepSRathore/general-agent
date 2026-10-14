"""Assembles the plan-and-execute graph.

Flow (dynamic routes are chosen by nodes returning `Command(goto=...)`):

    START -> load_context -> planner
    planner     -> clarify            (missing info: interrupt, ask the user)
                -> executor x N       (parallel fan-out via Send, one per ready task)
                -> synthesizer        (no tasks needed)
    clarify     -> planner
    executor    -> replanner          (runs once after each wave of executors)
    replanner   -> executor x N       (next wave: dependent or follow-up tasks)
                -> synthesizer
    synthesizer -> memory_writer -> END
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from general_agent.graph.deps import AgentDeps
from general_agent.graph.nodes.context import make_load_context
from general_agent.graph.nodes.dispatch import EXECUTOR_NODE
from general_agent.graph.nodes.executor import make_executor
from general_agent.graph.nodes.memory_writer import make_memory_writer
from general_agent.graph.nodes.planner import make_clarify, make_planner
from general_agent.graph.nodes.replanner import make_replanner
from general_agent.graph.nodes.synthesizer import make_synthesizer
from general_agent.state import AgentState

# Retries transient failures (network blips, rate limits) on nodes that call an LLM.
LLM_RETRY = RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2.0)


AgentGraph = CompiledStateGraph[AgentState, None, AgentState, AgentState]


def build_graph(
    deps: AgentDeps, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> AgentGraph:
    builder = StateGraph(AgentState)

    builder.add_node("load_context", make_load_context(deps))
    builder.add_node(
        "planner",
        make_planner(deps),
        retry_policy=LLM_RETRY,
        destinations=("clarify", EXECUTOR_NODE, "synthesizer"),
    )
    builder.add_node("clarify", make_clarify(deps), destinations=("planner",))
    builder.add_node(EXECUTOR_NODE, make_executor(deps))
    builder.add_node(
        "replanner",
        make_replanner(deps),
        retry_policy=LLM_RETRY,
        destinations=(EXECUTOR_NODE, "synthesizer"),
    )
    builder.add_node("synthesizer", make_synthesizer(deps), retry_policy=LLM_RETRY)
    builder.add_node("memory_writer", make_memory_writer(deps))

    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "planner")
    builder.add_edge(EXECUTOR_NODE, "replanner")  # runs once per wave, after all executors
    builder.add_edge("synthesizer", "memory_writer")
    builder.add_edge("memory_writer", END)

    return builder.compile(checkpointer=checkpointer, store=deps.store, name="general_agent")
