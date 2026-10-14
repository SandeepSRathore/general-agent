"""replanner node: runs after every executor wave.

1. If tasks are still waiting on dependencies that are now satisfied -> dispatch them
   (no LLM call needed).
2. Otherwise ask the LLM whether the results are sufficient or follow-up tasks are
   needed (bounded by `max_replans` to prevent infinite loops).
"""

from __future__ import annotations

import logging
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from general_agent import prompts
from general_agent.graph.deps import AgentDeps, format_results
from general_agent.graph.nodes.dispatch import pending_tasks, ready_sends, sanitize_tasks
from general_agent.state import AgentState, ReplanDecision

logger = logging.getLogger(__name__)

MAX_NEW_TASKS_PER_REPLAN = 3


def make_replanner(deps: AgentDeps) -> Any:
    structured = deps.planner_llm.with_structured_output(ReplanDecision)

    def replanner(state: AgentState) -> Command[Any]:
        plan = state.get("plan", [])
        results = state.get("results") or {}

        if sends := ready_sends(state):
            return Command(goto=sends)
        if pending_tasks(plan, results):  # unreachable after sanitize_tasks, but be safe
            logger.warning("unschedulable tasks left; synthesizing what we have")
            return Command(goto="synthesizer")

        replans = state.get("replans", 0)
        if replans >= deps.settings.max_replans:
            return Command(goto="synthesizer")

        ordered = [results[t["id"]] for t in plan]
        decision = cast(
            ReplanDecision,
            structured.invoke(
                [
                    SystemMessage(
                        prompts.REPLANNER_SYSTEM.format(untrusted_note=prompts.UNTRUSTED_NOTE)
                    ),
                    HumanMessage(
                        prompts.REPLANNER_INPUT.format(
                            input=state.get("input", ""), results=format_results(ordered)
                        )
                    ),
                ]
            ),
        )
        logger.info("replanner: %s (%s)", decision.action, decision.reasoning)

        if decision.action == "add_tasks" and decision.new_tasks:
            new = sanitize_tasks(
                decision.new_tasks, {t["id"] for t in plan}, MAX_NEW_TASKS_PER_REPLAN
            )
            new_plan = [*plan, *new]
            if sends := ready_sends(state, new_plan):
                return Command(update={"plan": new_plan, "replans": replans + 1}, goto=sends)
        return Command(goto="synthesizer")

    return replanner
