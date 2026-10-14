"""replanner node: runs after every executor wave.

1. If tasks are still waiting on dependencies that are now satisfied -> dispatch them
   (no LLM call needed).
2. Otherwise ask the LLM whether the results are sufficient or follow-up tasks are
   needed (bounded by `max_replans` to prevent infinite loops).
"""

from __future__ import annotations

import logging
import re
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from general_agent import prompts
from general_agent.graph.deps import AgentDeps, format_results
from general_agent.graph.nodes.dispatch import pending_tasks, ready_sends, sanitize_tasks
from general_agent.state import AgentState, ReplanDecision, TaskDict

logger = logging.getLogger(__name__)

MAX_NEW_TASKS_PER_REPLAN = 3
# Word-overlap (Jaccard) above which a proposed task counts as a repeat of an existing one.
DUPLICATE_SIMILARITY = 0.6


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def is_duplicate(description: str, plan: list[TaskDict]) -> bool:
    """Small models often re-propose a task that already ran; re-running it would just
    burn tokens and return the same result."""
    new = _words(description)
    for task in plan:
        old = _words(task["description"])
        if new and old and len(new & old) / len(new | old) >= DUPLICATE_SIMILARITY:
            return True
    return False


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
        any_failed = any(r["status"] == "failed" for r in results.values())
        if not any_failed and not deps.settings.replan_on_success:
            return Command(goto="synthesizer")  # saves an LLM call on every successful turn

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
            new = [t for t in new if not is_duplicate(t["description"], plan)]
            if not new:
                logger.info("replanner only proposed duplicate tasks; finishing")
                return Command(goto="synthesizer")
            new_plan = [*plan, *new]
            if sends := ready_sends(state, new_plan):
                return Command(update={"plan": new_plan, "replans": replans + 1}, goto=sends)
        return Command(goto="synthesizer")

    return replanner
