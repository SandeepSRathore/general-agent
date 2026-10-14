"""planner + clarify nodes.

planner: decomposes the request into a dependency-aware task list (structured output),
         then routes to: clarify (missing info) | executors (fan-out) | synthesizer.
clarify: human-in-the-loop. `interrupt()` pauses the graph; the CLI asks the user and
         resumes with the answer, and we loop back to the planner.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command, interrupt

from general_agent import prompts
from general_agent.graph.deps import AgentDeps, format_history, now_str
from general_agent.graph.nodes.dispatch import ready_sends, sanitize_tasks
from general_agent.state import AgentState, Plan, format_qa
from general_agent.tools import describe_tools
from general_agent.tools.ask_user import SAFETY_NOTE

logger = logging.getLogger(__name__)

MAX_CLARIFICATIONS = 2


def make_planner(deps: AgentDeps) -> Any:
    structured = deps.planner_llm.with_structured_output(Plan)
    tool_list = describe_tools(deps.tools)

    def planner(state: AgentState) -> Command[Any]:
        clarifications = state.get("clarifications", [])
        system = prompts.PLANNER_SYSTEM.format(
            now=now_str(),
            tools=tool_list,
            context=state.get("context", ""),
            history=format_history(state.get("messages", [])[:-1]),
            clarifications="\n".join(clarifications) or "(none)",
            max_tasks=deps.settings.max_tasks_per_plan,
            untrusted_note=prompts.UNTRUSTED_NOTE,
        )
        plan = cast(
            Plan, structured.invoke([SystemMessage(system), HumanMessage(state.get("input", ""))])
        )
        tasks = sanitize_tasks(plan.tasks, set(), deps.settings.max_tasks_per_plan)
        logger.info("plan: %s", [t["description"] for t in tasks])

        if plan.clarification_question and not tasks:
            if len(clarifications) < MAX_CLARIFICATIONS:
                return Command(
                    update={"pending_question": plan.clarification_question, "plan": []},
                    goto="clarify",
                )
            logger.info("clarification limit reached; answering with what we have")

        update = {"plan": tasks, "pending_question": ""}
        sends = ready_sends(state, tasks)
        return Command(update=update, goto=sends or "synthesizer")

    return planner


def make_clarify(deps: AgentDeps) -> Any:
    def clarify(state: AgentState) -> Command[Any]:
        question = state.get("pending_question", "")
        # interrupt() must come first: on resume the node re-runs from the top and
        # interrupt() then returns the user's answer instead of pausing.
        payload = {"type": "clarification", "question": question, "note": SAFETY_NOTE}
        answer = str(interrupt(payload)).strip()
        return Command(
            update={
                "clarifications": [format_qa(question, answer)],  # appended by reducer
                "messages": [AIMessage(question), HumanMessage(answer)],
                "pending_question": "",
            },
            goto="planner",
        )

    return clarify
