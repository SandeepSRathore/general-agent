"""executor node: runs ONE task with a ReAct tool-calling sub-agent.

Each task gets its own fresh sub-agent conversation, so tasks running in parallel never
see each other's scratch work. Only the final result is written back to the parent graph.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphBubbleUp, GraphRecursionError

from general_agent import prompts
from general_agent.graph.deps import AgentDeps, format_results, now_str
from general_agent.state import ExecutorInput, TaskResultDict

logger = logging.getLogger(__name__)


def make_executor(deps: AgentDeps) -> Any:
    agent = create_agent(
        model=deps.executor_llm,
        tools=deps.tools,
        system_prompt=prompts.EXECUTOR_SYSTEM.format(untrusted_note=prompts.UNTRUSTED_NOTE),
        name="task_executor",
    )
    # Each ReAct step is ~2 graph steps (model -> tools).
    recursion_limit = 2 * deps.settings.executor_max_steps + 1

    def executor(payload: ExecutorInput) -> dict[str, Any]:
        task = payload["task"]
        message = prompts.EXECUTOR_TASK.format(
            now=now_str(),
            description=task["description"],
            context=payload["context"],
            dependency_results=format_results(payload["dependency_results"])
            if payload["dependency_results"]
            else "(none)",
        )
        result: TaskResultDict = {
            "task_id": task["id"],
            "description": task["description"],
            "status": "done",
            "output": "",
            "tools_used": [],
        }
        try:
            out = agent.invoke(
                {"messages": [HumanMessage(message)]}, {"recursion_limit": recursion_limit}
            )
            messages = out["messages"]
            result["output"] = messages[-1].text or "(empty result)"
            result["tools_used"] = [
                call["name"] for m in messages if isinstance(m, AIMessage) for call in m.tool_calls
            ]
        except GraphBubbleUp:
            raise  # interrupts (e.g. write_file approval) must propagate to the parent graph
        except GraphRecursionError:
            result["status"] = "failed"
            result["output"] = "Stopped: the task needed more tool steps than allowed."
        except Exception as exc:  # isolate failures: one broken task must not sink the plan
            # Full traceback only in DEBUG; the failure itself is reported to the user.
            logger.warning(
                "task %s failed: %s", task["id"], exc, exc_info=logger.isEnabledFor(logging.DEBUG)
            )
            result["status"] = "failed"
            result["output"] = f"Task failed with {exc.__class__.__name__}: {exc}"
        return {"results": {task["id"]: result}}

    return executor
