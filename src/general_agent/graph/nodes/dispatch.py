"""Task scheduling helpers shared by the planner and replanner.

`ready_sends` implements dependency-aware parallelism: every task whose dependencies are
all finished is sent to its own executor in the same super-step, so independent tasks
run concurrently and dependent tasks wait for their inputs.
"""

from __future__ import annotations

from collections.abc import Mapping

from langgraph.types import Send

from general_agent.state import AgentState, ExecutorInput, Task, TaskDict

EXECUTOR_NODE = "executor"


def sanitize_tasks(tasks: list[Task], existing_ids: set[str], max_tasks: int) -> list[TaskDict]:
    """Normalise LLM-produced tasks so the schedule can never deadlock.

    - caps the number of tasks
    - makes ids unique
    - keeps only dependencies on tasks that appear EARLIER (rules out cycles and
      references to unknown ids)
    """
    seen = set(existing_ids)
    clean: list[TaskDict] = []
    for task in tasks[:max_tasks]:
        task_id = task.id.strip() or f"t{len(seen) + 1}"
        while task_id in seen:
            task_id = f"{task_id}_"
        clean.append(
            {
                "id": task_id,
                "description": task.description.strip(),
                "depends_on": [d for d in dict.fromkeys(task.depends_on) if d in seen],
                "suggested_tools": list(task.suggested_tools),
            }
        )
        seen.add(task_id)
    return clean


def pending_tasks(plan: list[TaskDict], results: Mapping[str, object]) -> list[TaskDict]:
    return [t for t in plan if t["id"] not in results]


def ready_sends(state: AgentState, plan: list[TaskDict] | None = None) -> list[Send]:
    plan = plan if plan is not None else state.get("plan", [])
    results = state.get("results") or {}
    sends = []
    for task in pending_tasks(plan, results):
        # A failed dependency still counts as finished: the dependent task gets the
        # failure message and can decide what to do, instead of hanging forever.
        if all(dep in results for dep in task["depends_on"]):
            payload: ExecutorInput = {
                "task": task,
                "dependency_results": [results[d] for d in task["depends_on"]],
                "context": state.get("context", ""),
                "user_id": state.get("user_id", ""),
            }
            sends.append(Send(EXECUTOR_NODE, payload))
    return sends
