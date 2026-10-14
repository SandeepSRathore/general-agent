"""Task scheduling helpers shared by the planner and replanner.

`ready_sends` implements dependency-aware parallelism: every task whose dependencies are
all finished is sent to its own executor in the same super-step, so independent tasks
run concurrently and dependent tasks wait for their inputs.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping

from langgraph.types import Send

from general_agent.state import AgentState, ExecutorInput, Task, TaskDict

logger = logging.getLogger(__name__)

EXECUTOR_NODE = "executor"

# Small models often ignore "don't create a combine task" in the prompt, so we enforce it
# in code: a task that only merges other results duplicates the synthesizer's job.
_SYNTHESIS = re.compile(
    r"^\s*(prepare|compile|combine|aggregate|assemble|consolidate|synthesi[sz]e|summari[sz]e"
    r"|present|format|draft|compose|provide|produce|create|give)\b"
    r".*\b(answer|response|reply|results|findings|everything|all of)\b",
    re.IGNORECASE | re.DOTALL,
)
_PRODUCES_ARTIFACT = re.compile(r"\b(file|save|write .* to|\.\w{2,4}\b)", re.IGNORECASE)
# "If X is not found, do Y" with no dependencies can't see whether X was found;
# failures are handled by the replanner instead.
_CONDITIONAL = re.compile(r"^\s*(if|in case|otherwise|should)\b", re.IGNORECASE)


def _is_redundant(task: Task, valid_deps: list[str]) -> str | None:
    """Return why a task should be dropped, or None to keep it."""
    text = task.description
    if len(valid_deps) >= 2 and _SYNTHESIS.search(text) and not _PRODUCES_ARTIFACT.search(text):
        return "only combines other results (the synthesizer does that)"
    if not valid_deps and _CONDITIONAL.search(text):
        return "conditional fallback without inputs (the replanner handles failures)"
    return None


def sanitize_tasks(tasks: list[Task], existing_ids: set[str], max_tasks: int) -> list[TaskDict]:
    """Normalise LLM-produced tasks so the schedule can never deadlock.

    - caps the number of tasks
    - makes ids unique
    - keeps only dependencies on tasks that appear EARLIER (rules out cycles and
      references to unknown ids)
    - drops redundant "combine the results" and dependency-less conditional tasks
    """
    seen = set(existing_ids)
    clean: list[TaskDict] = []
    for task in tasks[:max_tasks]:
        deps = [d for d in dict.fromkeys(task.depends_on) if d in seen]
        if reason := _is_redundant(task, deps):
            logger.info("dropping task %r: %s", task.description, reason)
            continue
        task_id = task.id.strip() or f"t{len(seen) + 1}"
        while task_id in seen:
            task_id = f"{task_id}_"
        clean.append(
            {
                "id": task_id,
                "description": task.description.strip(),
                "depends_on": deps,
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
