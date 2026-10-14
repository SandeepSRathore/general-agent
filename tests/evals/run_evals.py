"""Live evaluation suite (needs OPENAI_API_KEY; optional LangSmith upload).

    uv run python -m tests.evals.run_evals            # run locally, print a report
    LANGSMITH_TRACING=true uv run python -m tests.evals.run_evals   # also trace to LangSmith

Each case checks two things:
  1. plan quality  - did the planner create a sensible number of tasks?
  2. answer quality - an LLM-as-judge grades the final answer against the expectation.

It runs against throwaway databases in a temp folder, so your real memory is untouched.
Approval prompts are auto-denied (the write_file case checks the agent handles that).
"""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from general_agent.app import agent_session
from general_agent.config import PROJECT_ROOT, get_settings
from general_agent.llm import get_planner_llm
from general_agent.memory import documents

console = Console()


@dataclass
class Case:
    prompt: str
    expectation: str
    min_tasks: int
    max_tasks: int


CASES = [
    Case(
        "What is 17.5% of 2480? Also what's 2 to the power of 20?", "States 434 and 1048576.", 1, 2
    ),
    Case(
        "What time is it right now in Tokyo and in New York?",
        "Gives current times for both Tokyo and New York.",
        1,
        2,
    ),
    Case(
        "What's the weather in Paris and in Mumbai today?",
        "Reports current weather for both Paris and Mumbai.",
        1,
        2,
    ),
    Case("Hi! How are you?", "A brief friendly greeting; no tool results needed.", 0, 0),
    Case(
        "According to my notes, what does a checkpointer do?",
        "Explains that it saves graph state per thread_id (multi-turn, resume, recovery).",
        1,
        1,
    ),
    Case(
        "List the files in the knowledge folder and tell me how many there are.",
        "Lists agent-concepts.md and says there is 1 file.",
        1,
        2,
    ),
    Case(
        "Calculate 12*12, tell me the time in London, and summarise what ReAct means "
        "according to my notes.",
        "Gives 144, the London time, and a ReAct summary (reason + act tool loop).",
        3,
        3,
    ),
    Case(
        "Write a haiku about autumn to the file haiku.txt",
        "Says the write was not approved/cancelled (approval was denied).",
        1,
        2,
    ),
]


class Grade(BaseModel):
    reasoning: str = Field(description="Brief justification.")
    passed: bool = Field(description="True if the answer meets the expectation.")


JUDGE_PROMPT = """You are grading an AI assistant's answer.
Expectation: {expectation}
Be lenient on wording and exact live values (times, weather), strict on missing parts or
contradictions. The assistant can read the user's notes and live tools, which you cannot
see: do not fail an answer for extra plausible details unless they contradict the
expectation."""


def run_case(session: Any, case: Case) -> tuple[int, str]:
    config: Any = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 60}
    graph_input: Any = {"messages": [HumanMessage(case.prompt)]}
    while True:
        out = session.graph.invoke(graph_input, config)
        pending = out.get("__interrupt__") or ()
        if not pending:
            return len(out.get("plan", [])), out.get("final_answer", "")
        # auto-answer: deny approvals, give a generic clarification
        graph_input = Command(
            resume={
                i.id: "no" if i.value.get("type") == "approval" else "Use your best judgement."
                for i in pending
            }
        )


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    judge = get_planner_llm().with_structured_output(Grade)
    with tempfile.TemporaryDirectory() as tmp:
        settings = get_settings().model_copy(update={"data_dir": Path(tmp)})
        with agent_session(settings) as session:
            documents.ingest_directory(session.store, settings.knowledge_dir)
            table = Table("prompt", "tasks", "plan ok", "answer ok", "judge notes")
            passed = 0
            for case in CASES:
                n_tasks, answer = run_case(session, case)
                plan_ok = case.min_tasks <= n_tasks <= case.max_tasks
                grade: Any = judge.invoke(
                    [
                        SystemMessage(JUDGE_PROMPT.format(expectation=case.expectation)),
                        HumanMessage(f"Question: {case.prompt}\n\nAnswer:\n{answer}"),
                    ]
                )
                passed += plan_ok and grade.passed
                table.add_row(
                    case.prompt[:50],
                    str(n_tasks),
                    "✓" if plan_ok else "✗",
                    "✓" if grade.passed else "✗",
                    grade.reasoning[:80],
                )
    console.print(table)
    console.print(f"[bold]{passed}/{len(CASES)} cases fully passed[/bold]")


if __name__ == "__main__":
    main()
