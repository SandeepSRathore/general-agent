"""ask_user: lets an executor pause mid-task and ask the user a question.

Uses LangGraph's `interrupt()`, exactly like write_file approvals: the whole graph pauses,
the CLI/API shows the question, and the answer is returned to the executor on resume.
Several parallel executors can ask at once; all questions are answered in one resume.
"""

from __future__ import annotations

from langchain_core.tools import tool
from langgraph.types import interrupt

ANSWER_PREFIX = "The user answered: "


@tool
def ask_user(question: str) -> str:
    """Ask the user a short question when essential information is missing (e.g. their
    city) and is not in the context. Never guess or use placeholders like 'Unknown'."""
    # Nothing with side effects happens before interrupt(), so re-running on resume is safe.
    answer = interrupt({"type": "clarification", "question": question.strip()})
    return f"{ANSWER_PREFIX}{str(answer).strip()}"
