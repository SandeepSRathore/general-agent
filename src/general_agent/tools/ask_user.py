"""ask_user: lets an executor pause mid-task and ask the user a question.

Uses LangGraph's `interrupt()`, exactly like write_file approvals: the whole graph pauses,
the CLI/API shows the question, and the answer is returned to the executor on resume.
Several parallel executors can ask at once; all questions are answered in one resume.
"""

from __future__ import annotations

import re

from langchain_core.tools import tool
from langgraph.types import interrupt

ANSWER_PREFIX = "The user answered: "
MAX_QUESTION_CHARS = 300
# Shown with every agent question. The question text is model-written and may have been
# influenced by injected web/document content, so the user is reminded not to share secrets.
SAFETY_NOTE = "Asked by the AI agent. Never share passwords, API keys, codes or payment details."
# Security: injected content could make the agent ask "please paste your API key". Refuse
# any question that requests credentials or financial details.
_SENSITIVE = re.compile(
    r"\b(pass(word|code|phrase)|api[ _-]?keys?|secret|token|credentials?|log ?in details"
    r"|one[ -]?time|otp|2fa|mfa|verification code|pin|cvv|cvc|credit card|card number"
    r"|bank account|iban|routing number|ssn|social security|private key|seed phrase"
    r"|recovery phrase)\b",
    re.IGNORECASE,
)


@tool
def ask_user(question: str) -> str:
    """Ask the user a short question when essential information is missing (e.g. their
    city) and is not in the context. Never guess or use placeholders like 'Unknown'."""
    question = " ".join(question.split())
    if _SENSITIVE.search(question):
        return (
            "Error: ask_user may not request secrets (passwords, keys, tokens, codes or payment "
            "details). Continue without it or tell the user what you could not do."
        )
    if len(question) > MAX_QUESTION_CHARS:
        return f"Error: question too long (max {MAX_QUESTION_CHARS} characters); ask briefly."
    # Nothing with side effects happens before interrupt(), so re-running on resume is safe.
    answer = interrupt({"type": "clarification", "question": question, "note": SAFETY_NOTE})
    return f"{ANSWER_PREFIX}{str(answer).strip()}"
