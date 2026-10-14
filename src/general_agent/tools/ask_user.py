"""ask_user: lets an executor pause mid-task and ask the user a question.

Uses LangGraph's `interrupt()`, exactly like write_file approvals: the whole graph pauses,
the CLI/API shows the question, and the answer is returned to the executor on resume.
Several parallel executors can ask at once; all questions are answered in one resume.
"""

from __future__ import annotations

import re
import unicodedata

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
    r"\b(pass(word|code|phrase)?|passwd|pwd|(api|openai|secret|private|access|auth|"
    r"license)[ _-]?keys?|keys?|secret|tokens?|credentials?|log ?in details|sign[ -]?in"
    r"|one[ -]?time|otp|2fa|mfa|verification code|security code|auth(entication)? code|pin"
    r"|cvv|cvc|credit card|debit card|card number|bank account|account number|iban"
    r"|routing number|ssn|social security|seed phrase|recovery phrase|cookie|session id)\b",
    re.IGNORECASE,
)
# Invisible characters an attacker can insert to split a word ("pass\u200bword").
_INVISIBLE = re.compile("[\u00ad\u200b-\u200f\u2060-\u2064\ufeff]")


def is_safe_question(question: str) -> bool:
    """False if the question asks for secrets. NFKC folds look-alike characters (e.g.
    full-width letters) and invisible characters are removed before matching."""
    normalized = _INVISIBLE.sub("", unicodedata.normalize("NFKC", question))
    return not _SENSITIVE.search(normalized)


@tool
def ask_user(question: str) -> str:
    """Ask the user a short question when essential information is missing (e.g. their
    city) and is not in the context. Never guess or use placeholders like 'Unknown'."""
    question = " ".join(question.split())
    if not is_safe_question(question):
        return (
            "Error: ask_user may not request secrets (passwords, keys, tokens, codes or payment "
            "details). Continue without it or tell the user what you could not do."
        )
    if len(question) > MAX_QUESTION_CHARS:
        return f"Error: question too long (max {MAX_QUESTION_CHARS} characters); ask briefly."
    # Nothing with side effects happens before interrupt(), so re-running on resume is safe.
    answer = interrupt({"type": "clarification", "question": question, "note": SAFETY_NOTE})
    return f"{ANSWER_PREFIX}{str(answer).strip()}"
