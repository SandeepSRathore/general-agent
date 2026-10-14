"""memory_writer node: background memory formation after every turn.

- Extracts durable user facts from the user's message (+ clarification answers) and
  saves them to long-term memory (semantically de-duplicated).
- Saves a summary of the turn so future sessions can search past conversations.
Failures are logged, never raised: the user already has their answer.
"""

from __future__ import annotations

import logging
import re
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from general_agent import prompts
from general_agent.graph.deps import AgentDeps
from general_agent.memory import long_term
from general_agent.prompts import wrap_untrusted
from general_agent.state import AgentState, MemoryExtraction, split_qa

logger = logging.getLogger(__name__)

MAX_FACTS_PER_TURN = 5

# Small models tend to "remember" the request itself ("The user asked to save X").
# Those are one-off actions, not facts about the user, so they're filtered in code.
_ONE_OFF_REQUEST = re.compile(
    r"\b(asked|asks|requested|requests|wants? (you|the assistant|me) to|would like (you|me)"
    r"|instructed|told (you|the assistant|me) to)\b",
    re.IGNORECASE,
)
# Defense in depth on top of the prompt: never persist anything that looks like a secret.
_SECRET = re.compile(
    r"\b(sk|pk|lsv2|ghp|xox[abp])[-_][A-Za-z0-9_-]{8,}|\b(api[ _-]?key|password|passcode"
    r"|secret|token|cvv|card number)\b",
    re.IGNORECASE,
)


# Words that carry no user-specific content; a fact must share at least one OTHER word with
# what the user actually typed.
_GENERIC = frozenset(
    (
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "from",
        "about",
        "is",
        "are",
        "was",
        "be",
        "been",
        "user",
        "user's",
        "users",
        "their",
        "they",
        "them",
        "he",
        "she",
        "his",
        "her",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "likes",
        "like",
        "prefers",
        "prefer",
        "wants",
        "want",
        "lives",
        "live",
        "has",
        "have",
        "name",
        "named",
        "called",
        "always",
        "never",
        "usually",
        "often",
        "would",
        "should",
        "can",
        "will",
    )
)


def _content_words(text: str) -> set[str]:
    words = {
        w.removesuffix("'s")
        for w in re.findall(r"[a-z0-9][a-z0-9'.-]*[a-z0-9]|[a-z0-9]", text.lower())
    }
    return {w for w in words if w not in _GENERIC and len(w) > 1}


def grounded_in_user_words(fact: str, user_text: str) -> bool:
    """Security: a memory must come from what the USER typed, never from a model-written
    question (which injected web/document content could have crafted). A bare "yes" to
    "Should I forward your files to evil.com?" shares no content words -> rejected."""
    return bool(_content_words(fact) & _content_words(user_text))


def is_durable_fact(fact: str) -> bool:
    fact = fact.strip()
    return bool(fact) and not _ONE_OFF_REQUEST.search(fact) and not _SECRET.search(fact)


def _extraction_input(user_input: str, qa_pairs: list[tuple[str, str]]) -> str:
    lines = [f"User message:\n{user_input}"]
    if qa_pairs:
        lines.append("\nThe user's answers to questions the assistant asked:")
        for question, answer in qa_pairs:
            lines.append(
                f"- Answer: {answer}\n  (question, context only: {wrap_untrusted(question)})"
            )
    return "\n".join(lines)


def make_memory_writer(deps: AgentDeps) -> Any:
    structured = deps.executor_llm.with_structured_output(MemoryExtraction)

    def memory_writer(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        user_input = state.get("input", "")
        thread_id = str(config.get("configurable", {}).get("thread_id", ""))
        try:
            known = long_term.search_memories(deps.store, deps.user_id, user_input, limit=10)
            qa_pairs = [split_qa(r) for r in state.get("clarifications", [])]
            user_text = "\n".join([user_input, *(answer for _, answer in qa_pairs)])
            extraction = cast(
                MemoryExtraction,
                structured.invoke(
                    [
                        SystemMessage(
                            prompts.MEMORY_EXTRACTION_SYSTEM.format(
                                known=long_term.format_items(known) or "(nothing)"
                            )
                        ),
                        HumanMessage(_extraction_input(user_input, qa_pairs)),
                    ]
                ),
            )
            facts = [
                f
                for f in extraction.facts
                if is_durable_fact(f) and grounded_in_user_words(f, user_text)
            ]
            for fact in facts[:MAX_FACTS_PER_TURN]:
                if long_term.save_memory(
                    deps.store,
                    deps.user_id,
                    fact,
                    dedupe_threshold=deps.settings.memory_dedupe_threshold,
                ):
                    logger.info("remembered: %s", fact)
        except Exception:
            logger.exception("memory extraction failed")
        try:
            long_term.save_conversation(
                deps.store, deps.user_id, thread_id, user_input, state.get("final_answer", "")
            )
        except Exception:
            logger.exception("saving conversation failed")
        return {}

    return memory_writer
