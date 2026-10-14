"""memory_writer node: background memory formation after every turn.

- Extracts durable user facts from the user's message (+ clarification answers) and
  saves them to long-term memory (semantically de-duplicated).
- Saves a summary of the turn so future sessions can search past conversations.
Failures are logged, never raised: the user already has their answer.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from general_agent import prompts
from general_agent.graph.deps import AgentDeps
from general_agent.memory import long_term
from general_agent.state import AgentState, MemoryExtraction

logger = logging.getLogger(__name__)

MAX_FACTS_PER_TURN = 5


def make_memory_writer(deps: AgentDeps) -> Any:
    structured = deps.executor_llm.with_structured_output(MemoryExtraction)

    def memory_writer(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        user_input = state.get("input", "")
        thread_id = str(config.get("configurable", {}).get("thread_id", ""))
        try:
            known = long_term.search_memories(deps.store, deps.user_id, user_input, limit=10)
            user_text = "\n".join([user_input, *state.get("clarifications", [])])
            extraction = cast(
                MemoryExtraction,
                structured.invoke(
                    [
                        SystemMessage(
                            prompts.MEMORY_EXTRACTION_SYSTEM.format(
                                known=long_term.format_items(known) or "(nothing)"
                            )
                        ),
                        HumanMessage(user_text),
                    ]
                ),
            )
            for fact in extraction.facts[:MAX_FACTS_PER_TURN]:
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
