"""load_context: entry node of every turn.

Resets the per-turn working state and pulls relevant long-term memories and past
conversations via semantic search, so the planner starts with what we already know.
"""

from __future__ import annotations

import logging
from typing import Any

from general_agent.graph.deps import AgentDeps
from general_agent.memory import long_term
from general_agent.prompts import neutralize
from general_agent.state import AgentState

logger = logging.getLogger(__name__)


def make_load_context(deps: AgentDeps) -> Any:
    def load_context(state: AgentState) -> dict[str, Any]:
        humans = [m for m in state.get("messages", []) if m.type == "human"]
        user_input = humans[-1].text if humans else ""

        sections = []
        try:
            memories = long_term.search_memories(
                deps.store, deps.user_id, user_input, limit=deps.settings.memory_search_limit
            )
            conversations = long_term.search_conversations(
                deps.store, deps.user_id, user_input, limit=3
            )
        except Exception:  # memory is an enhancement; never fail the turn because of it
            logger.exception("memory lookup failed")
            memories, conversations = [], []
        if memories:
            sections.append("Known facts about the user:\n" + long_term.format_items(memories))
        if conversations:
            sections.append(
                "Possibly relevant past conversations:\n"
                + long_term.format_items(conversations, with_dates=True)
            )

        return {
            "input": user_input,
            "user_id": deps.user_id,
            "context": neutralize("\n\n".join(sections)) or "(nothing known yet)",
            # Reset per-turn fields (the checkpointer keeps the previous turn's values).
            "clarifications": None,  # extend_or_reset reducer treats None as "reset"
            "pending_question": "",
            "plan": [],
            "results": None,  # merge_results reducer treats None as "reset"
            "replans": 0,
            "final_answer": "",
        }

    return load_context
