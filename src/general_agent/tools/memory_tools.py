"""Semantic-search tools over long-term memory, past conversations and documents.

The store and user id are injected via a factory (closure) instead of globals, which
keeps the tools easy to test and safe to use for multiple users.

Writing memories is NOT a tool: the memory_writer graph node extracts facts automatically
after every turn ("background memory formation").
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool
from langgraph.store.base import BaseStore

from general_agent.memory import documents, long_term
from general_agent.prompts import wrap_untrusted


def make_memory_tools(store: BaseStore, user_id: str) -> list[BaseTool]:
    @tool
    def recall_memories(query: str) -> str:
        """Search long-term memory for facts and preferences about the user (location,
        name, likes, goals...) by meaning. Example query: 'where does the user live'."""
        items = long_term.search_memories(store, user_id, query, limit=5)
        return (
            wrap_untrusted(long_term.format_items(items))
            if items
            else "No relevant memories found."
        )

    @tool
    def search_past_conversations(query: str) -> str:
        """Semantically search earlier conversations with the user (across sessions).
        Use for questions like 'what did we discuss about X?'."""
        items = long_term.search_conversations(store, user_id, query, limit=4)
        if not items:
            return "No relevant past conversations found."
        return wrap_untrusted(long_term.format_items(items, with_dates=True))

    @tool
    def search_documents(query: str) -> str:
        """Semantic search over the indexed knowledge-base documents (files in knowledge/).
        Returns the most relevant passages with their source file."""
        items = documents.search_documents(store, query, limit=4)
        if not items:
            return "No matching documents. (Has `agent ingest` been run?)"
        blocks = [
            f"[source: {it.value['source']} | score {it.score or 0:.2f}]\n{it.value['text']}"
            for it in items
        ]
        return wrap_untrusted("\n\n".join(blocks))

    return [recall_memories, search_past_conversations, search_documents]
