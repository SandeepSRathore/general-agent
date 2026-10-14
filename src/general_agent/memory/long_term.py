"""Long-term memory API on top of the LangGraph store.

Namespaces:
    ("memories", user_id)       durable facts/preferences about the user
    ("conversations", user_id)  one summary item per completed turn
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime

from langgraph.store.base import BaseStore, SearchItem

MAX_CONVERSATION_CHARS = 2000


def memories_ns(user_id: str) -> tuple[str, ...]:
    return ("memories", user_id)


def conversations_ns(user_id: str) -> tuple[str, ...]:
    return ("conversations", user_id)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def save_memory(
    store: BaseStore, user_id: str, fact: str, *, dedupe_threshold: float = 0.9
) -> bool:
    """Save a fact unless a near-identical one already exists. Returns True if saved."""
    fact = fact.strip()
    if not fact:
        return False
    existing = store.search(memories_ns(user_id), query=fact, limit=1)
    if existing and (existing[0].score or 0) >= dedupe_threshold:
        return False
    key = hashlib.sha256(fact.lower().encode()).hexdigest()[:16]
    store.put(memories_ns(user_id), key, {"text": fact, "created_at": _now()})
    return True


def search_memories(store: BaseStore, user_id: str, query: str, limit: int = 5) -> list[SearchItem]:
    return store.search(memories_ns(user_id), query=query, limit=limit)


def list_memories(store: BaseStore, user_id: str, limit: int = 100) -> list[SearchItem]:
    return store.search(memories_ns(user_id), limit=limit)


def delete_memory(store: BaseStore, user_id: str, key: str) -> None:
    store.delete(memories_ns(user_id), key)


def save_conversation(
    store: BaseStore, user_id: str, thread_id: str, question: str, answer: str
) -> None:
    text = f"User asked: {question}\nAssistant answered: {answer}"[:MAX_CONVERSATION_CHARS]
    store.put(
        conversations_ns(user_id),
        str(uuid.uuid4()),
        {"text": text, "thread_id": thread_id, "created_at": _now()},
    )


def search_conversations(
    store: BaseStore, user_id: str, query: str, limit: int = 3
) -> list[SearchItem]:
    return store.search(conversations_ns(user_id), query=query, limit=limit)


def format_items(items: list[SearchItem], *, with_dates: bool = False) -> str:
    lines = []
    for item in items:
        date = f" ({item.value.get('created_at', '')[:10]})" if with_dates else ""
        lines.append(f"- {item.value['text']}{date}")
    return "\n".join(lines)
