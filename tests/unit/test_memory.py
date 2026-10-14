from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.store.sqlite import SqliteStore

from general_agent.memory import documents, long_term
from general_agent.memory.persistence import open_checkpointer
from general_agent.tools.memory_tools import make_memory_tools

USER = "u1"


def test_save_and_search_memories(store: SqliteStore) -> None:
    assert long_term.save_memory(store, USER, "The user lives in Pune.")
    assert long_term.save_memory(store, USER, "The user prefers green tea.")
    hits = long_term.search_memories(store, USER, "Which city does the user live in?", limit=1)
    assert hits[0].value["text"] == "The user lives in Pune."


def test_memory_dedupe(store: SqliteStore) -> None:
    assert long_term.save_memory(store, USER, "The user lives in Pune.")
    assert not long_term.save_memory(store, USER, "the user lives in pune")
    assert len(long_term.list_memories(store, USER)) == 1


def test_memories_are_isolated_per_user(store: SqliteStore) -> None:
    long_term.save_memory(store, "alice", "Alice likes cats.")
    assert long_term.search_memories(store, "bob", "cats") == []


def test_conversation_search(store: SqliteStore) -> None:
    long_term.save_conversation(store, USER, "t1", "weather in Pune?", "Sunny, 30C")
    long_term.save_conversation(store, USER, "t1", "15% of 2340?", "351")
    hits = long_term.search_conversations(store, USER, "what did we say about the weather", 1)
    assert "Sunny" in hits[0].value["text"]


def test_document_ingest_is_idempotent(store: SqliteStore, tmp_path: Path) -> None:
    (tmp_path / "langgraph.md").write_text("LangGraph checkpointers persist graph state. " * 40)
    (tmp_path / "cooking.txt").write_text("Boil pasta in salted water for ten minutes.")

    first = documents.ingest_directory(store, tmp_path, chunk_size=300, chunk_overlap=0)
    assert sorted(first.indexed) == ["cooking.txt", "langgraph.md"]
    assert first.chunks > 2

    second = documents.ingest_directory(store, tmp_path, chunk_size=300, chunk_overlap=0)
    assert second.indexed == [] and len(second.unchanged) == 2

    hits = documents.search_documents(store, "how do I cook pasta", limit=1)
    assert hits[0].value["source"] == "cooking.txt"

    (tmp_path / "cooking.txt").unlink()
    third = documents.ingest_directory(store, tmp_path, chunk_size=300, chunk_overlap=0)
    assert third.removed == ["cooking.txt"]
    assert all(
        h.value["source"] != "cooking.txt"
        for h in documents.search_documents(store, "pasta", limit=10)
    )


def test_memory_tools(store: SqliteStore, tmp_path: Path) -> None:
    long_term.save_memory(store, USER, "The user lives in Pune.")
    recall, past, docs = make_memory_tools(store, USER)
    assert "Pune" in recall.invoke({"query": "where does the user live"})
    assert "No relevant" in past.invoke({"query": "anything"})
    assert "agent ingest" in docs.invoke({"query": "anything"})


def test_checkpointer_roundtrip_with_strict_serde(tmp_path: Path) -> None:
    """Messages must survive the strict (safe-types-only) checkpoint serializer."""

    def reply(state: MessagesState) -> dict[str, list[AIMessage]]:
        return {"messages": [AIMessage("hi back")]}

    builder = StateGraph(MessagesState)
    builder.add_node("reply", reply)
    builder.add_edge(START, "reply")
    db = tmp_path / "cp.db"
    cfg = {"configurable": {"thread_id": "t"}}
    with open_checkpointer(db) as cp:
        builder.compile(checkpointer=cp).invoke({"messages": [HumanMessage("hi")]}, cfg)
    with open_checkpointer(db) as cp:  # fresh connection = like restarting the app
        state = builder.compile(checkpointer=cp).get_state(cfg)
    assert [m.content for m in state.values["messages"]] == ["hi", "hi back"]


def test_empty_queries_short_circuit(store: SqliteStore) -> None:
    long_term.save_memory(store, USER, "The user lives in Pune.")
    assert long_term.search_memories(store, USER, "  ") == []
    assert long_term.search_conversations(store, USER, "") == []
    assert documents.search_documents(store, "") == []


def test_tool_descriptions_disambiguate_notes_vs_conversations(store: SqliteStore) -> None:
    """Demo regression: nano searched past conversations for 'my notes'. The first line of
    each description (what the planner sees) must say which source it covers."""
    recall, past, docs = make_memory_tools(store, USER)
    first = {
        tool.name: (tool.description or "").splitlines()[0].lower() for tool in (recall, past, docs)
    }
    assert "notes" in first["search_documents"]
    assert "not" in first["search_past_conversations"]  # "...NOT notes/documents"
    assert "not" in first["recall_memories"]
