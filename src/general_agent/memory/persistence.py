"""Factories for the two SQLite-backed persistence layers.

- Checkpointer (short-term memory): saves graph state after every step, per `thread_id`.
  Enables multi-turn conversations, resuming after human-in-the-loop interrupts, and
  crash recovery.
- Store (long-term memory): a key/value store with a sqlite-vec vector index, shared
  across threads. Holds memories, past conversation summaries and document chunks, all
  searchable by meaning.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore


@contextmanager
def open_checkpointer(db_path: Path | str) -> Iterator[SqliteSaver]:
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    # allowed_msgpack_modules=None => strict mode: only built-in safe types are
    # deserialized, so a tampered DB file can't trigger arbitrary code execution.
    serde = JsonPlusSerializer(allowed_msgpack_modules=None)
    # check_same_thread=False: parallel graph branches run in worker threads;
    # SqliteSaver serializes access with its own lock.
    with closing(sqlite3.connect(str(db_path), check_same_thread=False)) as conn:
        saver = SqliteSaver(conn, serde=serde)
        saver.setup()
        yield saver


@contextmanager
def open_store(db_path: Path | str, embeddings: Embeddings, dims: int) -> Iterator[SqliteStore]:
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with SqliteStore.from_conn_string(
        str(db_path),
        # Only the "text" field of each item is embedded. NOTE: SqliteStore calls this key
        # `text_fields` (InMemoryStore/PostgresStore call it `fields`); with the wrong key
        # it silently embeds the whole JSON value instead.
        index={"dims": dims, "embed": embeddings, "text_fields": ["text"]},
    ) as store:
        store.setup()
        yield store
