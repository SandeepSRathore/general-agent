"""Document ingestion for RAG: load files from knowledge/, chunk them and index the chunks
in the store under ("documents",).

Ingestion is idempotent: a per-file content hash is kept under ("documents_meta",), so
unchanged files are skipped and changed files have their old chunks replaced.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.store.base import BaseStore, SearchItem
from pypdf import PdfReader

logger = logging.getLogger(__name__)

DOCS_NS = ("documents",)
META_NS = ("documents_meta",)
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


@dataclass
class IngestReport:
    indexed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    chunks: int = 0


def _load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8", errors="replace")


def _delete_chunks(store: BaseStore, source: str) -> None:
    while chunks := store.search(DOCS_NS, filter={"source": source}, limit=500):
        for item in chunks:
            store.delete(DOCS_NS, item.key)


def ingest_directory(store: BaseStore, directory: Path, *, chunk_size: int = 1000,
                     chunk_overlap: int = 150) -> IngestReport:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    report = IngestReport()
    seen: set[str] = set()

    for path in sorted(p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES):
        source = str(path.relative_to(directory))
        seen.add(source)
        text = _load_text(path)
        digest = hashlib.sha256(text.encode()).hexdigest()
        meta = store.get(META_NS, source)
        if meta and meta.value.get("hash") == digest:
            report.unchanged.append(source)
            continue

        _delete_chunks(store, source)
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            store.put(DOCS_NS, f"{source}#{i}", {"text": chunk, "source": source, "chunk": i})
        store.put(META_NS, source, {"hash": digest, "chunks": len(chunks)}, index=False)
        report.indexed.append(source)
        report.chunks += len(chunks)
        logger.info("indexed %s (%d chunks)", source, len(chunks))

    # Files deleted from disk are removed from the index too.
    for meta_item in store.search(META_NS, limit=10_000):
        if meta_item.key not in seen:
            _delete_chunks(store, meta_item.key)
            store.delete(META_NS, meta_item.key)
            report.removed.append(meta_item.key)
    return report


def search_documents(store: BaseStore, query: str, limit: int = 4) -> list[SearchItem]:
    return store.search(DOCS_NS, query=query, limit=limit)
