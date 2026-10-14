"""Shared fixtures. Everything runs offline: no OpenAI calls, no network."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterator

import pytest
from langchain_core.embeddings import Embeddings
from langgraph.store.sqlite import SqliteStore

from general_agent.memory.persistence import open_store

DIMS = 64
_STOPWORDS = {"the", "a", "an", "is", "in", "of", "to", "and", "i", "user", "what", "do", "my"}


class BagOfWordsEmbeddings(Embeddings):
    """Deterministic toy embedding: hashed bag of (lightly stemmed) words, L2-normalised.

    Texts that share words get a high cosine similarity, which is enough to test semantic
    ranking logic without a real model.
    """

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * DIMS
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word in _STOPWORDS:
                continue
            word = word.removesuffix("s")
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMS] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def store() -> Iterator[SqliteStore]:
    with open_store(":memory:", BagOfWordsEmbeddings(), DIMS) as s:
        yield s
