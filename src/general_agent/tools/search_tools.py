"""Web search via DuckDuckGo (no API key)."""

from __future__ import annotations

import logging

from ddgs import DDGS
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

MAX_SNIPPET_CHARS = 400


def search_web(query: str, max_results: int = 5) -> str:
    results = DDGS(timeout=10).text(query, max_results=max_results)
    if not results:
        return "No results found."
    blocks = [
        f"[{i}] {r.get('title', '')}\nURL: {r.get('href', '')}\n"
        f"{(r.get('body') or '')[:MAX_SNIPPET_CHARS]}"
        for i, r in enumerate(results, 1)
    ]
    return "<untrusted_data>\n" + "\n\n".join(blocks) + "\n</untrusted_data>"


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information (news, facts, prices, people, events).
    Returns titles, URLs and snippets. Cite URLs in your answer."""
    try:
        return search_web(query, max(1, min(max_results, 10)))
    except Exception as exc:  # ddgs raises several exception types (rate limits, timeouts)
        logger.warning("web search failed: %s", exc)
        return f"Error: web search failed ({exc.__class__.__name__}). Try rephrasing."
