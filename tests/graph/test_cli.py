"""The CLI turn runner: streaming render + human-in-the-loop resume loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.sqlite import SqliteStore

from general_agent import cli
from general_agent.app import AgentSession
from general_agent.config import Settings
from general_agent.graph.builder import build_graph
from general_agent.graph.deps import AgentDeps
from general_agent.state import Plan, Task
from general_agent.tools import get_tools
from general_agent.tools.files import FileSandbox
from tests.graph.fakes import ScriptedChatModel
from tests.graph.test_graph import default_executor, synth_handler


def test_run_turn_asks_user_and_returns_answer(
    store: SqliteStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plans = [
        Plan(clarification_question="Which city?", tasks=[]),
        Plan(
            clarification_question=None,
            tasks=[Task(id="t1", description="weather in Pune", depends_on=[], suggested_tools=[])],
        ),
    ]
    sandbox = FileSandbox(
        read_roots=[tmp_path], write_root=tmp_path, project_root=tmp_path, max_bytes=100
    )
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    deps = AgentDeps(
        planner_llm=ScriptedChatModel(handler=synth_handler, structured={Plan: plans}),
        executor_llm=ScriptedChatModel(handler=default_executor),
        store=store,
        tools=get_tools(store, "u", sandbox=sandbox),
        settings=settings,
        user_id="u",
    )
    session = AgentSession(build_graph(deps, InMemorySaver()), store, settings)

    asked: list[str] = []

    def fake_prompt(question: str, **_: Any) -> str:
        asked.append(question)
        return "Pune"

    monkeypatch.setattr(cli.Prompt, "ask", fake_prompt)
    answer = cli.run_turn(session, "thread-x", "What's the weather?")

    assert any("Which city?" in q for q in asked)
    assert answer.startswith("FINAL ANSWER")
    assert "done: weather in Pune" in answer
