"""HTTP API tests: FastAPI TestClient + scripted fake models (offline, $0)."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.sqlite import SqliteStore

from general_agent.api import create_app
from general_agent.app import AgentSession
from general_agent.config import Settings
from general_agent.graph.builder import build_graph
from general_agent.graph.deps import AgentDeps
from general_agent.memory import long_term
from general_agent.state import Plan, Task
from general_agent.tools import get_tools
from general_agent.tools.files import FileSandbox
from tests.graph.fakes import ScriptedChatModel, tool_call, tool_results
from tests.graph.test_graph import default_executor, synth_handler

USER = "api-user"


def task(id_: str, description: str) -> Task:
    return Task(id=id_, description=description, depends_on=[], suggested_tools=[])


def no_tasks() -> Plan:
    return Plan(clarification_question=None, tasks=[])


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "workspace").mkdir()
    (tmp_path / "knowledge").mkdir()
    return tmp_path


@pytest.fixture
def make_client(store: SqliteStore, workspace: Path) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def factory(
        plans: list[Plan],
        executor: Callable[[list[BaseMessage]], AIMessage] = default_executor,
        api_token: str | None = None,
        **settings_overrides: Any,
    ) -> TestClient:
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            user_id=USER,
            knowledge_dir=workspace / "knowledge",
            **settings_overrides,
        )
        sandbox = FileSandbox(
            read_roots=[workspace / "workspace", workspace / "knowledge"],
            write_root=workspace / "workspace",
            project_root=workspace,
            max_bytes=10_000,
        )
        deps = AgentDeps(
            planner_llm=ScriptedChatModel(handler=synth_handler, structured={Plan: list(plans)}),
            executor_llm=ScriptedChatModel(handler=executor),
            store=store,
            tools=get_tools(store, USER, sandbox=sandbox),
            settings=settings,
            user_id=USER,
        )

        @contextmanager
        def session_factory() -> Iterator[AgentSession]:
            yield AgentSession(build_graph(deps, InMemorySaver()), store, settings)

        client = TestClient(
            create_app(session_factory, api_token=api_token, allowed_hosts=["testserver"])
        )
        client.__enter__()  # run the lifespan (opens the agent session)
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


def sse_events(text: str) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


# --------------------------------------------------------------------------- basics


def test_health_needs_no_token(make_client: Any) -> None:
    client = make_client([], api_token="s3cret")
    assert client.get("/health").json() == {"status": "ok"}


def test_chat_completes_with_plan_results_and_answer(make_client: Any) -> None:
    plan = Plan(
        clarification_question=None,
        tasks=[task("t1", "time in Tokyo"), task("t2", "calc 15% of 2340")],
    )
    client = make_client([plan])

    res = client.post("/chat", json={"message": "time in tokyo and 15% of 2340"})

    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "completed"
    assert len(body["thread_id"]) == 12
    assert [t["id"] for t in body["plan"]] == ["t1", "t2"]
    assert {r["task_id"] for r in body["results"]} == {"t1", "t2"}
    assert body["answer"].startswith("FINAL ANSWER")
    assert "351" in body["answer"]
    assert body["interrupts"] == []


def test_thread_history_and_follow_up_turn(make_client: Any) -> None:
    client = make_client([no_tasks(), no_tasks()])
    first = client.post("/chat", json={"message": "hello"}).json()
    client.post("/chat", json={"message": "again", "thread_id": first["thread_id"]})

    thread = client.get(f"/threads/{first['thread_id']}").json()
    assert [m["role"] for m in thread["messages"]] == ["user", "assistant", "user", "assistant"]
    assert thread["messages"][2]["content"] == "again"
    assert thread["interrupts"] == []


def test_unknown_thread_is_404(make_client: Any) -> None:
    assert make_client([]).get("/threads/nope").status_code == 404


@pytest.mark.parametrize(
    "payload",
    [{"message": ""}, {"message": "hi", "thread_id": "bad id!"}, {"thread_id": "abc"}],
)
def test_request_validation(make_client: Any, payload: dict[str, Any]) -> None:
    assert make_client([]).post("/chat", json=payload).status_code == 422


def test_message_length_limit(make_client: Any) -> None:
    client = make_client([], api_max_message_chars=10)
    assert client.post("/chat", json={"message": "x" * 11}).status_code == 422


# ------------------------------------------------------------- human-in-the-loop


def test_clarification_interrupt_then_resume(make_client: Any) -> None:
    plans = [
        Plan(clarification_question="Which city?", tasks=[]),
        Plan(clarification_question=None, tasks=[task("t1", "weather in Pune")]),
    ]
    client = make_client(plans)

    first = client.post("/chat", json={"message": "What's the weather?"}).json()
    assert first["status"] == "interrupted"
    assert first["answer"] is None
    [pending] = first["interrupts"]
    assert pending["type"] == "clarification" and pending["question"] == "Which city?"

    thread_id = first["thread_id"]
    # a new message while questions are pending is rejected
    busy = client.post("/chat", json={"message": "hi", "thread_id": thread_id})
    assert busy.status_code == 409
    assert busy.json()["detail"]["interrupts"][0]["id"] == pending["id"]
    # answers must cover exactly the pending ids
    wrong = client.post(f"/chat/{thread_id}/resume", json={"answers": {"nope": "Pune"}})
    assert wrong.status_code == 422
    assert wrong.json()["detail"]["expected_ids"] == [pending["id"]]

    done = client.post(f"/chat/{thread_id}/resume", json={"answers": {pending["id"]: "Pune"}})
    body = done.json()
    assert body["status"] == "completed"
    assert "done: weather in Pune" in body["answer"]
    # nothing left to resume
    again = client.post(f"/chat/{thread_id}/resume", json={"answers": {pending["id"]: "x"}})
    assert again.status_code == 409


@pytest.mark.parametrize(("answer", "should_exist"), [(True, True), ("no", False)])
def test_write_approval_over_http(
    make_client: Any, workspace: Path, answer: Any, should_exist: bool
) -> None:
    def executor(messages: list[BaseMessage]) -> AIMessage:
        if results := tool_results(messages):
            return AIMessage(results[-1])
        return tool_call("write_file", {"path": "note.txt", "content": "hello"})

    plan = Plan(clarification_question=None, tasks=[task("t1", "write hello to note.txt")])
    client = make_client([plan], executor=executor)

    first = client.post("/chat", json={"message": "save a note"}).json()
    [pending] = first["interrupts"]
    assert pending["type"] == "approval" and pending["path"] == "workspace/note.txt"
    assert not (workspace / "workspace" / "note.txt").exists()  # nothing before approval

    resumed = client.post(
        f"/chat/{first['thread_id']}/resume", json={"answers": {pending["id"]: answer}}
    ).json()
    assert (workspace / "workspace" / "note.txt").exists() is should_exist
    # live-API regression: /resume returned "plan": [] because the planner doesn't re-run
    assert [t["id"] for t in resumed["plan"]] == ["t1"]


# ------------------------------------------------------------------------ streaming


def test_chat_stream_emits_events_in_order(make_client: Any) -> None:
    plan = Plan(clarification_question=None, tasks=[task("t1", "calc 15% of 2340")])
    client = make_client([plan])

    with client.stream("POST", "/chat/stream", json={"message": "15% of 2340"}) as res:
        assert res.headers["content-type"].startswith("text/event-stream")
        events = sse_events(res.read().decode())

    kinds = [e["type"] for e in events]
    assert kinds[0] == "start" and kinds[-1] == "done"
    assert kinds.index("plan") < kinds.index("task_result") < kinds.index("answer")
    assert events[-1]["answer"].startswith("FINAL ANSWER")


def test_stream_interrupt_and_streamed_resume(make_client: Any) -> None:
    plans = [
        Plan(clarification_question="Which city?", tasks=[]),
        Plan(clarification_question=None, tasks=[task("t1", "weather in Pune")]),
    ]
    client = make_client(plans)

    with client.stream("POST", "/chat/stream", json={"message": "weather?"}) as res:
        events = sse_events(res.read().decode())
    thread_id = events[0]["thread_id"]
    assert events[-1]["type"] == "interrupted"
    intr_id = events[-1]["interrupts"][0]["id"]

    with client.stream(
        "POST", f"/chat/{thread_id}/resume/stream", json={"answers": {intr_id: "Pune"}}
    ) as res:
        events = sse_events(res.read().decode())
    assert events[-1]["type"] == "done"


def test_busy_thread_returns_409(make_client: Any) -> None:
    client = make_client([no_tasks()])
    release = client.app.state.locks.acquire("t-busy")  # type: ignore[attr-defined]
    try:
        res = client.post("/chat", json={"message": "hi", "thread_id": "t-busy"})
        assert res.status_code == 409
    finally:
        release()
    assert client.post("/chat", json={"message": "hi", "thread_id": "t-busy"}).status_code == 200


def test_agent_error_returns_500_without_leaking_details(make_client: Any) -> None:
    client = make_client([])  # no scripted plan -> the fake planner raises
    res = client.post("/chat", json={"message": "hi"})
    assert res.status_code == 500
    assert res.json()["detail"] == "Agent failed: AssertionError"


# ----------------------------------------------------------------------------- auth


def test_token_required_when_configured(make_client: Any) -> None:
    client = make_client([no_tasks()], api_token="s3cret")
    assert client.get("/memories").status_code == 401
    bad = client.get("/memories", headers={"Authorization": "Bearer wrong"})
    assert bad.status_code == 401
    assert bad.headers["www-authenticate"] == "Bearer"
    ok = client.get("/memories", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


# --------------------------------------------------------------------------- memory


def test_list_and_clear_memories(make_client: Any, store: SqliteStore) -> None:
    long_term.save_memory(store, USER, "The user lives in Pune.")
    client = make_client([])
    assert [m["text"] for m in client.get("/memories").json()] == ["The user lives in Pune."]
    assert client.delete("/memories").json() == {"deleted": 1}
    assert client.get("/memories").json() == []


def test_ingest_endpoint(make_client: Any, workspace: Path) -> None:
    (workspace / "knowledge" / "notes.md").write_text("ReAct means reason and act.")
    client = make_client([])
    first = client.post("/ingest").json()
    assert first["indexed"] == ["notes.md"] and first["chunks"] == 1
    assert client.post("/ingest").json()["unchanged"] == ["notes.md"]


# ------------------------------------------------- browser attacks (security review)


def test_dns_rebinding_host_is_rejected(make_client: Any) -> None:
    """evil.com re-resolved to 127.0.0.1 arrives with Host: evil.com -> refuse."""
    client = make_client([])
    res = client.get("/memories", headers={"Host": "evil.com:8000"})
    assert res.status_code == 400
    for host in ("localhost:8000", "127.0.0.1:8000", "[::1]:8000"):
        assert client.get("/memories", headers={"Host": host}).status_code == 200


@pytest.mark.parametrize("origin", ["https://evil.com", "null", "http://127.0.0.1.evil.com"])
def test_cross_origin_browser_requests_are_rejected(make_client: Any, origin: str) -> None:
    client = make_client([no_tasks()])
    res = client.post("/chat", json={"message": "hi"}, headers={"Origin": origin})
    assert res.status_code == 403


def test_same_origin_browser_requests_work(make_client: Any) -> None:
    """The built-in /docs page (same origin) must keep working."""
    client = make_client([no_tasks()])
    res = client.post(
        "/chat",
        json={"message": "hi"},
        headers={"Origin": "http://127.0.0.1:8000", "Host": "127.0.0.1:8000"},
    )
    assert res.status_code == 200


def test_form_encoded_bodies_are_rejected(make_client: Any) -> None:
    """A cross-site HTML form can only send form/text bodies, never application/json."""
    client = make_client([no_tasks()])
    res = client.post("/chat", content=b'{"message": "hi"}', headers={"Content-Type": "text/plain"})
    assert res.status_code == 415
