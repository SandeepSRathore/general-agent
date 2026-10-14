"""End-to-end graph behaviour with scripted fake models (no network, no API key)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.sqlite import SqliteStore
from langgraph.types import Command

from general_agent.config import Settings
from general_agent.graph.builder import build_graph
from general_agent.graph.deps import AgentDeps
from general_agent.memory import long_term
from general_agent.state import MemoryExtraction, Plan, ReplanDecision, Task
from general_agent.tools import get_tools
from general_agent.tools.files import FileSandbox
from tests.graph.fakes import ScriptedChatModel, first_human_text, tool_call, tool_results

USER = "tester"


def task(id_: str, description: str, depends_on: list[str] | None = None) -> Task:
    return Task(id=id_, description=description, depends_on=depends_on or [], suggested_tools=[])


def synth_handler(messages: list[BaseMessage]) -> AIMessage:
    """Fake synthesizer: echoes the task-results block so tests can assert on it."""
    body = messages[-1].text
    return AIMessage("FINAL ANSWER\n" + body.split("Task results:")[-1])


def default_executor(messages: list[BaseMessage]) -> AIMessage:
    """Fake executor: calls the calculator for 'calc' tasks, otherwise answers directly."""
    task_text = first_human_text(messages)
    if "calc" in task_text:
        if results := tool_results(messages):
            return AIMessage(f"Result is {results[-1]}")
        return tool_call("calculator", {"expression": "0.15 * 2340"})
    return AIMessage("done: " + task_text.split("Task: ")[1].splitlines()[0])


@pytest.fixture
def sandbox(tmp_path: Path) -> FileSandbox:
    (tmp_path / "workspace").mkdir()
    (tmp_path / "knowledge").mkdir()
    return FileSandbox(
        read_roots=[tmp_path / "workspace", tmp_path / "knowledge"],
        write_root=tmp_path / "workspace",
        project_root=tmp_path,
        max_bytes=10_000,
    )


@pytest.fixture
def make_agent(
    store: SqliteStore, sandbox: FileSandbox
) -> Iterator[Callable[..., tuple[CompiledStateGraph, ScriptedChatModel, ScriptedChatModel]]]:
    def factory(
        plans: list[Plan],
        executor: Callable[[list[BaseMessage]], AIMessage] = default_executor,
        replans: list[ReplanDecision] | None = None,
        memories: list[MemoryExtraction] | None = None,
        **settings_overrides: Any,
    ) -> tuple[CompiledStateGraph, ScriptedChatModel, ScriptedChatModel]:
        planner_llm = ScriptedChatModel(
            handler=synth_handler,
            structured={Plan: list(plans), ReplanDecision: list(replans or [])},
        )
        executor_llm = ScriptedChatModel(
            handler=executor, structured={MemoryExtraction: list(memories or [])}
        )
        settings = Settings(_env_file=None, **settings_overrides)  # type: ignore[call-arg]
        deps = AgentDeps(
            planner_llm=planner_llm,
            executor_llm=executor_llm,
            store=store,
            tools=get_tools(store, USER, sandbox=sandbox),
            settings=settings,
            user_id=USER,
        )
        return build_graph(deps, checkpointer=InMemorySaver()), planner_llm, executor_llm

    yield factory


def cfg(thread: str = "t1") -> dict[str, Any]:
    return {"configurable": {"thread_id": thread}, "recursion_limit": 60}


def ask(text: str) -> dict[str, Any]:
    return {"messages": [HumanMessage(text)]}


# ----------------------------------------------------------------------------- tests


def test_parallel_tasks_then_dependent_task(make_agent: Any) -> None:
    plan = Plan(
        clarification_question=None,
        tasks=[
            task("t1", "Get the time in Tokyo"),
            task("t2", "calc 15% of 2340"),
            task("t3", "Explain the Tokyo time", depends_on=["t1"]),
        ],
    )
    graph, planner_llm, _ = make_agent([plan])

    waves: list[set[str]] = []
    for chunk in graph.stream(ask("time in tokyo, 15% of 2340"), cfg(), stream_mode="updates"):
        if "executor" in chunk:
            # One "executor" update per task; group consecutive ones into waves.
            ids = set(chunk["executor"]["results"])
            if waves and waves[-1] != {"__closed__"}:
                waves[-1] |= ids
            else:
                waves.append(ids)
        elif "replanner" in chunk and waves:
            waves.append({"__closed__"})
    waves = [w for w in waves if w != {"__closed__"}]

    assert waves == [{"t1", "t2"}, {"t3"}]  # t1 & t2 parallel; t3 waits for t1
    state = graph.get_state(cfg()).values
    assert all(r["status"] == "done" for r in state["results"].values())
    # executor's own words + the raw tool output as ground truth for the synthesizer
    assert state["results"]["t2"]["output"] == "Result is 351\n\nRaw tool output:\n[calculator] 351"
    assert "Raw tool output" not in state["results"]["t1"]["output"]  # no tools used
    assert state["results"]["t2"]["tools_used"] == ["calculator"]
    assert state["final_answer"].startswith("FINAL ANSWER")
    assert "Result is 351" in state["final_answer"]
    assert planner_llm.structured_calls == ["Plan", "ReplanDecision"]


def test_dependency_results_are_passed_to_dependent_task(make_agent: Any) -> None:
    seen: dict[str, str] = {}

    def executor(messages: list[BaseMessage]) -> AIMessage:
        text = first_human_text(messages)
        if "second" in text:
            seen["second"] = text
        return AIMessage("FIRST-OUTPUT" if "first" in text else "ok")

    plan = Plan(
        clarification_question=None,
        tasks=[task("a", "first step"), task("b", "second step", ["a"])],
    )
    graph, *_ = make_agent([plan], executor=executor)
    graph.invoke(ask("do both"), cfg())
    assert "FIRST-OUTPUT" in seen["second"]


def test_no_tasks_goes_straight_to_synthesizer_and_saves_memory(
    make_agent: Any, store: SqliteStore
) -> None:
    graph, planner_llm, _ = make_agent(
        [Plan(clarification_question=None, tasks=[])],
        memories=[MemoryExtraction(facts=["The user lives in Pune."])],
    )
    out = graph.invoke(ask("Remember that I live in Pune"), cfg())
    assert out["final_answer"].startswith("FINAL ANSWER")
    assert planner_llm.structured_calls == ["Plan"]  # no replanner without tasks
    assert [m.value["text"] for m in long_term.list_memories(store, USER)] == [
        "The user lives in Pune."
    ]
    convs = long_term.search_conversations(store, USER, "Pune", limit=5)
    assert len(convs) == 1 and convs[0].value["thread_id"] == "t1"


def test_memories_are_loaded_into_planner_context(make_agent: Any, store: SqliteStore) -> None:
    long_term.save_memory(store, USER, "The user lives in Pune.")
    graph, *_ = make_agent([Plan(clarification_question=None, tasks=[])])
    out = graph.invoke(ask("what's the weather where I live"), cfg())
    assert "The user lives in Pune." in out["context"]


def test_clarification_interrupt_and_resume(make_agent: Any) -> None:
    plans = [
        Plan(clarification_question="Which city do you mean?", tasks=[]),
        Plan(clarification_question=None, tasks=[task("t1", "weather in Pune")]),
    ]
    graph, *_ = make_agent(plans)

    first = graph.invoke(ask("What's the weather?"), cfg())
    interrupts = first["__interrupt__"]
    assert interrupts[0].value == {"type": "clarification", "question": "Which city do you mean?"}

    final = graph.invoke(Command(resume="Pune"), cfg())
    assert final["clarifications"] == ["Q: Which city do you mean?\nA: Pune"]
    assert "done: weather in Pune" in final["final_answer"]
    texts = [m.text for m in final["messages"]]
    assert texts[:3] == ["What's the weather?", "Which city do you mean?", "Pune"]


def test_clarification_loop_is_bounded(make_agent: Any) -> None:
    ask_again = Plan(clarification_question="More details?", tasks=[])
    graph, *_ = make_agent([ask_again, ask_again, ask_again])
    graph.invoke(ask("do the thing"), cfg())
    graph.invoke(Command(resume="a"), cfg())
    out = graph.invoke(Command(resume="b"), cfg())
    assert "__interrupt__" not in out  # third request for clarification is not honoured
    assert out["final_answer"].startswith("FINAL ANSWER")


@pytest.mark.parametrize(("answer", "should_exist"), [("yes", True), ("no", False)])
def test_write_file_requires_approval(
    make_agent: Any, sandbox: FileSandbox, answer: str, should_exist: bool
) -> None:
    def executor(messages: list[BaseMessage]) -> AIMessage:
        if results := tool_results(messages):
            return AIMessage(results[-1])
        return tool_call("write_file", {"path": "notes.txt", "content": "hello"})

    plan = Plan(clarification_question=None, tasks=[task("t1", "write hello to notes.txt")])
    graph, *_ = make_agent([plan], executor=executor)

    first = graph.invoke(ask("save a note"), cfg())
    payload = first["__interrupt__"][0].value
    assert payload["type"] == "approval" and payload["path"] == "workspace/notes.txt"

    final = graph.invoke(Command(resume=answer), cfg())
    target = sandbox.write_root / "notes.txt"
    assert target.exists() is should_exist
    expected = "Wrote 5 characters" if should_exist else "Write cancelled"
    assert expected in final["results"]["t1"]["output"]


def test_parallel_approvals_resume_together(make_agent: Any, sandbox: FileSandbox) -> None:
    def executor(messages: list[BaseMessage]) -> AIMessage:
        if results := tool_results(messages):
            return AIMessage(results[-1])
        name = "a.txt" if "file a" in first_human_text(messages) else "b.txt"
        return tool_call("write_file", {"path": name, "content": name})

    plan = Plan(
        clarification_question=None, tasks=[task("t1", "write file a"), task("t2", "write file b")]
    )
    graph, *_ = make_agent([plan], executor=executor)
    first = graph.invoke(ask("write two files"), cfg())
    pending = first["__interrupt__"]
    assert len(pending) == 2
    graph.invoke(Command(resume={i.id: "yes" for i in pending}), cfg())
    assert (sandbox.write_root / "a.txt").exists() and (sandbox.write_root / "b.txt").exists()


def test_replanning_is_bounded(make_agent: Any) -> None:
    more = ReplanDecision(
        reasoning="need more", action="add_tasks", new_tasks=[task("x", "another lookup")]
    )
    graph, planner_llm, _ = make_agent(
        [Plan(clarification_question=None, tasks=[task("t1", "lookup")])],
        replans=[more, more, more],
        max_replans=2,
    )
    out = graph.invoke(ask("research"), cfg())
    assert len(out["plan"]) == 3  # original + 2 replans, the 3rd request is ignored
    assert out["replans"] == 2
    assert planner_llm.structured_calls.count("ReplanDecision") == 2


def test_failing_task_is_isolated(make_agent: Any) -> None:
    def executor(messages: list[BaseMessage]) -> AIMessage:
        if "explode" in first_human_text(messages):
            raise RuntimeError("tool backend down")
        return AIMessage("fine")

    plan = Plan(
        clarification_question=None, tasks=[task("t1", "explode please"), task("t2", "be fine")]
    )
    graph, *_ = make_agent([plan], executor=executor)
    out = graph.invoke(ask("two things"), cfg())
    assert out["results"]["t1"]["status"] == "failed"
    assert "tool backend down" in out["results"]["t1"]["output"]
    assert out["results"]["t2"]["status"] == "done"
    assert out["final_answer"].startswith("FINAL ANSWER")


def test_second_turn_resets_working_state_but_keeps_history(make_agent: Any) -> None:
    plans = [
        Plan(clarification_question=None, tasks=[task("t1", "first")]),
        Plan(clarification_question=None, tasks=[task("t9", "second")]),
    ]
    graph, *_ = make_agent(plans)
    graph.invoke(ask("turn one"), cfg())
    out = graph.invoke(ask("turn two"), cfg())
    assert set(out["results"]) == {"t9"}
    assert [m.type for m in out["messages"]] == ["human", "ai", "human", "ai"]
