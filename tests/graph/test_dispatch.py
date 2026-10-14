from general_agent.graph.nodes.dispatch import ready_sends, sanitize_tasks
from general_agent.state import AgentState, Task


def t(id_: str, deps: list[str] | None = None) -> Task:
    return Task(id=id_, description=f"do {id_}", depends_on=deps or [], suggested_tools=[])


def test_sanitize_drops_forward_unknown_and_self_dependencies() -> None:
    clean = sanitize_tasks([t("a", ["b"]), t("b", ["a", "zzz", "b"])], set(), 10)
    assert clean[0]["depends_on"] == []  # 'b' comes later -> dropped (prevents cycles)
    assert clean[1]["depends_on"] == ["a"]


def test_sanitize_dedupes_ids_and_caps_count() -> None:
    clean = sanitize_tasks([t("a"), t("a"), t("c")], {"c"}, 2)
    assert [c["id"] for c in clean] == ["a", "a_"]


def test_ready_sends_respects_dependencies() -> None:
    plan = sanitize_tasks([t("a"), t("b"), t("c", ["a", "b"])], set(), 10)
    state: AgentState = {"plan": plan, "results": {}, "context": "ctx", "user_id": "u"}
    assert [s.arg["task"]["id"] for s in ready_sends(state)] == ["a", "b"]

    done = {"task_id": "a", "description": "", "status": "done", "output": "A", "tools_used": []}
    failed = {**done, "task_id": "b", "status": "failed"}
    state["results"] = {"a": done, "b": failed}  # type: ignore[typeddict-item]
    sends = ready_sends(state)
    assert [s.arg["task"]["id"] for s in sends] == ["c"]
    assert [r["task_id"] for r in sends[0].arg["dependency_results"]] == ["a", "b"]


def test_sanitize_drops_synthesis_task() -> None:
    """Demo regression: nano added a 'prepare the answer' task; the synthesizer does that."""
    tasks = [
        t("t1"),
        t("t2"),
        Task(
            id="t3",
            description="Prepare a concise answer combining the results above",
            depends_on=["t1", "t2"],
            suggested_tools=[],
        ),
    ]
    assert [c["id"] for c in sanitize_tasks(tasks, set(), 10)] == ["t1", "t2"]


def test_sanitize_keeps_real_tasks_that_look_similar() -> None:
    tasks = [
        t("t1"),
        t("t2"),
        # depends on 2 tasks but saves a file -> real work, keep it
        Task(
            id="t3",
            description="Write a summary of the results to report.txt",
            depends_on=["t1", "t2"],
            suggested_tools=["write_file"],
        ),
        # summarizing something specific with no dependencies -> keep
        Task(
            id="t4",
            description="Summarize what the notes say about ReAct",
            depends_on=[],
            suggested_tools=[],
        ),
    ]
    assert [c["id"] for c in sanitize_tasks(tasks, set(), 10)] == ["t1", "t2", "t3", "t4"]


def test_sanitize_drops_conditional_fallback_without_dependencies() -> None:
    """Demo regression: 'If notes are not found, ...' with no deps can't check its condition."""
    tasks = [
        t("t1"),
        Task(
            id="t2",
            description="If notes are not found, use search_documents instead",
            depends_on=[],
            suggested_tools=[],
        ),
    ]
    assert [c["id"] for c in sanitize_tasks(tasks, set(), 10)] == ["t1"]
