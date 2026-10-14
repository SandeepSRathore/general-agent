"""Command-line interface.

agent chat            interactive chat (plan -> parallel tasks -> one answer)
agent ask "..."       one-shot question
agent ingest          index knowledge/ for semantic document search
agent memories        list (or --clear) long-term memories
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from typing import Annotated, Any

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from general_agent import service
from general_agent.app import AgentSession, agent_session, open_memory_store
from general_agent.config import PROJECT_ROOT, get_settings
from general_agent.logging_setup import configure_logging
from general_agent.memory import documents, long_term

app = typer.Typer(add_completion=False, help="Plan-and-execute AI agent (LangGraph).")
console = Console()

HELP = """[bold]Commands[/bold]
  /new        start a new conversation thread (long-term memory is kept)
  /memories   show what the agent remembers about you
  /ingest     (re)index files in knowledge/ for document search
  /thread     show the current thread id
  /help       show this help
  /quit       exit"""


# ------------------------------------------------------------------------- setup


@app.callback()
def _setup() -> None:
    # Export .env to os.environ so LangSmith tracing vars (LANGSMITH_*) take effect.
    load_dotenv(PROJECT_ROOT / ".env")
    configure_logging(get_settings().log_level)


def _require_api_key() -> None:
    if not (get_settings().openai_api_key or os.environ.get("OPENAI_API_KEY")):
        console.print(
            "[red]OPENAI_API_KEY is not set.[/red] Copy .env.example to .env and add your key."
        )
        raise typer.Exit(1)


# --------------------------------------------------------------------- rendering

# C0/C1 control characters (incl. ESC, which starts ANSI terminal sequences), keeping \t \n.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def safe(text: object) -> str:
    """Make LLM/tool-controlled text safe to print: strip terminal control sequences and
    escape Rich markup, so model output can't restyle or spoof the UI (e.g. the approval
    prompt)."""
    return escape(_CONTROL_CHARS.sub("", str(text)))


def _render_plan(plan: list[dict[str, Any]], title: str = "Plan") -> None:
    if not plan:
        console.print("[dim]Plan: no tool tasks needed; answering directly.[/dim]")
        return
    table = Table(title=title, title_justify="left", show_lines=False, expand=False)
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("task")
    table.add_column("after", style="magenta")
    for t in plan:
        table.add_row(
            safe(t["id"]), safe(t["description"]), safe(", ".join(t["depends_on"]) or "-")
        )
    console.print(table)


def _render_result(result: dict[str, Any]) -> None:
    ok = result["status"] == "done"
    icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
    tools = (
        f" [dim](tools: {safe(', '.join(dict.fromkeys(result['tools_used'])))})[/dim]"
        if result["tools_used"]
        else ""
    )
    console.print(f"  {icon} {safe(result['task_id'])}: {safe(result['description'])}{tools}")


def _ask_human(payload: dict[str, Any]) -> Any:
    """Ask the user about one interrupt (see service.interrupt_to_dict for the shape)."""
    if payload.get("type") == "approval":
        # The question is built here from structured fields, never taken verbatim from the
        # payload, and the preview is rendered as plain text, not markup.
        action, path = safe(payload.get("action", "action")), safe(payload.get("path", "?"))
        console.print(
            Panel(
                Text(_CONTROL_CHARS.sub("", str(payload.get("preview", "")))),
                title=f"Preview: {action} → {path}",
                border_style="yellow",
            )
        )
        approved = Confirm.ask(
            f"[bold]Allow {action} to [yellow]{path}[/yellow]?[/bold]", default=False
        )
        return "yes" if approved else "no"
    console.print("[yellow]🤔 The agent needs more information.[/yellow]")
    return Prompt.ask(
        f"[bold yellow]{safe(payload.get('question', 'Please clarify'))}[/bold yellow]"
    )


# ------------------------------------------------------------------------- runner


def run_turn(session: AgentSession, thread_id: str, user_input: str | None) -> str:
    """Run one user turn to completion, handling any number of human-in-the-loop pauses."""
    config = service.make_config(thread_id, session.settings.recursion_limit)
    graph_input: Any = service.message_input(user_input) if user_input else None

    while True:
        answer, interrupts = _render_events(service.stream_turn(session.graph, config, graph_input))
        if not interrupts:
            return answer
        graph_input = service.resume_input({i["id"]: _ask_human(i) for i in interrupts})


def _answer_panel(text: str) -> Panel:
    return Panel(Markdown(_CONTROL_CHARS.sub("", text)), title="Answer", border_style="green")


def _render_events(events: Iterator[service.Event]) -> tuple[str, list[dict[str, Any]]]:
    """Render one streamed turn; returns (answer, pending interrupts)."""
    answer = ""
    interrupts: list[dict[str, Any]] = []
    live: Live | None = None
    status = console.status("[dim]Thinking…[/dim]", spinner="dots")
    status.start()
    try:
        for event in events:
            kind = event["type"]
            if kind == "token":
                if live is None:
                    status.stop()
                    live = Live(console=console, refresh_per_second=12)
                    live.start()
                answer += event["text"]
                live.update(_answer_panel(answer))
            elif kind == "plan":
                status.stop()
                _render_plan(event["plan"])
                status.update("[dim]Running tasks…[/dim]")
                status.start()
            elif kind == "task_result":
                _render_result(event["result"])
            elif kind == "replan":
                status.stop()
                console.print("[magenta]↻ Replanning: adding follow-up tasks[/magenta]")
                _render_plan(event["plan"], title="Updated plan")
                status.start()
            elif kind == "answer":
                status.update("[dim]Saving memories…[/dim]")
                if live is None:  # model did not stream: render the full answer once
                    status.stop()
                    answer = event["text"]
                    console.print(_answer_panel(answer))
                    status.start()
            elif kind == "interrupted":
                interrupts = event["interrupts"]
    finally:
        if live is not None:
            live.stop()
        status.stop()
    return answer, interrupts


# ----------------------------------------------------------------------- commands


@app.command()
def chat(
    thread: Annotated[str | None, typer.Option(help="Resume an existing thread id.")] = None,
) -> None:
    """Interactive chat session."""
    _require_api_key()
    thread_id = thread or service.new_thread_id()
    with agent_session() as session:
        console.print(
            Panel.fit(
                "[bold]General Agent[/bold] — plan → parallel tasks → one answer\n"
                f"[dim]thread {thread_id} · planner {session.settings.planner_model} · "
                f"executor {session.settings.executor_model}[/dim]\nType /help for commands.",
                border_style="blue",
            )
        )
        # Finish any pause left over from a previous session on this thread.
        if session.graph.get_state({"configurable": {"thread_id": thread_id}}).interrupts:
            console.print("[yellow]Resuming an unfinished turn on this thread…[/yellow]")
            run_turn(session, thread_id, None)

        while True:
            try:
                text = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\nBye!")
                return
            if not text:
                continue
            if text.startswith("/"):
                command = text.split()[0].lower()
                if command in ("/quit", "/exit"):
                    console.print("Bye!")
                    return
                if command == "/new":
                    thread_id = service.new_thread_id()
                    console.print(f"[green]New thread {thread_id}[/green]")
                elif command == "/thread":
                    console.print(f"thread id: {thread_id}")
                elif command == "/memories":
                    _print_memories(session)
                elif command == "/ingest":
                    _ingest(session)
                else:
                    console.print(HELP)
                continue
            try:
                run_turn(session, thread_id, text)
            except KeyboardInterrupt:
                console.print("[yellow]Cancelled.[/yellow]")
            except Exception as exc:  # keep the REPL alive; details go to the log
                console.print(f"[red]Error: {exc.__class__.__name__}: {safe(exc)}[/red]")


@app.command()
def ask(
    question: str,
    thread: Annotated[str | None, typer.Option(help="Thread id (default: new).")] = None,
) -> None:
    """Ask a single question and print the answer."""
    _require_api_key()
    with agent_session() as session:
        run_turn(session, thread or service.new_thread_id(), question)


LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port.")] = 8000,
) -> None:
    """Run the HTTP API (FastAPI). Interactive docs at http://HOST:PORT/docs."""
    _require_api_key()
    if host not in LOCAL_HOSTS and not get_settings().api_token:
        console.print(
            f"[red]Refusing to listen on {safe(host)} without API_TOKEN.[/red] Anyone on the "
            "network could use your OpenAI key. Set API_TOKEN in .env or bind to 127.0.0.1."
        )
        raise typer.Exit(1)
    import uvicorn

    console.print(f"Docs: http://{safe(host)}:{port}/docs")
    uvicorn.run("general_agent.api:create_app", factory=True, host=host, port=port)


@app.command()
def ingest() -> None:
    """Index files in knowledge/ (md, txt, pdf) for semantic document search."""
    _require_api_key()
    settings = get_settings()
    with open_memory_store(settings) as store:
        _ingest_into(store, settings.knowledge_dir)


@app.command()
def memories(
    clear: Annotated[bool, typer.Option("--clear", help="Delete all memories.")] = False,
) -> None:
    """List what the agent remembers about you."""
    _require_api_key()
    settings = get_settings()
    with open_memory_store(settings) as store:
        items = long_term.list_memories(store, settings.user_id)
        if clear:
            confirmed = bool(items) and Confirm.ask(f"Delete {len(items)} memories?", default=False)
            if confirmed:
                for item in items:
                    long_term.delete_memory(store, settings.user_id, item.key)
                console.print("[green]Memories cleared.[/green]")
            return
        _print_memory_items(items)


# ------------------------------------------------------------------------ helpers


def _print_memories(session: AgentSession) -> None:
    _print_memory_items(long_term.list_memories(session.store, session.settings.user_id))


def _print_memory_items(items: list[Any]) -> None:
    if not items:
        console.print("[dim]No memories yet. Tell me about yourself![/dim]")
        return
    for item in items:
        created = safe(item.value.get("created_at", "")[:10])
        console.print(f"• {safe(item.value['text'])} [dim]({created})[/dim]")


def _ingest(session: AgentSession) -> None:
    _ingest_into(session.store, session.settings.knowledge_dir)


def _ingest_into(store: Any, directory: Any) -> None:
    with console.status("Indexing documents…"):
        report = documents.ingest_directory(store, directory)
    console.print(
        f"[green]Indexed {len(report.indexed)} file(s), {report.chunks} chunks.[/green] "
        f"Unchanged: {len(report.unchanged)}. Removed: {len(report.removed)}."
    )
    for name in report.indexed:
        console.print(f"  + {safe(name)}")


if __name__ == "__main__":
    app()
