"""HTTP API for the agent (FastAPI).

    uv run agent serve            # http://127.0.0.1:8000, interactive docs at /docs

Human-in-the-loop over HTTP: when the agent needs input, a turn returns
`status: "interrupted"` with the pending questions; answer them with
`POST /chat/{thread_id}/resume`. Nothing is written to disk without an explicit "yes".

Endpoints are plain `def` (not `async def`) on purpose: the graph and SQLite are
synchronous, so FastAPI runs each request in its worker thread pool.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractContextManager, asynccontextmanager
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Path, Request, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from general_agent import service
from general_agent.app import AgentSession, agent_session
from general_agent.config import get_settings
from general_agent.memory import documents, long_term

logger = logging.getLogger(__name__)

SessionFactory = Callable[[], AbstractContextManager[AgentSession]]
THREAD_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
ThreadId = Annotated[str, Path(pattern=THREAD_ID_PATTERN)]

# ------------------------------------------------------------------------- schemas


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="The user's message.")
    thread_id: str | None = Field(
        default=None,
        pattern=THREAD_ID_PATTERN,
        description="Continue this conversation; omit to start a new one.",
    )


class ResumeRequest(BaseModel):
    answers: dict[str, str | bool] = Field(
        min_length=1,
        description="Answer for every pending interrupt, keyed by interrupt id. "
        'Clarifications take text; approvals take true/false or "yes"/"no".',
    )


class TurnResponse(BaseModel):
    thread_id: str
    status: Literal["completed", "interrupted"]
    answer: str | None = Field(description="Final answer (null while interrupted).")
    plan: list[dict[str, Any]] = Field(description="Tasks the planner created this turn.")
    results: list[dict[str, Any]] = Field(description="Executor results, in completion order.")
    interrupts: list[dict[str, Any]] = Field(description="Pending questions for the user.")


class MessageOut(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ThreadResponse(BaseModel):
    thread_id: str
    messages: list[MessageOut]
    interrupts: list[dict[str, Any]]


class MemoryOut(BaseModel):
    key: str
    text: str
    created_at: str | None


class IngestResponse(BaseModel):
    indexed: list[str]
    unchanged: list[str]
    removed: list[str]
    chunks: int


# ---------------------------------------------------------------- per-thread lock


class ThreadLocks:
    """One turn at a time per conversation thread (different threads run concurrently).

    A second request for a busy thread gets 409 instead of racing on its checkpoint.
    """

    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def acquire(self, thread_id: str) -> Callable[[], None]:
        with self._guard:
            lock = self._locks.setdefault(thread_id, threading.Lock())
        if not lock.acquire(blocking=False):
            raise HTTPException(
                status.HTTP_409_CONFLICT, "This thread is already processing a request."
            )
        return lock.release


# ----------------------------------------------------------- browser-attack guard

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _hostname(value: str) -> str | None:
    """'127.0.0.1:8000' / '[::1]:8000' / 'http://evil.com' -> bare lowercase host name."""
    try:
        return urlsplit(value if "//" in value else f"//{value}").hostname
    except ValueError:
        return None


class BrowserAttackGuard:
    """Protects a local API from malicious web pages open in the user's browser.

    - DNS rebinding: evil.com re-resolves to 127.0.0.1, so the browser treats our API as
      same-origin and lets the page read responses. Rejecting unknown Host headers stops it.
    - CSRF: a page can't read cross-site responses but can still *send* requests (and
      spend API credit). Browsers attach an Origin header to those; foreign origins are
      rejected, and request bodies must be JSON (which cross-site forms can't send).
    Non-browser clients (curl, scripts) send no Origin and are unaffected.
    """

    def __init__(self, app: ASGIApp, allowed_hosts: frozenset[str]) -> None:
        self.app = app
        self.allowed_hosts = allowed_hosts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        problem: tuple[int, str] | None = None
        if _hostname(headers.get("host", "")) not in self.allowed_hosts:
            problem = (400, "Invalid Host header.")
        elif (origin := headers.get("origin")) is not None and (
            _hostname(origin) not in self.allowed_hosts
        ):
            problem = (403, "Cross-origin requests are not allowed.")
        elif (
            scope["method"] in ("POST", "PUT", "PATCH")
            and int(headers.get("content-length") or 0) > 0
            and not headers.get("content-type", "").startswith("application/json")
        ):
            problem = (415, "Request body must be application/json.")
        if problem:
            await JSONResponse({"detail": problem[1]}, status_code=problem[0])(scope, receive, send)
            return
        await self.app(scope, receive, send)


# -------------------------------------------------------------------- dependencies

_bearer = HTTPBearer(auto_error=False)


def require_token(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    expected: str | None = request.app.state.api_token
    if expected is None:
        return
    given = credentials.credentials if credentials else ""
    # constant-time comparison: no timing side channel on the token
    if not secrets.compare_digest(given.encode(), expected.encode()):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Missing or invalid bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_session(request: Request) -> AgentSession:
    session: AgentSession = request.app.state.session
    return session


def get_locks(request: Request) -> ThreadLocks:
    locks: ThreadLocks = request.app.state.locks
    return locks


Session = Annotated[AgentSession, Depends(get_session)]
Locks = Annotated[ThreadLocks, Depends(get_locks)]

# ------------------------------------------------------------------------ helpers


def _config(session: AgentSession, thread_id: str) -> Any:
    return service.make_config(thread_id, session.settings.recursion_limit)


def _check_message(session: AgentSession, message: str) -> None:
    limit = session.settings.api_max_message_chars
    if len(message) > limit:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"Message too long (max {limit} chars)."
        )


def _start_input(session: AgentSession, body: ChatRequest) -> tuple[str, Any]:
    _check_message(session, body.message)
    thread_id = body.thread_id or service.new_thread_id()
    if pending := service.pending_interrupts(session.graph, _config(session, thread_id)):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            {"message": "This thread is waiting for answers; use /resume.", "interrupts": pending},
        )
    return thread_id, service.message_input(body.message)


def _resume_input(session: AgentSession, thread_id: str, body: ResumeRequest) -> Any:
    pending = service.pending_interrupts(session.graph, _config(session, thread_id))
    if not pending:
        raise HTTPException(status.HTTP_409_CONFLICT, "Nothing to resume on this thread.")
    expected = {i["id"] for i in pending}
    if set(body.answers) != expected:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            {"message": "Answer every pending interrupt by id.", "expected_ids": sorted(expected)},
        )
    return service.resume_input(dict(body.answers))


def _collect(session: AgentSession, thread_id: str, graph_input: Any) -> TurnResponse:
    plan: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    answer: str | None = None
    interrupts: list[dict[str, Any]] = []
    try:
        for event in service.stream_turn(session.graph, _config(session, thread_id), graph_input):
            kind = event["type"]
            if kind in ("plan", "replan"):
                plan = event["plan"]
            elif kind == "task_result":
                results.append(event["result"])
            elif kind == "done":
                answer = event["answer"]
            elif kind == "interrupted":
                interrupts = event["interrupts"]
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("turn failed on thread %s", thread_id)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Agent failed: {exc.__class__.__name__}"
        ) from exc
    if not plan:
        # On /resume the planner doesn't run again, so report the plan from saved state.
        plan = list(session.graph.get_state(_config(session, thread_id)).values.get("plan", []))
    return TurnResponse(
        thread_id=thread_id,
        status="interrupted" if interrupts else "completed",
        answer=None if interrupts else answer,
        plan=plan,
        results=results,
        interrupts=interrupts,
    )


def _sse(event: dict[str, Any]) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


def _stream_response(
    session: AgentSession, thread_id: str, graph_input: Any, release: Callable[[], None]
) -> StreamingResponse:
    def events() -> Iterator[str]:
        try:
            yield _sse({"type": "start", "thread_id": thread_id})
            config = _config(session, thread_id)
            for event in service.stream_turn(session.graph, config, graph_input):
                yield _sse(event)
        except Exception as exc:
            logger.exception("streamed turn failed on thread %s", thread_id)
            yield _sse({"type": "error", "message": f"Agent failed: {exc.__class__.__name__}"})
        finally:
            release()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------------- routes

router = APIRouter(dependencies=[Depends(require_token)])


@router.post("/chat", response_model=TurnResponse, tags=["chat"])
def chat(body: ChatRequest, session: Session, locks: Locks) -> TurnResponse:
    """Send a message and wait for the full turn (plan -> parallel tasks -> answer)."""
    thread_id = body.thread_id or service.new_thread_id()
    release = locks.acquire(thread_id)
    try:
        _, graph_input = _start_input(session, body.model_copy(update={"thread_id": thread_id}))
        return _collect(session, thread_id, graph_input)
    finally:
        release()


@router.post("/chat/{thread_id}/resume", response_model=TurnResponse, tags=["chat"])
def resume(
    thread_id: ThreadId, body: ResumeRequest, session: Session, locks: Locks
) -> TurnResponse:
    """Answer the pending clarification/approval questions and continue the turn."""
    release = locks.acquire(thread_id)
    try:
        return _collect(session, thread_id, _resume_input(session, thread_id, body))
    finally:
        release()


@router.post("/chat/stream", tags=["chat"], response_class=StreamingResponse)
def chat_stream(body: ChatRequest, session: Session, locks: Locks) -> StreamingResponse:
    """Like /chat, but streams Server-Sent Events: start, plan, task_result, replan,
    token, answer, and finally done or interrupted (or error)."""
    thread_id = body.thread_id or service.new_thread_id()
    release = locks.acquire(thread_id)
    try:
        _, graph_input = _start_input(session, body.model_copy(update={"thread_id": thread_id}))
    except BaseException:
        release()
        raise
    return _stream_response(session, thread_id, graph_input, release)


@router.post("/chat/{thread_id}/resume/stream", tags=["chat"], response_class=StreamingResponse)
def resume_stream(
    thread_id: ThreadId, body: ResumeRequest, session: Session, locks: Locks
) -> StreamingResponse:
    """Like /resume, but streams Server-Sent Events."""
    release = locks.acquire(thread_id)
    try:
        graph_input = _resume_input(session, thread_id, body)
    except BaseException:
        release()
        raise
    return _stream_response(session, thread_id, graph_input, release)


@router.get("/threads/{thread_id}", response_model=ThreadResponse, tags=["threads"])
def get_thread(thread_id: ThreadId, session: Session) -> ThreadResponse:
    """Conversation history (short-term memory) and any pending questions."""
    state = session.graph.get_state(_config(session, thread_id))
    if not state.values:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown thread.")
    roles = {"human": "user", "ai": "assistant"}
    messages = [
        MessageOut(role=roles[m.type], content=m.text)
        for m in state.values.get("messages", [])
        if m.type in roles
    ]
    interrupts = [service.interrupt_to_dict(i) for i in state.interrupts]
    return ThreadResponse(thread_id=thread_id, messages=messages, interrupts=interrupts)


@router.get("/memories", response_model=list[MemoryOut], tags=["memory"])
def list_memories(session: Session) -> list[MemoryOut]:
    """What the agent remembers about the user (no API cost)."""
    items = long_term.list_memories(session.store, session.settings.user_id)
    return [
        MemoryOut(key=i.key, text=i.value["text"], created_at=i.value.get("created_at"))
        for i in items
    ]


@router.delete("/memories", tags=["memory"])
def clear_memories(session: Session) -> dict[str, int]:
    """Delete all long-term memories for the user."""
    items = long_term.list_memories(session.store, session.settings.user_id)
    for item in items:
        long_term.delete_memory(session.store, session.settings.user_id, item.key)
    return {"deleted": len(items)}


@router.post("/ingest", response_model=IngestResponse, tags=["memory"])
def ingest(session: Session) -> IngestResponse:
    """Re-index knowledge/ for document search (embeds changed chunks only)."""
    report = documents.ingest_directory(session.store, session.settings.knowledge_dir)
    return IngestResponse(
        indexed=report.indexed,
        unchanged=report.unchanged,
        removed=report.removed,
        chunks=report.chunks,
    )


# --------------------------------------------------------------------------- app


def create_app(
    session_factory: SessionFactory | None = None,
    api_token: str | None = None,
    allowed_hosts: list[str] | None = None,
) -> FastAPI:
    """App factory. Tests pass a factory that builds the agent with fake models."""
    settings = get_settings()
    if api_token is None and (secret := settings.api_token):
        api_token = secret.get_secret_value()
    extra_hosts = allowed_hosts if allowed_hosts is not None else settings.api_allowed_host_list

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One agent session (models, SQLite store and checkpointer) for the app's lifetime.
        with (session_factory or agent_session)() as session:
            app.state.session = session
            app.state.locks = ThreadLocks()
            yield

    app = FastAPI(
        title="General Agent API",
        version="0.1.0",
        description="Plan-and-execute agent: parallel tool use, SQLite memory, semantic "
        "search and human-in-the-loop.",
        lifespan=lifespan,
    )
    app.state.api_token = api_token or None

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(router)
    app.add_middleware(BrowserAttackGuard, allowed_hosts=LOCAL_HOSTS | set(extra_hosts))
    return app
