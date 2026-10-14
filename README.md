# General Agent: plan-and-execute with LangGraph

A learning-oriented but production-style AI agent built with **LangChain + LangGraph**. It:

- **decomposes** one prompt into tasks (planner), **runs them in parallel** (executors),
  **checks** the results (replanner), and returns **one combined answer** (synthesizer)
- uses tools: current time, weather, calculator, web search, read/list/write files,
  memory recall, past-conversation search and document search
- remembers you with **SQLite memory**: short-term (per conversation thread) and
  long-term (across sessions)
- supports **semantic search** over memories, past conversations and your documents
- **asks you** when information is missing, and before writing files (human-in-the-loop)

## Quick start

```bash
# 1. Install (requires uv: https://docs.astral.sh/uv/)
uv sync

# 2. Configure
cp .env.example .env        # then put your OPENAI_API_KEY in .env

# 3. Index the sample knowledge base (files in knowledge/)
uv run agent ingest

# 4. Chat
uv run agent chat
```

Other commands: `uv run agent ask "..."` (one-shot), `uv run agent memories [--clear]`.
Inside chat: `/new`, `/memories`, `/ingest`, `/thread`, `/help`, `/quit`.

### Things to try

```
Remember that I live in Pune and I prefer short answers.
What time is it in Tokyo, what's the weather where I live, what's 15% of 2340,
  and according to my notes what is a checkpointer?
What's the weather?                 ← with no city known, the agent asks you
Write a 3-line summary of ReAct to react.txt   ← asks for approval first
What did we talk about earlier regarding weather?   ← past-conversation search
```

## How it works

```
 user ─▶ load_context ─▶ planner ──(needs info?)──▶ clarify ──▶ planner
                            │                     (interrupt: ask user)
                            ├──(no tasks)──────────────────────────┐
                            ▼                                      │
                 ┌── executor(t1) ─┐                               │
    Send() ────▶ ├── executor(t2) ─┼─▶ replanner ──(more tasks)──▶ executors…
   parallel      └── executor(t3) ─┘       │                       │
                                           ▼                       ▼
                                      synthesizer ◀────────────────┘
                                           │  (streams the final answer)
                                           ▼
                                     memory_writer ─▶ END
                           (extract facts + save the conversation turn)
```

| Node | File | What it does |
|---|---|---|
| `load_context` | `graph/nodes/context.py` | Resets per-turn state; semantic search over memories and past conversations |
| `planner` | `graph/nodes/planner.py` | Structured output `Plan{tasks, clarification_question}`; routes with `Command` |
| `clarify` | `graph/nodes/planner.py` | `interrupt()` asks you a question, then loops back to the planner |
| `executor` | `graph/nodes/executor.py` | A ReAct sub-agent (`create_agent`) that runs ONE task with tools |
| `replanner` | `graph/nodes/replanner.py` | Dispatches dependent tasks; may add follow-ups (bounded) |
| `synthesizer` | `graph/nodes/synthesizer.py` | Merges every result into one answer |
| `memory_writer` | `graph/nodes/memory_writer.py` | Background memory: extracts durable facts and stores the turn |

### Key LangGraph concepts used

- **`Send` (map-reduce / fan-out):** `dispatch.ready_sends` sends every task whose
  dependencies are finished to its own `executor` in the same step, so they run in
  parallel. Results merge through a custom reducer (`state.merge_results`).
- **`Command(goto=..., update=...)`:** nodes choose the next step dynamically
  (clarify, fan-out, or synthesize) instead of relying on fixed edges.
- **`interrupt()` + `Command(resume=...)`:** human-in-the-loop. The graph pauses and the
  CLI asks you, then resumes. It works even inside a parallel executor's tool call
  (`write_file`), and several pending interrupts can be resumed at once.
- **Checkpointer (`SqliteSaver`, `data/checkpoints.db`):** short-term memory per
  `thread_id`. It makes multi-turn chat and resuming after interrupts possible.
- **Store (`SqliteStore` + sqlite-vec, `data/memory.db`):** long-term memory with
  semantic search. Namespaces:
  `("memories", user)`, `("conversations", user)`, `("documents",)`.
- **`RetryPolicy`** on LLM nodes, and `recursion_limit` at the graph and sub-agent level.

## Project layout

```
src/general_agent/
  config.py          typed settings (.env), every limit in one place
  llm.py             model factories (init_chat_model, so the provider can be swapped)
  prompts.py         all prompts, plus untrusted-data wrapping
  state.py           graph state (TypedDicts) + LLM output schemas (Pydantic)
  app.py             composition root (wires everything)
  cli.py             Typer + Rich terminal UI
  graph/             builder.py, deps.py, nodes/*
  memory/            persistence.py (SQLite), long_term.py, documents.py (RAG ingest)
  tools/             time, weather, math, search, files (sandboxed), memory tools
tests/
  unit/              tools, memory, prompts (offline)
  graph/             whole-graph behaviour with scripted fake LLMs (offline)
  evals/run_evals.py live LLM-as-judge evaluation (needs API key)
knowledge/           your documents for semantic search (md, txt, pdf)
workspace/           the only folder the agent may write to
data/                SQLite databases (gitignored)
```

## Industry practices applied

| Practice | Where |
|---|---|
| Structured outputs instead of parsing JSON by hand | `state.py` schemas + `with_structured_output` |
| Dependency injection (testable nodes, no globals) | `graph/deps.py`, `make_*` node factories |
| Offline, deterministic tests with fake LLMs and embeddings | `tests/graph/fakes.py`, `tests/conftest.py` |
| Tool sandboxing (path traversal and symlink escapes blocked) | `tools/files.py` |
| Approval before side effects | `write_file` → `interrupt()` |
| Safe math (AST whitelist, bounded results; never `eval`) | `tools/math_tools.py` |
| Prompt-injection hygiene (untrusted data wrapped and its delimiters neutralized) | `prompts.py` |
| Hardened checkpoint deserialization (strict msgpack) | `memory/persistence.py` |
| Bounded loops: max tasks, replans, clarifications, executor steps | `config.py` |
| Failure isolation: one failed task doesn't sink the plan | `executor.py` |
| Retries and timeouts (LLM SDK, `RetryPolicy`, HTTP) | `llm.py`, `builder.py`, `weather.py` |
| Observability with LangSmith tracing (opt-in) | `.env`: `LANGSMITH_TRACING=true` |
| Lint, format and strict typing | `ruff`, `mypy --strict` |

## Development

```bash
uv run pytest                         # offline test suite
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run python -m tests.evals.run_evals    # live evals (uses your API key, small cost)
```

### Tracing with LangSmith

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in `.env`. Every run then shows up at
smith.langchain.com as a trace tree: planner → executors (with their tool calls) →
replanner → synthesizer.

### Changing models or provider

By default both roles use **`gpt-5-nano` with `REASONING_EFFORT=minimal`**, the cheapest
setup, which trades quality for cost. `gpt-5-nano` shuts down on 2026-12-11; switch to
`openai:gpt-5.6-luna` (the cheapest model that isn't deprecated) before then. For better plans and
answers, try `PLANNER_MODEL=openai:gpt-5.6-terra` and `EXECUTOR_MODEL=openai:gpt-5.4-mini`.

Models are `provider:model` strings in `.env` (`PLANNER_MODEL`, `EXECUTOR_MODEL`). To use
another provider, install its package (e.g. `uv add langchain-anthropic`) and set, for
example, `PLANNER_MODEL=anthropic:claude-sonnet-5-5`. Embeddings are configured in `llm.py`.
If you change `EMBEDDING_DIMS` or the embedding model, delete `data/memory.db` and re-run
`agent ingest`, because vectors from different models are not comparable.

## Ideas for extending (learning exercises)

1. Add a tool (e.g. currency conversion): write it in `tools/`, register it in
   `tools/__init__.py`, and add a test.
2. Stream executor progress with `stream_mode="custom"` and `get_stream_writer()`.
3. Add a FastAPI endpoint that reuses `app.agent_session()` and `cli.run_turn` logic.
4. Swap SQLite for Postgres (`PostgresSaver` / `PostgresStore`) for multi-user deployment.
5. Add memory TTLs or a "forget X" command.
