# General Agent — Plan-and-Execute agent with LangGraph, SQLite memory & semantic search

## Context
You're learning to build AI agents with LangChain + LangGraph. `~/MyLearnings/general-agent` is empty (and not a git repo yet).
The goal is a well-structured reference project that:
- takes one prompt, **splits it into tasks (planner)**, **runs the tasks (executor, with independent tasks in parallel)**, **checks the results (replanner)** and **returns one combined answer (synthesizer)**
- uses tools: current time, weather, read file, web search, calculator, list/write files, memory and document search
- keeps **SQLite memory**: the current conversation (checkpointer) plus long-term memory that lasts across sessions (store)
- supports **semantic search** over (a) long-term memories, (b) local documents (RAG), (c) past conversations
- **asks you (human-in-the-loop)** when it's missing information, and before it writes a file
- follows common industry practice: typed config, structured outputs, sandboxed tools, retries and timeouts, tracing, tests, linting

### Your decisions (from Q&A)
| Topic | Choice |
|---|---|
| LLM | OpenAI via `init_chat_model` (model names set in `.env`) |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dims) |
| Semantic search over | memories + local documents + past conversations |
| Extra tools | web search (DuckDuckGo), calculator, list/write files (sandboxed) |
| Missing info | ask the user via LangGraph `interrupt()` |
| Weather | Open-Meteo (free, no key) |
| Interface | CLI chat (Typer + Rich) |
| Quality | pytest suite + optional LangSmith tracing/evals |

## Architecture

```
            ┌──────────────┐
 user ───▶  │ load_context │  semantic search: memories + past conversations
            └──────┬───────┘
                   ▼
            ┌──────────────┐  structured output: Plan{tasks[], clarification?}
            │   planner    │──── needs info? ──▶ clarify (interrupt → ask user) ──┐
            └──────┬───────┘◀──────────────────────────────────────────────────────┘
                   ▼
            ┌──────────────┐  Send() fan-out to every task whose deps are met
            │   dispatch   │──────────────┬──────────────┐
            └──────▲───────┘              ▼              ▼
                   │               executor(task1)  executor(task2)   ← ReAct sub-agent with tools
                   │                      └──────┬───────┘                (parallel)
                   │                             ▼
                   │                      ┌──────────────┐
                   └──── more tasks ───── │  replanner   │  all done? failed? need more? (max 2 replans)
                                          └──────┬───────┘
                                                 ▼
                                          ┌──────────────┐
                                          │ synthesizer  │  one combined answer
                                          └──────┬───────┘
                                                 ▼
                                          ┌──────────────┐
                                          │memory_writer │  pull out lasting facts + save the conversation turn (embedded)
                                          └──────────────┘
```

- **Short-term memory:** `SqliteSaver` checkpointer (`data/agent.db`), keyed by `thread_id`. `/new` in the CLI starts a new thread.
- **Long-term memory and semantic index:** `langgraph.store.sqlite.SqliteStore` with `index={"dims":1536,"embed":OpenAIEmbeddings, "fields":["text"]}`. It uses sqlite-vec under the hood, so all three search targets live in the same DB under separate namespaces:
  - `("memories", user_id)`: facts and preferences ("My city is Pune")
  - `("conversations", user_id)`: a summary of each past turn (question and answer)
  - `("documents",)`: chunks from `./knowledge/*.{md,txt,pdf}`
- **Executor** = `langchain.agents.create_agent` (a ReAct loop), with the tool registry and a per-task step limit. It returns a `TaskResult{task_id, status, output, sources}`.
- **Parallelism:** `dispatch` returns `[Send("executor", {...task})]` for every task whose `depends_on` tasks are all done. Results merge into `state.results` through a dict-merge reducer.
- **HITL:** `interrupt()` in the `clarify` node, and inside the `write_file` tool for approval. The CLI loops over `stream.interrupts`, asks you, and resumes with `Command(resume={id: answer})`, which also works for parallel interrupts.

### State (`state.py`)
`AgentState(TypedDict)`: `messages` (add_messages), `user_id`, `input`, `context` (retrieved memories), `plan: list[Task]`, `results: Annotated[dict[str, TaskResult], merge]`, `replans: int`, `final_answer`.
Pydantic: `Task{id, description, depends_on: list[str], suggested_tools}`, `Plan{tasks, clarification_question: str|None}`, `ReplanDecision{action: "continue"|"add_tasks"|"finish", new_tasks}`, `TaskResult`.

## Project layout (uv, src layout, Python 3.12 pinned; 3.14 is too new for parts of the stack)
```
general-agent/
  pyproject.toml          # deps + ruff + mypy + pytest config
  .env.example  .gitignore  README.md (architecture + how to run + learning notes)
  docs/design.md          # copy of this design
  knowledge/sample.md     # sample doc for RAG
  workspace/              # sandbox for file tools
  data/                   # agent.db (gitignored)
  src/general_agent/
    config.py             # pydantic-settings: OPENAI_API_KEY, PLANNER_MODEL, EXECUTOR_MODEL, EMBEDDING_MODEL, DB_PATH, WORKSPACE_DIR, limits
    logging_setup.py      # structured logging (rich handler)
    llm.py                # get_planner_llm(), get_executor_llm(), get_embeddings()
    prompts.py            # all system prompts in one place
    state.py
    memory/
      persistence.py      # open_checkpointer(), open_store() context managers (STRICT_MSGPACK on)
      long_term.py        # save_memory/search_memories/save_conversation/search_conversations
      documents.py        # ingest(knowledge_dir): load → RecursiveCharacterTextSplitter → store.put (idempotent by content hash)
    tools/
      __init__.py         # get_tools() registry
      time_tools.py       # get_current_time(timezone)
      weather.py          # get_weather(city): Open-Meteo geocode + forecast (httpx, timeout, retry)
      files.py            # read_file, list_directory, write_file(interrupt approval); path-traversal safe, size limits
      calculator.py       # safe AST evaluator (no eval)
      web_search.py       # ddgs search, top-k, truncated
      memory_tools.py     # remember(fact), recall(query), search_past_conversations(query) (uses get_store())
      docs_search.py      # search_documents(query)
    graph/
      nodes/{context,planner,clarify,dispatch,executor,replanner,synthesizer,memory_writer}.py
      builder.py          # build_graph(checkpointer, store) → compiled graph; RetryPolicy on LLM nodes
    cli.py                # Typer app: `agent chat`, `agent ingest`, `agent memories`; Rich panels show plan → task progress → final answer
  tests/
    unit/  test_calculator, test_files_sandbox, test_time, test_weather (respx mock), test_long_term (fake embeddings)
    graph/ test_plan_execute (GenericFakeChatModel: planner→2 parallel tasks→synth), test_clarify_interrupt, test_dispatch_deps
    evals/ run_evals.py   # optional LangSmith dataset of ~8 multi-part prompts + LLM-as-judge
```

Main dependencies: `langgraph`, `langchain`, `langchain-openai`, `langgraph-checkpoint-sqlite`, `sqlite-vec`, `langchain-text-splitters`, `pypdf`, `httpx`, `ddgs`, `pydantic-settings`, `typer`, `rich`, `python-dotenv`. Dev dependencies: `pytest`, `pytest-asyncio`, `respx`, `ruff`, `mypy`.

## Industry practices included
- Structured outputs (`with_structured_output`) for the planner and replanner, so there's no fragile JSON parsing
- Configuration through env/pydantic-settings, with no hard-coded keys; `.env.example` committed and `.env` gitignored
- Tool safety: file tools are locked to `workspace/` and `knowledge/` (resolved paths, no `..` escapes), a size cap on reads, approval required for writes, and the calculator uses AST parsing instead of `eval`
- Resilience: HTTP timeouts and retries, `RetryPolicy` on LLM nodes, executor step limit, max tasks per plan (8), max 2 replans, `recursion_limit`
- Tool errors come back to the LLM as messages rather than crashing the graph; a failed task is marked `failed` and the synthesizer says so honestly
- Prompt-injection hygiene: tool and document outputs are wrapped and labelled as untrusted data in prompts
- Checkpoint deserialization hardened with `LANGGRAPH_STRICT_MSGPACK=true`
- Observability: LangSmith tracing switches on when `LANGSMITH_TRACING=true` is set; node-level logging otherwise
- Tests run offline with fake LLMs and embeddings; ruff and mypy are clean

## Implementation steps
1. `git init`; `uv init --package`; pin Python 3.12; add deps; set up ruff, mypy and pytest; add `.gitignore` and `.env.example`; copy this design to `docs/design.md`; commit.
2. **API spike (one-off check):** confirm the installed `SqliteStore` supports `index=` vector search and check the current `create_agent` and `stream_events`/interrupt APIs. If `SqliteStore` can't do vector search, fall back to a small `sqlite-vec` wrapper behind the same `long_term.py` interface.
3. config, logging, llm factories, state models (with tests for the reducers).
4. Tools one at a time, each with tests first (calculator → files sandbox → time → weather → web search).
5. Memory layer: persistence, long_term, documents ingest, then memory and docs search tools (tests use fake embeddings).
6. Graph nodes and builder; graph tests with fake chat models (parallel fan-out, dependency order, clarify interrupt and resume, replan limit).
7. CLI REPL with Rich, interrupt handling, and the `/new`, `/memories`, `/ingest`, `/quit` commands.
8. README with an architecture diagram, setup steps, example prompts and a "how it works" walkthrough for learning; optional evals script.
9. Code review pass, then verify (below). Commit at each step.

## Verification
- `uv run ruff check . && uv run mypy src && uv run pytest` all pass (offline).
- Manual end-to-end run with your `OPENAI_API_KEY`:
  1. `uv run agent ingest` indexes `knowledge/sample.md`
  2. `uv run agent chat`, then: *"Remember that I live in Pune."* The memory is stored.
  3. *"What time is it in Tokyo, what's the weather where I live, what's 15% of 2340, and summarize knowledge/sample.md"* The plan shows 4 tasks, the tasks run in parallel, and one combined answer comes back with Pune's weather pulled from memory.
  4. *"What's the weather?"* in a fresh setup with no memory: the agent asks which city (interrupt), then resumes.
  5. *"Write a summary to notes.txt"*: approval prompt, then the file appears in `workspace/`.
  6. Restart the CLI and ask *"what did we talk about earlier regarding weather?"* The answer comes from past-conversation semantic search.
  7. Optional: run with LangSmith on and check the trace tree.

## Things I'll need from you during implementation
- Put `OPENAI_API_KEY` in `.env` yourself (I won't handle the key). Optionally add `LANGSMITH_API_KEY`.
- Confirm default model names: I'll suggest the current OpenAI defaults after checking their docs (a stronger model for the planner and synthesizer, a cheaper one for executors). Both can be changed in `.env`.
