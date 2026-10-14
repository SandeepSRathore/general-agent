# AI Agent Concepts (sample knowledge base)

This file is indexed by `agent ingest` so the agent can answer questions about it with
semantic search (try: "according to my notes, what is a checkpointer?").

## Plan-and-execute
A plan-and-execute agent first asks a planner model to break a request into smaller tasks,
then runs each task with an executor. Independent tasks can run in parallel. A replanner
reviews the results and may add follow-up tasks. Finally a synthesizer writes one answer.
Compared with a single ReAct loop, this keeps long, multi-part requests organised and lets
you use a cheaper model for the executors.

## ReAct
ReAct ("reason + act") is a loop where the model decides on a tool call, sees the tool's
result, and repeats until it can answer. In this project every executor is a small ReAct
agent created with `langchain.agents.create_agent`.

## Checkpointer (short-term memory)
A LangGraph checkpointer saves the graph state after every step, keyed by a `thread_id`.
It enables multi-turn conversations, resuming after a human-in-the-loop interrupt, and
recovering from crashes. This project uses `SqliteSaver`, stored in data/checkpoints.db.

## Store (long-term memory)
A LangGraph store is a key/value database shared across threads, organised by namespaces
such as ("memories", user_id). With an embedding index it supports semantic search: items
are found by meaning rather than exact words. This project uses `SqliteStore` with the
sqlite-vec extension, stored in data/memory.db.

## Human-in-the-loop
`interrupt()` pauses a graph and surfaces a payload to the caller. The caller later resumes
with `Command(resume=value)`, and `interrupt()` returns that value. Code before an interrupt
re-runs on resume, so it must be idempotent.

## Embeddings
An embedding turns text into a vector of numbers so that texts with similar meaning are
close together. Cosine similarity measures how close two vectors are (1.0 = identical
direction). This project uses OpenAI's text-embedding-3-small with 1536 dimensions.
