"""Scripted fake chat model for offline graph tests.

- `handler(messages) -> AIMessage` decides each plain/tool-calling response. Routing by
  message content (instead of a shared queue) keeps parallel executors deterministic.
- `structured[Schema]` is a queue of objects returned by `with_structured_output(Schema)`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from general_agent.state import MemoryExtraction, ReplanDecision

DEFAULTS: dict[type[BaseModel], Callable[[], BaseModel]] = {
    ReplanDecision: lambda: ReplanDecision(reasoning="done", action="finish", new_tasks=[]),
    MemoryExtraction: lambda: MemoryExtraction(facts=[]),
}


class ScriptedChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    handler: Callable[[list[BaseMessage]], AIMessage] = lambda _: AIMessage("ok")
    structured: dict[type[BaseModel], list[BaseModel]] = Field(default_factory=dict)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _structured_calls: list[str] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.handler(messages))])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedChatModel:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:
        def respond(_: Any) -> BaseModel:
            with self._lock:
                self._structured_calls.append(schema.__name__)
                queue = self.structured.get(schema, [])
                if queue:
                    return queue.pop(0)
            if schema in DEFAULTS:
                return DEFAULTS[schema]()
            raise AssertionError(f"No scripted response left for {schema.__name__}")

        return RunnableLambda(respond)

    @property
    def structured_calls(self) -> list[str]:
        return list(self._structured_calls)


def first_human_text(messages: list[BaseMessage]) -> str:
    return next(m.text for m in messages if m.type == "human")


def tool_results(messages: list[BaseMessage]) -> list[str]:
    return [m.text for m in messages if m.type == "tool"]


def tool_call(name: str, args: dict[str, Any], call_id: str = "call_1") -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])
