"""synthesizer node: merges all task results into ONE final answer for the user."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from general_agent import prompts
from general_agent.graph.deps import AgentDeps, format_history, format_results, now_str
from general_agent.state import AgentState


def make_synthesizer(deps: AgentDeps) -> Any:
    def synthesizer(state: AgentState) -> dict[str, Any]:
        results = state.get("results") or {}
        ordered = [results[t["id"]] for t in state.get("plan", []) if t["id"] in results]
        response = deps.planner_llm.invoke(
            [
                SystemMessage(
                    prompts.SYNTHESIZER_SYSTEM.format(
                        now=now_str(), untrusted_note=prompts.UNTRUSTED_NOTE
                    )
                ),
                HumanMessage(
                    prompts.SYNTHESIZER_INPUT.format(
                        history=format_history(state.get("messages", [])[:-1]),
                        input=state.get("input", ""),
                        context=state.get("context", ""),
                        results=format_results(ordered),
                    )
                ),
            ]
        )
        answer = response.text
        return {"final_answer": answer, "messages": [AIMessage(answer)]}

    return synthesizer
