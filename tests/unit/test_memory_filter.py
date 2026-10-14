import pytest

from general_agent.graph.nodes.memory_writer import is_durable_fact


@pytest.mark.parametrize(
    "fact",
    [
        "The user lives in Pune.",
        "The user prefers short answers.",
        "The user's name is Sam.",
        "The user is learning LangGraph.",
    ],
)
def test_durable_facts_are_kept(fact: str) -> None:
    assert is_durable_fact(fact)


@pytest.mark.parametrize(
    "fact",
    [
        "The user asked to save a 2-line summary of ReAct to react.txt.",  # from the demo
        "The user requested the weather in Paris.",
        "The user wants the assistant to write a haiku.",
        "User asked what time it is in Tokyo.",
        "The user's API key is sk-abc123def456ghi789.",
        "",
    ],
)
def test_one_off_requests_and_secrets_are_rejected(fact: str) -> None:
    assert not is_durable_fact(fact)
