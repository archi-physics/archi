import pytest

from src.archi.pipelines.agents.agent_spec import AgentSpecError, load_agent_spec_from_text


def test_explicit_empty_tools_list_is_a_no_tools_agent():
    spec = load_agent_spec_from_text("---\nname: D\ntools: []\n---\n\nAnswer from your own knowledge.\n")
    assert spec.tools == []
    assert spec.prompt.strip() == "Answer from your own knowledge."


def test_missing_tools_key_is_still_rejected():
    with pytest.raises(AgentSpecError):
        load_agent_spec_from_text("---\nname: D\n---\n\nprompt\n")
