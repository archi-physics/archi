from __future__ import annotations

from typing import Any

from src.archi.pipelines.agents import agent_spec as agent_spec_module
from src.services.jira_ticket_responder import config as responder_config


def build_archi_for_jira(
    agent_config: responder_config.JiraAgentConfig,
) -> Any:
    from src.archi.archi import archi

    try:
        agent_spec = agent_spec_module.select_agent_spec(agent_config.agents_dir)
    except agent_spec_module.AgentSpecError as exc:
        raise ValueError(f"Failed to load Jira agent spec: {exc}") from exc

    return archi(
        pipeline=agent_config.agent_class,
        agent_spec=agent_spec,
        default_provider=agent_config.default_provider,
        default_model=agent_config.default_model,
    )
