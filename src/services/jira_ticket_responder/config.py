from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.utils import jira as jira_utils

DEFAULT_ELIGIBLE_STATUSES = ("Open", "In Progress")
JIRA_PROMPT_CONTEXT_WINDOW_SAFETY_MARGIN = 0.15
JIRA_PROMPT_CHARS_PER_TOKEN = 3


@dataclass(frozen=True)
class JiraServiceConfig:
    url: str
    projects: list[str]
    visible_to_role: Optional[str]
    poll_interval_minutes: int
    lookback_days: int
    eligible_statuses: list[str]
    respond_to_mentions: bool
    mention_allowed_roles: list[str]

    @classmethod
    def from_config(cls, raw_config: dict) -> "JiraServiceConfig":
        if not isinstance(raw_config, dict) or not raw_config:
            raise ValueError(
                "Missing required config section: services.jira_ticket_responder"
            )

        required = ["url", "projects"]
        missing = [key for key in required if raw_config.get(key) in (None, "")]
        if missing:
            raise ValueError(
                f"Missing required services.jira_ticket_responder fields: {', '.join(missing)}"
            )

        projects = jira_utils.parse_jira_project_keys(
            raw_config["projects"],
            "services.jira_ticket_responder.projects must be a non-empty list of Jira project keys.",
        )

        poll_interval = cls._parse_positive_int(
            raw_config.get("poll_interval_minutes", 1),
            "poll_interval_minutes",
            1,
        )
        lookback_days = cls._parse_positive_int(
            raw_config.get("lookback_days", 7),
            "lookback_days",
            7,
        )
        eligible_statuses = raw_config.get("eligible_statuses") or list(
            DEFAULT_ELIGIBLE_STATUSES
        )
        respond_to_mentions = False
        if "respond_to_mentions" in raw_config:
            respond_to_mentions = cls._parse_bool(
                raw_config["respond_to_mentions"],
                "respond_to_mentions",
            )
        mention_allowed_roles = cls._parse_optional_role_list(
            raw_config.get("mention_allowed_roles"),
            "mention_allowed_roles",
        )

        url = str(raw_config["url"]).strip()
        visible_to_role = cls._parse_optional_role(
            raw_config.get("visible_to_role"), "visible_to_role"
        )
        if not url:
            raise ValueError("services.jira_ticket_responder.url must not be empty.")

        return cls(
            url=url,
            projects=projects,
            visible_to_role=visible_to_role,
            poll_interval_minutes=poll_interval,
            lookback_days=lookback_days,
            eligible_statuses=eligible_statuses,
            respond_to_mentions=respond_to_mentions,
            mention_allowed_roles=mention_allowed_roles,
        )

    @staticmethod
    def _parse_positive_int(value: object, field_name: str, default: int) -> int:
        if value in (None, ""):
            value = default
        error = (
            f"services.jira_ticket_responder.{field_name} must be a positive integer."
        )
        if isinstance(value, bool):
            raise ValueError(error)
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(error) from exc
        if parsed <= 0:
            raise ValueError(error)
        return parsed

    @staticmethod
    def _parse_bool(value: object, field_name: str) -> bool:
        error = f"services.jira_ticket_responder.{field_name} must be a boolean."
        if not isinstance(value, bool):
            raise ValueError(error)
        return value

    @staticmethod
    def _parse_optional_role(value: object, field_name: str) -> Optional[str]:
        if value in (None, ""):
            return None
        error = (
            f"services.jira_ticket_responder.{field_name} must be a non-empty string."
        )
        if not isinstance(value, str):
            raise ValueError(error)
        role = value.strip()
        if not role:
            raise ValueError(error)
        return role

    @staticmethod
    def _parse_optional_role_list(value: object, field_name: str) -> list[str]:
        if value in (None, ""):
            return []
        error = (
            f"services.jira_ticket_responder.{field_name} must be a list of "
            "non-empty Jira project role names."
        )
        if not isinstance(value, list):
            raise ValueError(error)
        roles = []
        for value_role in value:
            if not isinstance(value_role, str):
                raise ValueError(error)
            role = value_role.strip()
            if not role:
                raise ValueError(error)
            roles.append(role)
        return roles


@dataclass(frozen=True)
class JiraAgentConfig:
    agent_class: str
    agents_dir: Path
    default_provider: str
    default_model: str
    prompt_max_chars: int

    @property
    def model_provider(self) -> str:
        return f"{self.default_provider}/{self.default_model}"


def resolve_jira_agent_config(services_config: dict) -> JiraAgentConfig:
    jira_config = services_config.get("jira_ticket_responder", {}) or {}
    chat_config = services_config.get("chat_app", {}) or {}

    agent_class = jira_config.get("agent_class") or "CMSCompOpsAgent"
    agents_dir = Path(
        jira_config.get("agents_dir")
        or chat_config.get("agents_dir")
        or "/root/archi/agents"
    )
    default_provider = jira_config.get("default_provider") or chat_config.get(
        "default_provider"
    )
    default_model = jira_config.get("default_model") or chat_config.get("default_model")
    if not default_provider or not default_model:
        raise ValueError(
            "Jira ticket responder requires default_provider and default_model in services.jira_ticket_responder or services.chat_app."
        )

    model_context_window = resolve_model_context_window(
        str(default_provider), str(default_model)
    )

    return JiraAgentConfig(
        agent_class=str(agent_class),
        agents_dir=agents_dir,
        default_provider=str(default_provider),
        default_model=str(default_model),
        prompt_max_chars=context_window_to_prompt_max_chars(model_context_window),
    )


def resolve_model_context_window(default_provider: str, default_model: str) -> int:
    from src.archi.providers import get_provider

    provider = get_provider(default_provider)
    model_info = provider.get_model_info(default_model)
    if model_info is None:
        raise ValueError(
            "Jira ticket responder could not resolve context window for "
            f"{default_provider}/{default_model}."
        )
    context_window = int(model_info.context_window)
    if context_window <= 0:
        raise ValueError(
            "Jira ticket responder resolved an invalid context window for "
            f"{default_provider}/{default_model}: {context_window}."
        )
    return context_window


def context_window_to_prompt_max_chars(context_window: int) -> int:
    if context_window <= 0:
        raise ValueError("context_window must be a positive integer.")
    prompt_token_budget = int(
        context_window * (1 - JIRA_PROMPT_CONTEXT_WINDOW_SAFETY_MARGIN)
    )
    return prompt_token_budget * JIRA_PROMPT_CHARS_PER_TOKEN
