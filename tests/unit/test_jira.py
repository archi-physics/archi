from contextlib import AbstractContextManager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.archi.utils.output_dataclass import PipelineOutput
from src.interfaces import jira as jira_interface
from src.services.jira_ticket_responder import config as responder_config
from src.services.jira_ticket_responder import formatting as responder_fmt
from src.services.jira_ticket_responder import prompts as responder_prompts
from src.services.jira_ticket_responder import service as responder_service
from src.services.jira_ticket_responder import store as responder_store


class _FakeArchi:
    def __init__(self, answer="  Use the documented fix.  ", result=None, failure=None):
        self.answer = answer
        self.result = result
        self.failure = failure
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure is not None:
            raise self.failure
        if self.result is not None:
            return self.result
        return SimpleNamespace(answer=self.answer, source_documents=[])


class _FakeIssueClient:
    def __init__(
        self,
        order=None,
        fail_post=False,
        issues=None,
        recent_comments_by_issue=None,
        fail_recent_comments=False,
        response_comment_id="jira-response-1",
        project_role_actors=None,
        role_authorization=None,
        fail_project_roles=False,
    ):
        self.order = order if order is not None else []
        self.fail_post = fail_post
        self.issues = issues if issues is not None else []
        self.recent_comments_by_issue = (
            recent_comments_by_issue if recent_comments_by_issue is not None else {}
        )
        self.fail_recent_comments = fail_recent_comments
        self.response_comment_id = response_comment_id
        self.project_role_actors = project_role_actors or []
        self.role_authorization = role_authorization or {}
        self.fail_project_roles = fail_project_roles
        self.searches = []
        self.recent_comment_fetches = []
        self.project_role_fetches = []
        self.role_authorization_checks = []
        self.posted = []

    def search_recent_issues(self, projects, lookback_days, eligible_statuses):
        self.searches.append((projects, lookback_days, eligible_statuses))
        return list(self.issues)

    def post_comment(self, issue_key, body, visible_to_role):
        self.order.append("post")
        if self.fail_post:
            raise RuntimeError("post failed")
        self.posted.append((issue_key, body, visible_to_role))
        if self.response_comment_id is None:
            return None
        return self.response_comment_id

    def fetch_recent_comments(self, issue_key):
        self.recent_comment_fetches.append(issue_key)
        if self.fail_recent_comments:
            raise RuntimeError("recent comments failed")
        return list(self.recent_comments_by_issue.get(issue_key, []))

    def comment_mentions_authenticated_user(self, comment):
        return "[~cmsai]" in comment.body

    def comment_authored_by_authenticated_user(self, comment):
        return comment.author.get("name") == "cmsai"

    def fetch_project_role_actors(self, project_key, role_names):
        self.project_role_fetches.append((project_key, role_names))
        if self.fail_project_roles:
            raise RuntimeError("project roles failed")
        return self.project_role_actors

    def comment_author_matches_project_role_actors(self, comment, actors):
        self.role_authorization_checks.append((comment.author, actors))
        return self.role_authorization.get(comment.author.get("name"), False)


class _FakeTriggerStore:
    def __init__(self, denied_keys=None, fail_claim_keys=None, fail_answer_keys=None):
        self.denied_keys = set(denied_keys or [])
        self.fail_claim_keys = set(fail_claim_keys or [])
        self.fail_answer_keys = set(fail_answer_keys or [])
        self.claims = []
        self.answered = []
        self.failed = []
        self.posted_but_unconfirmed = []
        self.last_errors = []
        self.linked_conversations = []

    def claim_trigger(
        self, *, trigger_key, trigger_type, issue_key, trigger_comment_id
    ):
        self.claims.append((trigger_key, trigger_type, issue_key, trigger_comment_id))
        if trigger_key in self.fail_claim_keys:
            raise RuntimeError("claim failed")
        return trigger_key not in self.denied_keys

    def mark_answered(self, trigger_key, response_comment_id):
        if trigger_key in self.fail_answer_keys:
            raise RuntimeError("answer update failed")
        self.answered.append((trigger_key, response_comment_id))

    def mark_failed(self, trigger_key, last_error):
        self.failed.append((trigger_key, last_error))

    def mark_posted_but_unconfirmed(
        self,
        *,
        trigger_key,
        trigger_type,
        issue_key,
        trigger_comment_id,
        response_comment_id,
        last_error,
    ):
        self.posted_but_unconfirmed.append(
            (
                trigger_key,
                trigger_type,
                issue_key,
                trigger_comment_id,
                response_comment_id,
                last_error,
            )
        )
        self.failed.append((trigger_key, last_error))

    def record_last_error(self, trigger_key, last_error):
        self.last_errors.append((trigger_key, last_error))

    def link_conversation(self, trigger_key, conversation_id):
        self.linked_conversations.append((trigger_key, conversation_id))


class _FakeCursor:
    def __init__(self, execute_side_effect=None, fetchone_values=None, rowcount=1):
        self.executed = []
        self.execute_side_effect = execute_side_effect
        self.fetchone_values = list(fetchone_values or [(42,)])
        self.rowcount = rowcount

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, query, params=None):
        if self.execute_side_effect is not None:
            raise self.execute_side_effect
        self.executed.append((query, params))

    def fetchone(self):
        if not self.fetchone_values:
            return None
        return self.fetchone_values.pop(0)


class _FakeConnection:
    def __init__(self, cursor=None):
        self.cursor_instance = cursor or _FakeCursor()
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class _FakeConnectionContext(AbstractContextManager):
    def __init__(self, connection):
        self.connection = connection
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self.connection

    def __exit__(self, exc_type, exc_value, traceback):
        self.exited = True
        return False


class _FakeConnectionPool:
    def __init__(self):
        self.connection = _FakeConnection()
        self.connection_context = None
        self.released = []

    def get_connection(self):
        self.connection_context = _FakeConnectionContext(self.connection)
        return self.connection_context

    def release_connection(self, conn):
        self.released.append(conn)


def _raw_issue(
    *,
    key="CMSTZ-1",
    status="Open",
):
    fields = SimpleNamespace(
        summary="Broken transfer",
        description="Transfers fail with timeout.",
        status=SimpleNamespace(name=status),
    )
    return SimpleNamespace(key=key, fields=fields)


def _normalize_sql(sql_text):
    return " ".join(sql_text.replace(";", "").split())


def _service(
    issue_client,
    archi_instance,
    projects=None,
    eligible_statuses=None,
    respond_to_mentions=False,
    mention_allowed_roles=None,
    visible_to_role="Developers",
    trigger_store=None,
    prompt_max_chars=1_000_000,
):
    config = responder_config.JiraServiceConfig(
        url="https://jira.example/",
        projects=projects if projects is not None else ["CMSTZ"],
        visible_to_role=visible_to_role,
        poll_interval_minutes=1,
        lookback_days=7,
        eligible_statuses=(
            eligible_statuses
            if eligible_statuses is not None
            else ["Open", "In Progress"]
        ),
        respond_to_mentions=respond_to_mentions,
        mention_allowed_roles=mention_allowed_roles or [],
    )
    return responder_service.JiraTicketResponderService(
        config=config,
        issue_client=issue_client,
        archi_instance=archi_instance,
        postgres_factory=SimpleNamespace(connection_pool=None),
        trigger_store=trigger_store or _FakeTriggerStore(),
        agent_config=SimpleNamespace(
            agent_class="CMSCompOpsAgent",
            model_provider="openai/gpt-5",
            prompt_max_chars=prompt_max_chars,
        ),
    )


def _expected_archi_prompt(ticket_prompt):
    return responder_prompts.build_archi_answer_prompt(ticket_prompt)


class TestJiraServiceConfig:
    def test_from_config_defaults_poll_interval_minutes_to_one(self):
        config = responder_config.JiraServiceConfig.from_config(
            {
                "url": "https://jira.example/",
                "projects": ["CMSTZ", "CMSDM"],
                "visible_to_role": "Developers",
            }
        )

        assert config.poll_interval_minutes == 1
        assert config.lookback_days == 7
        assert config.projects == ["CMSTZ", "CMSDM"]
        assert config.eligible_statuses == ["Open", "In Progress"]
        assert config.respond_to_mentions is False
        assert config.mention_allowed_roles == []
        assert config.visible_to_role == "Developers"

    def test_from_config_defaults_role_filters_to_all_roles(self):
        config = responder_config.JiraServiceConfig.from_config(
            {
                "url": "https://jira.example/",
                "projects": ["CMSTZ"],
            }
        )

        assert config.mention_allowed_roles == []
        assert config.visible_to_role is None

    def test_from_config_reads_poll_interval_lookback_statuses_and_mentions(
        self,
    ):
        config = responder_config.JiraServiceConfig.from_config(
            {
                "url": "https://jira.example/",
                "projects": ["CMSTZ"],
                "visible_to_role": "Developers",
                "poll_interval_minutes": 5,
                "lookback_days": 14,
                "eligible_statuses": ["Open", "Triaged"],
                "respond_to_mentions": True,
                "mention_allowed_roles": ["Developers", "Administrators"],
            }
        )

        assert config.poll_interval_minutes == 5
        assert config.lookback_days == 14
        assert config.eligible_statuses == ["Open", "Triaged"]
        assert config.respond_to_mentions is True
        assert config.mention_allowed_roles == ["Developers", "Administrators"]

    def test_from_config_defaults_empty_optional_fields(self):
        config = responder_config.JiraServiceConfig.from_config(
            {
                "url": "https://jira.example/",
                "projects": ["CMSTZ"],
                "visible_to_role": "Developers",
                "lookback_days": "",
                "eligible_statuses": "",
            }
        )

        assert config.lookback_days == 7
        assert config.eligible_statuses == ["Open", "In Progress"]

    @pytest.mark.parametrize("value", [0, -1, "x", True])
    def test_from_config_rejects_invalid_lookback_days(self, value):
        with pytest.raises(
            ValueError, match="services.jira_ticket_responder.lookback_days"
        ):
            responder_config.JiraServiceConfig.from_config(
                {
                    "url": "https://jira.example/",
                    "projects": ["CMSTZ"],
                    "visible_to_role": "Developers",
                    "poll_interval_minutes": 1,
                    "lookback_days": value,
                }
            )

    @pytest.mark.parametrize("value", ["true", "false", 1, 0, None])
    def test_from_config_rejects_non_boolean_respond_to_mentions(self, value):
        with pytest.raises(
            ValueError, match="services.jira_ticket_responder.respond_to_mentions"
        ):
            responder_config.JiraServiceConfig.from_config(
                {
                    "url": "https://jira.example/",
                    "projects": ["CMSTZ"],
                    "visible_to_role": "Developers",
                    "respond_to_mentions": value,
                }
            )

    @pytest.mark.parametrize(
        "value",
        ["Developers", [""], ["Developers", 7], [None]],
    )
    def test_from_config_rejects_invalid_mention_allowed_roles(self, value):
        with pytest.raises(
            ValueError,
            match="services.jira_ticket_responder.mention_allowed_roles",
        ):
            responder_config.JiraServiceConfig.from_config(
                {
                    "url": "https://jira.example/",
                    "projects": ["CMSTZ"],
                    "mention_allowed_roles": value,
                }
            )

    @pytest.mark.parametrize("value", [[], 7, "   "])
    def test_from_config_rejects_invalid_visible_to_role(self, value):
        with pytest.raises(
            ValueError,
            match="services.jira_ticket_responder.visible_to_role",
        ):
            responder_config.JiraServiceConfig.from_config(
                {
                    "url": "https://jira.example/",
                    "projects": ["CMSTZ"],
                    "visible_to_role": value,
                }
            )

    @pytest.mark.parametrize(
        "projects",
        [
            [],
            "CMSTZ",
            ["CMSTZ", ""],
            ["CMSTZ", 7],
            ["CMSTZ, CMSDM"],
            ["cms"],
            ["2013PROJECT"],
            ["PRODUCT-2012"],
        ],
    )
    def test_from_config_rejects_invalid_projects(self, projects):
        with pytest.raises(ValueError, match="services.jira_ticket_responder.projects"):
            responder_config.JiraServiceConfig.from_config(
                {
                    "url": "https://jira.example/",
                    "projects": projects,
                    "visible_to_role": "Developers",
                    "poll_interval_minutes": 1,
                }
            )


class TestJiraAgentConfig:
    def test_resolve_agent_config_prefers_jira_provider_and_model(self, monkeypatch):
        monkeypatch.setattr(
            responder_config, "resolve_model_context_window", Mock(return_value=128000)
        )

        config = responder_config.resolve_jira_agent_config(
            {
                "jira_ticket_responder": {
                    "default_provider": "openai",
                    "default_model": "gpt-5",
                },
                "chat_app": {
                    "default_provider": "anthropic",
                    "default_model": "claude-sonnet-4-20250514",
                    "agents_dir": "/chat/agents",
                },
            }
        )

        assert config.default_provider == "openai"
        assert config.default_model == "gpt-5"
        assert str(config.agents_dir) == "/chat/agents"
        assert config.prompt_max_chars == 326400

    def test_resolve_agent_config_falls_back_to_chat_provider_and_model(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            responder_config, "resolve_model_context_window", Mock(return_value=128000)
        )

        config = responder_config.resolve_jira_agent_config(
            {
                "jira_ticket_responder": {},
                "chat_app": {
                    "default_provider": "openai",
                    "default_model": "gpt-5",
                },
            }
        )

        assert config.agent_class == "CMSCompOpsAgent"
        assert config.default_provider == "openai"
        assert config.default_model == "gpt-5"
        assert config.prompt_max_chars == 326400

    def test_resolve_agent_config_requires_resolved_provider_and_model(self):
        with pytest.raises(
            ValueError, match="services.jira_ticket_responder or services.chat_app"
        ):
            responder_config.resolve_jira_agent_config(
                {"jira_ticket_responder": {}, "chat_app": {}}
            )

    def test_resolve_model_context_window_requires_known_model(self, monkeypatch):
        class FakeProvider:
            def get_model_info(self, model_name):
                return None

        from src.archi import providers

        monkeypatch.setattr(
            providers, "get_provider", Mock(return_value=FakeProvider())
        )

        with pytest.raises(
            ValueError,
            match="could not resolve context window for openai/not-a-real-model",
        ):
            responder_config.resolve_model_context_window("openai", "not-a-real-model")


class TestJiraTicketPrompt:
    def test_build_archi_answer_prompt_adds_operator_style_instructions(self):
        prompt = responder_prompts.build_archi_answer_prompt("Issue payload.")

        assert prompt.startswith(
            "You are Archi, a CMS Computing Operations assistant answering Jira tickets."
        )
        assert (
            "Do not claim that you performed, will perform, approved, created, "
            "granted, staged, invalidated, deleted, retried, or changed anything"
            in prompt
        )
        assert prompt.endswith("\n\nIssue payload.")

    def test_build_archi_answer_prompt_preserves_evaluated_resolution_policy(self):
        prompt = responder_prompts.build_archi_answer_prompt("Issue payload.")

        expected_policies = (
            "Before drafting, extract the operator resolution from the evidence.",
            "Prefer historical answer evidence from the same Jira project",
            "Separate what the operator told the requester from what the operator "
            "or another service actually did.",
            "Preserve negative policy boundaries, routing decisions, ownership "
            "boundaries, exception criteria, limits, prerequisites, and follow-up "
            "requirements",
            "Do not replace a user-facing URL with an internal OKG node id.",
            "starting with the current Jira project when possible",
            "Do not convert a routing decision into an offer for Archi or CompOps "
            "to act.",
            "state that boundary first",
            "answer definitively only when the available tools provide that evidence",
        )

        for policy in expected_policies:
            assert policy in prompt

    def test_build_ticket_prompt_excludes_comments(self):
        issue = jira_interface.JiraIssue(
            key="CMSTZ-7",
            summary="Storage is unavailable",
            description="The site reports storage errors.",
            status_name="Open",
        )

        prompt = responder_prompts.build_ticket_prompt(issue)

        assert prompt == (
            "Suggest a solution to this problem.\n\n"
            "Issue:\n"
            "CMSTZ-7\n\n"
            "Summary:\n"
            "Storage is unavailable\n\n"
            "Status:\n"
            "Open\n\n"
            "Description:\n"
            "The site reports storage errors."
        )

    def test_build_mention_prompt_separates_triggering_comment_and_context(self):
        issue = jira_interface.JiraIssue(
            key="CMSTZ-7",
            summary="Storage is unavailable",
            description="The site reports storage errors.",
            status_name="Open",
        )
        triggering_comment = jira_interface.JiraComment(
            id="7382883",
            body="[~cmsai] can you answer this?",
            author={"name": "human-user"},
            created="2026-06-26T10:00:00.000+0200",
            updated="2026-06-26T10:01:00.000+0200",
        )
        service_context = jira_interface.JiraComment(
            id="7382884",
            body="Earlier Archi answer for context.",
            author={"name": "cmsai"},
            created="2026-06-26T09:00:00.000+0200",
            updated="2026-06-26T09:00:00.000+0200",
        )

        prompt = responder_prompts.build_mention_prompt(
            issue,
            triggering_comment,
            [triggering_comment, service_context],
            max_prompt_chars=1_000_000,
        )

        assert prompt == (
            "Answer the Jira comment in the 'Triggering Comment:' section. "
            "Use the issue fields and recent comments as context.\n\n"
            "Issue:\n"
            "CMSTZ-7\n\n"
            "Summary:\n"
            "Storage is unavailable\n\n"
            "Status:\n"
            "Open\n\n"
            "Description:\n"
            "The site reports storage errors.\n\n"
            "Triggering Comment:\n"
            "Comment ID: 7382883\n"
            "Author: name=human-user\n"
            "Created: 2026-06-26T10:00:00.000+0200\n"
            "Updated: 2026-06-26T10:01:00.000+0200\n"
            "Body:\n"
            "[~cmsai] can you answer this?\n\n"
            "Recent comments context (newest first):\n"
            "Comment ID: 7382884\n"
            "Author: name=cmsai\n"
            "Created: 2026-06-26T09:00:00.000+0200\n"
            "Updated: 2026-06-26T09:00:00.000+0200\n"
            "Body:\n"
            "Earlier Archi answer for context."
        )
        assert prompt.count("Comment ID: 7382883") == 1
        assert "TRIGGERING COMMENT TO ANSWER" not in prompt
        assert "Context comment" not in prompt

    def test_build_mention_prompt_reports_no_additional_context_after_skipping_trigger(
        self,
    ):
        issue = jira_interface.JiraIssue(
            key="CMSTZ-7",
            summary="Storage is unavailable",
            description="The site reports storage errors.",
            status_name="Open",
        )
        triggering_comment = jira_interface.JiraComment(
            id="7382883",
            body="[~cmsai] can you answer this?",
            author={"name": "human-user"},
            created="2026-06-26T10:00:00.000+0200",
            updated="2026-06-26T10:01:00.000+0200",
        )

        prompt = responder_prompts.build_mention_prompt(
            issue,
            triggering_comment,
            [triggering_comment],
            max_prompt_chars=1_000_000,
        )

        assert (
            "Recent comments context (newest first):\n"
            "No additional recent comments were fetched."
        ) in prompt

    def test_build_mention_prompt_includes_full_comment_bodies_without_hard_caps(self):
        issue = jira_interface.JiraIssue(
            key="CMSTZ-7",
            summary="Storage is unavailable",
            description="The site reports storage errors.",
            status_name="Open",
        )
        triggering_comment = jira_interface.JiraComment(
            id="7382883",
            body="T" * 8500,
            author={"name": "human-user"},
            created="",
            updated="",
        )
        context_comment = jira_interface.JiraComment(
            id="7382884",
            body="C" * 4500,
            author={"name": "another-user"},
            created="",
            updated="",
        )

        prompt = responder_prompts.build_mention_prompt(
            issue,
            triggering_comment,
            [triggering_comment, context_comment],
            max_prompt_chars=20_000,
        )

        assert "[truncated " not in prompt
        assert "T" * 8500 in prompt
        assert "C" * 4500 in prompt

    def test_build_mention_prompt_stops_at_total_budget_and_logs(self, caplog):
        caplog.set_level("INFO", logger=responder_prompts.logger.name)

        issue = jira_interface.JiraIssue(
            key="CMSTZ-7",
            summary="Storage is unavailable",
            description="The site reports storage errors.",
            status_name="Open",
        )
        triggering_comment = jira_interface.JiraComment(
            id="7382883",
            body="[~cmsai] can you answer this?",
            author={"name": "human-user"},
            created="",
            updated="",
        )
        comments = [
            jira_interface.JiraComment(
                id=str(7382884 + index),
                body=f"context {index} " + ("x" * 200),
                author={"name": f"user-{index}"},
                created="",
                updated="",
            )
            for index in range(3)
        ]

        prompt = responder_prompts.build_mention_prompt(
            issue,
            triggering_comment,
            [triggering_comment, *comments],
            max_prompt_chars=900,
        )

        assert "TRIGGERING COMMENT TO ANSWER" not in prompt
        assert prompt.count("Comment ID: 7382883") == 1
        assert "context 0 " in prompt
        assert "context 1 " not in prompt
        assert "context 2 " not in prompt
        assert "Omitted 2 older Jira comments from prompt" in caplog.text

    def test_build_mention_prompt_keeps_required_fields_when_over_budget(self):
        issue = jira_interface.JiraIssue(
            key="CMSTZ-7",
            summary="Storage is unavailable",
            description="The site reports storage errors.",
            status_name="Open",
        )
        triggering_comment = jira_interface.JiraComment(
            id="7382883",
            body="[~cmsai] " + ("x" * 1000),
            author={"name": "human-user"},
            created="",
            updated="",
        )

        prompt = responder_prompts.build_mention_prompt(
            issue,
            triggering_comment,
            [triggering_comment],
            max_prompt_chars=500,
        )

        assert len(prompt) > 500
        assert "[~cmsai] " + ("x" * 1000) in prompt
        assert "No additional recent comments were fetched." in prompt


class TestJiraIssueEligibility:
    def test_open_issue_is_eligible(self):
        issue = jira_interface.JiraIssue(
            key="CMSTZ-7",
            summary="Open",
            description="Open",
            status_name="Open",
        )

        assert responder_service.is_issue_eligible(issue, ["Open", "Triaged"]) is True

    def test_configured_issue_status_is_eligible(self):
        issue = jira_interface.JiraIssue(
            key="CMSTZ-7",
            summary="Triaged",
            description="Triaged",
            status_name="Triaged",
        )

        assert responder_service.is_issue_eligible(issue, ["Open", "Triaged"]) is True

    @pytest.mark.parametrize("status", ["Closed", "Resolved", "To Do"])
    def test_disallowed_status_is_not_eligible(self, status):
        issue = jira_interface.JiraIssue(
            key="CMSTZ-1",
            summary="Wrong status",
            description="Wrong status",
            status_name=status,
        )

        assert responder_service.is_issue_eligible(issue, ["Open", "Triaged"]) is False


class TestJiraCommentTraceFormatting:
    def test_builds_standard_wiki_panels_for_reasoning_and_tool_calls(self):
        result = PipelineOutput(
            answer="Use the documented fix.",
            messages=[
                SimpleNamespace(
                    additional_kwargs={
                        "reasoning_content": "Checked the transfer logs."
                    },
                    tool_calls=[
                        {
                            "id": "call-1",
                            "name": "retriever",
                            "args": {"query": "transfer timeout"},
                        }
                    ],
                ),
                SimpleNamespace(
                    tool_call_id="call-1",
                    content="Found the transfer timeout runbook.",
                ),
            ],
        )

        comment_body = responder_fmt.build_jira_comment_body(
            "Use the documented fix.", result
        )

        assert comment_body == (
            "Use the documented fix.\n\n"
            "{panel:title=Reasoning trace}\n"
            "{noformat}\n"
            "Checked the transfer logs.\n"
            "{noformat}\n"
            "{panel}\n\n"
            "{panel:title=Tool calls}\n"
            "{noformat}\n"
            "Tool call 1: retriever\n"
            "Input:\n"
            "{\n"
            '  "query": "transfer timeout"\n'
            "}\n"
            "\n"
            "Output:\n"
            "Found the transfer timeout runbook.\n"
            "{noformat}\n"
            "{panel}"
        )
        assert "{expand" not in comment_body

    def test_escapes_noformat_macro_inside_trace_text(self):
        result = PipelineOutput(
            answer="Use the documented fix.",
            messages=[
                SimpleNamespace(
                    additional_kwargs={
                        "reasoning_content": "A tool returned {noformat} in text."
                    },
                    tool_calls=[],
                ),
            ],
        )

        comment_body = responder_fmt.build_jira_comment_body(
            "Use the documented fix.", result
        )

        assert comment_body == (
            "Use the documented fix.\n\n"
            "{panel:title=Reasoning trace}\n"
            "{noformat}\n"
            "A tool returned { noformat } in text.\n"
            "{noformat}\n"
            "{panel}"
        )

    def test_caps_tool_calls_panel_as_one_closed_wiki_section(self):
        result = PipelineOutput(
            answer="Use the documented fix.",
            messages=[
                SimpleNamespace(
                    tool_calls=[
                        {
                            "id": f"call-{index}",
                            "name": "retriever",
                            "args": {"query": f"query-{index}", "payload": "x" * 1000},
                        }
                        for index in range(10)
                    ],
                ),
                *[
                    SimpleNamespace(
                        tool_call_id=f"call-{index}",
                        content=f"result-{index} " + ("y" * 1000),
                    )
                    for index in range(10)
                ],
            ],
        )

        comment_body = responder_fmt.build_jira_comment_body(
            "Use the documented fix.", result
        )

        assert "[truncated " in comment_body
        assert comment_body.endswith("{panel}")
        assert comment_body.count("{panel:title=Tool calls}") == 1
        assert comment_body.count("{noformat}") == 2
        assert len(comment_body) < responder_fmt.JIRA_TRACE_SECTION_MAX_CHARS + 200


class TestJiraTriggerStore:
    def test_ensure_schema_creates_trigger_table_and_indexes(self):
        pool = _FakeConnectionPool()
        store = responder_store.JiraTriggerStore(SimpleNamespace(connection_pool=pool))

        store.ensure_schema()

        executed_sql = "\n".join(
            query.strip() for query, _params in pool.connection.cursor_instance.executed
        )
        assert "CREATE TABLE IF NOT EXISTS jira_responder_triggers" in executed_sql
        assert "trigger_key TEXT PRIMARY KEY" in executed_sql
        assert (
            "trigger_type TEXT NOT NULL CHECK (trigger_type IN ('issue','mention_comment'))"
            in executed_sql
        )
        assert (
            "conversation_id INTEGER REFERENCES conversation_metadata(conversation_id) ON DELETE SET NULL"
            in executed_sql
        )
        assert (
            "CREATE INDEX IF NOT EXISTS idx_jira_responder_triggers_issue"
            in executed_sql
        )
        assert (
            "CREATE INDEX IF NOT EXISTS idx_jira_responder_triggers_status"
            in executed_sql
        )
        assert pool.connection.commits == 1
        assert pool.connection.rollbacks == 0
        assert pool.connection_context.entered is True
        assert pool.connection_context.exited is True

    def test_ensure_schema_rolls_back_and_raises_on_failure(self):
        cursor = _FakeCursor(execute_side_effect=RuntimeError("ddl failed"))
        connection = _FakeConnection(cursor=cursor)
        pool = _FakeConnectionPool()
        pool.connection = connection
        store = responder_store.JiraTriggerStore(SimpleNamespace(connection_pool=pool))

        with pytest.raises(RuntimeError, match="ddl failed"):
            store.ensure_schema()

        assert connection.commits == 0
        assert connection.rollbacks == 1
        assert pool.connection_context.entered is True
        assert pool.connection_context.exited is True

    def test_schema_statements_match_fresh_deployment_sql_files(self):
        repo_root = Path(__file__).resolve().parents[2]
        startup_sql = _normalize_sql(
            "\n".join(responder_store.JIRA_RESPONDER_SCHEMA_STATEMENTS)
        )

        for relative_path in (
            "src/cli/templates/init.sql",
            "tests/smoke/init-test.sql",
        ):
            sql_text = (repo_root / relative_path).read_text()
            assert startup_sql in _normalize_sql(sql_text), relative_path

    def test_claim_trigger_inserts_new_issue_trigger_as_answering(self):
        cursor = _FakeCursor(fetchone_values=[None])
        connection = _FakeConnection(cursor=cursor)
        pool = _FakeConnectionPool()
        pool.connection = connection
        store = responder_store.JiraTriggerStore(SimpleNamespace(connection_pool=pool))

        should_answer = store.claim_trigger(
            trigger_key="issue:CMSTZ-1",
            trigger_type="issue",
            issue_key="CMSTZ-1",
            trigger_comment_id=None,
        )

        assert should_answer is True
        assert connection.commits == 1
        assert connection.rollbacks == 0
        assert "SELECT status, retry_used, updated_at" in cursor.executed[0][0]
        assert cursor.executed[0][1] == ("issue:CMSTZ-1",)
        assert "INSERT INTO jira_responder_triggers" in cursor.executed[1][0]
        assert cursor.executed[1][1] == (
            "issue:CMSTZ-1",
            "issue",
            "CMSTZ-1",
            None,
        )
        assert "'answering', FALSE" in cursor.executed[1][0]

    @pytest.mark.parametrize("status", ["answered", "failed"])
    def test_claim_trigger_skips_terminal_states(self, status):
        cursor = _FakeCursor(
            fetchone_values=[(status, False, datetime.now(timezone.utc))]
        )
        connection = _FakeConnection(cursor=cursor)
        pool = _FakeConnectionPool()
        pool.connection = connection
        store = responder_store.JiraTriggerStore(SimpleNamespace(connection_pool=pool))

        should_answer = store.claim_trigger(
            trigger_key="comment:7382883",
            trigger_type="mention_comment",
            issue_key="CMSTZ-1",
            trigger_comment_id="7382883",
        )

        assert should_answer is False
        assert len(cursor.executed) == 1
        assert connection.commits == 1
        assert connection.rollbacks == 0

    def test_claim_trigger_skips_fresh_answering_without_retry(self):
        cursor = _FakeCursor(
            fetchone_values=[
                (
                    "answering",
                    False,
                    datetime.now(timezone.utc) - timedelta(seconds=59),
                )
            ]
        )
        connection = _FakeConnection(cursor=cursor)
        pool = _FakeConnectionPool()
        pool.connection = connection
        store = responder_store.JiraTriggerStore(SimpleNamespace(connection_pool=pool))

        should_answer = store.claim_trigger(
            trigger_key="comment:7382883",
            trigger_type="mention_comment",
            issue_key="CMSTZ-1",
            trigger_comment_id="7382883",
        )

        assert should_answer is False
        assert len(cursor.executed) == 1
        assert connection.commits == 1
        assert connection.rollbacks == 0

    def test_claim_trigger_retries_stale_answering_once(self):
        cursor = _FakeCursor(
            fetchone_values=[
                (
                    "answering",
                    False,
                    datetime.now(timezone.utc) - timedelta(seconds=61),
                )
            ]
        )
        connection = _FakeConnection(cursor=cursor)
        pool = _FakeConnectionPool()
        pool.connection = connection
        store = responder_store.JiraTriggerStore(SimpleNamespace(connection_pool=pool))

        should_answer = store.claim_trigger(
            trigger_key="comment:7382883",
            trigger_type="mention_comment",
            issue_key="CMSTZ-1",
            trigger_comment_id="7382883",
        )

        assert should_answer is True
        assert len(cursor.executed) == 2
        assert "SET retry_used = TRUE" in cursor.executed[1][0]
        assert cursor.executed[1][1] == ("comment:7382883",)
        assert connection.commits == 1
        assert connection.rollbacks == 0

    def test_claim_trigger_skips_retried_answering_before_final_timeout(self):
        cursor = _FakeCursor(
            fetchone_values=[
                (
                    "answering",
                    True,
                    datetime.now(timezone.utc) - timedelta(seconds=599),
                )
            ]
        )
        connection = _FakeConnection(cursor=cursor)
        pool = _FakeConnectionPool()
        pool.connection = connection
        store = responder_store.JiraTriggerStore(SimpleNamespace(connection_pool=pool))

        should_answer = store.claim_trigger(
            trigger_key="comment:7382883",
            trigger_type="mention_comment",
            issue_key="CMSTZ-1",
            trigger_comment_id="7382883",
        )

        assert should_answer is False
        assert len(cursor.executed) == 1
        assert connection.commits == 1
        assert connection.rollbacks == 0

    def test_claim_trigger_marks_retried_stale_answering_failed(self):
        cursor = _FakeCursor(
            fetchone_values=[
                (
                    "answering",
                    True,
                    datetime.now(timezone.utc) - timedelta(seconds=601),
                )
            ]
        )
        connection = _FakeConnection(cursor=cursor)
        pool = _FakeConnectionPool()
        pool.connection = connection
        store = responder_store.JiraTriggerStore(SimpleNamespace(connection_pool=pool))

        should_answer = store.claim_trigger(
            trigger_key="comment:7382883",
            trigger_type="mention_comment",
            issue_key="CMSTZ-1",
            trigger_comment_id="7382883",
        )

        assert should_answer is False
        assert len(cursor.executed) == 2
        assert "SET status = 'failed'" in cursor.executed[1][0]
        assert cursor.executed[1][1] == (
            "Trigger remained answering after the final stale timeout.",
            "comment:7382883",
        )
        assert connection.commits == 1
        assert connection.rollbacks == 0

    def test_trigger_status_update_helpers_commit_expected_fields(self):
        pool = _FakeConnectionPool()
        store = responder_store.JiraTriggerStore(SimpleNamespace(connection_pool=pool))

        store.mark_answered("comment:7382883", "8001")
        store.mark_posted_but_unconfirmed(
            trigger_key="comment:7382887",
            trigger_type="mention_comment",
            issue_key="CMSTZ-1",
            trigger_comment_id="7382887",
            response_comment_id="8002",
            last_error="posted but not marked",
        )
        store.mark_failed("comment:7382884", "post failed")
        store.record_last_error("comment:7382885", "persist failed")
        store.link_conversation("comment:7382886", 42)

        executed = pool.connection.cursor_instance.executed
        assert "SET status = 'answered'" in executed[0][0]
        assert executed[0][1] == ("8001", "comment:7382883")
        assert "ON CONFLICT (trigger_key) DO UPDATE" in executed[1][0]
        assert executed[1][1] == (
            "comment:7382887",
            "mention_comment",
            "CMSTZ-1",
            "7382887",
            "posted but not marked",
            "8002",
        )
        assert "SET status = 'failed'" in executed[2][0]
        assert executed[2][1] == ("post failed", "comment:7382884")
        assert "SET last_error = %s" in executed[3][0]
        assert executed[3][1] == ("persist failed", "comment:7382885")
        assert "SET conversation_id = %s" in executed[4][0]
        assert executed[4][1] == (42, "comment:7382886")
        assert pool.connection.commits == 5
        assert pool.connection.rollbacks == 0

    def test_trigger_status_update_helpers_raise_when_no_rows_change(self):
        cursor = _FakeCursor(rowcount=0)
        connection = _FakeConnection(cursor=cursor)
        pool = _FakeConnectionPool()
        pool.connection = connection
        store = responder_store.JiraTriggerStore(SimpleNamespace(connection_pool=pool))

        with pytest.raises(RuntimeError, match="Expected exactly one Jira responder"):
            store.mark_failed("comment:missing", "not found")

        assert connection.commits == 0
        assert connection.rollbacks == 1


class TestJiraTicketResponderService:
    def test_poll_once_searches_multiple_projects_and_answers_each_eligible_issue(self):
        issue_client = _FakeIssueClient(
            issues=[
                _raw_issue(key="CMSTZ-1"),
                _raw_issue(key="CMSDM-2"),
            ]
        )
        archi_instance = _FakeArchi()
        trigger_store = _FakeTriggerStore()
        service = _service(
            issue_client,
            archi_instance,
            projects=["CMSTZ", "CMSDM"],
            trigger_store=trigger_store,
        )
        service.persist_interaction = Mock(side_effect=[101, 102])

        service.poll_once()

        assert issue_client.searches == [
            (["CMSTZ", "CMSDM"], 7, ["Open", "In Progress"])
        ]
        assert issue_client.recent_comment_fetches == []
        assert trigger_store.claims == [
            ("issue:CMSTZ-1", "issue", "CMSTZ-1", None),
            ("issue:CMSDM-2", "issue", "CMSDM-2", None),
        ]
        assert len(archi_instance.calls) == 2
        assert archi_instance.calls[0]["history"][0][1] == _expected_archi_prompt(
            "Suggest a solution to this problem.\n\n"
            "Issue:\n"
            "CMSTZ-1\n\n"
            "Summary:\n"
            "Broken transfer\n\n"
            "Status:\n"
            "Open\n\n"
            "Description:\n"
            "Transfers fail with timeout."
        )
        assert archi_instance.calls[1]["history"][0][1] == _expected_archi_prompt(
            "Suggest a solution to this problem.\n\n"
            "Issue:\n"
            "CMSDM-2\n\n"
            "Summary:\n"
            "Broken transfer\n\n"
            "Status:\n"
            "Open\n\n"
            "Description:\n"
            "Transfers fail with timeout."
        )
        assert issue_client.posted == [
            ("CMSTZ-1", "Use the documented fix.", "Developers"),
            ("CMSDM-2", "Use the documented fix.", "Developers"),
        ]
        assert trigger_store.answered == [
            ("issue:CMSTZ-1", "jira-response-1"),
            ("issue:CMSDM-2", "jira-response-1"),
        ]
        assert trigger_store.linked_conversations == [
            ("issue:CMSTZ-1", 101),
            ("issue:CMSDM-2", 102),
        ]
        assert service.persist_interaction.call_count == 2
        service.persist_interaction.assert_any_call(
            "Jira issue CMSTZ-1",
            archi_instance.calls[0]["history"][0][1],
            "Use the documented fix.",
            [],
        )
        service.persist_interaction.assert_any_call(
            "Jira issue CMSDM-2",
            archi_instance.calls[1]["history"][0][1],
            "Use the documented fix.",
            [],
        )

    def test_poll_once_reuses_role_authorization_within_project(self):
        first_mention = jira_interface.JiraComment(
            id="7382883",
            body="[~cmsai] first request.",
            author={"name": "developer"},
            created="",
            updated="",
        )
        second_mention = jira_interface.JiraComment(
            id="7382884",
            body="[~cmsai] second request.",
            author={"name": "developer"},
            created="",
            updated="",
        )
        actors = [{"type": "atlassian-group-role-actor", "name": "jira-devs"}]
        issue_client = _FakeIssueClient(
            issues=[_raw_issue(key="CMSTZ-1"), _raw_issue(key="CMSTZ-2")],
            recent_comments_by_issue={
                "CMSTZ-1": [first_mention],
                "CMSTZ-2": [second_mention],
            },
            project_role_actors=actors,
            role_authorization={"developer": True},
        )
        trigger_store = _FakeTriggerStore(
            denied_keys={"issue:CMSTZ-1", "issue:CMSTZ-2"}
        )
        service = _service(
            issue_client,
            _FakeArchi(),
            respond_to_mentions=True,
            mention_allowed_roles=["Developers"],
            trigger_store=trigger_store,
        )
        service.persist_interaction = Mock(side_effect=[101, 102])

        service.poll_once()

        assert issue_client.project_role_fetches == [("CMSTZ", ["Developers"])]
        assert issue_client.role_authorization_checks == [
            ({"name": "developer"}, actors)
        ]
        assert [claim[0] for claim in trigger_store.claims] == [
            "issue:CMSTZ-1",
            "comment:7382883",
            "issue:CMSTZ-2",
            "comment:7382884",
        ]

    def test_poll_once_refreshes_role_authorization_on_next_poll(self):
        first_mention = jira_interface.JiraComment(
            id="7382883",
            body="[~cmsai] first request.",
            author={"name": "developer"},
            created="",
            updated="",
        )
        second_mention = jira_interface.JiraComment(
            id="7382884",
            body="[~cmsai] request after role removal.",
            author={"name": "developer"},
            created="",
            updated="",
        )
        actors = [{"type": "atlassian-group-role-actor", "name": "jira-devs"}]
        issue_client = _FakeIssueClient(
            issues=[_raw_issue(key="CMSTZ-1")],
            recent_comments_by_issue={"CMSTZ-1": [first_mention]},
            project_role_actors=actors,
            role_authorization={"developer": True},
        )
        trigger_store = _FakeTriggerStore(denied_keys={"issue:CMSTZ-1"})
        service = _service(
            issue_client,
            _FakeArchi(),
            respond_to_mentions=True,
            mention_allowed_roles=["Developers"],
            trigger_store=trigger_store,
        )
        service.persist_interaction = Mock(return_value=101)

        service.poll_once()
        issue_client.recent_comments_by_issue["CMSTZ-1"] = [second_mention]
        issue_client.role_authorization["developer"] = False
        service.poll_once()

        assert issue_client.project_role_fetches == [
            ("CMSTZ", ["Developers"]),
            ("CMSTZ", ["Developers"]),
        ]
        assert [claim[0] for claim in trigger_store.claims] == [
            "issue:CMSTZ-1",
            "comment:7382883",
            "issue:CMSTZ-1",
        ]

    @patch("src.services.jira_ticket_responder.service.psycopg2.extras.execute_values")
    def test_poll_once_e2e_answers_multiple_projects_and_persists(self, execute_values):
        issue_client = _FakeIssueClient(
            issues=[
                _raw_issue(key="CMSTZ-1"),
                _raw_issue(key="CMSDM-2"),
            ]
        )
        pool = _FakeConnectionPool()
        pool.connection = _FakeConnection(
            cursor=_FakeCursor(fetchone_values=[(101,), (102,)])
        )
        trigger_store = _FakeTriggerStore()
        service = _service(issue_client, _FakeArchi(), projects=["CMSTZ", "CMSDM"])
        service.postgres_factory = SimpleNamespace(connection_pool=pool)
        service.trigger_store = trigger_store

        service.poll_once()

        assert issue_client.searches == [
            (["CMSTZ", "CMSDM"], 7, ["Open", "In Progress"])
        ]
        assert issue_client.posted == [
            ("CMSTZ-1", "Use the documented fix.", "Developers"),
            ("CMSDM-2", "Use the documented fix.", "Developers"),
        ]
        assert pool.connection.commits == 2
        assert pool.connection.rollbacks == 0
        assert pool.connection.cursor_instance.executed[0][1][0] == "Jira issue CMSTZ-1"
        assert pool.connection.cursor_instance.executed[1][1][0] == "Jira issue CMSDM-2"
        assert trigger_store.linked_conversations == [
            ("issue:CMSTZ-1", 101),
            ("issue:CMSDM-2", 102),
        ]
        assert execute_values.call_count == 2

    def test_process_issue_posts_before_persisting(self):
        order = []
        issue_client = _FakeIssueClient(order=order)
        archi_instance = _FakeArchi()
        trigger_store = _FakeTriggerStore()
        service = _service(issue_client, archi_instance, trigger_store=trigger_store)
        service.persist_interaction = Mock(
            side_effect=lambda *args: order.append("persist") or 42
        )

        processed = service.process_issue(_raw_issue())

        assert processed is True
        assert archi_instance.calls == [
            {
                "history": [
                    (
                        "User",
                        _expected_archi_prompt(
                            "Suggest a solution to this problem.\n\n"
                            "Issue:\n"
                            "CMSTZ-1\n\n"
                            "Summary:\n"
                            "Broken transfer\n\n"
                            "Status:\n"
                            "Open\n\n"
                            "Description:\n"
                            "Transfers fail with timeout."
                        ),
                    )
                ]
            }
        ]
        assert issue_client.posted == [
            ("CMSTZ-1", "Use the documented fix.", "Developers")
        ]
        assert order == ["post", "persist"]
        assert trigger_store.answered == [("issue:CMSTZ-1", "jira-response-1")]
        assert trigger_store.linked_conversations == [("issue:CMSTZ-1", 42)]
        service.persist_interaction.assert_called_once_with(
            "Jira issue CMSTZ-1",
            archi_instance.calls[0]["history"][0][1],
            "Use the documented fix.",
            [],
        )

    def test_process_issue_posts_answer_with_trace_but_persists_plain_answer(self):
        result = PipelineOutput(
            answer="  Use the documented fix.  ",
            messages=[
                SimpleNamespace(
                    additional_kwargs={"reasoning_content": "Matched symptoms."},
                    tool_calls=[
                        {
                            "id": "call-1",
                            "name": "local_files",
                            "args": {"path": "runbook.md"},
                        }
                    ],
                ),
                SimpleNamespace(tool_call_id="call-1", content="Runbook section 2"),
            ],
            source_documents=[],
        )
        issue_client = _FakeIssueClient()
        archi_instance = _FakeArchi(result=result)
        trigger_store = _FakeTriggerStore()
        service = _service(issue_client, archi_instance, trigger_store=trigger_store)
        service.persist_interaction = Mock(return_value=42)

        processed = service.process_issue(_raw_issue())

        assert processed is True
        assert issue_client.posted == [
            (
                "CMSTZ-1",
                "Use the documented fix.\n\n"
                "{panel:title=Reasoning trace}\n"
                "{noformat}\n"
                "Matched symptoms.\n"
                "{noformat}\n"
                "{panel}\n\n"
                "{panel:title=Tool calls}\n"
                "{noformat}\n"
                "Tool call 1: local_files\n"
                "Input:\n"
                "{\n"
                '  "path": "runbook.md"\n'
                "}\n"
                "\n"
                "Output:\n"
                "Runbook section 2\n"
                "{noformat}\n"
                "{panel}",
                "Developers",
            )
        ]
        assert trigger_store.answered == [("issue:CMSTZ-1", "jira-response-1")]
        service.persist_interaction.assert_called_once_with(
            "Jira issue CMSTZ-1",
            archi_instance.calls[0]["history"][0][1],
            "Use the documented fix.",
            [],
        )

    def test_post_failure_marks_trigger_failed_and_skips_persistence(self):
        order = []
        issue_client = _FakeIssueClient(order=order, fail_post=True)
        archi_instance = _FakeArchi()
        trigger_store = _FakeTriggerStore()
        service = _service(issue_client, archi_instance, trigger_store=trigger_store)
        service.persist_interaction = Mock()

        processed = service.process_issue(_raw_issue())

        assert processed is False
        assert archi_instance.calls == [
            {
                "history": [
                    (
                        "User",
                        _expected_archi_prompt(
                            "Suggest a solution to this problem.\n\n"
                            "Issue:\n"
                            "CMSTZ-1\n\n"
                            "Summary:\n"
                            "Broken transfer\n\n"
                            "Status:\n"
                            "Open\n\n"
                            "Description:\n"
                            "Transfers fail with timeout."
                        ),
                    )
                ]
            }
        ]
        assert order == ["post"]
        assert trigger_store.answered == []
        assert trigger_store.failed == [
            ("issue:CMSTZ-1", "Failed to post Jira comment: post failed")
        ]
        service.persist_interaction.assert_not_called()

    def test_answered_transition_failure_marks_terminal_failed_and_skips_persistence(
        self,
    ):
        issue_client = _FakeIssueClient()
        trigger_store = _FakeTriggerStore(fail_answer_keys={"issue:CMSTZ-1"})
        service = _service(issue_client, _FakeArchi(), trigger_store=trigger_store)
        service.persist_interaction = Mock()

        processed = service.process_issue(_raw_issue())

        assert processed is False
        assert issue_client.posted == [
            ("CMSTZ-1", "Use the documented fix.", "Developers")
        ]
        assert trigger_store.answered == []
        assert trigger_store.failed == [
            (
                "issue:CMSTZ-1",
                "Jira comment was posted but marking trigger answered failed: "
                "answer update failed",
            )
        ]
        assert trigger_store.posted_but_unconfirmed == [
            (
                "issue:CMSTZ-1",
                "issue",
                "CMSTZ-1",
                None,
                "jira-response-1",
                "Jira comment was posted but marking trigger answered failed: "
                "answer update failed",
            )
        ]
        assert trigger_store.linked_conversations == []
        service.persist_interaction.assert_not_called()

    def test_persistence_failure_keeps_trigger_answered_and_records_error(self):
        issue_client = _FakeIssueClient()
        trigger_store = _FakeTriggerStore()
        service = _service(issue_client, _FakeArchi(), trigger_store=trigger_store)
        service.persist_interaction = Mock(side_effect=RuntimeError("db failed"))

        processed = service.process_issue(_raw_issue())

        assert processed is True
        assert issue_client.posted == [
            ("CMSTZ-1", "Use the documented fix.", "Developers")
        ]
        assert trigger_store.answered == [("issue:CMSTZ-1", "jira-response-1")]
        assert trigger_store.linked_conversations == []
        assert trigger_store.last_errors == [
            (
                "issue:CMSTZ-1",
                "Failed to persist Jira interaction after posting comment: db failed",
            )
        ]
        service.persist_interaction.assert_called_once_with(
            "Jira issue CMSTZ-1",
            _expected_archi_prompt(
                "Suggest a solution to this problem.\n\n"
                "Issue:\n"
                "CMSTZ-1\n\n"
                "Summary:\n"
                "Broken transfer\n\n"
                "Status:\n"
                "Open\n\n"
                "Description:\n"
                "Transfers fail with timeout."
            ),
            "Use the documented fix.",
            [],
        )

    def test_archi_empty_answer_skips_posting(self):
        issue_client = _FakeIssueClient()
        trigger_store = _FakeTriggerStore()
        service = _service(
            issue_client, _FakeArchi(answer="   "), trigger_store=trigger_store
        )
        service.persist_interaction = Mock()

        processed = service.process_issue(_raw_issue())

        assert processed is False
        assert issue_client.posted == []
        assert trigger_store.failed == [("issue:CMSTZ-1", "Archi returned no answer.")]
        service.persist_interaction.assert_not_called()

    def test_denied_issue_trigger_skips_without_archi_call(self):
        issue_client = _FakeIssueClient()
        archi_instance = _FakeArchi()
        trigger_store = _FakeTriggerStore(denied_keys={"issue:CMSTZ-1"})
        service = _service(issue_client, archi_instance, trigger_store=trigger_store)
        service.persist_interaction = Mock()

        processed = service.process_issue(_raw_issue())

        assert processed is False
        assert trigger_store.claims == [("issue:CMSTZ-1", "issue", "CMSTZ-1", None)]
        assert archi_instance.calls == []
        assert issue_client.posted == []

    def test_failed_and_answered_issue_triggers_skip_without_archi_call(self):
        issue_client = _FakeIssueClient()
        archi_instance = _FakeArchi()
        trigger_store = _FakeTriggerStore(denied_keys={"issue:CMSTZ-1"})
        service = _service(issue_client, archi_instance, trigger_store=trigger_store)
        service.persist_interaction = Mock()

        processed = service.process_issue(_raw_issue())

        assert processed is False
        assert archi_instance.calls == []
        assert issue_client.posted == []
        service.persist_interaction.assert_not_called()

    def test_claim_failure_skips_without_archi_call(self):
        issue_client = _FakeIssueClient()
        archi_instance = _FakeArchi()
        trigger_store = _FakeTriggerStore(fail_claim_keys={"issue:CMSTZ-1"})
        service = _service(issue_client, archi_instance, trigger_store=trigger_store)
        service.persist_interaction = Mock()

        processed = service.process_issue(_raw_issue())

        assert processed is False
        assert archi_instance.calls == []
        assert issue_client.posted == []
        service.persist_interaction.assert_not_called()

    def test_archi_failure_marks_trigger_failed(self):
        issue_client = _FakeIssueClient()
        archi_instance = _FakeArchi(failure=RuntimeError("model failed"))
        trigger_store = _FakeTriggerStore()
        service = _service(issue_client, archi_instance, trigger_store=trigger_store)
        service.persist_interaction = Mock()

        processed = service.process_issue(_raw_issue())

        assert processed is False
        assert issue_client.posted == []
        assert trigger_store.failed == [
            (
                "issue:CMSTZ-1",
                "Archi failed while answering trigger: model failed",
            )
        ]
        service.persist_interaction.assert_not_called()

    def test_respond_to_mentions_false_does_not_fetch_recent_comments(self):
        comment = jira_interface.JiraComment(
            id="7382883",
            body="[~cmsai] please check this.",
            author={"name": "human-user"},
            created="2026-06-26T10:00:00.000+0200",
            updated="2026-06-26T10:00:00.000+0200",
        )
        issue_client = _FakeIssueClient(recent_comments_by_issue={"CMSTZ-1": [comment]})
        trigger_store = _FakeTriggerStore()
        service = _service(issue_client, _FakeArchi(), trigger_store=trigger_store)
        service.persist_interaction = Mock(return_value=42)

        processed = service.process_issue(_raw_issue())

        assert processed is True
        assert issue_client.recent_comment_fetches == []
        assert trigger_store.claims == [("issue:CMSTZ-1", "issue", "CMSTZ-1", None)]

    def test_respond_to_mentions_true_answers_human_mention_once(self):
        human_mention = jira_interface.JiraComment(
            id="7382883",
            body="[~cmsai] please check this transfer.",
            author={"name": "human-user"},
            created="2026-06-26T10:00:00.000+0200",
            updated="2026-06-26T10:00:00.000+0200",
        )
        service_mention = jira_interface.JiraComment(
            id="7382884",
            body="[~cmsai] service-authored context.",
            author={"name": "cmsai"},
            created="2026-06-26T10:01:00.000+0200",
            updated="2026-06-26T10:01:00.000+0200",
        )
        issue_client = _FakeIssueClient(
            recent_comments_by_issue={"CMSTZ-1": [service_mention, human_mention]}
        )
        trigger_store = _FakeTriggerStore(denied_keys={"issue:CMSTZ-1"})
        service = _service(
            issue_client,
            _FakeArchi(),
            respond_to_mentions=True,
            trigger_store=trigger_store,
        )
        service.persist_interaction = Mock(return_value=42)

        processed = service.process_issue(_raw_issue())

        assert processed is True
        assert issue_client.recent_comment_fetches == ["CMSTZ-1"]
        assert trigger_store.claims == [
            ("issue:CMSTZ-1", "issue", "CMSTZ-1", None),
            ("comment:7382883", "mention_comment", "CMSTZ-1", "7382883"),
        ]
        assert issue_client.posted == [
            ("CMSTZ-1", "Use the documented fix.", "Developers")
        ]
        assert trigger_store.answered == [("comment:7382883", "jira-response-1")]
        service.persist_interaction.assert_called_once()
        assert service.persist_interaction.call_args.args[0] == (
            "Jira comment 7382883 on issue CMSTZ-1"
        )
        mention_prompt = service.persist_interaction.call_args.args[1]
        assert "Triggering Comment:" in mention_prompt
        assert "TRIGGERING COMMENT TO ANSWER" not in mention_prompt
        assert mention_prompt.count("Comment ID: 7382883") == 1
        assert "Comment ID: 7382884" in mention_prompt
        assert "[~cmsai] service-authored context." in mention_prompt

    def test_empty_mention_allowed_roles_allows_any_human_author(self):
        human_mention = jira_interface.JiraComment(
            id="7382883",
            body="[~cmsai] please check this transfer.",
            author={"name": "human-user"},
            created="",
            updated="",
        )
        issue_client = _FakeIssueClient(
            recent_comments_by_issue={"CMSTZ-1": [human_mention]}
        )
        trigger_store = _FakeTriggerStore(denied_keys={"issue:CMSTZ-1"})
        service = _service(
            issue_client,
            _FakeArchi(),
            respond_to_mentions=True,
            mention_allowed_roles=[],
            trigger_store=trigger_store,
        )
        service.persist_interaction = Mock(return_value=42)

        assert service.process_issue(_raw_issue()) is True
        assert issue_client.project_role_fetches == []
        assert trigger_store.claims[-1] == (
            "comment:7382883",
            "mention_comment",
            "CMSTZ-1",
            "7382883",
        )

    def test_mention_allowed_roles_only_answers_author_in_any_configured_role(self):
        allowed_mention = jira_interface.JiraComment(
            id="7382883",
            body="[~cmsai] allowed request.",
            author={"name": "developer"},
            created="",
            updated="",
        )
        denied_mention = jira_interface.JiraComment(
            id="7382884",
            body="[~cmsai] denied request.",
            author={"name": "reporter"},
            created="",
            updated="",
        )
        actors = [{"type": "atlassian-group-role-actor", "name": "jira-devs"}]
        issue_client = _FakeIssueClient(
            recent_comments_by_issue={"CMSTZ-1": [denied_mention, allowed_mention]},
            project_role_actors=actors,
            role_authorization={"developer": True, "reporter": False},
        )
        trigger_store = _FakeTriggerStore(denied_keys={"issue:CMSTZ-1"})
        service = _service(
            issue_client,
            _FakeArchi(),
            respond_to_mentions=True,
            mention_allowed_roles=["Developers", "Administrators"],
            trigger_store=trigger_store,
        )
        service.persist_interaction = Mock(return_value=42)

        assert service.process_issue(_raw_issue()) is True
        assert issue_client.project_role_fetches == [
            ("CMSTZ", ["Developers", "Administrators"])
        ]
        assert [claim[0] for claim in trigger_store.claims] == [
            "issue:CMSTZ-1",
            "comment:7382883",
        ]

    def test_mention_role_lookup_failure_fails_closed(self):
        human_mention = jira_interface.JiraComment(
            id="7382883",
            body="[~cmsai] please check this transfer.",
            author={"name": "developer"},
            created="",
            updated="",
        )
        issue_client = _FakeIssueClient(
            recent_comments_by_issue={"CMSTZ-1": [human_mention]},
            fail_project_roles=True,
        )
        trigger_store = _FakeTriggerStore()
        service = _service(
            issue_client,
            _FakeArchi(),
            respond_to_mentions=True,
            mention_allowed_roles=["Developers"],
            trigger_store=trigger_store,
        )
        service.persist_interaction = Mock(return_value=42)

        assert service.process_issue(_raw_issue()) is True
        assert trigger_store.claims == [("issue:CMSTZ-1", "issue", "CMSTZ-1", None)]
        assert issue_client.posted == [
            ("CMSTZ-1", "Use the documented fix.", "Developers")
        ]

    def test_poll_once_caches_project_role_lookup_failure(self):
        human_mention = jira_interface.JiraComment(
            id="7382883",
            body="[~cmsai] please check this transfer.",
            author={"name": "developer"},
            created="",
            updated="",
        )
        issue_client = _FakeIssueClient(
            issues=[_raw_issue(key="CMSTZ-1"), _raw_issue(key="CMSTZ-2")],
            recent_comments_by_issue={
                "CMSTZ-1": [human_mention],
                "CMSTZ-2": [human_mention],
            },
            fail_project_roles=True,
        )
        trigger_store = _FakeTriggerStore()
        service = _service(
            issue_client,
            _FakeArchi(),
            respond_to_mentions=True,
            mention_allowed_roles=["Developers"],
            trigger_store=trigger_store,
        )
        service.persist_interaction = Mock(side_effect=[101, 102])

        service.poll_once()

        assert issue_client.project_role_fetches == [("CMSTZ", ["Developers"])]
        assert [claim[0] for claim in trigger_store.claims] == [
            "issue:CMSTZ-1",
            "issue:CMSTZ-2",
        ]
        assert issue_client.posted == [
            ("CMSTZ-1", "Use the documented fix.", "Developers"),
            ("CMSTZ-2", "Use the documented fix.", "Developers"),
        ]

    def test_mention_prompt_over_budget_marks_trigger_failed_without_archi_call(self):
        human_mention = jira_interface.JiraComment(
            id="7382883",
            body="[~cmsai] " + ("x" * 1000),
            author={"name": "human-user"},
            created="2026-06-26T10:00:00.000+0200",
            updated="2026-06-26T10:00:00.000+0200",
        )
        issue_client = _FakeIssueClient(
            recent_comments_by_issue={"CMSTZ-1": [human_mention]}
        )
        archi_instance = _FakeArchi()
        trigger_store = _FakeTriggerStore(denied_keys={"issue:CMSTZ-1"})
        service = _service(
            issue_client,
            archi_instance,
            respond_to_mentions=True,
            trigger_store=trigger_store,
            prompt_max_chars=500,
        )
        service.persist_interaction = Mock(return_value=42)

        processed = service.process_issue(_raw_issue())

        assert processed is False
        assert trigger_store.claims == [
            ("issue:CMSTZ-1", "issue", "CMSTZ-1", None),
            ("comment:7382883", "mention_comment", "CMSTZ-1", "7382883"),
        ]
        assert trigger_store.failed[0][0] == "comment:7382883"
        assert trigger_store.failed[0][1].startswith(
            "Jira mention prompt exceeds model-derived prompt budget: prompt_chars="
        )
        assert trigger_store.failed[0][1].endswith(" budget_chars=500")
        assert archi_instance.calls == []
        assert issue_client.posted == []
        service.persist_interaction.assert_not_called()

    def test_service_authored_mentions_are_not_triggers(self):
        service_mention = jira_interface.JiraComment(
            id="7382884",
            body="[~cmsai] service-authored context.",
            author={"name": "cmsai"},
            created="",
            updated="",
        )
        issue_client = _FakeIssueClient(
            recent_comments_by_issue={"CMSTZ-1": [service_mention]}
        )
        trigger_store = _FakeTriggerStore(denied_keys={"issue:CMSTZ-1"})
        service = _service(
            issue_client,
            _FakeArchi(),
            respond_to_mentions=True,
            trigger_store=trigger_store,
        )
        service.persist_interaction = Mock()

        processed = service.process_issue(_raw_issue())

        assert processed is False
        assert issue_client.recent_comment_fetches == ["CMSTZ-1"]
        assert trigger_store.claims == [("issue:CMSTZ-1", "issue", "CMSTZ-1", None)]
        assert issue_client.posted == []
        service.persist_interaction.assert_not_called()

    def test_issue_and_mention_triggers_can_both_answer_in_one_process_call(self):
        human_mention = jira_interface.JiraComment(
            id="7382883",
            body="[~cmsai] please check this transfer.",
            author={"name": "human-user"},
            created="",
            updated="",
        )
        issue_client = _FakeIssueClient(
            recent_comments_by_issue={"CMSTZ-1": [human_mention]}
        )
        trigger_store = _FakeTriggerStore()
        service = _service(
            issue_client,
            _FakeArchi(),
            respond_to_mentions=True,
            trigger_store=trigger_store,
        )
        service.persist_interaction = Mock(side_effect=[101, 102])

        processed = service.process_issue(_raw_issue())

        assert processed is True
        assert trigger_store.claims == [
            ("issue:CMSTZ-1", "issue", "CMSTZ-1", None),
            ("comment:7382883", "mention_comment", "CMSTZ-1", "7382883"),
        ]
        assert len(issue_client.posted) == 2
        assert trigger_store.answered == [
            ("issue:CMSTZ-1", "jira-response-1"),
            ("comment:7382883", "jira-response-1"),
        ]
        assert trigger_store.linked_conversations == [
            ("issue:CMSTZ-1", 101),
            ("comment:7382883", 102),
        ]

    def test_comment_fetch_failure_keeps_issue_trigger_result(self):
        issue_client = _FakeIssueClient(fail_recent_comments=True)
        trigger_store = _FakeTriggerStore()
        service = _service(
            issue_client,
            _FakeArchi(),
            respond_to_mentions=True,
            trigger_store=trigger_store,
        )
        service.persist_interaction = Mock(return_value=42)

        processed = service.process_issue(_raw_issue())

        assert processed is True
        assert issue_client.recent_comment_fetches == ["CMSTZ-1"]
        assert issue_client.posted == [
            ("CMSTZ-1", "Use the documented fix.", "Developers")
        ]
        assert trigger_store.claims == [("issue:CMSTZ-1", "issue", "CMSTZ-1", None)]


class TestJiraTicketResponderPersistence:
    @patch("src.services.jira_ticket_responder.service.psycopg2.extras.execute_values")
    def test_persist_interaction_uses_context_managed_pool_connection(
        self, execute_values
    ):
        service = _service(_FakeIssueClient(), _FakeArchi())
        pool = _FakeConnectionPool()
        service.postgres_factory = SimpleNamespace(connection_pool=pool)

        conversation_id = service.persist_interaction(
            "Jira issue CMSTZ-1",
            "Suggest a solution.",
            "Use the documented fix.",
            [],
        )

        assert conversation_id == 42
        assert pool.connection_context.entered is True
        assert pool.connection_context.exited is True
        assert pool.released == []
        assert pool.connection.commits == 1
        assert pool.connection.rollbacks == 0
        assert pool.connection.cursor_instance.executed[0][1][0] == "Jira issue CMSTZ-1"
        execute_values.assert_called_once()

    @patch("src.services.jira_ticket_responder.service.psycopg2.extras.execute_values")
    def test_persist_interaction_rolls_back_and_exits_context_on_insert_failure(
        self, execute_values
    ):
        service = _service(_FakeIssueClient(), _FakeArchi())
        pool = _FakeConnectionPool()
        service.postgres_factory = SimpleNamespace(connection_pool=pool)
        execute_values.side_effect = RuntimeError("insert failed")

        with pytest.raises(RuntimeError, match="insert failed"):
            service.persist_interaction(
                "Jira issue CMSTZ-1",
                "Suggest a solution.",
                "Use the documented fix.",
                [],
            )

        assert pool.connection_context.entered is True
        assert pool.connection_context.exited is True
        assert pool.released == []
        assert pool.connection.commits == 0
        assert pool.connection.rollbacks == 1


class TestJiraIssueClient:
    @patch("src.interfaces.jira.jira.JIRA")
    def test_constructor_validates_login_before_polling(self, jira_cls):
        jira_cls.return_value.myself.side_effect = RuntimeError("bad auth")

        with pytest.raises(RuntimeError, match="Failed to log in to Jira"):
            jira_interface.JiraIssueClient("https://jira.example/", "pat")

        jira_cls.assert_called_once_with(
            "https://jira.example/", token_auth="pat", timeout=30
        )
        jira_cls.return_value.myself.assert_called_once_with()

    @patch("src.interfaces.jira.jira.JIRA")
    def test_constructor_stores_authenticated_account_id(self, jira_cls):
        jira_cls.return_value.myself.return_value = {"accountId": "service-account-id"}

        client = jira_interface.JiraIssueClient("https://jira.example/", "pat")

        assert client.user_identities == {"accountId": "service-account-id"}
        jira_cls.return_value.myself.assert_called_once_with()

    @patch("src.interfaces.jira.jira.JIRA")
    def test_constructor_stores_data_center_user_key(self, jira_cls):
        jira_cls.return_value.myself.return_value = {
            "key": "service-user-key",
            "name": "service-user-name",
        }

        client = jira_interface.JiraIssueClient("https://jira.example/", "pat")

        assert client.user_identities == {
            "key": "service-user-key",
            "name": "service-user-name",
        }
        jira_cls.return_value.myself.assert_called_once_with()

    @patch("src.interfaces.jira.jira.JIRA")
    def test_constructor_falls_back_to_data_center_user_name(self, jira_cls):
        jira_cls.return_value.myself.return_value = {"name": "service-user-name"}

        client = jira_interface.JiraIssueClient("https://jira.example/", "pat")

        assert client.user_identities == {"name": "service-user-name"}
        jira_cls.return_value.myself.assert_called_once_with()

    @patch("src.interfaces.jira.jira.JIRA")
    def test_constructor_rejects_missing_user_identity(self, jira_cls):
        jira_cls.return_value.myself.return_value = {"displayName": "Service User"}

        with pytest.raises(
            RuntimeError, match="Failed to resolve Jira service account identity"
        ):
            jira_interface.JiraIssueClient("https://jira.example/", "pat")

        jira_cls.return_value.myself.assert_called_once_with()

    def test_search_recent_issues_uses_rolling_lookback_and_does_not_request_comments(
        self,
    ):
        client = object.__new__(jira_interface.JiraIssueClient)
        client.client = SimpleNamespace(search_issues=Mock(return_value=[]))

        list(client.search_recent_issues(["CMSTZ", "IF"], 7, ["Open", 'Blocked "QA"']))

        client.client.search_issues.assert_called_once_with(
            'project in ("CMSTZ", "IF") AND status in ("Open", "Blocked \\"QA\\"") AND updated >= "-7d" ORDER BY updated ASC',
            startAt=0,
            maxResults=100,
            fields=["summary", "description", "status"],
        )

    def test_fetch_recent_comments_reads_newest_fifty_and_converts_comments(self):
        client = object.__new__(jira_interface.JiraIssueClient)
        client.client = SimpleNamespace(
            _get_json=Mock(
                return_value={
                    "comments": [
                        {
                            "id": 7382883,
                            "body": "Please check this [~cmsai].",
                            "author": {"name": "human-user"},
                            "created": "2026-06-26T10:00:00.000+0200",
                            "updated": "2026-06-26T10:01:00.000+0200",
                        },
                        {
                            "body": "Missing IDs are not stable mention triggers.",
                            "author": {"name": "human-user"},
                        },
                        {
                            "id": "7382884",
                            "body": "Comment with defaults.",
                            "author": {"name": "human-user"},
                        },
                        {
                            "id": "7382885",
                            "body": "[~cmsai] no author should be skipped.",
                        },
                        {
                            "id": "7382886",
                            "body": {"content": "[~cmsai]"},
                            "author": {"name": "human-user"},
                        },
                        {
                            "id": "7382887",
                            "body": "[~cmsai] author without stable identity should be skipped.",
                            "author": {"displayName": "Human User"},
                        },
                    ]
                }
            )
        )

        comments = client.fetch_recent_comments("CMSTZ-1")

        assert comments == [
            jira_interface.JiraComment(
                id="7382883",
                body="Please check this [~cmsai].",
                author={"name": "human-user"},
                created="2026-06-26T10:00:00.000+0200",
                updated="2026-06-26T10:01:00.000+0200",
            ),
            jira_interface.JiraComment(
                id="7382884",
                body="Comment with defaults.",
                author={"name": "human-user"},
                created="",
                updated="",
            ),
        ]
        assert jira_interface.JIRA_RECENT_COMMENT_LIMIT == 50
        client.client._get_json.assert_called_once_with(
            "issue/CMSTZ-1/comment",
            params={"startAt": 0, "maxResults": 50, "orderBy": "-created"},
        )

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            ("Please check this [~cmsai].", True),
            ("Please check this [~JIRAUSER106000].", False),
            ("Please check this [~CMSAI].", False),
            ("Please check @cmsai.", False),
        ],
    )
    def test_comment_mentions_authenticated_user_matches_name_only(
        self, body, expected
    ):
        client = object.__new__(jira_interface.JiraIssueClient)
        client.user_identities = {"key": "JIRAUSER106000", "name": "cmsai"}
        comment = jira_interface.JiraComment(
            id="7382883",
            body=body,
            author={"name": "human-user"},
            created="",
            updated="",
        )

        assert client.comment_mentions_authenticated_user(comment) is expected

    def test_comment_authored_by_authenticated_user_uses_jira_identity_match(self):
        client = object.__new__(jira_interface.JiraIssueClient)
        client.user_identities = {
            "key": "JIRAUSER106000",
            "name": "cmsai",
        }
        service_comment = jira_interface.JiraComment(
            id="7382883",
            body="[~cmsai] should not trigger on our own comment.",
            author={"key": "JIRAUSER106000"},
            created="",
            updated="",
        )
        human_comment = jira_interface.JiraComment(
            id="7382884",
            body="[~cmsai] please check this.",
            author={"name": "human-user"},
            created="",
            updated="",
        )

        assert client.comment_authored_by_authenticated_user(service_comment) is True
        assert client.comment_authored_by_authenticated_user(human_comment) is False

    def test_mention_trigger_comment_ignores_service_authored_comment(self):
        client = object.__new__(jira_interface.JiraIssueClient)
        client.user_identities = {
            "key": "JIRAUSER106000",
            "name": "cmsai",
        }
        service_comment = jira_interface.JiraComment(
            id="7382883",
            body="[~cmsai] should not trigger on our own comment.",
            author={"name": "cmsai"},
            created="",
            updated="",
        )
        human_comment = jira_interface.JiraComment(
            id="7382884",
            body="[~cmsai] please check this.",
            author={"name": "human-user"},
            created="",
            updated="",
        )
        unmentioned_comment = jira_interface.JiraComment(
            id="7382885",
            body="Please check this.",
            author={"name": "human-user"},
            created="",
            updated="",
        )

        assert (
            responder_service.is_mention_trigger_comment(client, service_comment)
            is False
        )
        assert (
            responder_service.is_mention_trigger_comment(client, human_comment) is True
        )
        assert (
            responder_service.is_mention_trigger_comment(client, unmentioned_comment)
            is False
        )

    def test_post_comment_uses_role_visibility(self):
        client = object.__new__(jira_interface.JiraIssueClient)
        client.client = SimpleNamespace(
            add_comment=Mock(return_value=SimpleNamespace(id=8001))
        )

        response_comment_id = client.post_comment("CMSTZ-1", "Fix it", "Developers")

        assert response_comment_id == "8001"
        client.client.add_comment.assert_called_once_with(
            "CMSTZ-1",
            "Fix it",
            visibility={"type": "role", "value": "Developers"},
        )

    def test_post_comment_without_role_posts_public_comment(self):
        client = object.__new__(jira_interface.JiraIssueClient)
        client.client = SimpleNamespace(
            add_comment=Mock(return_value=SimpleNamespace(id=8001))
        )

        response_comment_id = client.post_comment("CMSTZ-1", "Fix it", None)

        assert response_comment_id == "8001"
        client.client.add_comment.assert_called_once_with("CMSTZ-1", "Fix it")

    def test_fetch_project_role_actors_returns_actors_for_all_selected_roles(self):
        client = object.__new__(jira_interface.JiraIssueClient)
        client.client = SimpleNamespace(
            project_roles=Mock(
                return_value={
                    "Developers": {"id": "10001"},
                    "Administrators": {"id": "10002"},
                }
            ),
            project_role=Mock(
                side_effect=[
                    SimpleNamespace(raw={"actors": [{"name": "dev-user"}]}),
                    SimpleNamespace(raw={"actors": [{"name": "admin-user"}]}),
                ]
            ),
        )

        actors = client.fetch_project_role_actors(
            "CMSTZ", ["Developers", "Administrators"]
        )

        assert actors == [{"name": "dev-user"}, {"name": "admin-user"}]

    def test_fetch_project_role_actors_rejects_unknown_configured_role(self):
        client = object.__new__(jira_interface.JiraIssueClient)
        client.client = SimpleNamespace(
            project_roles=Mock(return_value={"Developers": {"id": "10001"}})
        )

        with pytest.raises(ValueError, match="Unknown Jira project roles for CMSTZ"):
            client.fetch_project_role_actors("CMSTZ", ["Developers", "Administrators"])

    def test_comment_author_matches_direct_project_role_actor(self):
        client = object.__new__(jira_interface.JiraIssueClient)
        client.client = SimpleNamespace(user=Mock())
        comment = jira_interface.JiraComment(
            id="7382883",
            body="[~cmsai] please check this.",
            author={"key": "JIRAUSER123", "name": "developer"},
            created="",
            updated="",
        )
        actors = [
            {
                "type": "atlassian-user-role-actor",
                "name": "JIRAUSER123",
            }
        ]

        assert client.comment_author_matches_project_role_actors(comment, actors)
        client.client.user.assert_not_called()

    def test_comment_author_matches_group_project_role_actor(self):
        client = object.__new__(jira_interface.JiraIssueClient)
        client.client = SimpleNamespace(
            user=Mock(
                return_value=SimpleNamespace(
                    raw={
                        "groups": {
                            "items": [
                                {"name": "jira-users"},
                                {"name": "jira-developers"},
                            ]
                        }
                    }
                )
            )
        )
        comment = jira_interface.JiraComment(
            id="7382883",
            body="[~cmsai] please check this.",
            author={"key": "JIRAUSER123", "name": "developer"},
            created="",
            updated="",
        )
        actors = [
            {
                "type": "atlassian-group-role-actor",
                "name": "jira-developers",
            }
        ]

        assert client.comment_author_matches_project_role_actors(comment, actors)
        client.client.user.assert_called_once_with("developer", expand="groups")

    def test_comment_author_does_not_match_unrelated_project_role_group(self):
        client = object.__new__(jira_interface.JiraIssueClient)
        client.client = SimpleNamespace(
            user=Mock(
                return_value=SimpleNamespace(
                    raw={"groups": {"items": [{"name": "jira-users"}]}}
                )
            )
        )
        comment = jira_interface.JiraComment(
            id="7382883",
            body="[~cmsai] please check this.",
            author={"key": "JIRAUSER123", "name": "reporter"},
            created="",
            updated="",
        )
        actors = [
            {
                "type": "atlassian-group-role-actor",
                "name": "jira-developers",
            }
        ]

        assert not client.comment_author_matches_project_role_actors(comment, actors)

    def test_post_comment_returns_none_without_exposed_id(self):
        client = object.__new__(jira_interface.JiraIssueClient)
        client.client = SimpleNamespace(add_comment=Mock(return_value=object()))

        response_comment_id = client.post_comment("CMSTZ-1", "Fix it", "Developers")

        assert response_comment_id is None
