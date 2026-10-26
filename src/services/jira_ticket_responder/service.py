from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import psycopg2.extras

from src.interfaces import jira as jira_interface
from src.utils.conversation_service import Message
from src.utils.logging import get_logger
from src.utils.postgres_service_factory import PostgresServiceFactory
from src.utils.sql import SQL_CREATE_CONVERSATION, SQL_INSERT_CONVO

from . import config as responder_config
from . import formatting as responder_formatting
from . import prompts as responder_prompts
from . import store as trigger_store_module

logger = get_logger(__name__)


class JiraTicketResponderService:
    def __init__(
        self,
        *,
        config: responder_config.JiraServiceConfig,
        issue_client: jira_interface.JiraIssueClient,
        archi_instance: Any,
        postgres_factory: PostgresServiceFactory,
        trigger_store: trigger_store_module.JiraTriggerStore,
        agent_config: responder_config.JiraAgentConfig,
    ) -> None:
        self.config = config
        self.issue_client = issue_client
        self.archi = archi_instance
        self.postgres_factory = postgres_factory
        self.trigger_store = trigger_store
        self.agent_config = agent_config

    def poll_once(self) -> None:
        role_actor_cache: dict[
            tuple[str, tuple[str, ...]], Optional[list[dict[str, Any]]]
        ] = {}
        author_role_matches: dict[tuple[str, ...], bool] = {}
        for raw_issue in self.issue_client.search_recent_issues(
            self.config.projects,
            self.config.lookback_days,
            self.config.eligible_statuses,
        ):
            issue_key = str(getattr(raw_issue, "key", "<unknown>"))
            try:
                self.process_issue(
                    raw_issue,
                    role_actor_cache=role_actor_cache,
                    author_role_matches=author_role_matches,
                )
            except Exception:
                logger.error(
                    "Failed to process Jira issue %s", issue_key, exc_info=True
                )

    def process_issue(
        self,
        raw_issue: Any,
        *,
        role_actor_cache: Optional[
            dict[tuple[str, tuple[str, ...]], Optional[list[dict[str, Any]]]]
        ] = None,
        author_role_matches: Optional[dict[tuple[str, ...], bool]] = None,
    ) -> bool:
        role_actor_cache = role_actor_cache if role_actor_cache is not None else {}
        author_role_matches = (
            author_role_matches if author_role_matches is not None else {}
        )
        issue = jira_interface.extract_issue(raw_issue)
        if not is_issue_eligible(issue, self.config.eligible_statuses):
            return False

        answered_any = self._answer_issue_trigger(issue)
        if not self.config.respond_to_mentions:
            return answered_any

        try:
            comments = self.issue_client.fetch_recent_comments(issue.key)
        except Exception:
            logger.error(
                "Failed to fetch Jira comments for mention scanning on issue %s.",
                issue.key,
                exc_info=True,
            )
            return answered_any

        for comment in comments:
            if not is_mention_trigger_comment(self.issue_client, comment):
                continue
            if self.config.mention_allowed_roles:
                project_key = jira_project_key(issue.key)
                role_cache_key = (
                    project_key,
                    tuple(self.config.mention_allowed_roles),
                )
                if role_cache_key not in role_actor_cache:
                    try:
                        role_actor_cache[role_cache_key] = (
                            self.issue_client.fetch_project_role_actors(
                                project_key,
                                self.config.mention_allowed_roles,
                            )
                        )
                    except Exception:
                        role_actor_cache[role_cache_key] = None
                        logger.error(
                            "Failed to resolve configured Jira mention roles for issue %s.",
                            issue.key,
                            exc_info=True,
                        )
                        return answered_any
                role_actors = role_actor_cache[role_cache_key]
                if role_actors is None:
                    return answered_any
                author_key = (
                    project_key,
                    *self.config.mention_allowed_roles,
                    *(
                        jira_interface.jira_user_identity_value(comment.author, field)
                        for field in jira_interface.JIRA_USER_IDENTITY_FIELDS
                    ),
                )
                if author_key not in author_role_matches:
                    try:
                        author_role_matches[author_key] = (
                            self.issue_client.comment_author_matches_project_role_actors(
                                comment,
                                role_actors,
                            )
                        )
                    except Exception:
                        logger.error(
                            "Failed to authorize Jira mention comment %s on issue %s.",
                            comment.id,
                            issue.key,
                            exc_info=True,
                        )
                        author_role_matches[author_key] = False
                if not author_role_matches[author_key]:
                    logger.info(
                        "Skipping Jira mention comment %s on issue %s because its "
                        "author is not in an allowed project role.",
                        comment.id,
                        issue.key,
                    )
                    continue
            answered_any = (
                self._answer_mention_trigger(issue, comment, comments) or answered_any
            )
        return answered_any

    def _answer_issue_trigger(self, issue: jira_interface.JiraIssue) -> bool:
        return self._answer_trigger(
            trigger_key=issue_trigger_key(issue.key),
            trigger_type="issue",
            issue=issue,
            trigger_comment_id=None,
            prompt=responder_prompts.build_ticket_prompt(issue),
            conversation_title=f"Jira issue {issue.key}",
        )

    def _answer_mention_trigger(
        self,
        issue: jira_interface.JiraIssue,
        comment: jira_interface.JiraComment,
        comments: list[jira_interface.JiraComment],
    ) -> bool:
        trigger_key = comment_trigger_key(comment.id)
        trigger_type = "mention_comment"
        if not self._claim_trigger(
            trigger_key=trigger_key,
            trigger_type=trigger_type,
            issue=issue,
            trigger_comment_id=comment.id,
        ):
            return False

        prompt_payload_budget = max(
            1,
            self.agent_config.prompt_max_chars
            - responder_prompts.archi_answer_prompt_overhead_chars(),
        )
        prompt = responder_prompts.build_mention_prompt(
            issue,
            comment,
            comments,
            prompt_payload_budget,
        )
        archi_prompt = responder_prompts.build_archi_answer_prompt(prompt)
        if len(archi_prompt) > self.agent_config.prompt_max_chars:
            last_error = (
                "Jira mention prompt exceeds model-derived prompt budget: "
                f"prompt_chars={len(archi_prompt)} "
                f"budget_chars={self.agent_config.prompt_max_chars}"
            )
            self._mark_trigger_failed(trigger_key, last_error)
            logger.warning(
                "Skipping Jira responder trigger %s because %s",
                trigger_key,
                last_error,
            )
            return False

        return self._answer_claimed_trigger(
            trigger_key=trigger_key,
            trigger_type=trigger_type,
            issue=issue,
            trigger_comment_id=comment.id,
            prompt=prompt,
            conversation_title=f"Jira comment {comment.id} on issue {issue.key}",
        )

    def _claim_trigger(
        self,
        *,
        trigger_key: str,
        trigger_type: str,
        issue: jira_interface.JiraIssue,
        trigger_comment_id: Optional[str],
    ) -> bool:
        try:
            should_answer = self.trigger_store.claim_trigger(
                trigger_key=trigger_key,
                trigger_type=trigger_type,
                issue_key=issue.key,
                trigger_comment_id=trigger_comment_id,
            )
        except Exception:
            logger.error(
                "Failed to claim Jira responder trigger %s.", trigger_key, exc_info=True
            )
            return False
        if not should_answer:
            logger.debug("Skipping Jira responder trigger %s.", trigger_key)
            return False
        return True

    def _answer_trigger(
        self,
        *,
        trigger_key: str,
        trigger_type: str,
        issue: jira_interface.JiraIssue,
        trigger_comment_id: Optional[str],
        prompt: str,
        conversation_title: str,
    ) -> bool:
        if not self._claim_trigger(
            trigger_key=trigger_key,
            trigger_type=trigger_type,
            issue=issue,
            trigger_comment_id=trigger_comment_id,
        ):
            return False

        return self._answer_claimed_trigger(
            trigger_key=trigger_key,
            trigger_type=trigger_type,
            issue=issue,
            trigger_comment_id=trigger_comment_id,
            prompt=prompt,
            conversation_title=conversation_title,
        )

    def _answer_claimed_trigger(
        self,
        *,
        trigger_key: str,
        trigger_type: str,
        issue: jira_interface.JiraIssue,
        trigger_comment_id: Optional[str],
        prompt: str,
        conversation_title: str,
    ) -> bool:
        archi_prompt = responder_prompts.build_archi_answer_prompt(prompt)
        try:
            result = self.archi(history=[("User", archi_prompt)])
        except Exception as exc:
            self._mark_trigger_failed(
                trigger_key,
                responder_formatting.trigger_error_message(
                    "Archi failed while answering trigger", exc
                ),
            )
            logger.error(
                "Archi failed while answering Jira trigger %s.",
                trigger_key,
                exc_info=True,
            )
            return False

        answer = responder_formatting.extract_answer(result)
        if answer is None:
            self._mark_trigger_failed(trigger_key, "Archi returned no answer.")
            logger.warning(
                "Skipping Jira trigger %s because Archi returned no answer.",
                trigger_key,
            )
            return False

        jira_comment_body = responder_formatting.build_jira_comment_body(answer, result)
        try:
            response_comment_id = self.issue_client.post_comment(
                issue.key,
                jira_comment_body,
                self.config.visible_to_role,
            )
        except Exception as exc:
            self._mark_trigger_failed(
                trigger_key,
                responder_formatting.trigger_error_message(
                    "Failed to post Jira comment", exc
                ),
            )
            logger.error(
                "Failed to post Jira comment for trigger %s.",
                trigger_key,
                exc_info=True,
            )
            return False

        if not self._mark_trigger_answered(
            trigger_key=trigger_key,
            trigger_type=trigger_type,
            issue_key=issue.key,
            trigger_comment_id=trigger_comment_id,
            response_comment_id=response_comment_id,
        ):
            return False
        try:
            source_documents = getattr(result, "source_documents", []) or []
            conversation_id = self.persist_interaction(
                conversation_title, archi_prompt, answer, source_documents
            )
            self.trigger_store.link_conversation(trigger_key, conversation_id)
        except Exception as exc:
            logger.error(
                "Failed to persist Jira interaction for trigger %s after posting comment.",
                trigger_key,
                exc_info=True,
            )
            self._record_trigger_error(
                trigger_key,
                responder_formatting.trigger_error_message(
                    "Failed to persist Jira interaction after posting comment", exc
                ),
            )
        return True

    def _mark_trigger_answered(
        self,
        *,
        trigger_key: str,
        trigger_type: str,
        issue_key: str,
        trigger_comment_id: Optional[str],
        response_comment_id: Optional[str],
    ) -> bool:
        try:
            self.trigger_store.mark_answered(trigger_key, response_comment_id)
            return True
        except Exception as exc:
            logger.error(
                "Failed to mark Jira responder trigger %s answered after posting.",
                trigger_key,
                exc_info=True,
            )
            last_error = responder_formatting.trigger_error_message(
                "Jira comment was posted but marking trigger answered failed", exc
            )
            try:
                self.trigger_store.mark_posted_but_unconfirmed(
                    trigger_key=trigger_key,
                    trigger_type=trigger_type,
                    issue_key=issue_key,
                    trigger_comment_id=trigger_comment_id,
                    response_comment_id=response_comment_id,
                    last_error=last_error,
                )
            except Exception:
                logger.error(
                    "Failed to mark Jira responder trigger %s terminal after posted comment.",
                    trigger_key,
                    exc_info=True,
                )
            return False

    def _mark_trigger_failed(self, trigger_key: str, last_error: str) -> None:
        try:
            self.trigger_store.mark_failed(trigger_key, last_error)
        except Exception:
            logger.error(
                "Failed to mark Jira responder trigger %s failed.",
                trigger_key,
                exc_info=True,
            )

    def _record_trigger_error(self, trigger_key: str, last_error: str) -> None:
        try:
            self.trigger_store.record_last_error(trigger_key, last_error)
        except Exception:
            logger.error(
                "Failed to record Jira responder trigger %s error.",
                trigger_key,
                exc_info=True,
            )

    def persist_interaction(
        self,
        conversation_title: str,
        prompt: str,
        answer: str,
        source_documents: Iterable[Any],
    ) -> int:
        now = datetime.now(timezone.utc)
        client_id = "jira"
        archi_version = os.getenv("APP_VERSION", "unknown")
        link, context = responder_formatting.format_source_context(source_documents)

        with self.postgres_factory.connection_pool.get_connection() as conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        SQL_CREATE_CONVERSATION,
                        (conversation_title, now, now, client_id, archi_version, None),
                    )
                    conversation_id = cursor.fetchone()[0]
                    messages = [
                        Message(
                            conversation_id=conversation_id,
                            sender="User",
                            content=prompt,
                            ts=now,
                            model_used=self.agent_config.model_provider,
                            pipeline_used=self.agent_config.agent_class,
                            archi_service="Jira",
                        ),
                        Message(
                            conversation_id=conversation_id,
                            sender="archi",
                            content=answer,
                            link=link,
                            context=context,
                            ts=now,
                            model_used=self.agent_config.model_provider,
                            pipeline_used=self.agent_config.agent_class,
                            archi_service="Jira",
                        ),
                    ]
                    values = [
                        (
                            message.archi_service,
                            message.conversation_id,
                            message.sender,
                            message.content,
                            message.link or "",
                            message.context or "",
                            message.ts,
                            message.model_used,
                            message.pipeline_used,
                        )
                        for message in messages
                    ]
                    psycopg2.extras.execute_values(cursor, SQL_INSERT_CONVO, values)
                conn.commit()
                return int(conversation_id)
            except Exception:
                conn.rollback()
                raise


def is_issue_eligible(
    issue: jira_interface.JiraIssue, eligible_statuses: Iterable[str]
) -> bool:
    if issue.status_name not in eligible_statuses:
        logger.debug(
            "Skipping Jira issue %s with status %s.", issue.key, issue.status_name
        )
        return False
    return True


def is_mention_trigger_comment(
    issue_client: jira_interface.JiraIssueClient,
    comment: jira_interface.JiraComment,
) -> bool:
    return issue_client.comment_mentions_authenticated_user(
        comment
    ) and not issue_client.comment_authored_by_authenticated_user(comment)


def issue_trigger_key(issue_key: str) -> str:
    return f"issue:{issue_key}"


def comment_trigger_key(comment_id: str) -> str:
    return f"comment:{comment_id}"


def jira_project_key(issue_key: str) -> str:
    project_key, separator, issue_number = issue_key.rpartition("-")
    if not separator or not project_key or not issue_number:
        raise ValueError(f"Invalid Jira issue key: {issue_key}")
    return project_key
