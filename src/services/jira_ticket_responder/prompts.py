from __future__ import annotations

from typing import Any

from src.interfaces.jira import JiraComment, JiraIssue
from src.utils.logging import get_logger

logger = get_logger(__name__)


JIRA_ANSWERING_INSTRUCTIONS = """You are Archi, a CMS Computing Operations assistant answering Jira tickets.

Answer the ticket the way CompOps operators historically answered similar Jira tickets. Base your answer on retrieved Jira tickets, operator comments, and documentation evidence from the available MCP tools. Do not answer from general knowledge when historical CompOps behavior is available.

Before drafting, extract the operator resolution from the evidence. Do this internally, not as a separate section in the final answer:

- Identify the user's request type and the closest operator-resolved evidence.
- Prefer historical answer evidence from the same Jira project as the current ticket. If the best available historical evidence comes from another Jira project, you may still answer, but explicitly say that the similar issue was usually answered in that other project rather than the current project.
- Separate what the operator told the requester from what the operator or another service actually did.
- Preserve negative policy boundaries, routing decisions, ownership boundaries, exception criteria, limits, prerequisites, and follow-up requirements when they are part of the operator resolution.
- If evidence includes a procedure or documentation page, include the documentation name and URL when available. Do not replace a user-facing URL with an internal OKG node id.
- If evidence conflicts, prefer the closest match by request type, data tier/workflow class, ownership boundary, and policy era. If there is no close match, say the ticket cannot be auto-answered reliably.
- If the visible request asks for current operational status, a specific rule/dataset state, or a concrete owner/account, answer definitively only when the available tools provide that evidence. Otherwise say that a live/operator check is needed instead of guessing from a similar ticket.

Use this decision order:

1. Search for the current ticket and similar historical tickets, starting with the current Jira project when possible. Identify the final operator resolution, not only the first plausible related procedure.
2. If operators historically routed the request to another team, queue, system, or self-service process, keep that routing as the answer. Do not convert a routing decision into an offer for Archi or CompOps to act.
3. If operators historically declined, constrained, or treated the request as an exception, state that boundary first. Then give the approved path or escalation path only if the evidence supports it.
4. If operators historically recommended a self-service solution that the requester can do themselves, recommend that solution. Include the relevant procedure, documentation link, command pattern, caveats, prerequisites, limits, and follow-up expectations when the evidence supports them.
5. If a self-service solution is not supported and operators historically needed missing information before acting, ask the same kind of clarifying question. Ask only for information that is necessary for an operator to proceed.
6. If the historical operator solution is explicitly trackable to a tool Archi can use, use that tool before answering and report the result.
7. If Archi has no suitable tool, or the evidence is not enough to give a complete historical operator-style answer, say that this ticket cannot be auto-answered reliably and that the requester should wait for a CompOps operator.

Do not claim that you performed, will perform, approved, created, granted, staged, invalidated, deleted, retried, or changed anything unless you actually used an available tool that performs that action and the tool result confirms it. When prior tickets show operators performed an action, describe it as historical operator behavior, not as something Archi has done.

Prefer concise operator-style answers. Cite ticket IDs or documentation names when useful, but do not include long retrieval details."""


def build_archi_answer_prompt(ticket_prompt: str) -> str:
    return f"{JIRA_ANSWERING_INSTRUCTIONS}\n\n{ticket_prompt}"


def archi_answer_prompt_overhead_chars() -> int:
    return len(build_archi_answer_prompt(""))


def build_ticket_prompt(issue: JiraIssue) -> str:
    return (
        "Suggest a solution to this problem.\n\n"
        "Issue:\n"
        f"{issue.key}\n\n"
        "Summary:\n"
        f"{issue.summary}\n\n"
        "Status:\n"
        f"{issue.status_name}\n\n"
        "Description:\n"
        f"{issue.description}"
    )


def build_mention_prompt(
    issue: JiraIssue,
    triggering_comment: JiraComment,
    comments: list[JiraComment],
    max_prompt_chars: int,
) -> str:
    if max_prompt_chars <= 0:
        raise ValueError("max_prompt_chars must be a positive integer.")

    prompt_prefix = (
        "Answer the Jira comment in the 'Triggering Comment:' section. "
        "Use the issue fields and recent comments as context.\n\n"
        "Issue:\n"
        f"{issue.key}\n\n"
        "Summary:\n"
        f"{issue.summary}\n\n"
        "Status:\n"
        f"{issue.status_name}\n\n"
        "Description:\n"
        f"{issue.description}\n\n"
        "Triggering Comment:\n"
        f"{format_jira_comment_for_prompt(triggering_comment)}\n\n"
        "Recent comments context (newest first):\n"
    )
    return prompt_prefix + format_jira_comments_context(
        comments,
        triggering_comment.id,
        prompt_prefix_chars=len(prompt_prefix),
        max_prompt_chars=max_prompt_chars,
    )


def format_jira_comments_context(
    comments: list[JiraComment],
    triggering_comment_id: str,
    *,
    prompt_prefix_chars: int,
    max_prompt_chars: int,
) -> str:
    if max_prompt_chars <= 0:
        raise ValueError("max_prompt_chars must be a positive integer.")
    if not comments:
        return "No recent comments were fetched."
    formatted_comments = []
    context_comments = []
    skipped_trigger_comment = False
    for comment in comments:
        if comment.id == triggering_comment_id:
            skipped_trigger_comment = True
        else:
            context_comments.append(comment)

    if not context_comments:
        if skipped_trigger_comment:
            return "No additional recent comments were fetched."
        return "No recent comments were fetched."

    current_chars = 0
    separator = "\n\n"
    omitted_count = 0
    for index, comment in enumerate(context_comments):
        formatted_comment = format_jira_comment_for_prompt(comment)
        next_chars = len(formatted_comment)
        if formatted_comments:
            next_chars += len(separator)
        if prompt_prefix_chars + current_chars + next_chars > max_prompt_chars:
            omitted_count = len(context_comments) - index
            logger.info(
                "Omitted %s older Jira comments from prompt because the "
                "model-derived prompt budget would be exceeded: "
                "prompt_chars=%s next_comment_chars=%s budget_chars=%s",
                omitted_count,
                prompt_prefix_chars + current_chars,
                next_chars,
                max_prompt_chars,
            )
            break
        formatted_comments.append(formatted_comment)
        current_chars += next_chars

    if not formatted_comments:
        return "No additional recent comments fit in the prompt budget."
    return separator.join(formatted_comments)


def format_jira_comment_for_prompt(comment: JiraComment) -> str:
    return (
        f"Comment ID: {comment.id}\n"
        f"Author: {format_jira_comment_author(comment.author)}\n"
        f"Created: {comment.created}\n"
        f"Updated: {comment.updated}\n"
        "Body:\n"
        f"{comment.body}"
    )


def format_jira_comment_author(author: dict[str, Any]) -> str:
    for field in ("displayName", "name", "key", "accountId"):
        value = str(author.get(field) or "").strip()
        if value:
            return f"{field}={value}"
    return "unknown"
