"""A model-written reply for ticket-triage (board d704: "add a Luna reply drafter").

With `proposer.drafter` set in ops.yaml, the "Suggested reply" of a triage draft is written by a model from
the records the run already read: the ticket (its text is read again here from the pinned generation, since
the run's context leaves it out of the run history), the related tickets, the documentation excerpts and the
ops-record precedents, each with its node id. One call per run, with no tools.

    automations:
      ticket-triage:
        proposer:
          kind: ticket-triage
          drafter:
            model: gpt-6-luna
            url: https://api.openai.com/v1      # default; OpenAI's Responses API
            api_key_env: OPENAI_API_KEY         # default; the name of the variable, never the key
            max_output_tokens: 1000             # default
            price: {input_per_million: 0.10, output_per_million: 0.50, cached_input_per_million: 0.01}

`price` is required: the run's cost is recorded against the pack budget (cost_per_run, cost_per_day), and a
drafter whose cost cannot be counted is refused. A reply the model gives without token usage is charged at
its bound (every input character as a token, plus `max_output_tokens`).

The reply must cite at least one record as `[node id]`, only records the run read, and hold no link that
is not in those records. A reply that breaks any of these, or a model that fails, is not used: the draft keeps
the template reply and its summary says why. These checks constrain the reply's form, not its truth: text in a
ticket can still steer what the model writes, so the person who approves `post_reply` reads the reply (the
escalation shows it as it will be posted) and is the check on its content. Ops records are not shown to the
model: the reply goes to the person who opened the ticket. Email
addresses are removed from the reply. Posting it on the ticket is a separate L2 action (`post_reply`,
archi.tools.tickets) that runs only after a person approves.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from archi.enrichment.anonymizer import redact_email_addresses

DEFAULT_URL = "https://api.openai.com/v1"
DEFAULT_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_MAX_OUTPUT = 1000
MAX_TEXT = 4000  # characters of the ticket text, and of each excerpt, sent to the model
CITED = re.compile(r"\[([A-Za-z_][A-Za-z0-9_]*:[^\]\s]+)\]")
URL = re.compile(r"https?://[^\s)>\]]+")

INSTRUCTIONS = """\
You write the first reply to a support ticket, for an operator to review before it is posted on the ticket.
Use only the records below. Cite every fact you take from a record with its id in square brackets, exactly
as given, for example [tracker_ticket:redmine:4799]. Do not cite anything else. Do not include any link that is not
in the records. If the records do not answer
the ticket, say what the operators need to know to help, and do not guess. Do not promise actions or dates.
Write plain text, at most 200 words, addressed to the person who opened the ticket. Do not sign it."""


class DrafterError(RuntimeError):
    """The drafter is configured wrongly, or its reply cannot be used."""


@dataclass(frozen=True)
class Reply:
    text: str
    cited: tuple[str, ...]
    cost: float


@dataclass(frozen=True)
class Price:
    input_per_million: float
    output_per_million: float
    cached_input_per_million: float

    def cost(self, prompt: int, cached: int, completion: int) -> float:
        fresh = max(prompt - cached, 0)
        return (fresh * self.input_per_million + cached * self.cached_input_per_million
                + completion * self.output_per_million) / 1e6


def _price(config: Mapping[str, Any]) -> Price:
    p = config.get("price")
    if not isinstance(p, Mapping):
        raise DrafterError("proposer.drafter.price is required ({input_per_million, output_per_million, "
                           "cached_input_per_million}): the run's cost must be counted against the budget")
    try:
        inp, out = float(p["input_per_million"]), float(p["output_per_million"])
        cached = float(p.get("cached_input_per_million", inp))
    except (KeyError, TypeError, ValueError) as exc:
        raise DrafterError(f"proposer.drafter.price is incomplete: {exc}") from None
    if min(inp, out, cached) < 0:
        raise DrafterError("proposer.drafter.price may not be negative")
    return Price(inp, out, cached)


def _clip(text: Any, limit: int = MAX_TEXT) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit] + " ..."


def records_text(ticket: Mapping[str, Any], description: str, related: Sequence[Mapping[str, Any]],
                 docs: Sequence[Mapping[str, Any]], precedents: Sequence[Mapping[str, Any]]) -> str:
    """The user message: the ticket and the records the reply may cite, each with its node id."""
    lines = [f"Ticket [{ticket['node_id']}] {ticket['system']} {ticket['key']}: {ticket.get('subject') or ''}",
             f"Status: {ticket.get('status') or '-'}", f"Text: {_clip(description) or '(no text)'}", "",
             "Related tickets:" if related else "Related tickets: none"]
    for r in related:
        lines.append(f"- [{r['node_id']}] {r['system']} {r['key']}: {r.get('subject') or ''} "
                     f"(status {r.get('status') or '-'}; resolution: {_clip(r.get('resolution'), 600) or 'none'})")
    lines.append("Documentation:" if docs else "Documentation: none")
    for d in docs:
        link = f" <{d['url']}>" if d.get("url") else ""
        lines.append(f"- [{d['node_id']}] {d.get('title') or ''}{link}: {_clip(d.get('excerpt'), 1200)}")
    lines.append("Earlier operational records:" if precedents else "Earlier operational records: none")
    for p in precedents:
        lines.append(f"- [{p['node_id']}] {p.get('title') or ''}: {_clip(p.get('summary'), 600)}")
    return "\n".join(lines)


class Drafter:
    """Writes the reply; `make_model(url, model, api_key=..., max_completion_tokens=...)` is
    archi.evaluation.agent.make_model, replaced by a fake in the tests."""

    def __init__(self, config: Mapping[str, Any], *, env: Optional[Mapping[str, str]] = None,
                 make_model: Optional[Callable[..., Any]] = None):
        self.model_name = str(config.get("model") or "")
        if not self.model_name:
            raise DrafterError("proposer.drafter.model is required")
        self.url = str(config.get("url") or DEFAULT_URL)
        self.key_env = str(config.get("api_key_env") or DEFAULT_KEY_ENV)
        self.max_output = int(config.get("max_output_tokens") or DEFAULT_MAX_OUTPUT)
        self.price = _price(config)
        self.env = os.environ if env is None else env
        if make_model is None:
            from archi.evaluation.agent import make_model
        self._make_model = make_model

    def bound(self, prompt_chars: int) -> float:
        """The most a call can cost: every character a token, and the whole output allowance."""
        return self.price.cost(prompt_chars, 0, self.max_output)

    def write(self, ticket: Mapping[str, Any], description: str, related: Sequence[Mapping[str, Any]],
              docs: Sequence[Mapping[str, Any]], precedents: Sequence[Mapping[str, Any]],
              allowed: Sequence[str]) -> Reply:
        """One model call. Raises DrafterError (with the cost already spent, `.cost`) when the reply cannot
        be used."""
        key = self.env.get(self.key_env)
        if not key:
            raise DrafterError(f"environment variable {self.key_env} (the model's API key) is not set")
        user = records_text(ticket, description, related, docs, precedents)
        model = self._make_model(self.url, self.model_name, api_key=key, temperature=None,
                                 max_completion_tokens=self.max_output)
        messages = [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": user}]
        try:
            reply = model.complete(messages, [])
        except Exception as exc:  # the request may have been billed: charge the bound when it was sent
            err = DrafterError(f"the model call failed ({type(exc).__name__}: {str(exc)[:300]})")
            err.cost = self.bound(len(INSTRUCTIONS) + len(user)) if getattr(exc, "sent", True) else 0.0
            raise err from None
        if getattr(reply, "usage_reported", True):
            cost = self.price.cost(reply.prompt_tokens, reply.cached_tokens, reply.completion_tokens)
        else:
            cost = self.bound(len(INSTRUCTIONS) + len(user))
        text = redact_email_addresses(str(reply.content or "")).strip()
        cited = tuple(dict.fromkeys(CITED.findall(text)))
        unknown = [c for c in cited if c not in set(allowed)]
        links = [u.rstrip(".,;:") for u in URL.findall(text)]
        foreign = [u for u in links if u not in user]
        problem = ("the model gave an empty reply" if not text
                   else f"the reply cites records the run did not read: {', '.join(unknown[:5])}" if unknown
                   else "the reply cites no record" if not cited
                   else f"the reply has links that are in no record: {', '.join(foreign[:3])}" if foreign
                   else "the reply was cut off at max_output_tokens" if reply.finish_reason == "length"
                   else "")
        if problem:
            err = DrafterError(problem)
            err.cost = cost
            raise err
        return Reply(text, cited, cost)
