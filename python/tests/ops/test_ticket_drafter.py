"""The ticket-triage reply drafter (board d704) and the post_reply live tool, with a fake model and fake
ticket systems: no network, no paid call."""
import pytest

from archi.evaluation.agent import ChatReply, ModelError
from archi.ops.proposers import ProposeRequest
from archi.tools import tickets
from archi.tools._live import ActRefused
from archi.triage.context import read_ticket_context
from archi.triage.drafter import Drafter, DrafterError, Price
from archi.triage.proposer import TriageProposer
from test_ticket_tools import CONTEXT, ENV, approved, endpoint, instance_dir, link, system  # noqa: F401
from test_ticket_triage import NAME, graph_with_arrivals

PRICE = {"input_per_million": 0.10, "output_per_million": 0.50, "cached_input_per_million": 0.01}
CONFIG = {"model": "gpt-6-luna", "price": PRICE, "max_output_tokens": 1000}
KEY_ENV = {"OPENAI_API_KEY": "sk-test-not-a-key"}


class FakeModel:
    """Answers `text` (or raises `error`); records what it was built with and sent."""

    def __init__(self, text="", *, error=None, usage=True, finish="stop", prompt=6000, cached=1000, completion=400):
        self.text, self.error, self.usage, self.finish = text, error, usage, finish
        self.tokens = (prompt, cached, completion)
        self.built, self.sent = [], []

    def __call__(self, url, model, **kw):
        self.built.append((url, model, kw))
        return self

    def complete(self, messages, tools):
        self.sent.append((messages, tools))
        if self.error:
            raise self.error
        prompt, cached, completion = self.tokens
        return ChatReply(self.text, prompt_tokens=prompt, cached_tokens=cached, completion_tokens=completion,
                         finish_reason=self.finish, usage_reported=self.usage)


def context():
    g = graph_with_arrivals()
    d = read_ticket_context(g, "redmine:4820", g.generation, {"doc_subtypes": ["servicenow_kb_article"]})
    return g, d


def propose(model, level, config=None):
    g, d = context()
    drafter = Drafter(CONFIG, env=KEY_ENV, make_model=model)
    proposer = TriageProposer(config, drafter=drafter, read_text=lambda t, gen: "Jobs fail with ENOSPC on /scratch.")
    out = proposer.propose(ProposeRequest(NAME, {"id": "s", "entity": "redmine:4820"}, d["context"], g.generation, level))
    return out, set(d["node_ids"])


GOOD = ("A similar ticket was solved by cleaning the scratch area [tracker_ticket:redmine:4799]. "
        "Please check your quota first.")


def test_the_model_writes_the_reply_from_the_records_it_may_cite():
    model = FakeModel(GOOD)
    out, known = propose(model, 1)
    p = out.proposal
    body = p["payload"]["body"]
    assert "Suggested reply (written by gpt-6-luna; check it before it is posted):" in body and GOOD in body
    assert "tracker_ticket:redmine:4799" in p["evidence_ids"] and set(p["evidence_ids"]) <= known
    (url, name, kw), = model.built
    assert url == "https://api.openai.com/v1" and name == "gpt-6-luna" and kw["max_completion_tokens"] == 1000
    assert kw["api_key"] == "sk-test-not-a-key" and kw["temperature"] is None
    (messages, tools), = model.sent
    assert tools == [] and "[tracker_ticket:redmine:4820]" in messages[1]["content"]
    assert "ENOSPC" in messages[1]["content"]  # the ticket's text, read again from the graph
    assert out.cost == pytest.approx(Price(0.10, 0.50, 0.01).cost(6000, 1000, 400)) and out.cost > 0


@pytest.mark.parametrize("model, why", [
    (FakeModel("See [tracker_ticket:redmine:9999] for the fix."), "cites records the run did not read"),
    (FakeModel(""), "empty reply"),
    (FakeModel(GOOD, finish="length"), "cut off"),
    (FakeModel(error=ModelError("HTTP 500", status=500, sent=True)), "the model call failed"),
])
def test_a_reply_that_cannot_be_used_leaves_the_template_and_says_why(model, why):
    out, _ = propose(model, 1)
    p = out.proposal
    assert "Suggested reply:" in p["payload"]["body"] and "Thank you for the report." in p["payload"]["body"]
    assert "the model reply was not used" in p["summary"] and why in p["summary"]
    assert out.cost > 0  # a sent call is charged even when its reply is not used


def test_a_reply_without_usage_is_charged_at_its_bound_and_addresses_are_removed():
    model = FakeModel(GOOD + " Write to ada@example.org.", usage=False)
    out, _ = propose(model, 1)
    assert "ada@example.org" not in out.proposal["payload"]["body"]
    (messages, _), = model.sent
    chars = len(messages[0]["content"]) + len(messages[1]["content"])
    assert out.cost == pytest.approx(Price(0.10, 0.50, 0.01).cost(chars, 0, 1000))


def test_the_drafter_needs_a_model_a_price_and_its_key():
    with pytest.raises(DrafterError, match="model is required"):
        Drafter({"price": PRICE}, env=KEY_ENV, make_model=FakeModel())
    with pytest.raises(DrafterError, match="price is required"):
        Drafter({"model": "gpt-6-luna"}, env=KEY_ENV, make_model=FakeModel())
    model = FakeModel(GOOD)
    out, _ = propose(model, 1)  # with the key
    drafter = Drafter(CONFIG, env={}, make_model=model)
    with pytest.raises(DrafterError, match="OPENAI_API_KEY"):
        drafter.write({"node_id": "n", "system": "redmine", "key": "1"}, "", [], [], [], ["n"])


def test_at_l2_with_post_replies_the_reply_is_proposed_for_posting():
    config = {"systems": {"redmine": {**endpoint("redmine"), "post_replies": True}}}
    out, _ = propose(FakeModel(GOOD), 2, config)
    p = out.proposal
    assert p["action_class"] == "post_reply" and p["payload"]["reply"] == GOOD
    assert p["payload"]["endpoint"] == endpoint("redmine") and p["payload"]["ticket_url"] == link("redmine", "4820")
    assert "cannot withdraw" in p["rollback"]


def test_without_post_replies_or_a_used_reply_l2_proposes_as_before():
    config = {"systems": {"redmine": endpoint("redmine")}}
    assert propose(FakeModel(GOOD), 2, config)[0].proposal["action_class"] == "assign_ticket"
    config = {"systems": {"redmine": {**endpoint("redmine"), "post_replies": True}}}
    assert propose(FakeModel(""), 2, config)[0].proposal["action_class"] == "assign_ticket"


# --- the live tool --------------------------------------------------------------------------------


def post(system_name, ticket, **extra):
    return {"system": system_name, "ticket": ticket, "ticket_url": link(system_name, ticket),
            "reply": "Please check your quota [tracker_ticket:redmine:4799].", "endpoint": endpoint(system_name),
            **extra}


@pytest.mark.parametrize("payload, key", [
    (post("redmine", "4820"), ("redmine", "4820")),
    (post("jira", "OPS-15"), ("jira", "OPS-15")),
    (post("servicenow", "INC0010003", table="incident"), ("servicenow", "INC0010003")),
])
def test_an_approved_reply_is_posted_once_with_the_approval_named(payload, key):
    s = system()
    out = tickets.post_reply(payload=payload, context=CONTEXT, session=s.session(), env=ENV, verify=approved)
    assert out["status"] == "succeeded" and "esc-abc" in out["result"]
    (note,) = s.ticket(*key)["notes"]
    assert note.startswith("Please check your quota") and "approval by operator-x in escalation esc-abc" in note


def test_post_reply_refuses_without_approval_or_on_another_server():
    s = system()
    with pytest.raises(ActRefused):
        tickets.post_reply(payload=post("redmine", "4820"), context=CONTEXT, session=s.session(), env=ENV,
                           verify=lambda c, p, a: None)
    other = {**post("redmine", "4820"), "ticket_url": "https://evil.example/issues/4820",
             "endpoint": {"url": "https://evil.example", "token_env": "RM_TOKEN"}}
    with pytest.raises(Exception, match="ops.yaml configures"):
        tickets.post_reply(payload=other, context=CONTEXT, session=s.session(), env=ENV, verify=approved)
    assert s.requests == []


def test_a_post_with_no_answer_is_unknown_not_failed():
    s = system()
    s.fail_writes = True
    out = tickets.post_reply(payload=post("redmine", "4820"), context=CONTEXT, session=s.session(), env=ENV,
                             verify=approved)
    assert out["status"] == "unknown" and "check the ticket before posting again" in out["result"]


@pytest.mark.parametrize("payload, key", [
    (post("redmine", "4820"), ("redmine", "4820")),
    (post("jira", "OPS-15"), ("jira", "OPS-15")),
    (post("servicenow", "INC0010003", table="incident"), ("servicenow", "INC0010003")),
])
def test_a_repeat_finds_its_own_note_and_posts_nothing(payload, key):
    """A crash after the post and before the attempt was recorded runs the tool again for the same escalation."""
    s = system()
    for _ in range(2):
        out = tickets.post_reply(payload=payload, context=CONTEXT, session=s.session(), env=ENV, verify=approved)
        assert out["status"] == "succeeded"
    assert len(s.ticket(*key)["notes"]) == 1 and "already posted" in out["result"]
