"""req.r3-context-quality.context-eval, the raw-document baseline: the agent arms' loop, caps,
spend ledger, retries, gold resolution, scorer and judge, all on fakes (no model call, no
network).

The sandbox itself is tested in test_raw_sandbox.py and the corpus in test_raw_corpus.py.
"""
import asyncio
import json
from pathlib import Path

import pytest
import yaml

from archi.bundles.cli import main
from archi.evaluation import agent as agent_mod
from archi.evaluation import raw_arm
from archi.evaluation.agent import ChatReply, ModelError
from archi.evaluation.context import parse_context
from archi.evaluation.raw_arm import (
    AGENT_BASH,
    AGENT_GRAPH,
    ANSWERED,
    CAPPED,
    ERROR,
    JUDGE_OFF,
    NO_ANSWER_KEYS,
    OUTPUT_LIMIT,
    STOPPED,
    AgentPlan,
    Caps,
    FatalModelError,
    GoldError,
    GoldRecord,
    Judge,
    Ledger,
    Metered,
    Price,
    SpendError,
    ToolEvent,
    build_prompt,
    content_fingerprints,
    resolve_gold,
    run_agent_arms,
    run_try,
    score_evidence,
)
from archi.evaluation.raw_corpus import verify_corpus
from evaluation.raw_support import (
    NODES,
    FakeMcp,
    ScriptedModel,
    failing,
    http_error,
    reply,
    search_rows,
    write_corpus,
)

PRICE = Price(1.0, 10.0)  # $1 / $10 per million tokens
NO_WAIT = Caps(min_call_seconds=0)


def metered(tmp_path, script, *, cap=10.0, price=PRICE, max_output=1000, slept=None):
    ledger = Ledger(tmp_path / "ledger.jsonl", cap)
    return Metered(ScriptedModel(script), price, ledger, max_output,
                   sleep=(slept.append if slept is not None else (lambda s: None)))


def one_try(model, tools, *, caps=Caps(), clock=None, arm=AGENT_GRAPH):
    specs = asyncio.run(tools.list_tools())
    kwargs = {"clock": clock} if clock else {}
    return asyncio.run(run_try(model, tools, specs, arm=arm, system="sys", question_id="q1",
                               question="Why do transfers fail?", attempt=1, caps=caps, **kwargs))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


# --- the loop, the caps and the ledger -------------------------------------------------------


def test_the_loop_runs_tools_until_the_model_answers_and_ledgers_every_call(tmp_path):
    tools = FakeMcp({"transfer failures": search_rows("jira:DEMO-1")})

    def script(messages, turn):
        if turn == 1:
            return reply("", ("search", {"query": "transfer failures"}))
        if turn == 2:
            assert messages[-1]["role"] == "tool" and "jira:DEMO-1" in messages[-1]["content"]
            return reply("", ("inspect", {"target": {"kind": "node", "node_id": "jira:DEMO-1"}}))
        return reply("Stuck transfers.\nSources: jira:DEMO-1", prompt=2000, completion=50)

    model = metered(tmp_path, script)
    result = one_try(model, tools)
    assert result.status == ANSWERED and result.turns == 3 and result.tool_calls == 2
    assert result.answer.startswith("Stuck transfers") and all(e.ok for e in result.events)
    ledger = rows(tmp_path / "ledger.jsonl")
    assert len(ledger) == 3 and {r["outcome"] for r in ledger} == {"answered"}
    assert {r["arm"] for r in ledger} == {AGENT_GRAPH} and {r["model"] for r in ledger} == {"fake-model"}
    assert ledger[0]["prompt_tokens"] == 1000 and ledger[0]["completion_tokens"] == 100
    assert ledger[0]["cost_usd"] == pytest.approx((1000 * 1 + 100 * 10) / 1e6)
    assert result.cost == pytest.approx(sum(r["cost_usd"] for r in ledger))


def test_the_turn_cap_ends_a_try(tmp_path):
    model = metered(tmp_path, lambda m, t: reply("", ("search", {"query": "x"})))
    result = one_try(model, FakeMcp(), caps=Caps(max_turns=3))
    assert result.status == CAPPED and result.reason == "3 model turns" and result.turns == 3


def test_the_tool_call_cap_ends_a_try_mid_turn(tmp_path):
    model = metered(tmp_path, lambda m, t: reply("", *[("search", {"query": str(i)}) for i in range(5)]))
    result = one_try(model, FakeMcp(), caps=Caps(max_tool_calls=7))
    assert result.status == CAPPED and result.reason == "7 tool calls" and result.tool_calls == 7


def test_the_time_cap_ends_a_try(tmp_path):
    now = [0.0]

    def clock():
        now[0] += 400.0
        return now[0]

    model = metered(tmp_path, lambda m, t: reply("", ("search", {"query": "x"})))
    result = one_try(model, FakeMcp(), caps=Caps(try_seconds=900), clock=clock)
    assert result.status == CAPPED and result.reason.startswith("900 seconds")


def test_no_call_starts_with_under_a_minute_left(tmp_path):
    model = metered(tmp_path, lambda m, t: reply("done"))
    result = one_try(model, FakeMcp(), caps=Caps(try_seconds=50))
    assert result.status == CAPPED and "under 60 s left" in result.reason and model.model.calls == 0


def test_a_model_call_never_waits_longer_than_the_try_has_left(tmp_path):
    seen = []

    class Slow(ScriptedModel):
        timeout = 600.0

        def complete(self, messages, tools):
            seen.append(self.timeout)
            return super().complete(messages, tools)

    model = Metered(Slow(lambda m, t: reply("done")), PRICE, Ledger(tmp_path / "l.jsonl", 1.0), 100)
    one_try(model, FakeMcp(), caps=Caps(try_seconds=90))
    assert seen and seen[0] <= 90 and model.model.timeout == 600.0


def test_a_tool_response_is_cut_to_the_output_limit_both_arms(tmp_path):
    tools = FakeMcp({"big": "x" * 5000})
    model = metered(tmp_path, lambda m, t: reply("", ("search", {"query": "big"})) if t == 1 else reply("done"))
    result = one_try(model, tools, caps=Caps(output_bytes=1000))
    assert len(result.events[0].text) < 1100 and result.events[0].text.endswith("[cut at 1000 bytes]")
    assert result.events[0].ok  # the visible part is still a response


def test_an_empty_answer_cut_by_the_output_limit_is_a_named_failure(tmp_path):
    model = metered(tmp_path, lambda m, t: ChatReply("", (), 10, 1000, finish_reason="length"))
    result = one_try(model, FakeMcp())
    assert result.status == ERROR and result.reason == OUTPUT_LIMIT and result.answer == ""


@pytest.mark.parametrize("cap", [None, 0, -1, float("nan"), float("inf"), True])
def test_the_ledger_refuses_to_start_without_a_cap(tmp_path, cap):
    with pytest.raises(SpendError, match="without a spend cap"):
        Ledger(tmp_path / "ledger.jsonl", cap)


def test_a_price_is_dollars_per_million_tokens_and_a_tiny_one_is_refused():
    assert Price.parse("1.25/10") == Price(1.25, 10.0)
    assert Price.parse("1.25/10/0.125").cost(1_000_000, 0, 1_000_000) == pytest.approx(0.125)
    for bad in ("", "1", "a/b", "1/2/3/4", "-1/2"):
        with pytest.raises(SpendError):
            Price.parse(bad)
    for tiny in ("0/0", "0.000001/0.00001", "1/0.01"):
        with pytest.raises(SpendError, match="--allow-low-price"):
            Price.parse(tiny).check_plausible(allow_low=False)
        Price.parse(tiny).check_plausible(allow_low=True)
    Price.parse("0.05/0.4").check_plausible(allow_low=False)


def test_the_cap_stops_before_a_call_that_could_cross_it(tmp_path):
    # Each call's bound is about 0.0005 + 1000 output tokens at $10/M = $0.01; at $0.025 the third
    # call of the run could cross the cap, so it is never made.
    script = lambda m, t: reply("", ("search", {"query": "x"}), prompt=1000, completion=1000)
    model = metered(tmp_path, script, cap=0.025)
    result = one_try(model, FakeMcp())
    assert result.status == STOPPED and "cap" in result.reason
    assert model.model.calls == 2 and len(rows(tmp_path / "ledger.jsonl")) == 2
    assert model.ledger.spent <= 0.025


# --- model failures: retries, charges, aborts ------------------------------------------------


def test_429_and_5xx_are_retried_with_backoff_and_every_attempt_is_ledgered(tmp_path):
    errors = [http_error(429), http_error(503)]

    def script(messages, turn):
        if errors:
            raise errors.pop(0)
        return reply("done")

    slept = []
    model = metered(tmp_path, script, slept=slept)
    result = one_try(model, FakeMcp())
    assert result.status == ANSWERED and model.model.calls == 3 and slept == [4.0, 8.0]
    ledger = rows(tmp_path / "ledger.jsonl")
    assert [r["outcome"] for r in ledger] == ["failed", "failed", "answered"]
    assert [r["status"] for r in ledger[:2]] == [429, 503]
    assert ledger[0]["cost_usd"] == ledger[0]["bound_usd"] > 0  # an unanswered call is charged at its bound


def test_retries_stop_after_three_and_the_try_is_an_error(tmp_path):
    slept = []
    model = metered(tmp_path, failing, slept=slept)
    result = one_try(model, FakeMcp())
    assert result.status == ERROR and model.model.calls == 4 and len(slept) == 3
    assert len(rows(tmp_path / "ledger.jsonl")) == 4


def test_a_context_length_error_ends_the_try_not_the_run(tmp_path):
    def script(m, t):
        raise http_error(400, "This model's maximum context length is 128000 tokens")

    result = one_try(metered(tmp_path, script), FakeMcp())
    assert result.status == ERROR and "maximum context length" in result.reason


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_another_4xx_is_fatal(tmp_path, status):
    def script(m, t):
        raise http_error(status, "invalid_api_key or bad request")

    with pytest.raises(FatalModelError):
        one_try(metered(tmp_path, script), FakeMcp())
    assert rows(tmp_path / "ledger.jsonl")[0]["outcome"] == "failed"


def test_a_call_that_never_reached_the_endpoint_is_not_charged(tmp_path):
    def script(m, t):
        raise ModelError("http://model: connection refused", sent=False)

    model = metered(tmp_path, script)
    assert one_try(model, FakeMcp()).status == ERROR
    assert rows(tmp_path / "ledger.jsonl") == [] and model.ledger.spent == 0


def test_a_reply_without_usage_is_charged_at_its_bound_with_a_warning(tmp_path):
    model = metered(tmp_path, lambda m, t: ChatReply("done", (), 0, 0, usage_reported=False))
    one_try(model, FakeMcp())
    row = rows(tmp_path / "ledger.jsonl")[0]
    assert row["cost_usd"] == row["bound_usd"] > 0 and "no token usage" in row["note"]
    assert model.ledger.warnings and "no token usage" in model.ledger.warnings[0]


def test_the_ledger_and_its_cap_carry_across_question_sets(tmp_path):
    ledger = Ledger(tmp_path / "a" / "ledger.jsonl", 1.0)
    model = Metered(ScriptedModel(lambda m, t: reply("done")), PRICE, ledger, 100)
    one_try(model, FakeMcp())
    ledger.redirect(tmp_path / "b" / "ledger.jsonl")
    one_try(model, FakeMcp())
    assert len(rows(tmp_path / "a" / "ledger.jsonl")) == len(rows(tmp_path / "b" / "ledger.jsonl")) == 1
    second = rows(tmp_path / "b" / "ledger.jsonl")[0]
    assert second["spent_usd"] == pytest.approx(ledger.spent) and ledger.calls == 2


# --- gold resolution -------------------------------------------------------------------------


def context_set(questions, bundle="demo"):
    return parse_context({"bundle": bundle, "questions": questions})


def corpus_and_keys(tmp_path):
    corpus = verify_corpus(write_corpus(tmp_path))
    return corpus, corpus.keys(), corpus.manifest_paths()


def resolve(tools, cset, keys, manifest, corpus=None):
    return asyncio.run(resolve_gold(tools, cset.scored, keys, manifest, corpus.records if corpus else None))


def test_gold_ids_resolve_through_inspect_to_corpus_files(tmp_path):
    corpus, keys, manifest = corpus_and_keys(tmp_path)
    cset = context_set([{"id": "q1", "question": "q?", "gold": [
        "jira:DEMO-1", "documentation_page:abc123", "slurm_partition:submit:gpu", "ops_incident:inc-1", "hn:42"]}])
    tools = FakeMcp()
    gold = resolve(tools, cset, keys, manifest, corpus).records["q1"]
    by_id = {g.node_id: g for g in gold}
    assert by_id["jira:DEMO-1"].path == "jira/DEMO-1.json" and by_id["jira:DEMO-1"].key == "DEMO-1"
    assert by_id["jira:DEMO-1"].units  # its description is six words
    assert by_id["documentation_page:abc123"].source == "docsite"  # found by its url, the subtype's key field
    # {"name": "gpu"} holds nothing distinctive: unavailable, never a miss
    assert by_id["slurm_partition:submit:gpu"].status == "unavailable"
    assert "distinctive" in by_id["slurm_partition:submit:gpu"].reason
    assert by_id["ops_incident:inc-1"].status == "unavailable" and "ops-record" in by_id["ops_incident:inc-1"].reason
    assert by_id["hn:42"].status == "unavailable" and "hypernews" in by_id["hn:42"].reason
    assert all(call[0] == "inspect" and call[2] == {"okg/result_format": "json"} for call in tools.calls)


@pytest.mark.parametrize("source", ["cms-physics.twiki-eos", "cms-physics.twiki_eos", "twiki-eos", "Demo.Twiki_EOS"])
def test_a_source_spelled_as_okg_writes_provenance_finds_its_corpus_source(tmp_path, source):
    keys = {"sources": {"twiki_eos": {"lookup": {"CMS:TopicOne": "twiki-eos/TopicOne.txt"}, "ambiguous": [],
                                      "natural_key": {"twiki-eos/TopicOne.txt": "TopicOne"}}},
            "left_out": {}}
    nodes = {"twiki:CMS:TopicOne": ("twiki_page", source, {})}
    cset = context_set([{"id": "q1", "question": "q?", "gold": ["twiki:CMS:TopicOne"]}])
    gold = asyncio.run(resolve_gold(FakeMcp(nodes=nodes), cset.scored, keys, frozenset())).records["q1"]
    assert gold[0].source == "twiki_eos" and gold[0].path == "twiki-eos/TopicOne.txt"


def test_a_left_out_source_matches_in_any_spelling(tmp_path):
    keys = {"sources": {}, "left_out": {"hypernews": "crawled live"}}
    nodes = {"hn:1": ("thread", "comp-ops.HyperNews", {})}
    cset = context_set([{"id": "q1", "question": "q?", "gold": ["hn:1"]}])
    gold = asyncio.run(resolve_gold(FakeMcp(nodes=nodes), cset.scored, keys, frozenset())).records["q1"]
    assert gold[0].status == "unavailable"


def test_key_candidates_use_only_the_subtypes_key_fields():
    info = raw_arm.NodeInfo("jira:DEMO-9", "jira_issue", "demo.jira", {"summary": "DEMO-1", "assignee": "DEMO-2"})
    assert raw_arm.key_candidates(info) == ["jira:DEMO-9", "DEMO-9"]


def test_a_gold_id_that_no_longer_resolves_refuses_the_run_by_name(tmp_path):
    _, keys, manifest = corpus_and_keys(tmp_path)
    cset = context_set([{"id": "q1", "question": "q?", "gold": ["jira:DEMO-1", "jira:GONE-9"]},
                        {"id": "q2", "question": "q?", "gold": ["mystery:1"]}])
    with pytest.raises(GoldError) as err:
        resolve(FakeMcp(), cset, keys, manifest)
    message = str(err.value)
    assert "q1: jira:GONE-9: inspect reported an error" in message
    assert "q2: mystery:1: its source demo.unknown_source is neither in the corpus nor listed as left out" in message
    assert "never scored as a miss" in message


def test_a_gold_id_whose_record_is_not_in_the_corpus_refuses_the_run(tmp_path):
    _, keys, manifest = corpus_and_keys(tmp_path)
    nodes = dict(NODES, **{"jira:DEMO-7": ("jira_issue", "demo.jira", {"key": "DEMO-7"})})
    cset = context_set([{"id": "q1", "question": "q?", "gold": ["jira:DEMO-7"]}])
    with pytest.raises(GoldError, match="no record of jira in the corpus has its key"):
        resolve(FakeMcp(nodes=nodes), cset, keys, manifest)


def test_gold_sources_override_resolution(tmp_path):
    corpus, keys, manifest = corpus_and_keys(tmp_path)
    cset = context_set([{"id": "q1", "question": "q?", "gold": ["chunk:ff01", "jira:GONE-9"],
                         "gold_sources": {"chunk:ff01": {"path": "jira/DEMO-2.json"},
                                          "jira:GONE-9": {"source": "jira", "key": "DEMO-1"}}}])
    tools = FakeMcp(nodes={})
    gold = resolve(tools, cset, keys, manifest, corpus).records["q1"]
    assert [(g.status, g.path, g.key) for g in gold] == [
        ("override", "jira/DEMO-2.json", "DEMO-2"), ("override", "jira/DEMO-1.json", "DEMO-1")]
    assert tools.calls == []  # inspect is never asked


def test_a_gold_sources_entry_that_names_nothing_refuses(tmp_path):
    _, keys, manifest = corpus_and_keys(tmp_path)
    cset = context_set([{"id": "q1", "question": "q?", "gold": ["x:1"],
                         "gold_sources": {"x:1": {"path": "jira/NOPE.json"}}}])
    with pytest.raises(GoldError, match="gold_sources path jira/NOPE.json is not in the corpus"):
        resolve(FakeMcp(), cset, keys, manifest)


def test_an_atom_only_a_live_system_answers_is_unavailable_and_a_mixed_one_counts_through_its_file(tmp_path):
    corpus, keys, manifest = corpus_and_keys(tmp_path)
    cset = context_set([{
        "id": "q1", "question": "q?",
        "gold": ["jira:DEMO-1", "jira:DEMO-2", "live:rule-count"],
        "gold_sources": {"live:rule-count": {"source": "jira", "key": "DEMO-2"}},
        "atoms": [
            {"id": "a1", "answer_source": "ticket", "gold": ["jira:DEMO-1"]},
            {"id": "a2", "answer_source": "ticket+live_rucio", "gold": ["jira:DEMO-2"]},
            {"id": "a3", "answer_source": "live_rucio", "gold": ["live:rule-count"]},
            {"id": "a4", "answer_source": "monitoring or live_dbs"},
        ]}])
    tools = FakeMcp()
    gold_map = resolve(tools, cset, keys, manifest, corpus)
    by_id = {g.node_id: g for g in gold_map.records["q1"]}
    assert by_id["live:rule-count"].status == "unavailable" and "live system" in by_id["live:rule-count"].reason
    assert by_id["jira:DEMO-2"].status == "resolved"  # the mixed atom counts through its ticket
    assert gold_map.live_atoms == {"q1": [{"atom": "a3", "answer_source": "live_rucio"},
                                          {"atom": "a4", "answer_source": "monitoring or live_dbs"}]}
    evidence = score_evidence([ev("Transfer failures at the test site. Upgrade the storage servers.")],
                              gold_map.records["q1"], manifest)
    assert evidence.recall == 1.0 and evidence.unavailable == ["live:rule-count"]


def test_live_only_atoms_are_listed_in_the_report(tmp_path):
    questions = [dict(QUESTIONS[0], atoms=[{"id": "a1", "answer_source": "ticket", "gold": ["jira:DEMO-1"]},
                                           {"id": "a2", "answer_source": "live_dbs"}])]
    _, data, _ = run_plan(tmp_path, questions=questions)
    assert data["live_only_atoms"] == {"q1": [{"atom": "a2", "answer_source": "live_dbs"}]}
    from archi.evaluation.raw_arm import render_agent_markdown

    assert "q1 a2 (live_dbs)" in "\n".join(render_agent_markdown(data, {"k": 3, "arms": {}}))


@pytest.mark.parametrize("row, message", [
    ({"atoms": [{"id": "a1", "gold": ["x:9"]}]}, "not among the question's gold ids"),
    ({"atoms": [{"id": "a1"}, {"id": "a1"}]}, "unique non-empty string"),
    ({"atoms": [{"id": "a1", "colour": "red"}]}, "takes id, answer_source, gold, text and required"),
    ({"gold_sources": {"other:1": {"path": "a"}}}, "not one of its gold ids"),
    ({"gold_sources": {"jira:DEMO-1": {"source": "jira"}}}, "must be {source, key} or {path}"),
    ({"answer": ""}, "`answer` must be a non-empty string"),
])
def test_bad_agent_keys_in_a_question_are_refused(row, message):
    from archi.evaluation.context import ContextSetError

    with pytest.raises(ContextSetError) as err:
        context_set([{"id": "q1", "question": "q?", "gold": ["jira:DEMO-1"], **row}])
    assert message in str(err.value)


# --- scoring ---------------------------------------------------------------------------------

DEMO1_FILE = b'{"key": "DEMO-1", "description": "Transfer failures at the test site", "status": "Open"}\n'
DEMO2_FILE = b'{"key": "DEMO-2", "description": "Upgrade the storage servers"}\n'


def gold_record(node_id, key, path, data):
    units = content_fingerprints(data, exclude=(key, node_id, path))
    return GoldRecord(node_id, "resolved", "jira", key, path, units=units)


DEMO1 = gold_record("jira:DEMO-1", "DEMO-1", "jira/DEMO-1.json", DEMO1_FILE)
DEMO2 = gold_record("jira:DEMO-2", "DEMO-2", "jira/DEMO-2.json", DEMO2_FILE)
GONE = GoldRecord("ops_incident:inc-1", "unavailable", "ops-record", reason="left out")
MANIFEST = frozenset({"jira/DEMO-1.json", "jira/DEMO-2.json", "docsite/x.json"})


def ev(text, ok=True, arguments=None):
    return ToolEvent("bash", arguments or {}, text, ok)


def test_fingerprints_come_from_values_never_field_names_or_the_key():
    assert content_fingerprints(DEMO1_FILE, exclude=("DEMO-1",)) == (("transfer failures at the test site",),)
    assert content_fingerprints(DEMO2_FILE, exclude=("DEMO-2",)) == (("upgrade the storage servers",),)
    assert content_fingerprints(b'{"name": "gpu", "state": "up"}') == ()


def test_a_short_value_never_fingerprints_however_long_its_characters():
    # two or three words are a name or a label, not content: "Alexandra Kowalczyk-Smith" is 25 characters
    data = b'{"key": "DEMO-7", "assignee": "Alexandra Kowalczyk-Smith", "project": "Computing Operations"}'
    assert content_fingerprints(data, exclude=("DEMO-7",)) == ()


SHARED_LINE = "from __future__ import annotations"
GOLD_CODE = (f"{SHARED_LINE}\n# Retry stuck transfers after the storage element comes back online\n"
             "def retry(transfer): return transfer.resubmit()\n")
OTHER_CODE = f"{SHARED_LINE}\n#!/usr/bin/env python3\ndef unrelated(): pass\n"


def shared_corpus(tmp_path, files):
    root = tmp_path / "records"
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root, frozenset(files)


def test_text_another_corpus_record_holds_is_dropped_from_the_gold_fingerprints(tmp_path):
    root, manifest = shared_corpus(tmp_path, {"code/a/retry.py": GOLD_CODE, "code/b/other.py": OTHER_CODE})
    gold = GoldRecord("file:a/retry.py", "resolved", "git_code", "a/retry.py", "code/a/retry.py",
                      units=content_fingerprints(GOLD_CODE.encode()))
    assert ("from future import annotations",) in gold.units
    kept = raw_arm.drop_shared({"q1": [gold]}, root, manifest)["q1"][0]
    assert ("from future import annotations",) not in kept.units and kept.shared == 1 and kept.available
    # a cat of the other file shows the shared line: it lists nothing and reads nothing
    evidence = score_evidence([ev(OTHER_CODE)], [kept], manifest)
    assert evidence.found == [] and evidence.recall == 0.0
    assert score_evidence([ev(GOLD_CODE)], [kept], manifest).found == ["file:a/retry.py"]


def test_a_record_whose_every_fingerprint_is_shared_is_unavailable_not_a_miss(tmp_path):
    meta = "%META:TOPICPARENT{name=\"WebHome\"}%\n%META:TOPICINFO{author=\"BaseUserMapping\" format=\"1.1\" version=\"1\"}%\n"
    root, manifest = shared_corpus(tmp_path, {"twiki/A.txt": meta, "twiki/B.txt": meta + "Other text here.\n"})
    gold = GoldRecord("twiki:A", "resolved", "twiki", "A", "twiki/A.txt", units=content_fingerprints(meta.encode()))
    assert gold.units
    kept = raw_arm.drop_shared({"q1": [gold]}, root, manifest)["q1"][0]
    assert kept.status == "unavailable" and "also appears in another corpus record" in kept.reason


TWO_PART = gold_record("jira:DEMO-5", "DEMO-5", "jira/DEMO-5.json",
                       b'{"key": "DEMO-5", "summary": "Transfers stuck at the test site", '
                       b'"description": "The storage element rejected writes after the upgrade", '
                       b'"resolution": "Restarted the gridftp doors on every storage node"}')
TWO_PART_MANIFEST = MANIFEST | {"jira/DEMO-5.json"}


def test_a_title_field_never_fingerprints():
    assert content_fingerprints(b'{"summary": "Transfers stuck at the test site", '
                                b'"fields": {"title": "Storage element rejected writes after upgrade"}}') == ()
    assert len(TWO_PART.units) == 2  # description and resolution; the summary is a title


def test_a_search_row_with_its_title_and_one_excerpt_only_lists_a_record_with_more_to_read():
    row = ("[1] jira:DEMO-5 (jira_issue) · source demo.jira\n    Transfers stuck at the test site\n"
           "    The storage element rejected writes after the upgrade")
    evidence = score_evidence([ev(row)], [TWO_PART], TWO_PART_MANIFEST)
    assert evidence.found == [] and evidence.listed == ["jira:DEMO-5"]
    # one grep hit is the same: one line of the record
    grep = '/corpus/jira/DEMO-5.json:  "description": "The storage element rejected writes after the upgrade",'
    assert score_evidence([ev(grep)], [TWO_PART], TWO_PART_MANIFEST).found == []
    # a second part of it in a later response: read, in either arm
    later = score_evidence([ev(row), ev("Restarted the gridftp doors on every storage node")], [TWO_PART],
                           TWO_PART_MANIFEST)
    assert later.found == ["jira:DEMO-5"]


def test_a_unicode_escape_and_the_character_it_stands_for_read_the_same():
    gold = gold_record("jira:DEMO-6", "DEMO-6", "jira/DEMO-6.json",
                       '{"key": "DEMO-6", "description": "The site’s storage element rejects writes"}'.encode())
    escaped = '{"description": "The site\\u2019s storage element rejects writes"}'  # json.dumps(ensure_ascii=True)
    assert score_evidence([ev(escaped)], [gold], MANIFEST).found == ["jira:DEMO-6"]


def test_gold_only_in_a_tool_input_never_counts():
    events = [ev("no match here", arguments={"command": "grep 'Transfer failures at the test site' /corpus"})]
    evidence = score_evidence(events, [DEMO1], MANIFEST)
    assert evidence.found == [] and evidence.recall == 0.0 and evidence.first_rank is None


def test_gold_only_in_the_final_answer_never_counts(tmp_path):
    tools = FakeMcp({"q": search_rows("jira:DEMO-2")})
    model = metered(tmp_path, lambda m, t: reply("", ("search", {"query": "q"})) if t == 1
                    else reply("Transfer failures at the test site.\nSources: jira:DEMO-1"))
    result = one_try(model, tools)
    evidence = score_evidence(result.events, [DEMO1, DEMO2], MANIFEST)
    assert evidence.found == ["jira:DEMO-2"] and evidence.missed == ["jira:DEMO-1"]


def test_gold_in_a_response_the_scorer_does_not_read_never_counts():
    evidence = score_evidence([ev("Transfer failures at the test site", ok=False)], [DEMO1], MANIFEST)
    assert evidence.found == []


@pytest.mark.parametrize("text", [
    '/corpus/jira/DEMO-1.json:2:  "description": "Transfer failures at the test site",',   # rg output
    '{\n  "key": "DEMO-1",\n  "description": "Transfer failures at the test site"\n}',      # cat
    '{"results": [{"node_id": "jira:DEMO-1", "excerpt": "Transfer failures at the test site"}]}',  # MCP JSON
    "[1] jira:DEMO-1 (jira_issue) · source demo.jira\n    TRANSFER failures — at the test-site.",   # reflowed
])
def test_content_of_the_record_in_a_response_reads_it(text):
    evidence = score_evidence([ev(text)], [DEMO1], MANIFEST)
    assert evidence.found == ["jira:DEMO-1"] and evidence.listed == []


@pytest.mark.parametrize("text", [
    "/corpus/jira/DEMO-1.json",                          # find
    "DEMO-1.json\nDEMO-2.json",                          # ls: names only
    "[1] jira:DEMO-1 (jira_issue) · source demo.jira",   # a search row with no excerpt
])
def test_a_path_key_or_id_alone_only_lists_the_record(text):
    evidence = score_evidence([ev(text)], [DEMO1], MANIFEST)
    assert evidence.found == [] and evidence.listed == ["jira:DEMO-1"] and evidence.recall == 0.0


def test_a_short_key_does_not_even_list():
    short = gold_record("site:T2_X", "T2_X", "cric/sites/T2_X.json", b'{"tier": "a site in the test region"}')
    evidence = score_evidence([ev("T2_X is down")], [short], MANIFEST | {"cric/sites/T2_X.json"})
    assert evidence.found == [] and evidence.listed == []


def test_a_nonzero_exit_or_cut_output_still_counts_if_it_was_read():
    text = "/corpus/jira/DEMO-1.json:  Transfer failures at the test site\n[exit code 2]"
    assert score_evidence([ev(text)], [DEMO1], MANIFEST).found == ["jira:DEMO-1"]


def test_mrr_ranks_the_first_gold_record_read_among_the_records_met():
    events = [ev("/corpus/docsite/x.json\n/corpus/jira/DEMO-2.json"),            # met: x; DEMO-2 listed
              ev("nothing"),
              ev("/corpus/jira/DEMO-1.json: Transfer failures at the test site")]  # read: DEMO-1
    evidence = score_evidence(events, [DEMO1, DEMO2, GONE], MANIFEST)
    assert evidence.found == ["jira:DEMO-1"] and evidence.listed == ["jira:DEMO-2"]
    assert evidence.first_rank == 2 and evidence.reciprocal_rank == pytest.approx(1 / 2)


def test_an_unavailable_gold_record_leaves_the_denominator_and_is_listed():
    evidence = score_evidence([ev("Transfer failures at the test site")], [DEMO1, DEMO2, GONE], MANIFEST)
    assert evidence.recall == 0.5 and evidence.unavailable == ["ops_incident:inc-1"]
    only_gone = score_evidence([ev("x")], [GONE], MANIFEST)
    assert only_gone.recall is None  # nothing to score: not a 0


# --- the prompt ------------------------------------------------------------------------------


def test_the_prompt_differs_between_arms_only_in_the_knowledge_paragraph():
    caps = Caps()
    bash = build_prompt(AGENT_BASH, caps, folders=["jira"], tools=["jq", "python3"]).split("\n\n")
    graph = build_prompt(AGENT_GRAPH, caps).split("\n\n")
    assert len(bash) == len(graph) == 4
    assert [p for i, p in enumerate(bash) if i != 1] == [p for i, p in enumerate(graph) if i != 1]
    assert bash[1] != graph[1] and "/corpus" in bash[1] and "jq, python3 are installed" in bash[1]
    assert "30 turns and 200 tool calls" in bash[2]


# --- a whole run: arms, judge, cap stop, errors -----------------------------------------------


QUESTIONS = [
    {"id": "q1", "question": "Why do transfers fail?", "gold": ["jira:DEMO-1"], "answer": "Stuck transfers."},
    {"id": "q2", "question": "What is upgraded?", "gold": ["jira:DEMO-2", "ops_incident:inc-1"]},
    {"id": "q3", "question": "Only an ops record", "gold": ["ops_incident:inc-1"]},
]


def agent_script(messages, turn):
    """Search for the question, then answer."""
    if messages[-1]["role"] == "user":
        return reply("", ("search", {"query": messages[-1]["content"]}))
    return reply("Stuck transfers.\nSources: jira:DEMO-1")


def judge_script(scores):
    it = iter(scores)
    return lambda messages, turn: reply(json.dumps({"score": next(it), "reason": "ok"}), prompt=500, completion=20)


def run_plan(tmp_path, *, cap=10.0, judge_scores=None, passes=1, questions=QUESTIONS, tries=1, script=agent_script):
    corpus, keys, manifest = corpus_and_keys(tmp_path)
    cset = context_set(questions)
    tools = FakeMcp({"Why do transfers fail?": search_rows("jira:DEMO-1"),
                     "What is upgraded?": search_rows("jira:DEMO-9", "jira:DEMO-2")})
    model = metered(tmp_path, script, cap=cap)
    judge = None
    if judge_scores is not None:
        judge = Judge(Metered(ScriptedModel(judge_script(judge_scores), "fake-judge"), Price(2.0, 8.0),
                              model.ledger, 1000), passes=passes)
    plan = AgentPlan((AGENT_GRAPH,), model, NO_WAIT, tries, corpus, judge=judge,
                     judge_requested=judge_scores is not None)
    gold = resolve(tools, cset, keys, manifest, corpus)
    report = asyncio.run(run_agent_arms(cset.scored, plan, tools, gold, manifest=manifest))
    return report, report.to_dict(), model


def test_a_run_scores_each_question_and_skips_one_with_only_unavailable_gold(tmp_path):
    report, data, _ = run_plan(tmp_path)
    s = data["arms"][AGENT_GRAPH]
    assert [q["id"] for q in data["questions"]] == ["q1", "q2"]  # q3 has no available gold
    assert s["tries"] == 2 and s["tries_scored"] == 2
    assert s["evidence_recall"] == 1.0  # q1 reads DEMO-1; q2 reads DEMO-2, its ops record is unavailable
    assert s["mrr"] == pytest.approx((1 + 0.5) / 2)  # q2 meets DEMO-9 first
    assert s["answer_score"] == JUDGE_OFF  # q1 has an answer key, but no judge ran
    assert data["gold"]["q2"]["ops_incident:inc-1"]["status"] == "unavailable"
    assert data["spend"]["calls"] == 4 and data["spend"]["stopped_at_cap"] is False
    assert data["corpus"]["digest"].startswith("sha256:") and "ops-record" in data["ops_record"]
    assert data["corpus"]["generation"] == "g-1" and data["corpus"]["generation_warning"] is None


def test_error_tries_are_listed_counted_and_left_out_of_the_averages(tmp_path):
    def script(messages, turn):
        if "upgraded" in messages[1]["content"]:
            raise http_error(400, "maximum context length exceeded")
        return agent_script(messages, turn)

    report, data, _ = run_plan(tmp_path, script=script)
    s = data["arms"][AGENT_GRAPH]
    assert s["failures"] == 1 and s["tries_scored"] == 1 and s["evidence_recall"] == 1.0
    assert [(e["question"], e["try"]) for e in s["error_tries"]] == [("q2", 1)]
    assert "maximum context length" in s["error_tries"][0]["reason"]
    from archi.evaluation.raw_arm import render_agent_markdown

    assert "error try, agent-graph: q2 try 1" in "\n".join(render_agent_markdown(data, {"k": 3, "arms": {}}))


def test_the_arms_take_turns_going_first(tmp_path, monkeypatch):
    corpus, keys, manifest = corpus_and_keys(tmp_path)
    cset = context_set(QUESTIONS)
    tools = FakeMcp({"Why do transfers fail?": search_rows("jira:DEMO-1"),
                     "What is upgraded?": search_rows("jira:DEMO-2")})
    monkeypatch.setattr(raw_arm, "BashTools", lambda sandbox: tools)  # no bwrap needed to check the order
    plan = AgentPlan((AGENT_BASH, AGENT_GRAPH), metered(tmp_path, agent_script), NO_WAIT, 1, corpus,
                     sandbox=type("Box", (), {"tools": []})())
    order = []
    asyncio.run(run_agent_arms(cset.scored, plan, tools, resolve(tools, cset, keys, manifest, corpus),
                               manifest=manifest, progress=lambda r, i, n: order.append((r.question, r.arm))))
    assert order == [("q1", AGENT_BASH), ("q1", AGENT_GRAPH), ("q2", AGENT_GRAPH), ("q2", AGENT_BASH)]


def broken_scope_box():
    """A sandbox whose systemd scope stopped starting mid-run: systemd-run fails before bwrap."""
    box = object.__new__(raw_arm.BashSandbox)
    box.command_seconds, box.output_bytes, box.tools = 10.0, 64 * 1024, []
    box._scope, box._scope_env = ["systemd-run"], {}
    box.argv = lambda command: ["/bin/sh", "-c", "echo 'Failed to connect to bus: No such file or directory' >&2; "
                                                 "exit 1"]
    return box


def test_a_scope_that_stops_starting_mid_run_aborts_the_run_instead_of_feeding_the_model(tmp_path):
    with pytest.raises(raw_arm.SandboxBroken, match="Failed to connect to bus"):
        broken_scope_box().run("ls")
    corpus, keys, manifest = corpus_and_keys(tmp_path)
    cset = context_set(QUESTIONS[:1])
    tools = FakeMcp()
    model = metered(tmp_path, lambda m, t: reply("", ("bash", {"command": "ls /corpus"})) if t == 1 else reply("x"))
    plan = AgentPlan((AGENT_BASH,), model, NO_WAIT, 1, corpus, sandbox=broken_scope_box())
    report = asyncio.run(run_agent_arms(cset.scored, plan, tools, resolve(tools, cset, keys, manifest, corpus),
                                        manifest=manifest))
    assert report.aborted and "sandbox stopped working" in report.aborted


def test_a_command_that_prints_systemds_failure_text_itself_is_only_output():
    box = broken_scope_box()
    box.argv = lambda command: (["/bin/sh", "-c", "true"] if command == "true" else
                                ["/bin/sh", "-c", "echo 'Failed to start transient scope unit: in an ops log'; exit 1"])
    result = box.run("cat ops.log; false")
    assert "in an ops log" in result.text and result.exit_code == 1


def test_a_servicenow_title_never_fingerprints():
    record = (b'{"number": "INC0012345", "short_description": "Storage element rejects writes after upgrade", '
              b'"description": "Restarted the gridftp doors on every storage node"}')
    assert content_fingerprints(record, exclude=("INC0012345",)) == (
        ("restarted the gridftp doors on every", "the gridftp doors on every storage", "gridftp doors on every storage node"),)


def test_the_report_counts_unavailable_gold_by_reason(tmp_path):
    _, data, _ = run_plan(tmp_path)
    assert sum(data["gold_unavailable_by_reason"].values()) == 2  # q2's and q3's ops record
    assert all("ops-record" in reason for reason in data["gold_unavailable_by_reason"])


def test_recall_and_mrr_compare_the_arms_only_on_pairs_both_finished():
    def done(arm, qid, recall_hit):
        ev_ = raw_arm.Evidence(found=["g"] if recall_hit else [], missed=[] if recall_hit else ["g"],
                               first_rank=1 if recall_hit else None)
        return raw_arm.TryResult(arm, qid, 1, evidence=ev_)

    plan = AgentPlan((AGENT_BASH, AGENT_GRAPH), None, Caps(), 1, None)
    report = raw_arm.AgentReport(plan, raw_arm.GoldMap({}))
    report.tries = {"q1": {AGENT_BASH: [done(AGENT_BASH, "q1", True)], AGENT_GRAPH: [done(AGENT_GRAPH, "q1", True)]},
                    # the spend stop cut q2 short in the graph arm: bash's q2 miss is left out of both
                    "q2": {AGENT_BASH: [done(AGENT_BASH, "q2", False)],
                           AGENT_GRAPH: [raw_arm.TryResult(AGENT_GRAPH, "q2", 1, status=STOPPED)]},
                    # an error try (a context-length 400, say) unpairs its question too
                    "q3": {AGENT_BASH: [done(AGENT_BASH, "q3", False)],
                           AGENT_GRAPH: [raw_arm.TryResult(AGENT_GRAPH, "q3", 1, status=ERROR)]}}
    assert report.paired() == {("q1", 1)}
    bash = report.summary(AGENT_BASH)
    assert bash["tries_scored"] == 1 and bash["tries_unpaired"] == 2 and bash["evidence_recall"] == 1.0
    assert report.summary(AGENT_GRAPH)["tries_scored"] == 1


def test_a_fatal_4xx_aborts_the_run_and_keeps_what_completed(tmp_path):
    def script(messages, turn):
        if "upgraded" in messages[1]["content"]:
            raise http_error(401, "invalid_api_key")
        return agent_script(messages, turn)

    report, data, _ = run_plan(tmp_path, script=script)
    assert "HTTP 401" in data["aborted"] and data["tries_run"] == 1


def test_with_no_answer_keys_the_judge_column_says_so_never_0(tmp_path):
    questions = [{k: v for k, v in q.items() if k != "answer"} for q in QUESTIONS]
    report, data, _ = run_plan(tmp_path, questions=questions, judge_scores=[])
    assert data["arms"][AGENT_GRAPH]["answer_score"] == NO_ANSWER_KEYS
    from archi.evaluation.raw_arm import render_agent_markdown

    md = "\n".join(render_agent_markdown(data, {"k": 3, "arms": {}}))
    assert "| no answer keys |" in md and "judge: no answer keys" in md


def test_two_judge_passes_are_averaged_and_ledgered_as_judge_calls(tmp_path):
    report, data, model = run_plan(tmp_path, judge_scores=[1, 0.5], passes=2)
    assert data["arms"][AGENT_GRAPH]["answer_score"] == 0.75
    judged = [r for r in rows(tmp_path / "ledger.jsonl") if r["role"] == "judge"]
    assert len(judged) == 2 and {r["model"] for r in judged} == {"fake-judge"} and judged[0]["arm"] == AGENT_GRAPH
    assert data["judge"]["passes"] == 2 and data["judge"]["cost_usd"] > 0


def test_a_run_stops_cleanly_at_the_cap_and_reports_what_completed(tmp_path):
    # Each call costs $0.002 and is bounded at about $0.0104 (1000 output tokens at $10/M): at a
    # $0.013 cap the first try's two calls go through and the second try's first could cross it.
    report, data, model = run_plan(tmp_path, cap=0.013, tries=2)
    spend = data["spend"]
    assert spend["stopped_at_cap"] is True and "cap" in spend["stop_reason"]
    assert spend["spent_usd"] <= 0.013 and len(rows(tmp_path / "ledger.jsonl")) == spend["calls"] == 2
    statuses = [t["status"] for q in data["questions"] for ts in q["arms"].values() for t in ts]
    assert statuses == [ANSWERED, STOPPED]
    s = data["arms"][AGENT_GRAPH]
    assert s["tries_scored"] == 1 and s["stopped_at_cap"] == 1
    assert data["tries_run"] == 2 and data["tries_planned"] == 4


def test_a_corpus_from_another_generation_is_flagged(tmp_path):
    corpus, keys, manifest = corpus_and_keys(tmp_path)
    cset = context_set(QUESTIONS[:1])
    tools = FakeMcp({"Why do transfers fail?": search_rows("jira:DEMO-1")})
    plan = AgentPlan((AGENT_GRAPH,), metered(tmp_path, agent_script), NO_WAIT, 1, corpus)
    gold = resolve(tools, cset, keys, manifest, corpus)
    report = asyncio.run(run_agent_arms(cset.scored, plan, tools, gold, manifest=manifest,
                                        evaluated_generations=["g-7"]))
    warning = report.to_dict()["corpus"]["generation_warning"]
    assert "built when generation g-1 was published" in warning and "g-7" in warning


# --- the command -----------------------------------------------------------------------------


class Endpoint(FakeMcp):
    opened = 0

    def __init__(self, url, token):
        super().__init__({"Why do transfers fail?": search_rows("jira:DEMO-1"),
                          "What is upgraded?": search_rows("jira:DEMO-2"),
                          "Only an ops record": search_rows("jira:DEMO-2")})
        Endpoint.opened += 1


class FakeChat:
    made = []
    fail_with = None

    def __init__(self, base_url, model, **kwargs):
        self.name, self.kwargs, self.calls = model, kwargs, 0
        FakeChat.made.append(self)

    def complete(self, messages, tools):
        self.calls += 1
        if FakeChat.fail_with is not None:
            raise FakeChat.fail_with
        if self.name.startswith("judge"):
            return reply('{"score": 1, "reason": "same"}', prompt=100, completion=10)
        return agent_script(messages, self.calls)


@pytest.fixture
def cli(monkeypatch, tmp_path):
    Endpoint.opened = 0
    FakeChat.made = []
    FakeChat.fail_with = None
    monkeypatch.setattr(agent_mod, "McpToolBox", Endpoint)
    monkeypatch.setattr(agent_mod, "OpenAIChat", FakeChat)
    monkeypatch.setenv("OKG_CHAT_MCP_TOKEN", "t0ken")
    monkeypatch.delenv("ARCHI_CONTEXT_EVAL_QUESTIONS", raising=False)
    questions = tmp_path / "questions.yaml"
    questions.write_text(yaml.safe_dump({"bundle": "demo", "k": 3, "questions": QUESTIONS}))
    return questions, write_corpus(tmp_path)


def agent_args(questions, corpus, out, *extra):
    return ["eval", "--context", "--questions", str(questions), "--mcp-url", "http://127.0.0.1:9/mcp",
            "--out", str(out), "--agents", "agent-graph", "--corpus", str(corpus), "--model", "m",
            "--model-price", "1/10", "--cap", "10", *extra]


def test_the_command_runs_the_agent_arms_beside_the_others_and_keeps_a_ledger(cli, tmp_path, capsys):
    questions, corpus = cli
    code = main(agent_args(questions, corpus, tmp_path / "out"))
    captured = capsys.readouterr()
    assert code == 0, captured.err
    data = json.loads((tmp_path / "out" / "report.json").read_text())
    assert set(data["arms"]) == {"graph", "baseline"}  # the no-model arms, untouched
    assert set(data["agent_arms"]["arms"]) == {AGENT_GRAPH}
    assert data["agent_arms"]["corpus"]["digest"] == verify_corpus(corpus).digest
    assert data["agent_arms"]["arms"][AGENT_GRAPH]["answer_score"] == JUDGE_OFF
    ledger = rows(tmp_path / "out" / "ledger.jsonl")
    assert ledger and {r["arm"] for r in ledger} == {AGENT_GRAPH}
    md = (tmp_path / "out" / "report.md").read_text()
    assert "## All arms side by side" in md and "| agent-graph |" in md and "| graph |" in md
    assert [c.name for c in FakeChat.made] == ["m"]  # no judge was built
    assert "spend: one call is bounded at $" in captured.out  # printed before the first call
    assert captured.out.index("spend: one call") < captured.out.index("agent-graph q1 try 1")


def test_the_judge_is_off_unless_asked(cli, tmp_path):
    questions, corpus = cli
    from archi.bundles.cli import build_parser

    args = build_parser().parse_args(agent_args(questions, corpus, tmp_path / "out"))
    assert args.judge is False and args.judge_passes == 1 and args.judge_reasoning_effort == "high"


def test_the_judge_runs_when_asked_and_keys_exist(cli, tmp_path, monkeypatch):
    questions, corpus = cli
    key = tmp_path / "judge.key"
    key.write_text("sk-judge-test\n")
    monkeypatch.setenv("ARCHI_EVAL_JUDGE_KEY_FILE", str(key))
    code = main(agent_args(questions, corpus, tmp_path / "out", "--judge", "--judge-model", "judge-m",
                           "--judge-price", "2/8", "--judge-url", "http://127.0.0.1:8/v1", "--judge-passes", "2",
                           "--judge-max-output-tokens", "4000"))
    assert code == 0
    data = json.loads((tmp_path / "out" / "report.json").read_text())
    assert data["agent_arms"]["arms"][AGENT_GRAPH]["answer_score"] == 1.0
    judge = next(c for c in FakeChat.made if c.name == "judge-m")
    assert judge.kwargs["reasoning_effort"] == "high" and judge.kwargs["api_key"] == "sk-judge-test"
    assert judge.kwargs["max_completion_tokens"] == 4000 and judge.kwargs["temperature"] is None
    assert data["agent_arms"]["judge"]["max_output_tokens"] == 4000
    assert "sk-judge-test" not in (tmp_path / "out" / "report.json").read_text()
    assert "sk-judge-test" not in (tmp_path / "out" / "ledger.jsonl").read_text()


@pytest.mark.parametrize("extra, message", [
    (["--cap", "0"], "refuse to start without a positive spend cap"),
    (["--cap", "-5"], "refuse to start without a positive spend cap"),
    (["--model-price", "0.000001/0.00001"], "--allow-low-price"),
])
def test_the_command_refuses_without_a_cap_or_with_a_tiny_price_and_spends_nothing(cli, tmp_path, capsys, extra,
                                                                                  message):
    questions, corpus = cli
    assert main(agent_args(questions, corpus, tmp_path / "out", *extra)) == 2
    assert message in capsys.readouterr().err
    assert FakeChat.made == [] and Endpoint.opened == 0 and not (tmp_path / "out").exists()


def test_the_command_has_no_default_cap_and_refuses_without_one(cli, tmp_path, capsys):
    questions, corpus = cli
    args = agent_args(questions, corpus, tmp_path / "out")
    at = args.index("--cap")
    assert main(args[:at] + args[at + 2:]) == 2
    assert "refuse to start without a positive spend cap: pass --cap DOLLARS" in capsys.readouterr().err
    assert FakeChat.made == [] and Endpoint.opened == 0 and not (tmp_path / "out").exists()


def test_the_command_refuses_without_a_price(cli, tmp_path, capsys):
    questions, corpus = cli
    args = [a for a in agent_args(questions, corpus, tmp_path / "out") if a not in ("--model-price", "1/10")]
    assert main(args) == 2 and "--model-price" in capsys.readouterr().err


def test_a_gold_id_that_does_not_resolve_refuses_the_command_before_any_model_call(cli, tmp_path, capsys):
    questions, corpus = cli
    data = yaml.safe_load(questions.read_text())
    data["questions"][0]["gold"].append("jira:GONE-9")
    questions.write_text(yaml.safe_dump(data))
    assert main(agent_args(questions, corpus, tmp_path / "out")) == 2
    err = capsys.readouterr().err
    assert "q1: jira:GONE-9" in err and "never scored as a miss" in err
    assert all(c.calls == 0 for c in FakeChat.made) and not (tmp_path / "out" / "report.json").exists()


def test_a_corpus_that_changed_is_refused(cli, tmp_path, capsys):
    questions, corpus = cli
    (corpus / "records" / "jira" / "DEMO-1.json").write_text("tampered\n")
    assert main(agent_args(questions, corpus, tmp_path / "out")) == 2
    assert "changed since the corpus was built" in capsys.readouterr().err


def test_a_paid_endpoint_needs_the_opt_in(cli, tmp_path, capsys):
    questions, corpus = cli
    assert main(agent_args(questions, corpus, tmp_path / "out", "--model-url", "https://api.example.com/v1")) == 2
    assert "--allow-remote-model" in capsys.readouterr().err


def test_a_remote_model_needs_its_key_file(cli, tmp_path, capsys, monkeypatch):
    questions, corpus = cli
    monkeypatch.delenv("ARCHI_EVAL_MODEL_KEY_FILE", raising=False)
    assert main(agent_args(questions, corpus, tmp_path / "out", "--model-url", "https://api.example.com/v1",
                           "--allow-remote-model")) == 2
    assert "ARCHI_EVAL_MODEL_KEY_FILE" in capsys.readouterr().err


def test_a_fatal_model_error_exits_2_with_the_report_written(cli, tmp_path, capsys):
    questions, corpus = cli
    FakeChat.fail_with = http_error(403, "forbidden")
    assert main(agent_args(questions, corpus, tmp_path / "out")) == 2
    assert "the agent arms were aborted" in capsys.readouterr().err
    data = json.loads((tmp_path / "out" / "report.json").read_text())
    assert "HTTP 403" in data["agent_arms"]["aborted"]
    assert rows(tmp_path / "out" / "ledger.jsonl")[0]["outcome"] == "failed"


def test_the_cap_stop_exits_3_with_the_report_written(cli, tmp_path, capsys):
    questions, corpus = cli
    code = main(agent_args(questions, corpus, tmp_path / "out", "--cap", "0.013", "--max-output-tokens", "1000"))
    assert code == 3
    assert "stopped at the spend cap" in capsys.readouterr().err
    data = json.loads((tmp_path / "out" / "report.json").read_text())
    assert data["agent_arms"]["spend"]["stopped_at_cap"] is True
    assert data["status"] == "pass"  # the graph arm's own status is unchanged


def test_on_an_instance_the_ledger_lands_in_the_run_directory_and_the_gate_still_reads(cli, tmp_path, monkeypatch):
    from archi.ops.context_gate import GateSpec, context_eval_reading, run_dirs
    from archi.stack.instance import Instance
    from evaluation.test_context import instance_at

    questions, corpus = cli
    inst = instance_at(tmp_path, ["demo"])
    monkeypatch.setattr(Instance, "load", classmethod(lambda cls, d: inst))
    code = main(["eval", "--context", "--questions", str(questions), "--agents", "agent-graph", "--corpus",
                 str(corpus), "--model", "m", "--model-price", "1/10", "--cap", "10"])
    assert code == 0
    base = tmp_path / "data" / "eval" / "context" / "demo"
    runs = run_dirs(base)
    assert len(runs) == 1 and (runs[0] / "ledger.jsonl").is_file() and (runs[0] / "report.json").is_file()
    assert [p.name for p in base.iterdir() if p.name.startswith(".")] == []
    reading = context_eval_reading(GateSpec.from_mapping({"min_context_eval_score": 0.1}), inst.dir)
    assert reading.problem is None


# --- what is sent to a model endpoint --------------------------------------------------------


class Posted:
    def __init__(self, body):
        self.status_code, self._body, self.headers, self.text = 200, body, {}, json.dumps(body)

    def json(self):
        return self._body


@pytest.mark.parametrize("url, sent", [("https://api.example.com/v1", False), ("http://127.0.0.1:11434/v1", True)])
def test_temperature_and_seed_never_go_to_a_remote_endpoint(monkeypatch, url, sent):
    import requests

    bodies = []

    def post(endpoint, json=None, headers=None, timeout=None):
        bodies.append(json)
        return Posted({"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    monkeypatch.setattr(requests, "post", post)
    reply_ = agent_mod.OpenAIChat(url, "m", api_key="k").complete([], [])
    assert ("temperature" in bodies[0]) is sent and ("seed" in bodies[0]) is sent
    assert reply_.finish_reason == "stop" and reply_.usage_reported


def test_a_reply_without_usage_says_so_and_an_http_error_keeps_its_status(monkeypatch):
    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: Posted({"choices": [{"message": {"content": "hi"}}]}))
    assert agent_mod.OpenAIChat("http://127.0.0.1:1/v1", "m").complete([], []).usage_reported is False

    class Refused(Posted):
        def __init__(self):
            super().__init__({})
            self.status_code, self.headers, self.text = 429, {"Retry-After": "7"}, "slow down"

    monkeypatch.setattr(requests, "post", lambda *a, **k: Refused())
    with pytest.raises(ModelError) as err:
        agent_mod.OpenAIChat("http://127.0.0.1:1/v1", "m").complete([], [])
    assert err.value.status == 429 and err.value.sent and err.value.retry_after == 7.0
