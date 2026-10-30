"""req.r3-context-quality.context-eval: `archi eval --context` over a fake MCP search tool.

No network, no model and no real question set: the fixture set and the search
results below are invented, in okg's text row format (`[n] <id> (<subtype>) · ...`)
and in its JSON form (`_meta` `okg/result_format: json`).
"""
import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from archi.bundles.cli import main
from archi.evaluation import agent as agent_mod
from archi.evaluation.agent import ToolSpec
from archi.evaluation.context import (
    BASELINE,
    EXECUTION_FAILED,
    FAIL,
    GRAPH,
    PASS,
    RESULT_META,
    ContextSetError,
    load_context,
    parse_context,
    read_page,
    run_context,
    publish_run,
    score_hits,
    write_report,
)
from archi.ops.context_gate import (
    GateFacts,
    GateSpec,
    GateSpecError,
    context_eval_dir,
    context_eval_reading,
    evaluate_gate,
    read_context_eval,
    run_dirs,
)

FIXTURE = Path(__file__).parent / "fixtures" / "context.yaml"
SEARCH = ToolSpec("search", "Find records by words", {"type": "object"})
Q1 = "Which service moves files between storage sites?"
Q2 = "How are dataset replica rules written?"
Q3 = "Where is a planned site downtime announced?"


def rows(*lines, generation="g-41"):
    return f'search ok · "q" · rows 1-{len(lines)} · generation {generation}\n\n' + "\n".join(lines) + "\n"


#: What the fake search returns as text, by arm (told apart by `rank_by`) and question.
GRAPH_RESULTS = {
    Q1: rows(
        '[1] documentation_page:fx-transfer (documentation_page) · "Transfers" · source t.docsite',
        "    The transfer service moves files.",
    ),
    Q2: rows(
        '[1] ticket:FX-7 (ticket) · "Rule syntax question" · source t.jira',
        '[2] documentation_page:fx-other (documentation_page) · "Other"',
        '[3] documentation_page:fx-rules (documentation_page) · "Replica rules"',
    ),
    Q3: rows(
        '[1] documentation_page:fx-other (documentation_page) · "Other"',
        '[2] documentation_page:fx-a (documentation_page) · "A"',
        '[3] documentation_page:fx-b (documentation_page) · "B"',
        '[4] event:fx-meeting-1 (event) · "Ops meeting"',
    ),
}
BASELINE_RESULTS = {
    Q1: rows("[1] document_chunk:c1 (document_chunk) · passage · in documentation_page:fx-transfer · source t.docsite"),
    Q2: rows(
        '[1] document_chunk:c2 (document_chunk) · passage · "see · in ticket:FX-7 · here" · in documentation_page:fx-other',
        "    the excerpt names ticket:FX-7, which is not a hit",
        "    [2] ticket:FX-7 (ticket) · an indented line of an excerpt is not a row",
        "[2] document_chunk:c3 (document_chunk) · passage · in documentation_page:fx-rules",
    ),
}


def as_json(text):
    """The same page as okg's JSON form."""
    page = read_page(text)
    results = [{"node_id": ids[0], **({"parent": {"node_id": ids[1]}} if len(ids) > 1 else {})}
               for ids in page.ranking]
    return json.dumps({"status": "ok", "pinned_generation_id": page.generation, "results": results})


class FakeSearch:
    """`fail` maps (arm, question) to an error text; `json` answers in okg's JSON form."""

    def __init__(self, specs=(SEARCH,), fail=None, json_replies=False):
        self.specs, self.calls, self.fail, self.json = list(specs), [], dict(fail or {}), json_replies

    async def list_tools(self):
        return self.specs

    async def call(self, name, arguments, *, meta=None):
        self.calls.append((name, dict(arguments), meta))
        arm = BASELINE if arguments.get("rank_by") == "self" else GRAPH
        error = self.fail.get((arm, arguments["query"]))
        if error:
            return error, True
        text = (BASELINE_RESULTS if arm == BASELINE else GRAPH_RESULTS).get(arguments["query"])
        if text is None:
            return "error · search: the lexical index is not built", True
        return (as_json(text) if self.json else text), False


def opener(tools):
    @asynccontextmanager
    async def open_tools():
        yield tools

    return open_tools


def run(cset, tools):
    return asyncio.run(run_context(cset, opener(tools), mcp_url="http://127.0.0.1:1/mcp"))


def fixture_raw():
    return yaml.safe_load(FIXTURE.read_text())


# --- the question file ---------------------------------------------------------------


def test_the_fixture_set_parses_and_keeps_the_smoke_shape():
    cset = load_context(FIXTURE)
    assert cset.bundle == "demo" and cset.k == 3 and cset.min_recall == 0.5
    assert [q.id for q in cset.scored] == ["transfer-tool", "replica-rules", "site-downtime"]
    assert cset.questions[-1].gold == ()
    assert cset.sha256.startswith("sha256:") and len(cset.sha256) == 71
    assert {a.name: a.arguments for a in cset.arms} == {
        GRAPH: {}, BASELINE: {"method": "lexical", "rank_by": "self", "expand_concepts": False}}


def base(**over):
    raw = {"bundle": "demo", "questions": [{"id": "a", "question": "q?", "gold": ["x:1"]}]}
    raw.update(over)
    return raw


@pytest.mark.parametrize("raw, message", [
    (base(questions=[{"id": "a", "question": "q?"}]), "no question has `gold`"),
    (base(k=0), "`k` must be"),
    (base(min_recall=1.5), "`min_recall` must be"),
    (base(extra=1), "unknown keys"),
    (base(questions=[{"id": "a", "question": "q?", "gold": "x:1"}]), "gold must be a list"),
    (base(questions=[{"id": "a", "question": "q?", "gold": ["x:1"], "expected": "y"}]), "unknown keys"),
    (base(arms={"baseline": {"tool": "search", "arguments": {"query": "fixed"}}}), "may not set ['query']"),
    (base(arms={"baseline": {"tool": "search", "args": {}}}), "takes only `tool` and `arguments`"),
    (base(bundle="../etc"), "cannot be used in a path"),
    (base(bundle="a/b"), "cannot be used in a path"),
    (base(bundle="a..b"), "cannot be used in a path"),
])
def test_a_bad_question_file_is_refused_by_name(raw, message):
    with pytest.raises(ContextSetError) as err:
        parse_context(raw)
    assert message in str(err.value)


def test_an_arm_in_the_file_replaces_the_default_of_that_name():
    cset = parse_context(base(arms={"baseline": {"arguments": {"method": "lexical", "where": {"type": "document_chunk"}}},
                                    "semantic": {"tool": "search", "arguments": {"method": "semantic"}}}))
    arms = {a.name: (a.tool, a.arguments, a.description) for a in cset.arms}
    assert arms[BASELINE] == ("search", {"method": "lexical", "where": {"type": "document_chunk"}},
                              "set by the question file")
    assert arms["semantic"][:2] == ("search", {"method": "semantic"}) and arms[GRAPH][:2] == ("search", {})
    assert "rolled up to their records" in arms[GRAPH][2] and "not graph links" in arms[GRAPH][2]


# --- reading a ranking ---------------------------------------------------------------


def test_rows_are_read_in_rank_order_with_the_record_of_a_passage_and_the_generation():
    page = read_page(rows("[2] b:2 (t) · in rec:9", '[1] a:1 (t) · "title · in fake:3 · x"',
                          "    excerpt mentions c:3", generation="g-7"))
    assert page.ranking == [("a:1",), ("b:2", "rec:9")] and page.generation == "g-7" and page.format == "text"


def test_an_indented_row_marker_in_an_excerpt_is_not_a_row():
    page = read_page(rows("[1] a:1 (t) · x", "    [2] gold:1 (t) · quoted from a document", "  [3] gold:2 (t)"))
    assert page.ranking == [("a:1",)]


def test_a_json_page_is_read_with_its_generation():
    payload = {"pinned_generation_id": 41, "results": [{"node_id": "a:1"}, {"node_id": "c:1", "parent": {"node_id": "rec:2"}}]}
    page = read_page(json.dumps(payload))
    assert page.ranking == [("a:1",), ("c:1", "rec:2")] and page.generation == "41" and page.format == "json"


def test_text_with_no_rows_ranks_nothing():
    assert read_page("search ok · 0 rows\n\nnothing matched documentation_page:22").ranking == []


def test_recall_and_first_rank_count_only_the_top_k():
    ranking = [("x:1",), ("c:1", "g:2"), ("g:1",)]
    hits = score_hits(("g:1", "g:2"), ranking, 2)
    assert hits.found == ("g:2",) and hits.missed == ("g:1",) and hits.first_rank == 2
    assert hits.recall == 0.5 and hits.reciprocal_rank == 0.5
    assert score_hits(("g:1", "g:2"), ranking, 3).recall == 1.0


# --- the run -------------------------------------------------------------------------


@pytest.mark.parametrize("json_replies", [False, True])
def test_both_arms_are_scored_and_the_graph_arm_is_the_score(json_replies):
    tools = FakeSearch(fail={(BASELINE, Q3): "error · search: the lexical index is not built"},
                       json_replies=json_replies)
    report = run(load_context(FIXTURE), tools)
    data = report.to_dict()
    assert report.status == PASS and report.exit_code == 0
    assert data["arms"][GRAPH]["recall_at_k"] == pytest.approx(0.6667, abs=1e-4)
    assert data["arms"][GRAPH]["mrr"] == pytest.approx(0.6667, abs=1e-4)
    assert data["arms"][BASELINE]["recall_at_k"] == pytest.approx(0.5)
    assert data["arms"][BASELINE]["errors"] == 1 and data["arms"][GRAPH]["errors"] == 0 and data["errors"] == 1
    assert data["score"] == pytest.approx(0.6667, abs=1e-4)
    # the delta is over the two questions both arms executed: graph 1.0 and 1.0, baseline 1.0 and 0.5
    assert data["baseline_delta_questions"] == 2 and data["baseline_delta"] == pytest.approx(0.25)
    assert data["generation"] == "g-41" and data["generations"] == ["g-41"]
    assert data["question_count"] == 3 and data["questions_sha256"] == load_context(FIXTURE).sha256
    assert data["unscored"] == ["no-gold-yet"] and len(data["questions"]) == 3
    assert "lexical index over chunks and attributes" in data["arms"][BASELINE]["description"]
    by_id = {q["id"]: q for q in data["questions"]}
    assert by_id["site-downtime"]["arms"][GRAPH]["missed"] == ["event:fx-meeting-1"]  # rank 4, k is 3
    assert by_id["site-downtime"]["arms"][BASELINE]["status"] == EXECUTION_FAILED
    rules = by_id["replica-rules"]["arms"]
    assert rules[GRAPH]["found"] == ["documentation_page:fx-rules", "ticket:FX-7"] and rules[GRAPH]["first_rank"] == 1
    # the passage's record counts; the ticket named in a title, an excerpt or an indented line does not
    assert rules[BASELINE]["found"] == ["documentation_page:fx-rules"] and rules[BASELINE]["first_rank"] == 2
    assert rules[GRAPH]["format"] == ("json" if json_replies else "text")
    # no model: one search call per arm per scored question, with the question, k and the JSON request
    assert len(tools.calls) == 6 and all(meta == RESULT_META for _, _, meta in tools.calls)
    assert tools.calls[0][:2] == ("search", {"query": Q1, "limit": 3})
    assert tools.calls[1][:2] == ("search", {"method": "lexical", "rank_by": "self", "expand_concepts": False,
                                             "query": Q1, "limit": 3})


def test_a_failed_graph_call_fails_the_run_and_withholds_the_score():
    report = run(load_context(FIXTURE), FakeSearch(fail={(GRAPH, Q2): "error · search: timed out"}))
    data = report.to_dict()
    assert report.status == EXECUTION_FAILED and report.exit_code == 2 and data["score"] is None
    assert data["errors"] == 2 and data["arms"][GRAPH]["errors"] == 1  # the graph call and the baseline's Q3
    assert data["baseline_delta_questions"] == 1


def test_below_min_recall_fails():
    report = run(parse_context({**fixture_raw(), "min_recall": 0.9}, FIXTURE),
                 FakeSearch(fail={(BASELINE, Q3): "x"}))
    assert report.status == FAIL and report.exit_code == 1 and report.score == pytest.approx(2 / 3)


def test_a_missing_search_tool_is_execution_failed_with_no_score():
    report = run(load_context(FIXTURE), FakeSearch(specs=[ToolSpec("inspect", "", {})]))
    assert report.status == EXECUTION_FAILED and report.exit_code == 2 and report.score is None
    assert "does not offer search" in report.error


def test_an_unreachable_endpoint_is_execution_failed():
    @asynccontextmanager
    async def broken():
        raise agent_mod.ToolError("connection refused")
        yield  # pragma: no cover

    report = asyncio.run(run_context(load_context(FIXTURE), broken))
    assert report.status == EXECUTION_FAILED and "connection refused" in report.error


# --- runs on disk and what the gate takes from them ---------------------------------


def write_run(data_root, tools=None, *, cset=None):
    report = run(cset or load_context(FIXTURE), tools or FakeSearch(fail={(BASELINE, Q3): "x"}))
    return report, publish_run(report, context_eval_dir(data_root, report.bundle))


def test_runs_are_kept_one_directory_each_with_a_collision_suffix(tmp_path):
    report = run(load_context(FIXTURE), FakeSearch(fail={(BASELINE, Q3): "x"}))
    report.finished_at = "2026-10-27T03:01:44Z"
    root = tmp_path / "eval" / "context" / "demo"
    names = [publish_run(report, root)[1].parent.name for _ in range(3)]
    assert names == ["20261027T030144Z", "20261027T030144Z-2", "20261027T030144Z-3"]
    assert sorted(p.name for p in root.iterdir()) == names  # no temporary directory left behind


def test_a_crash_while_writing_leaves_no_report_where_readers_look(tmp_path, monkeypatch):
    from archi.evaluation import context as context_mod

    def half_written(directory, report):
        (directory / "report.json").write_text('{"schema": "archi-context-eval-report/1", "bun')
        raise KeyboardInterrupt

    report = run(load_context(FIXTURE), FakeSearch(fail={(BASELINE, Q3): "x"}))
    root = context_eval_dir(tmp_path, "demo")
    monkeypatch.setattr(context_mod, "_write", half_written)
    with pytest.raises(KeyboardInterrupt):
        publish_run(report, root)
    assert list(root.iterdir()) == []
    # had the process died before cleaning up, the half-written run would sit in a hidden directory,
    # which neither the gate nor the instance-health reader looks at
    hidden = root / ".20261027T030144Z.tmp-1"
    hidden.mkdir()
    (hidden / "report.json").write_text("{half")
    assert "no context eval of bundle demo" in read_context_eval(tmp_path, "demo").problem


def test_out_files_are_replaced_whole(tmp_path):
    report = run(load_context(FIXTURE), FakeSearch(fail={(BASELINE, Q3): "x"}))
    md, js = write_report(report, tmp_path / "out")
    md, js = write_report(report, tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["report.json", "report.md"]
    assert json.loads(js.read_text())["bundle"] == "demo"


def test_the_report_and_the_gate(tmp_path):
    report, (md, js) = write_run(tmp_path)
    assert js.parent.parent == tmp_path / "eval" / "context" / "demo"
    data = json.loads(js.read_text())
    assert data["schema"] == "archi-context-eval-report/1" and data["status"] == PASS
    text = md.read_text()
    assert "# archi eval --context: demo" in text and "| baseline |" in text and "| site-downtime | 1 | 0/1 | error |" in text
    assert "generation read: g-41" in text and "roll-up" in text

    reading = read_context_eval(tmp_path, "demo")
    assert reading.problem is None and reading.score == pytest.approx(0.6667, abs=1e-4)
    assert reading.generation == "g-41" and reading.path == str(js)
    spec = GateSpec.from_mapping({"min_context_eval_score": 0.6})
    assert evaluate_gate(spec, GateFacts(context_eval_score=reading.score)).ok
    assert not evaluate_gate(GateSpec.from_mapping({"min_context_eval_score": 0.8}),
                             GateFacts(context_eval_score=reading.score)).ok


def test_the_gate_reads_only_its_bundle_and_only_the_newest_run(tmp_path):
    assert "no context eval of bundle demo" in read_context_eval(tmp_path, "demo").problem
    write_run(tmp_path)
    assert "no context eval of bundle other" in read_context_eval(tmp_path, "other").problem
    # a newer run that could not execute is not passed over for the older good one
    write_run(tmp_path, FakeSearch(fail={(GRAPH, Q1): "boom"}))
    problem = read_context_eval(tmp_path, "demo").problem
    assert "did not execute" in problem and "status 'execution_failed'" in problem


def edit_newest(data_root, change):
    js = run_dirs(data_root / "eval" / "context" / "demo")[-1] / "report.json"
    data = json.loads(js.read_text())
    change(data)
    js.write_text(json.dumps(data))


@pytest.mark.parametrize("change, message", [
    (lambda d: d.update(schema="archi-eval-report/1"), "has schema 'archi-eval-report/1'"),
    (lambda d: d.update(bundle="cern-team"), "is a run of bundle 'cern-team'"),
    (lambda d: d.pop("questions_sha256"), "hash of its question file"),
    (lambda d: d.update(question_count=4), "records 4 scored questions but has 3"),
    (lambda d: d["arms"]["graph"].update(errors=1), "graph arm whose every call executed"),
    (lambda d: d.update(score=1.5), "no score from 0 to 1"),
    (lambda d: d.update(finished_at="yesterday"), "no usable finished_at"),
    (lambda d: d.update(arms=["graph"]), "graph arm whose every call executed"),  # not a mapping: no raise
    (lambda d: d.update(arms="graph"), "graph arm whose every call executed"),
    (lambda d: d["arms"]["graph"].update(arguments={"method": "semantic"}), "informational only"),
    (lambda d: d["arms"]["graph"].update(tool="query"), "informational only"),
    (lambda d: d.update(finished_at=(datetime.now(timezone.utc) + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")),
     "in the future"),
])
def test_an_invalid_report_fails_the_gate_with_the_reason(tmp_path, change, message):
    write_run(tmp_path)
    edit_newest(tmp_path, change)
    reading = read_context_eval(tmp_path, "demo")
    assert reading.score is None and message in reading.problem


def test_a_finish_time_a_little_ahead_is_clock_skew_and_accepted(tmp_path):
    write_run(tmp_path)
    edit_newest(tmp_path, lambda d: d.update(
        finished_at=(datetime.now(timezone.utc) + timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M:%SZ")))
    assert read_context_eval(tmp_path, "demo").problem is None


def test_a_directory_that_cannot_be_listed_is_a_named_failure(tmp_path, monkeypatch):
    write_run(tmp_path)

    def refuse(self):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "iterdir", refuse)
    reading = read_context_eval(tmp_path, "demo")
    assert reading.score is None and "cannot list the context evals" in reading.problem


def test_any_other_error_while_checking_is_a_named_failure(tmp_path, monkeypatch):
    from archi.ops import context_gate

    write_run(tmp_path)
    monkeypatch.setattr(context_gate, "_parse_time", lambda value: 1 / 0)
    reading = read_context_eval(tmp_path, "demo")
    assert reading.score is None and "could not be checked (ZeroDivisionError" in reading.problem


def test_the_gate_can_pin_k_and_a_minimum_question_count(tmp_path):
    write_run(tmp_path)  # k 3, 3 scored questions
    assert read_context_eval(tmp_path, "demo", k=3, min_questions=3).problem is None
    assert "used k=3, not the k=10 the gate pins" in read_context_eval(tmp_path, "demo", k=10).problem
    assert "scored 3 questions, fewer than the 4" in read_context_eval(tmp_path, "demo", min_questions=4).problem
    spec = GateSpec.from_mapping({"min_context_eval_score": 0.5, "context_eval_k": 10, "context_eval_min_questions": 40})
    assert (spec.context_eval_k, spec.context_eval_min_questions) == (10, 40)


def test_an_unreadable_report_fails_with_the_reason(tmp_path):
    _, (_, js) = write_run(tmp_path)
    js.write_text("{not json")
    assert "cannot be read" in read_context_eval(tmp_path, "demo").problem


def test_a_report_older_than_max_age_days_fails(tmp_path):
    write_run(tmp_path)
    later = datetime.now(timezone.utc) + timedelta(days=15)
    assert "more than the 14 days" in read_context_eval(tmp_path, "demo", now=later).problem
    assert read_context_eval(tmp_path, "demo", max_age_days=30, now=later).problem is None
    spec = GateSpec.from_mapping({"min_context_eval_score": 0.5, "max_age_days": 30})
    assert spec.max_age_days == 30.0


@pytest.mark.parametrize("raw, message", [
    ({"min_context_eval_score": 0.5, "max_age_days": 0}, "max_age_days must be a number of days above 0"),
    ({"min_context_eval_score": 0.5, "max_age_days": float("inf")}, "at most 3650"),
    ({"min_context_eval_score": 0.5, "max_age_days": float("nan")}, "at most 3650"),
    ({"min_context_eval_score": 0.5, "max_age_days": 3651}, "at most 3650"),
    ({"min_context_eval_score": 0.5, "context_eval_bundle": "../x"}, "cannot be used in a path"),
    ({"min_context_eval_score": 0.5, "context_eval_k": 0}, "context_eval_k must be an integer from 1 to 100"),
    ({"min_context_eval_score": 0.5, "context_eval_k": 2.5}, "context_eval_k must be an integer"),
    ({"min_context_eval_score": 0.5, "context_eval_min_questions": 0}, "at least 1"),
    ({"max_age_days": 3, "sources": ["a.b"]}, "apply only with min_context_eval_score"),
    ({"context_eval_k": 10, "sources": ["a.b"]}, "context_eval_k apply only with min_context_eval_score"),
])
def test_bad_gate_keys_are_refused(raw, message):
    with pytest.raises(GateSpecError, match=None) as err:
        GateSpec.from_mapping(raw)
    assert message in str(err.value)


def instance_at(tmp_path, bundles):
    from archi.stack.instance import Instance, Ports

    inst = Instance(dir=tmp_path / "inst", name="demo", bundles=list(bundles), data_root=tmp_path / "data",
                    runtime="podman", ports=Ports(graph_db=1, mcp=4242, webui=3))
    inst.dir.mkdir()
    (inst.dir / ".env").write_text("OKG_CHAT_MCP_TOKEN=instance-token\n")
    inst.save()
    return inst


def test_the_gate_takes_the_instances_only_bundle_or_the_named_one(tmp_path):
    inst = instance_at(tmp_path, ["demo"])
    write_run(inst.data_root)
    spec = GateSpec.from_mapping({"min_context_eval_score": 0.5})
    assert context_eval_reading(spec, inst.dir).score == pytest.approx(0.6667, abs=1e-4)
    named = GateSpec.from_mapping({"min_context_eval_score": 0.5, "context_eval_bundle": "cern-team"})
    assert "is not a bundle of this instance" in context_eval_reading(named, inst.dir).problem


def test_an_instance_with_several_bundles_needs_context_eval_bundle(tmp_path):
    inst = instance_at(tmp_path, ["demo", "cern-team"])
    write_run(inst.data_root)
    spec = GateSpec.from_mapping({"min_context_eval_score": 0.5})
    assert "context_eval_bundle" in context_eval_reading(spec, inst.dir).problem
    named = GateSpec.from_mapping({"min_context_eval_score": 0.5, "context_eval_bundle": "demo"})
    assert context_eval_reading(named, inst.dir).problem is None


def test_a_problem_fails_the_check_by_name():
    spec = GateSpec.from_mapping({"min_context_eval_score": 0.5})
    result = evaluate_gate(spec, GateFacts(context_eval_score=0.9, context_eval_problem="the run is too old"))
    assert result.failing == ("context_eval",) and "the run is too old" in result.message


# --- the command ---------------------------------------------------------------------


class Endpoint(FakeSearch):
    seen = []
    failing = {(BASELINE, Q3): "error · search: the lexical index is not built"}

    def __init__(self, url, token):
        super().__init__(fail=Endpoint.failing)
        Endpoint.seen.append((url, token))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None


@pytest.fixture
def endpoint(monkeypatch):
    Endpoint.seen = []
    Endpoint.failing = {(BASELINE, Q3): "error · search: the lexical index is not built"}
    monkeypatch.delenv("ARCHI_CONTEXT_EVAL_QUESTIONS", raising=False)
    monkeypatch.setattr(agent_mod, "McpToolBox", Endpoint)
    monkeypatch.setattr(agent_mod, "OpenAIChat", None)  # a context run never builds a model
    monkeypatch.setenv("OKG_CHAT_MCP_TOKEN", "t0ken")


def test_the_command_with_mcp_url_writes_reports_where_no_gate_reads(endpoint, tmp_path, capsys):
    code = main(["eval", "--context", "--questions", str(FIXTURE), "--mcp-url", "http://127.0.0.1:9/mcp",
                 "--out", str(tmp_path / "out")])
    assert code == 0
    data = json.loads((tmp_path / "out" / "report.json").read_text())
    assert data["status"] == PASS and set(data["arms"]) == {GRAPH, BASELINE} and data["k"] == 3
    assert (tmp_path / "out" / "report.md").is_file()
    assert Endpoint.seen == [("http://127.0.0.1:9/mcp", "t0ken")]
    captured = capsys.readouterr()
    assert "recall@3 graph 0.667, baseline 0.500; 1 failed calls" in captured.out
    assert "no context gate reads this run" in captured.err


def test_k_on_the_command_line_overrides_the_file(endpoint, tmp_path):
    main(["eval", "--context", "--questions", str(FIXTURE), "--mcp-url", "http://127.0.0.1:9/mcp",
          "--out", str(tmp_path / "out"), "--k", "4"])
    data = json.loads((tmp_path / "out" / "report.json").read_text())
    assert data["k"] == 4 and data["arms"][GRAPH]["recall_at_k"] == 1.0


def test_on_an_instance_each_run_is_kept_under_the_data_root_for_the_gate(endpoint, monkeypatch, tmp_path, capsys):
    from archi.stack.instance import Instance

    inst = instance_at(tmp_path, ["demo"])
    monkeypatch.setattr(Instance, "load", classmethod(lambda cls, d: inst))
    assert main(["eval", "--context", "--questions", str(FIXTURE)]) == 0
    assert main(["eval", "--context", "--questions", str(FIXTURE)]) == 0
    assert Endpoint.seen == [("http://127.0.0.1:4242/mcp", "instance-token")] * 2
    runs = run_dirs(tmp_path / "data" / "eval" / "context" / "demo")
    assert len(runs) == 2 and all((r / "report.json").is_file() and (r / "report.md").is_file() for r in runs)
    assert not (inst.dir / "eval").exists()  # nothing in the instance repository
    spec = GateSpec.from_mapping({"min_context_eval_score": 0.5})
    assert context_eval_reading(spec, inst.dir).path == str(runs[-1] / "report.json")


def test_on_an_instance_a_set_of_another_bundle_is_refused(endpoint, monkeypatch, tmp_path, capsys):
    from archi.stack.instance import Instance

    inst = instance_at(tmp_path, ["cern-team"])
    monkeypatch.setattr(Instance, "load", classmethod(lambda cls, d: inst))
    assert main(["eval", "--context", "--questions", str(FIXTURE)]) == 2
    assert "bundle demo, which this instance does not list" in capsys.readouterr().err
    assert Endpoint.seen == [] and not (tmp_path / "data").exists()


def test_on_an_instance_a_set_that_overrides_the_graph_arm_is_refused_unless_written_elsewhere(
        endpoint, monkeypatch, tmp_path, capsys):
    from archi.stack.instance import Instance

    inst = instance_at(tmp_path, ["demo"])
    monkeypatch.setattr(Instance, "load", classmethod(lambda cls, d: inst))
    custom = tmp_path / "custom.yaml"
    custom.write_text(yaml.safe_dump({**fixture_raw(), "arms": {"graph": {"tool": "search",
                                                                          "arguments": {"method": "semantic"}}}}))
    assert main(["eval", "--context", "--questions", str(custom)]) == 2
    assert "overrides the graph arm" in capsys.readouterr().err
    assert Endpoint.seen == [] and not (tmp_path / "data").exists()
    # written with --out it runs, as an informational report the gate would refuse anyway
    main(["eval", "--context", "--questions", str(custom), "--out", str(tmp_path / "elsewhere")])
    data = json.loads((tmp_path / "elsewhere" / "report.json").read_text())
    assert data["arms"]["graph"]["arguments"] == {"method": "semantic"}
    # a file that spells out the default graph arm overrides nothing
    same = tmp_path / "same.yaml"
    same.write_text(yaml.safe_dump({**fixture_raw(), "arms": {"graph": {"tool": "search", "arguments": {}}}}))
    assert load_context(same).graph_overridden is False and load_context(custom).graph_overridden is True


def test_a_failed_graph_call_exits_2_and_the_gate_refuses_the_run(endpoint, monkeypatch, tmp_path, capsys):
    from archi.stack.instance import Instance

    inst = instance_at(tmp_path, ["demo"])
    monkeypatch.setattr(Instance, "load", classmethod(lambda cls, d: inst))
    Endpoint.failing = {(GRAPH, Q2): "error · search: timed out"}
    assert main(["eval", "--context", "--questions", str(FIXTURE)]) == 2
    assert "graph-arm search calls failed" in capsys.readouterr().err
    spec = GateSpec.from_mapping({"min_context_eval_score": 0.1})
    assert "did not execute" in context_eval_reading(spec, inst.dir).problem


def test_a_question_file_outside_the_repository_loads_from_the_environment(endpoint, monkeypatch, tmp_path):
    """A private set (an instance's operator questions) is named at run time, never committed."""
    private = tmp_path / "private" / "operator-questions.yaml"
    private.parent.mkdir()
    private.write_text(FIXTURE.read_text())
    monkeypatch.setenv("ARCHI_CONTEXT_EVAL_QUESTIONS", str(private))
    assert main(["eval", "--context", "--mcp-url", "http://127.0.0.1:9/mcp", "--out", str(tmp_path / "out")]) == 0
    data = json.loads((tmp_path / "out" / "report.json").read_text())
    assert data["questions_file"] == str(private) and data["status"] == PASS


def test_the_command_line_path_wins_over_the_environment(endpoint, monkeypatch, tmp_path):
    monkeypatch.setenv("ARCHI_CONTEXT_EVAL_QUESTIONS", str(tmp_path / "elsewhere.yaml"))
    assert main(["eval", "--context", "--questions", str(FIXTURE), "--mcp-url", "http://127.0.0.1:9/mcp",
                 "--out", str(tmp_path / "out")]) == 0


@pytest.mark.parametrize("via_env", [False, True])
def test_a_missing_question_file_exits_2_and_names_the_path(endpoint, monkeypatch, tmp_path, capsys, via_env):
    missing = tmp_path / "nowhere" / "questions.yaml"
    args = ["eval", "--context", "--mcp-url", "http://127.0.0.1:9/mcp", "--out", str(tmp_path / "out")]
    if via_env:
        monkeypatch.setenv("ARCHI_CONTEXT_EVAL_QUESTIONS", str(missing))
    else:
        args += ["--questions", str(missing)]
    assert main(args) == 2
    err = capsys.readouterr().err
    assert f"question file {missing} does not exist" in err and "ARCHI_CONTEXT_EVAL_QUESTIONS" in err
    assert not (tmp_path / "out").exists() and Endpoint.seen == []


def test_a_bundle_with_no_context_set_needs_a_question_file(endpoint, tmp_path, capsys):
    code = main(["eval", "--context", "--bundle", "cern-team", "--mcp-url", "http://127.0.0.1:9/mcp",
                 "--out", str(tmp_path / "out")])
    assert code == 2 and "ships no context set" in capsys.readouterr().err


def test_a_question_file_for_another_bundle_is_refused(endpoint, tmp_path, capsys):
    code = main(["eval", "--context", "--bundle", "cern-team", "--questions", str(FIXTURE),
                 "--mcp-url", "http://127.0.0.1:9/mcp"])
    assert code == 2 and "not cern-team" in capsys.readouterr().err


def test_the_context_run_needs_a_token(endpoint, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("OKG_CHAT_MCP_TOKEN")
    code = main(["eval", "--context", "--questions", str(FIXTURE), "--mcp-url", "http://127.0.0.1:9/mcp",
                 "--out", str(tmp_path / "out")])
    assert code == 2 and "no MCP bearer token" in capsys.readouterr().err and not (tmp_path / "out").exists()
