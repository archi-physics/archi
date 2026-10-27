"""req.r3-context-quality.context-eval: `archi eval --context` over a fake MCP search tool.

No network, no model and no real question set: the fixture set and the search
results below are invented, in okg's row format (`[n] <id> (<subtype>) · ...`).
"""
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from archi.bundles.cli import main
from archi.evaluation import agent as agent_mod
from archi.evaluation.agent import ToolSpec
from archi.evaluation.context import (
    BASELINE,
    EXECUTION_FAILED,
    FAIL,
    GRAPH,
    PASS,
    ContextSetError,
    load_context,
    parse_context,
    ranked_ids,
    record_score,
    run_context,
    score_hits,
    write_report,
)
from archi.health import signals
from archi.ops.context_gate import (
    ContextEvalRecordError,
    GateFacts,
    GateSpec,
    evaluate_gate,
    read_context_eval_score,
)

FIXTURE = Path(__file__).parent / "fixtures" / "context.yaml"
SEARCH = ToolSpec("search", "Find records by words", {"type": "object"})


def rows(*lines):
    return 'search ok · "q" · rows 1-%d\n\n' % len(lines) + "\n".join(lines) + "\n"


#: What the fake search returns, by arm (told apart by `rank_by`) and question.
GRAPH_RESULTS = {
    "Which service moves files between storage sites?": rows(
        '[1] documentation_page:fx-transfer (documentation_page) · "Transfers" · source t.docsite',
        "    The transfer service moves files.",
    ),
    "How are dataset replica rules written?": rows(
        '[1] ticket:FX-7 (ticket) · "Rule syntax question" · source t.jira',
        '[2] documentation_page:fx-other (documentation_page) · "Other"',
        '[3] documentation_page:fx-rules (documentation_page) · "Replica rules"',
    ),
    "Where is a planned site downtime announced?": rows(
        '[1] documentation_page:fx-other (documentation_page) · "Other"',
        '[2] documentation_page:fx-a (documentation_page) · "A"',
        '[3] documentation_page:fx-b (documentation_page) · "B"',
        '[4] event:fx-meeting-1 (event) · "Ops meeting"',
    ),
}
BASELINE_RESULTS = {
    "Which service moves files between storage sites?": rows(
        "[1] document_chunk:c1 (document_chunk) · passage · in documentation_page:fx-transfer · source t.docsite",
    ),
    "How are dataset replica rules written?": rows(
        '[1] document_chunk:c2 (document_chunk) · passage · "see · in ticket:FX-7 · here" · in documentation_page:fx-other',
        "    the excerpt names ticket:FX-7, which is not a hit",
        "[2] document_chunk:c3 (document_chunk) · passage · in documentation_page:fx-rules",
    ),
}


class FakeSearch:
    def __init__(self, specs=(SEARCH,)):
        self.specs, self.calls = list(specs), []

    async def list_tools(self):
        return self.specs

    async def call(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        table = BASELINE_RESULTS if arguments.get("rank_by") == "self" else GRAPH_RESULTS
        text = table.get(arguments["query"])
        if text is None:
            return "error · search: the lexical index is not built", True
        return text, False


def opener(tools):
    @asynccontextmanager
    async def open_tools():
        yield tools

    return open_tools


def run(cset, tools):
    return asyncio.run(run_context(cset, opener(tools), mcp_url="http://127.0.0.1:1/mcp"))


# --- the question file ---------------------------------------------------------------


def test_the_fixture_set_parses_and_keeps_the_smoke_shape():
    cset = load_context(FIXTURE)
    assert cset.bundle == "demo" and cset.k == 3 and cset.min_recall == 0.5
    assert [q.id for q in cset.scored] == ["transfer-tool", "replica-rules", "site-downtime"]
    assert cset.questions[-1].gold == ()
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
    (base(questions=[{"id": "a", "question": "q?", "gold": ["x:1"], "answer": "y"}]), "unknown keys"),
    (base(arms={"baseline": {"tool": "search", "arguments": {"query": "fixed"}}}), "may not set ['query']"),
    (base(arms={"baseline": {"tool": "search", "args": {}}}), "takes only `tool` and `arguments`"),
])
def test_a_bad_question_file_is_refused_by_name(raw, message):
    with pytest.raises(ContextSetError) as err:
        parse_context(raw)
    assert message in str(err.value)


def test_an_arm_in_the_file_replaces_the_default_of_that_name():
    cset = parse_context(base(arms={"baseline": {"arguments": {"method": "lexical", "where": {"type": "document_chunk"}}},
                                    "semantic": {"tool": "search", "arguments": {"method": "semantic"}}}))
    arms = {a.name: (a.tool, a.arguments) for a in cset.arms}
    assert arms[BASELINE] == ("search", {"method": "lexical", "where": {"type": "document_chunk"}})
    assert arms["semantic"] == ("search", {"method": "semantic"}) and arms[GRAPH] == ("search", {})


# --- reading a ranking ---------------------------------------------------------------


def test_rows_are_read_in_rank_order_with_the_record_of_a_passage():
    text = rows("[2] b:2 (t) · in rec:9", '[1] a:1 (t) · "title · in fake:3 · x"', "    excerpt mentions c:3")
    assert ranked_ids(text) == [("a:1",), ("b:2", "rec:9")]


def test_a_json_page_is_read_too():
    page = json.dumps({"results": [{"node_id": "a:1"}, {"node_id": "c:1", "parent": {"node_id": "rec:2"}}]})
    assert ranked_ids(page) == [("a:1",), ("c:1", "rec:2")]


def test_text_with_no_rows_ranks_nothing():
    assert ranked_ids("search ok · 0 rows\n\nnothing matched documentation_page:22") == []


def test_recall_and_first_rank_count_only_the_top_k():
    ranking = [("x:1",), ("c:1", "g:2"), ("g:1",)]
    hits = score_hits(("g:1", "g:2"), ranking, 2)
    assert hits.found == ("g:2",) and hits.missed == ("g:1",) and hits.first_rank == 2
    assert hits.recall == 0.5 and hits.reciprocal_rank == 0.5
    assert score_hits(("g:1", "g:2"), ranking, 3).recall == 1.0


# --- the run -------------------------------------------------------------------------


def test_both_arms_are_scored_and_the_graph_arm_is_the_score():
    tools = FakeSearch()
    report = run(load_context(FIXTURE), tools)
    data = report.to_dict()
    assert report.status == PASS and report.exit_code == 0
    assert data["arms"][GRAPH]["recall_at_k"] == pytest.approx(0.6667, abs=1e-4)
    assert data["arms"][GRAPH]["mrr"] == pytest.approx(0.6667, abs=1e-4)
    assert data["arms"][BASELINE]["recall_at_k"] == pytest.approx(0.5)
    assert data["arms"][BASELINE]["mrr"] == pytest.approx(0.5)
    assert data["arms"][BASELINE]["errors"] == 1 and data["arms"][GRAPH]["errors"] == 0
    assert data["score"] == pytest.approx(0.6667, abs=1e-4) and data["baseline_delta"] == pytest.approx(0.1667, abs=1e-4)
    assert data["unscored"] == ["no-gold-yet"] and len(data["questions"]) == 3
    by_id = {q["id"]: q for q in data["questions"]}
    assert by_id["site-downtime"]["arms"][GRAPH]["missed"] == ["event:fx-meeting-1"]  # rank 4, k is 3
    assert by_id["site-downtime"]["arms"][BASELINE]["status"] == EXECUTION_FAILED
    assert "index is not built" in by_id["site-downtime"]["arms"][BASELINE]["error"]
    rules = by_id["replica-rules"]["arms"]
    assert rules[GRAPH]["found"] == ["documentation_page:fx-rules", "ticket:FX-7"] and rules[GRAPH]["first_rank"] == 1
    # the passage's record counts; the ticket named in its title and excerpt does not
    assert rules[BASELINE]["found"] == ["documentation_page:fx-rules"] and rules[BASELINE]["first_rank"] == 2
    # no model: one search call per arm per scored question, with the question and k
    assert len(tools.calls) == 6
    assert tools.calls[0] == ("search", {"query": "Which service moves files between storage sites?", "limit": 3})
    assert tools.calls[1] == ("search", {"method": "lexical", "rank_by": "self", "expand_concepts": False,
                                         "query": "Which service moves files between storage sites?", "limit": 3})


def test_below_min_recall_fails():
    cset = load_context(FIXTURE)
    report = run(parse_context({**_raw(), "min_recall": 0.9}, cset.path), FakeSearch())
    assert report.status == FAIL and report.exit_code == 1 and report.score == pytest.approx(2 / 3)


def _raw():
    import yaml

    return yaml.safe_load(FIXTURE.read_text())


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


# --- the reports and the recorded score ----------------------------------------------


def test_reports_are_written_and_the_score_reaches_the_gate_and_the_health_monitor(tmp_path):
    report = run(load_context(FIXTURE), FakeSearch())
    md, js = write_report(report, tmp_path / "out")
    data = json.loads(js.read_text())
    assert data["schema"] == "archi-context-eval-report/1" and data["status"] == PASS
    text = md.read_text()
    assert "# archi eval --context: demo" in text and "| baseline |" in text and "| site-downtime | 1 | 0/1 | error |" in text

    instance = tmp_path / "instance"
    assert read_context_eval_score(instance) is None
    spec = GateSpec.from_mapping({"min_context_eval_score": 0.6})
    assert evaluate_gate(spec, GateFacts(context_eval_score=read_context_eval_score(instance))).failing == ("context_eval",)

    path = record_score(report, instance, report_path=js)
    assert path.parent == instance / "eval" / "reports" and path.name.startswith("context-demo-")
    record = json.loads(path.read_text())
    assert record["score"] == pytest.approx(0.6667, abs=1e-4) and record["baseline_recall_at_k"] == 0.5
    assert read_context_eval_score(instance) == record["score"]
    assert evaluate_gate(spec, GateFacts(context_eval_score=read_context_eval_score(instance))).ok
    assert not evaluate_gate(GateSpec.from_mapping({"min_context_eval_score": 0.8}),
                             GateFacts(context_eval_score=read_context_eval_score(instance))).ok
    # instance-health reads the same records: one is no trend yet
    assert signals.eval_problems(instance) == ([], "only one eval report exists; a trend needs two")


def test_the_latest_record_wins_and_a_broken_one_is_an_error(tmp_path):
    reports = tmp_path / "eval" / "reports"
    reports.mkdir(parents=True)
    (reports / "context-demo-a.json").write_text(json.dumps({"finished_at": "2026-10-26T10:00:00Z", "score": 0.9}))
    (reports / "context-demo-b.json").write_text(json.dumps({"finished_at": "2026-10-26T11:00:00Z", "score": 0.4}))
    assert read_context_eval_score(tmp_path) == 0.4
    (reports / "context-demo-c.json").write_text("{not json")
    with pytest.raises(ContextEvalRecordError, match="context-demo-c.json"):
        read_context_eval_score(tmp_path)


def test_a_run_that_could_not_execute_records_no_score(tmp_path):
    report = run(load_context(FIXTURE), FakeSearch(specs=[]))
    assert record_score(report, tmp_path) is None and not (tmp_path / "eval").exists()


# --- the command ---------------------------------------------------------------------


class Endpoint(FakeSearch):
    seen = []

    def __init__(self, url, token):
        super().__init__()
        Endpoint.seen.append((url, token))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None


@pytest.fixture
def endpoint(monkeypatch):
    Endpoint.seen = []
    monkeypatch.delenv("ARCHI_CONTEXT_EVAL_QUESTIONS", raising=False)
    monkeypatch.setattr(agent_mod, "McpToolBox", Endpoint)
    monkeypatch.setattr(agent_mod, "OpenAIChat", None)  # a context run never builds a model
    monkeypatch.setenv("OKG_CHAT_MCP_TOKEN", "t0ken")


def test_the_command_with_mcp_url_writes_reports_and_records_nothing(endpoint, tmp_path, capsys):
    code = main(["eval", "--context", "--questions", str(FIXTURE), "--mcp-url", "http://127.0.0.1:9/mcp",
                 "--out", str(tmp_path / "out")])
    assert code == 0
    data = json.loads((tmp_path / "out" / "report.json").read_text())
    assert data["status"] == PASS and set(data["arms"]) == {GRAPH, BASELINE} and data["k"] == 3
    assert (tmp_path / "out" / "report.md").is_file()
    assert Endpoint.seen == [("http://127.0.0.1:9/mcp", "t0ken")]
    captured = capsys.readouterr()
    assert "recall@3 graph 0.667, baseline 0.500" in captured.out and "no score is recorded" in captured.err


def test_k_on_the_command_line_overrides_the_file(endpoint, tmp_path):
    main(["eval", "--context", "--questions", str(FIXTURE), "--mcp-url", "http://127.0.0.1:9/mcp",
          "--out", str(tmp_path / "out"), "--k", "4"])
    data = json.loads((tmp_path / "out" / "report.json").read_text())
    assert data["k"] == 4 and data["arms"][GRAPH]["recall_at_k"] == 1.0


def test_the_command_on_an_instance_records_the_score_for_the_gate(endpoint, monkeypatch, tmp_path, capsys):
    from archi.stack.instance import Instance, Ports

    inst = Instance(dir=tmp_path / "inst", name="demo", bundles=["cern-team"], data_root=tmp_path / "data",
                    runtime="podman", ports=Ports(graph_db=1, mcp=4242, webui=3))
    inst.dir.mkdir()
    (inst.dir / ".env").write_text("OKG_CHAT_MCP_TOKEN=instance-token\n")
    monkeypatch.setattr(Instance, "load", classmethod(lambda cls, d: inst))
    assert main(["eval", "--context", "--questions", str(FIXTURE)]) == 0
    assert Endpoint.seen == [("http://127.0.0.1:4242/mcp", "instance-token")]
    assert (tmp_path / "data" / "eval" / "context" / "demo" / "report.json").is_file()
    assert read_context_eval_score(inst.dir) == pytest.approx(0.6667, abs=1e-4)
    assert "recorded for the context gate" in capsys.readouterr().out


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
