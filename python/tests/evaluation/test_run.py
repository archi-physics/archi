"""The agent loop, the run and the report, with a fake model and a fake MCP (no network)."""
import asyncio
import json
import subprocess
import sys
from contextlib import asynccontextmanager

import pytest

from archi.evaluation import agent as agent_mod
from archi.evaluation.agent import ChatReply, ModelError, ToolCall, ToolError, ToolSpec
from archi.evaluation.runner import EXECUTION_FAILED, FAIL, PASS, run_smoke, write_report
from archi.evaluation.smoke import parse_smoke

SEARCH = ToolSpec("search", "Search the graph", {"type": "object", "properties": {"query": {"type": "string"}}})
RESULT = (
    'search ok · "q" · source gp.docsite · rows 1-1\n\n'
    '[1] documentation_page:22 (documentation_page) · "Intro · FTS3"\n'
    "    FTS3 is the File Transfer Service.\n"
)


class FakeTools:
    def __init__(self, specs=(SEARCH,), text=RESULT, fail=False):
        self.specs, self.text, self.fail, self.calls = list(specs), text, fail, []

    async def list_tools(self):
        return self.specs

    async def call(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if self.fail:
            raise RuntimeError("boom")
        return self.text, False


def opener(tools):
    @asynccontextmanager
    async def open_tools():
        yield tools

    return open_tools


class FakeModel:
    """Replies are scripted per question: a list of ChatReply, consumed in order."""

    name = "fake:1b"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    def complete(self, messages, tools):
        self.seen.append((list(messages), tools))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def searches(query="FTS3"):
    return ChatReply("", (ToolCall("c1", "search", {"query": query}),), 10, 2)


def answers(text):
    return ChatReply(text, (), 20, 5)


def smoke_of(*ids, threshold=1):
    rows = [{"id": i, "question": f"question {i}?", "must_cite": {"sources": ["docsite"]}} for i in ids]
    return parse_smoke({"bundle": "demo", "pass_threshold": threshold, "questions": rows})


def run(smoke, model, tools, **kw):
    return asyncio.run(run_smoke(smoke, model, opener(tools), model_url="http://127.0.0.1:11434/v1",
                                 mcp_url="http://127.0.0.1:1/mcp", **kw))


def test_a_cited_answer_passes_and_the_trace_is_recorded():
    model = FakeModel(searches(), answers("FTS3 is a transfer service.\nSources: documentation_page:22"))
    report = run(smoke_of("a"), model, FakeTools())
    q = report.questions[0]
    assert (q.status, q.reasons) == (PASS, [])
    assert q.cited == [{"id": "documentation_page:22", "subtype": "documentation_page", "source": "gp.docsite"}]
    assert q.tool_calls == [{"name": "search", "arguments": {"query": "FTS3"}, "chars": len(RESULT), "is_error": False}]
    assert (q.prompt_tokens, q.completion_tokens) == (30, 7)
    assert report.status == PASS and report.exit_code == 0 and report.tools == ["search"]
    assert report.prompt_hash.startswith("sha256:")


def test_the_model_sees_the_system_prompt_the_question_and_the_tool_result():
    model = FakeModel(searches(), answers("done documentation_page:22"))
    run(smoke_of("a"), model, FakeTools())
    first, tools = model.seen[0]
    assert first[0]["role"] == "system" and first[1] == {"role": "user", "content": "question a?"}
    assert tools[0]["function"]["name"] == "search"
    second, _ = model.seen[1]
    assert second[-1]["role"] == "tool" and "FTS3 is the File Transfer Service." in second[-1]["content"]


def test_an_answer_with_no_citation_fails_not_errors():
    report = run(smoke_of("a"), FakeModel(searches(), answers("FTS3 is a transfer service.")), FakeTools())
    q = report.questions[0]
    assert q.status == FAIL and "cited no node from source docsite" in q.reasons[0]
    assert report.status == FAIL and report.exit_code == 1


def test_a_made_up_id_fails():
    report = run(smoke_of("a"), FakeModel(searches(), answers("Sources: documentation_page:7")), FakeTools())
    assert report.questions[0].status == FAIL


def test_an_answer_without_any_tool_call_cannot_cite():
    report = run(smoke_of("a"), FakeModel(answers("From memory. Sources: documentation_page:22")), FakeTools())
    assert report.questions[0].status == FAIL and report.questions[0].tool_calls == []


def test_zero_mcp_tools_is_execution_failed_for_the_whole_run():
    model = FakeModel()
    report = run(smoke_of("a", "b"), model, FakeTools(specs=()))
    assert [q.status for q in report.questions] == [EXECUTION_FAILED] * 2
    assert report.status == EXECUTION_FAILED and report.exit_code == 2
    assert "no tools" in report.error and model.seen == []


def test_an_unreachable_mcp_endpoint_is_execution_failed():
    @asynccontextmanager
    async def refused():
        raise ToolError("http://127.0.0.1:1/mcp: connection refused")
        yield  # pragma: no cover

    model = FakeModel()
    report = asyncio.run(run_smoke(smoke_of("a"), model, refused))
    assert report.status == EXECUTION_FAILED and "connection refused" in report.error


def test_a_model_error_is_execution_failed_for_that_question_only():
    model = FakeModel(ModelError("model not found"), searches(), answers("documentation_page:22"))
    report = run(smoke_of("a", "b", threshold=1), model, FakeTools())
    assert [q.status for q in report.questions] == [EXECUTION_FAILED, PASS]
    assert report.questions[0].error == "model not found"
    assert report.status == PASS


def test_every_question_erroring_makes_the_run_execution_failed():
    model = FakeModel(ModelError("down"), ModelError("down"))
    report = run(smoke_of("a", "b"), model, FakeTools())
    assert report.status == EXECUTION_FAILED and report.exit_code == 2


def test_the_threshold_decides_pass_or_fail():
    def result(threshold):
        model = FakeModel(searches(), answers("documentation_page:22"), searches(), answers("no cite"))
        return run(smoke_of("a", "b", threshold=threshold), model, FakeTools())

    assert result(1).status == PASS and result(1).passed == 1
    assert result(2).status == FAIL and result(2).exit_code == 1


def test_a_failing_tool_is_reported_to_the_model_not_raised():
    model = FakeModel(searches(), answers("could not read it"))
    report = run(smoke_of("a"), model, FakeTools(fail=True))
    assert report.questions[0].tool_calls[0]["is_error"] is True
    assert "error: RuntimeError: boom" in model.seen[1][0][-1]["content"]


def test_an_unknown_tool_name_is_an_error_the_model_can_see():
    bad = ChatReply("", (ToolCall("c1", "delete_everything", {}),))
    model = FakeModel(bad, answers("documentation_page:22"))
    tools = FakeTools()
    report = run(smoke_of("a"), model, tools)
    assert tools.calls == [] and report.questions[0].tool_calls[0]["is_error"]


def test_the_step_limit_ends_a_model_that_never_answers():
    loop = [searches() for _ in range(3)]
    report = run(smoke_of("a"), FakeModel(*loop), FakeTools(), max_steps=3)
    q = report.questions[0]
    assert q.status == FAIL and "step limit" in q.reasons[0] and len(q.tool_calls) == 3


def test_long_tool_output_is_cut_for_the_model_but_ids_come_from_what_it_saw():
    text = RESULT + "x" * 5000 + "\n[2] documentation_page:99 (documentation_page) · source gp.docsite\n"
    model = FakeModel(searches(), answers("documentation_page:99"))
    report = run(smoke_of("a"), model, FakeTools(text=text), max_tool_chars=200)
    assert report.questions[0].status == FAIL
    assert model.seen[1][0][-1]["content"].endswith("[cut]")


def test_the_report_is_written_as_markdown_and_json(tmp_path):
    model = FakeModel(searches(), answers("FTS3.\nSources: documentation_page:22"), searches(), answers("none"))
    report = run(smoke_of("a", "b", threshold=2), model, FakeTools())
    md, js = write_report(report, tmp_path / "out")
    data = json.loads(js.read_text())
    assert data["schema"] == "archi-eval-report/1"
    assert {k: data[k] for k in ("bundle", "status", "passed", "total", "pass_threshold", "model")} == {
        "bundle": "demo", "status": "fail", "passed": 1, "total": 2, "pass_threshold": 2, "model": "fake:1b"}
    assert data["tools"] == ["search"] and data["prompt_hash"] == report.prompt_hash
    assert "#619" in data["note"]
    first = data["questions"][0]
    assert set(first) == {"id", "question", "status", "reasons", "cited", "lookups", "tool_calls",
                          "prompt_tokens", "completion_tokens", "seconds", "answer", "error"}
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["report.json", "report.md"]
    assert first["cited"][0]["id"] == "documentation_page:22"
    text = md.read_text()
    assert "# archi eval: demo" in text and "**fail**: 1 of 2" in text and "| a | pass |" in text
    assert "#619" in text


def test_the_prompt_hash_changes_with_the_tools():
    one = agent_mod.prompt_hash([SEARCH])
    two = agent_mod.prompt_hash([SEARCH, ToolSpec("inspect", "Read", {})])
    assert one != two and one == agent_mod.prompt_hash([SEARCH])


@pytest.mark.parametrize("url", ["http://127.0.0.1:11434/v1", "http://localhost:11500/v1", "http://ollama:11434/v1",
                                 "http://10.0.0.5:11434/v1", "http://192.168.1.2/v1"])
def test_local_model_endpoints_are_allowed(url):
    agent_mod.require_local(url)


@pytest.mark.parametrize("url", ["https://api.openai.com/v1", "https://api.anthropic.com/v1", "http://8.8.8.8/v1"])
def test_a_model_endpoint_off_this_network_is_refused(url):
    with pytest.raises(ModelError, match="no paid calls"):
        agent_mod.require_local(url)


def test_the_eval_harness_is_not_imported_by_other_commands():
    code = (
        "import sys\n"
        "from archi.bundles.cli import build_parser\n"
        "build_parser(load_ops=False).parse_args(['bundles', 'list'])\n"
        "print('archi.evaluation' in sys.modules, 'mcp' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False False"


def test_the_system_message_carries_the_server_instructions_and_the_chat_prompt():
    tools = FakeTools()
    tools.instructions = "Use search first; filter by type only with a type the graph card lists."
    model = FakeModel(searches(), answers("documentation_page:22"))
    report = run(smoke_of("a"), model, tools, chat_prompt="You are the team chat.", chat_prompt_file="p.md")
    system = model.seen[0][0][0]["content"]
    # The answer format comes last, after the chat prompt's own citation style.
    assert system.endswith(agent_mod.ANSWER_FORMAT)
    assert system.index("Use search first") < system.index("You are the team chat.")
    assert system.index("You are the team chat.") < system.index(agent_mod.ANSWER_FORMAT)
    assert report.mcp_instructions_hash == agent_mod.sha(tools.instructions)
    assert report.chat_prompt_hash == agent_mod.sha("You are the team chat.")
    assert report.system_prompt_hash == agent_mod.sha(system)
    data = report.to_dict()
    assert data["chat_prompt_file"] == "p.md" and data["mcp_instructions_hash"].startswith("sha256:")


def test_a_server_without_instructions_and_no_chat_prompt_is_recorded_as_none():
    model = FakeModel(searches(), answers("documentation_page:22"))
    report = run(smoke_of("a"), model, FakeTools())
    assert model.seen[0][0][0]["content"] == agent_mod.ANSWER_FORMAT
    assert report.mcp_instructions_hash is None and report.chat_prompt_hash is None


def test_the_answer_format_examples_are_no_id_a_smoke_set_expects():
    import re
    from pathlib import Path

    examples = re.findall(r"\b[a-z_]+:[\w-]+", agent_mod.ANSWER_FORMAT)
    assert examples
    repo = Path(__file__).resolve().parents[3]
    for path in sorted((repo / "bundles").glob("*/eval/smoke.yaml")):
        text = path.read_text(encoding="utf-8")
        for example in examples:
            assert example.split(":", 1)[1] not in text, (example, path)


class ListFails(FakeTools):
    async def list_tools(self):
        raise RuntimeError("session dropped")


def test_an_unexpected_error_is_execution_failed_and_keeps_the_report():
    report = run(smoke_of("a", "b"), FakeModel(), ListFails())
    assert report.status == EXECUTION_FAILED and report.exit_code == 2
    assert "session dropped" in report.error
    assert [q.status for q in report.questions] == [EXECUTION_FAILED, EXECUTION_FAILED]


def test_a_run_that_stops_midway_keeps_the_answered_questions(monkeypatch):
    async def broken_one(question, *a, **kw):
        if question.id == "b":
            raise ValueError("model reply was not a mapping")
        return await real_one(question, *a, **kw)

    from archi.evaluation import runner as runner_mod

    real_one = runner_mod._one
    monkeypatch.setattr(runner_mod, "_one", broken_one)
    model = FakeModel(searches(), answers("FTS3 moves files.\nSources: documentation_page:22"))
    report = run(smoke_of("a", "b", "c"), model, FakeTools())
    assert [q.id for q in report.questions] == ["a", "b", "c"]
    assert [q.status for q in report.questions] == [PASS, EXECUTION_FAILED, EXECUTION_FAILED]
    assert report.exit_code == 2 and "not a mapping" in report.error


def test_a_reply_whose_message_is_not_a_mapping_is_a_model_error(monkeypatch):
    class Response:
        status_code = 200
        text = '{"choices": [{"message": "hi"}]}'

        def json(self):
            return json.loads(self.text)

    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **kw: Response())
    with pytest.raises(ModelError, match="unreadable reply"):
        agent_mod.OpenAIChat("http://127.0.0.1:1/v1", "m").complete([], [])


FILTER = ToolSpec("filter", "Filter the graph", {"type": "object", "properties": {"where": {"type": "object"}}})
INSPECT = ToolSpec("inspect", "Read one record", {"type": "object", "properties": {"target": {"type": "object"}}})
#: what `filter` returns when the model selects only `node_id` and `attrs`: no subtype, no source
FILTER_BARE = "filter ok · 1-1 of 1 · complete\n[1] ops_incident:inc-1\n  title: GPU errors on gpu03\n"
INSPECT_INCIDENT = (
    "inspect ok · ops_incident:inc-1 (ops_incident) · complete\n"
    "source workflow:archi-ops\n"
    "  --- text\nGPU errors on gpu03, see ticket:9\n"
)


class ScriptedTools(FakeTools):
    """Answers `filter` with a bare row and `inspect` with the full record."""

    def __init__(self, specs=(FILTER, INSPECT), inspect_fails=False):
        super().__init__(specs=specs)
        self.inspect_fails = inspect_fails

    async def call(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if name == "inspect":
            if self.inspect_fails:
                raise RuntimeError("server went away")
            return INSPECT_INCIDENT, False
        return FILTER_BARE, False


def filters():
    return ChatReply("", (ToolCall("c1", "filter", {"where": {"type": "ops_incident"}}),), 10, 2)


def incident_smoke():
    rows = [{"id": "q", "question": "Any open incident on gpu03?", "must_cite": {"subtypes": ["ops_incident"]}}]
    return parse_smoke({"bundle": "demo", "pass_threshold": 1, "questions": rows})


def test_a_cited_node_whose_subtype_no_tool_said_is_looked_up_with_inspect():
    tools = ScriptedTools()
    report = run(incident_smoke(), FakeModel(filters(), answers("Yes. Sources: ops_incident:inc-1")), tools)
    q = report.questions[0]
    assert (q.status, q.lookups) == (PASS, [{"id": "ops_incident:inc-1", "ok": True}])
    assert q.cited == [{"id": "ops_incident:inc-1", "subtype": "ops_incident", "source": "workflow:archi-ops"}]
    assert tools.calls[-1] == ("inspect", {"target": {"kind": "node", "node_id": "ops_incident:inc-1"}})
    assert report.to_dict()["questions"][0]["lookups"] == [{"id": "ops_incident:inc-1", "ok": True}]


def test_a_lookup_does_not_credit_ids_the_model_was_never_given():
    # ticket:9 appears only in the lookup's text, never in a result the model saw.
    tools = ScriptedTools()
    rows = [{"id": "q", "question": "x?", "must_cite": {"ids": ["ticket:9"]}}]
    smoke = parse_smoke({"bundle": "demo", "pass_threshold": 1, "questions": rows})
    report = run(smoke, FakeModel(filters(), answers("See ops_incident:inc-1 and ticket:9")), tools)
    assert report.questions[0].status == FAIL


def test_no_lookup_without_an_inspect_tool_and_no_lookup_when_the_row_said_enough():
    tools = ScriptedTools(specs=(FILTER,))
    report = run(incident_smoke(), FakeModel(filters(), answers("Sources: ops_incident:inc-1")), tools)
    assert report.questions[0].status == FAIL and report.questions[0].lookups == []
    assert [c[0] for c in tools.calls] == ["filter"]
    report = run(smoke_of("a"), FakeModel(searches(), answers("Sources: documentation_page:22")), FakeTools())
    assert report.questions[0].lookups == []


def test_a_failed_lookup_fails_the_question_not_the_run():
    tools = ScriptedTools(inspect_fails=True)
    report = run(incident_smoke(), FakeModel(filters(), answers("Sources: ops_incident:inc-1")), tools)
    q = report.questions[0]
    assert q.status == FAIL and "cited no node of subtype ops_incident" in q.reasons[0]
    assert q.lookups == [{"id": "ops_incident:inc-1", "ok": False, "error": "RuntimeError: server went away"}]
    assert report.error is None


#: an incident whose text mentions a ticket the model never read
FILTER_MENTION = (
    "filter ok · 1-1 of 1 · complete\n[1] ops_incident:inc-1 (ops_incident) · source workflow:archi-ops\n"
    "  ticket: tracker_ticket:redmine:4690\n"
)


class MentionTools(ScriptedTools):
    async def call(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if name == "inspect":
            return ("inspect ok · tracker_ticket:redmine:4690 (ticket) · complete\nsource px.tickets\n"), False
        return FILTER_MENTION, False


def test_an_id_seen_only_inside_another_record_is_never_looked_up():
    tools = MentionTools()
    rows = [{"id": "q", "question": "Which ticket?", "must_cite": {"sources": ["tickets"]}}]
    smoke = parse_smoke({"bundle": "demo", "pass_threshold": 1, "questions": rows})
    report = run(smoke, FakeModel(filters(), answers("Sources: tracker_ticket:redmine:4690")), tools)
    q = report.questions[0]
    assert q.status == FAIL and q.lookups == []
    assert [c[0] for c in tools.calls] == ["filter"]


def test_no_lookup_once_the_check_is_already_met():
    tools = MentionTools()
    report = run(incident_smoke(), FakeModel(filters(), answers("ops_incident:inc-1, see tracker_ticket:redmine:4690")),
                 tools)
    assert report.questions[0].status == PASS and report.questions[0].lookups == []


# --- dated reports (#854): every default run is kept, the newest `keep` stay -----------------------


def fixture_report(started_at, passed=1, total=2):
    from archi.evaluation.runner import QuestionResult, Report

    return Report(bundle="demo", model="fake:1b", model_url="http://127.0.0.1:11434/v1",
                  mcp_url="http://127.0.0.1:18080/mcp", pass_threshold=1, smoke_file="bundles/demo/eval/smoke.yaml",
                  started_at=started_at,
                  questions=[QuestionResult(f"q{i}", f"question {i}?", PASS if i < passed else FAIL)
                             for i in range(total)])


def test_two_default_runs_leave_two_dated_reports_and_latest_names_the_newest(tmp_path):
    from archi.evaluation.runner import publish_report

    root = tmp_path / "eval" / "demo"
    md1, js1, gone1 = publish_report(fixture_report("2026-10-30T10:00:00Z"), root)
    md2, js2, gone2 = publish_report(fixture_report("2026-10-31T10:00:00Z", passed=2), root)
    assert gone1 == gone2 == []
    assert js1 == root / "20261030T100000Z" / "report.json" and js2 == root / "20261031T100000Z" / "report.json"
    assert md1.is_file() and js1.is_file() and md2.is_file() and js2.is_file()
    assert json.loads(js1.read_text())["passed"] == 1 and json.loads(js2.read_text())["passed"] == 2
    latest = json.loads((root / "latest.json").read_text())
    assert latest == {"schema": "archi-eval-latest/1", "run": "20261031T100000Z",
                      "report": "20261031T100000Z/report.json", "started_at": "2026-10-31T10:00:00Z",
                      "status": "pass"}
    assert sorted(p.name for p in root.iterdir()) == ["20261030T100000Z", "20261031T100000Z", "latest.json"]


def test_two_runs_in_the_same_second_both_stay(tmp_path):
    from archi.evaluation.runner import publish_report

    publish_report(fixture_report("2026-10-30T10:00:00Z"), tmp_path)
    _, js, _ = publish_report(fixture_report("2026-10-30T10:00:00Z"), tmp_path)
    assert js.parent.name == "20261030T100000Z-2"
    assert json.loads((tmp_path / "latest.json").read_text())["run"] == "20261030T100000Z-2"


def test_only_the_newest_keep_runs_stay_and_nothing_else_is_deleted(tmp_path):
    from archi.evaluation.runner import publish_report

    (tmp_path / "report.json").write_text("{}")  # an older archi wrote straight into the bundle folder
    (tmp_path / "2026-10-01").mkdir()  # a folder a person made with --out
    (tmp_path / "2026-10-01" / "report.json").write_text("{}")
    deleted = []
    for day in range(1, 6):
        deleted += publish_report(fixture_report(f"2026-10-{day:02d}T10:00:00Z"), tmp_path, keep=3)[2]
    assert [p.name for p in deleted] == ["20261001T100000Z", "20261002T100000Z"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "2026-10-01", "20261003T100000Z", "20261004T100000Z", "20261005T100000Z", "latest.json", "report.json"]


def test_keep_zero_keeps_every_run(tmp_path):
    from archi.evaluation.runner import publish_report

    for day in range(1, 5):
        publish_report(fixture_report(f"2026-10-{day:02d}T10:00:00Z"), tmp_path, keep=0)
    assert len([p for p in tmp_path.iterdir() if p.is_dir()]) == 4


def test_a_run_leaves_no_hidden_temporary_directory(tmp_path):
    from archi.evaluation.runner import publish_report

    publish_report(fixture_report("2026-10-30T10:00:00Z"), tmp_path)
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_a_negative_keep_or_a_bad_start_time_is_refused(tmp_path):
    from archi.evaluation.runner import publish_report

    with pytest.raises(ValueError):
        publish_report(fixture_report("2026-10-30T10:00:00Z"), tmp_path, keep=-1)
    with pytest.raises(ValueError):
        publish_report(fixture_report("2026-10-30T10:00:00+02:00"), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_the_default_output_is_dated_and_an_explicit_out_keeps_its_meaning(tmp_path):
    from pathlib import Path

    from archi.cli.evaluate import report_target

    root = Path("/data/eval")
    assert report_target(None, root, "demo", 1) == (root / "demo", True)
    assert report_target(None, root, "demo", 2) == (root / "demo", True)
    assert report_target(tmp_path, root, "demo", 1) == (tmp_path, False)
    assert report_target(tmp_path, root, "demo", 2) == (tmp_path / "demo", False)


def test_keep_is_on_the_parser_and_its_default_lives_in_one_number():
    import argparse

    from archi.cli import evaluate
    from archi.evaluation import runner

    parser = argparse.ArgumentParser()
    evaluate.add_parser(parser.add_subparsers(), lambda p: None)
    assert parser.parse_args(["eval", "--bundle", "x"]).keep is None  # resolved to DEFAULT_KEEP at run time
    assert parser.parse_args(["eval", "--keep", "0"]).keep == 0
    assert evaluate.DEFAULT_KEEP == runner.DEFAULT_KEEP == 20


def test_two_default_command_runs_leave_two_reports_that_instance_health_compares(tmp_path, monkeypatch):
    """#854 done-when: two `archi eval` runs with no --out leave two reports, and instance-health compares
    them. Drives the command's own `_run_one` (the model and MCP are replaced by a canned run)."""
    import argparse
    from types import SimpleNamespace

    from archi.cli import evaluate
    from archi.evaluation import runner as runner_mod
    from archi.health import signals

    starts = iter(["2026-10-30T10:00:00Z", "2026-10-31T10:00:00Z"])

    async def canned_run(smoke, model, tools, **kwargs):
        return fixture_report(next(starts), passed=2)

    monkeypatch.setattr(runner_mod, "run_smoke", canned_run)
    args = argparse.Namespace(model_url="http://127.0.0.1:11434/v1", no_think=False, max_steps=8, keep=20)
    smoke = SimpleNamespace(bundle="demo", questions=["q0", "q1"])
    eval_root = tmp_path / "eval"
    for _ in range(2):
        out, dated = evaluate.report_target(None, eval_root, "demo", 1)
        assert evaluate._run_one(smoke, "fake:1b", args, "http://127.0.0.1:1/mcp", "t", out, dated=dated) == 0
    runs = sorted(p.name for p in (eval_root / "demo").iterdir() if p.is_dir())
    assert runs == ["20261030T100000Z", "20261031T100000Z"]
    problems, notes = signals.eval_problems(eval_root)
    assert problems == [] and "eval-trend" not in [n["signal"] for n in notes]
    assert [r.path for r in signals.read_eval_runs(eval_root)] == [
        "demo/20261030T100000Z/report.json", "demo/20261031T100000Z/report.json"]


def test_with_out_the_report_is_replaced_in_place(tmp_path, monkeypatch):
    import argparse
    from types import SimpleNamespace

    from archi.cli import evaluate
    from archi.evaluation import runner as runner_mod

    starts = iter(["2026-10-30T10:00:00Z", "2026-10-31T10:00:00Z"])

    async def canned_run(smoke, model, tools, **kwargs):
        return fixture_report(next(starts))

    monkeypatch.setattr(runner_mod, "run_smoke", canned_run)
    args = argparse.Namespace(model_url="http://127.0.0.1:11434/v1", no_think=False, max_steps=8, keep=20)
    smoke = SimpleNamespace(bundle="demo", questions=["q0", "q1"])
    for _ in range(2):
        out, dated = evaluate.report_target(tmp_path / "mine", tmp_path / "eval", "demo", 1)
        evaluate._run_one(smoke, "fake:1b", args, "http://127.0.0.1:1/mcp", "t", out, dated=dated)
    assert sorted(p.name for p in (tmp_path / "mine").iterdir()) == ["report.json", "report.md"]
    assert json.loads((tmp_path / "mine" / "report.json").read_text())["started_at"] == "2026-10-31T10:00:00Z"
    assert not (tmp_path / "eval").exists()


def test_a_report_pruned_while_instance_health_reads_is_skipped_not_an_error(tmp_path, monkeypatch):
    from archi.evaluation.runner import publish_report
    from archi.health import signals

    publish_report(fixture_report("2026-10-30T10:00:00Z"), tmp_path / "demo")
    _, gone, _ = publish_report(fixture_report("2026-10-31T10:00:00Z"), tmp_path / "demo")
    real_rglob = type(tmp_path).rglob

    def rglob_then_prune(self, pattern):
        found = list(real_rglob(self, pattern))
        gone.unlink()  # `archi eval --keep` removed this run after the walk listed it
        return iter(found)

    monkeypatch.setattr(type(tmp_path), "rglob", rglob_then_prune)
    assert [r.path for r in signals.read_eval_runs(tmp_path)] == ["demo/20261030T100000Z/report.json"]
