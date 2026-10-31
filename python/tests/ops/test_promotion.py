"""req.r3-ops-runtime.shadow: promotion to live needs a passing shadow report, then a PACT change that a
person approves. `archi ops promote` opens the change and stops; the runtime enforces the approval."""
import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from archi.ops.promotion import PromotionRefused, change_id, promote, readiness_junit
from archi.ops.shadow import promotion_readiness

NOW = datetime(2026, 10, 25, 14, 30, 5, tzinfo=timezone.utc)
GOOD = {"scored": 40, "answered": 30, "response_coverage": 0.75, "agreement_rate": 0.9, "false_proposal_rate_upper": 0.05}
BAD = {"scored": 3, "answered": 1, "response_coverage": 0.33, "agreement_rate": 0.5, "false_proposal_rate_upper": 0.4}
THRESHOLDS = {"min_proposals": 30, "min_answered": 20, "min_response_coverage": 0.5, "min_agreement": 0.8,
              "max_false_rate": 0.1}


def git(directory, *args):
    return subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.email", "ops@example.test")
    git(tmp_path, "config", "user.name", "ops")
    (tmp_path / "ops.yaml").write_text(yaml.safe_dump({"automations": {"triage": {"mode": "shadow", "timeout_seconds": 60}}}))
    (tmp_path / "other.txt").write_text("unrelated\n")
    git(tmp_path, "add", "ops.yaml", "other.txt")
    git(tmp_path, "commit", "-q", "-m", "baseline")
    return tmp_path


class FakeOkg:
    """Stands in for `okg pact new / evidence ingest / gate task-done`; `fail` names a verb that exits 1."""

    def __init__(self, root: Path, fail=None):
        self.root, self.fail, self.calls, self.envs = root, fail, [], []

    def __call__(self, cmd, **kw):
        if cmd[0] == "git":
            return subprocess.run(cmd, **kw)
        self.calls.append(cmd)
        self.envs.append(kw.get("env") or {})
        verb = cmd[cmd.index("--root") + 2]
        if verb == self.fail:
            return subprocess.CompletedProcess(cmd, 1, "tip: ignore me\n", f"{verb} refused")
        if verb == "new":
            change_dir = self.root / "pact" / "changes" / cmd[cmd.index("new") + 1]
            change_dir.mkdir(parents=True)
            (change_dir / "pact.yaml").write_text("id: x\n")
        return subprocess.CompletedProcess(cmd, 0, "{}", "")


def run(repo, report=GOOD, fail=None, **kw):
    okg = FakeOkg(repo, fail)
    result = promote(repo, "triage", report=report, readiness=promotion_readiness(report, **THRESHOLDS),
                     thresholds=THRESHOLDS, principal="jdoe", now=NOW, run=okg, okg="okg",
                     dsn="postgresql://x", deployment="dep", **kw)
    return result, okg


def test_a_passing_report_opens_the_change_and_stops_without_approving(repo):
    result, okg = run(repo, note="reviewed the 40 incidents")
    change = "promote-triage-20261025t143005z"
    assert result["change"] == change == change_id("triage", NOW) and result["state"] == "awaiting_review"
    new, ingest, gate = okg.calls  # no decide: the approval is a human step archi never runs
    assert new[:5] == ["okg", "pact", "--root", str(repo), "new"] and new[5] == change
    assert "scored=40" in new[new.index("--intent") + 1] and "min_proposals" in new[new.index("--intent") + 1]
    assert ingest[4:7] == ["evidence", "ingest", change]
    assert ingest[ingest.index("--pytest-report") + 1] == f"pact/changes/{change}/readiness.xml"
    assert gate[4:7] == ["gate", "task-done", change] and gate[7] == f"task.{change}"
    assert not any("decide" in c for c in okg.calls)
    assert okg.envs[-1]["OKG_PACT_GRAPH_DEPLOYMENT"] == "dep" and okg.envs[-1]["OKG_DSN"] == "postgresql://x"
    ops = yaml.safe_load((repo / "ops.yaml").read_text())["automations"]["triage"]
    assert ops == {"mode": "live", "timeout_seconds": 60, "promotion": change}  # a request, until a person approves
    change_dir = repo / "pact" / "changes" / change
    assert json.loads((change_dir / "shadow-report.json").read_text())["scored"] == 40
    assert json.loads((change_dir / "thresholds.json").read_text()) == THRESHOLDS
    assert git(repo, "log", "-1", "--format=%s").strip() == f"request promotion of triage ({change})"
    assert sorted(git(repo, "show", "--name-only", "--format=", "HEAD").split()) == sorted(
        ["ops.yaml", f"pact/changes/{change}/pact.yaml", f"pact/changes/{change}/shadow-report.json",
         f"pact/changes/{change}/thresholds.json", f"pact/changes/{change}/readiness.xml",
         f"pact/changes/{change}/promotion.json"])
    record = json.loads((change_dir / "promotion.json").read_text())
    assert record == {"automation": "triage", "change": change, "requested_at": "2026-10-25T14:30:05Z",
                      "requested_by": "jdoe"}
    assert git(repo, "status", "--porcelain").strip() == ""
    assert "okg pact decide" in result["next"] and "shadow" in result["next"]


def test_the_junit_file_says_what_the_readiness_check_said():
    ready = ET.fromstring(readiness_junit(promotion_readiness(GOOD, **THRESHOLDS), GOOD, THRESHOLDS))
    case = ready.find(".//testcase")
    assert case.get("name") == "test_promotion_readiness" and case.find("failure") is None
    assert "min_agreement" in case.find("system-out").text and "scored=40" in case.find("system-out").text
    not_ready = ET.fromstring(readiness_junit(promotion_readiness(BAD, **THRESHOLDS), BAD, THRESHOLDS))
    failure = not_ready.find(".//testcase/failure")
    assert failure is not None and "only 3 scored incidents" in failure.get("message")
    assert not_ready.find("testsuite").get("failures") == "1"


def test_a_failing_report_is_refused_and_nothing_is_written(repo):
    before = git(repo, "rev-parse", "HEAD")
    okg = FakeOkg(repo)
    with pytest.raises(PromotionRefused, match="not ready to promote: only 3 scored incidents; need 30"):
        promote(repo, "triage", report=BAD, readiness=promotion_readiness(BAD, **THRESHOLDS), thresholds=THRESHOLDS,
                principal="jdoe", now=NOW, run=okg, okg="okg")
    assert okg.calls == [] and git(repo, "rev-parse", "HEAD") == before
    assert yaml.safe_load((repo / "ops.yaml").read_text())["automations"]["triage"]["mode"] == "shadow"
    assert not (repo / "pact").exists()


def test_an_empty_report_is_refused():
    with pytest.raises(PromotionRefused, match="not ready"):
        promote(Path("."), "triage", report={}, readiness=promotion_readiness({}), principal="x", now=NOW,
                run=FakeOkg(Path(".")), okg="okg")


def test_a_directory_that_is_not_a_git_repository_is_refused(tmp_path):
    (tmp_path / "ops.yaml").write_text(yaml.safe_dump({"automations": {"triage": {}}}))
    okg = FakeOkg(tmp_path)
    with pytest.raises(PromotionRefused, match="not a git repository"):
        promote(tmp_path, "triage", report=GOOD, readiness=promotion_readiness(GOOD, **THRESHOLDS), principal="x",
                now=NOW, run=okg, okg="okg")
    assert okg.calls == [] and "live" not in (tmp_path / "ops.yaml").read_text()


def test_an_automation_ops_yaml_does_not_know_is_refused(repo):
    with pytest.raises(PromotionRefused, match="no automations.nope"):
        promote(repo, "nope", report=GOOD, readiness=promotion_readiness(GOOD, **THRESHOLDS), principal="x",
                now=NOW, run=FakeOkg(repo), okg="okg")


@pytest.mark.parametrize("fail, what", [("new", "new"), ("evidence", "evidence ingest"), ("gate", "gate task-done")])
def test_when_okg_pact_fails_ops_yaml_is_unchanged_and_nothing_is_committed(repo, fail, what):
    before = git(repo, "rev-parse", "HEAD")
    with pytest.raises(PromotionRefused, match=f"okg pact {what} failed") as excinfo:
        run(repo, fail=fail)
    assert "tip: ignore me" not in str(excinfo.value) and f"{fail} refused" in str(excinfo.value)
    assert git(repo, "rev-parse", "HEAD") == before
    assert yaml.safe_load((repo / "ops.yaml").read_text())["automations"]["triage"] == {"mode": "shadow", "timeout_seconds": 60}


def test_the_cli_refuses_without_a_passing_report_and_passes_the_thresholds_on(monkeypatch, capsys, repo):
    from archi.bundles.cli import build_parser
    from archi.cli import ops
    from archi.ops import shadow

    monkeypatch.setattr(ops, "make_io", lambda args: type("Io", (), {"deployment": "dep", "dsn": "x"})())
    monkeypatch.setattr(shadow, "okg_source", lambda *a, **k: None)
    monkeypatch.setattr(shadow, "score_automation", lambda *a, **k: BAD)
    args = build_parser().parse_args(["ops", "--instance", str(repo), "promote", "triage"])
    assert args.func(args) == 2
    assert "not ready to promote" in capsys.readouterr().err
    assert yaml.safe_load((repo / "ops.yaml").read_text())["automations"]["triage"]["mode"] == "shadow"

    seen = {}
    import archi.ops.promotion as promotion

    monkeypatch.setattr(shadow, "score_automation", lambda *a, **k: GOOD)
    monkeypatch.setattr(promotion, "promote", lambda *a, **k: seen.update(k) or {"change": "c", "next": "n"})
    args = build_parser().parse_args(["ops", "--instance", str(repo), "promote", "triage", "--min-proposals", "5"])
    assert args.func(args) == 0
    assert seen["thresholds"]["min_proposals"] == 5 and seen["thresholds"]["max_false_rate"] == 0.1
