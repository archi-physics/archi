"""req.r3-ops-self-monitoring.health: one issue per problem, deduplicated by key; update; close; local backend."""
import subprocess
from datetime import datetime, timezone

import pytest

from archi.health.local import ISSUES_DIR, LocalTracker
from archi.health.problems import Problem
from archi.health.steps import draft_issue
from archi.health.tracker import OPEN, RESOLVED, TrackerError, needs_update, reconcile, render_body

T = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)


def git(repo, *args):
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.org")
    git(tmp_path, "config", "user.name", "t")
    (tmp_path / "archi.yaml").write_text("name: t\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def problem(subject="jira.jira", detail="stale since 2026-09-20", **kw):
    return Problem(key=f"source-freshness/{subject}", family="doctor", title=f"Source {subject} is stale",
                   detail=detail, **kw)


def tracker(repo):
    return LocalTracker(repo, now=lambda: T)


def commits(repo):
    return git(repo, "log", "--format=%s").splitlines()


def issue_files(repo):
    return sorted(p.name for p in (repo / ISSUES_DIR).glob("*.md"))


# --- the acceptance: a stale source yields exactly one issue across two runs ------------


def test_a_stale_source_yields_exactly_one_issue_across_two_runs(repo):
    scope = ["doctor", "monitor"]
    first = reconcile(tracker(repo), [problem()], scope)
    second = reconcile(tracker(repo), [problem()], scope)
    assert [(c.key, c.action) for c in first] == [("source-freshness/jira.jira", "created")]
    assert [(c.key, c.action) for c in second] == [("source-freshness/jira.jira", "unchanged")]
    assert issue_files(repo) == ["source-freshness--jira.jira.md"]
    assert len(commits(repo)) == 2  # init, and one for the issue; the second run wrote and committed nothing
    assert git(repo, "status", "--porcelain") == ""


def test_the_issue_is_a_committed_markdown_file_with_the_key_and_the_detail(repo):
    reconcile(tracker(repo), [problem()], ["doctor"])
    text = (repo / ISSUES_DIR / "source-freshness--jira.jira.md").read_text()
    assert text.startswith("---\nkey: source-freshness/jira.jira\nfamily: doctor\ntitle: Source jira.jira is stale\nstate: open\n")
    assert "stale since 2026-09-20" in text and "opened" in text
    assert "health: open source-freshness/jira.jira" in commits(repo)[0]


# --- update and close ---------------------------------------------------------------------


def test_a_changed_problem_updates_the_same_issue(repo):
    reconcile(tracker(repo), [problem(detail="stale since 2026-09-20")], ["doctor"])
    (change,) = reconcile(tracker(repo), [problem(detail="stale since 2026-09-20, still")], ["doctor"])
    assert change.action == "updated" and issue_files(repo) == ["source-freshness--jira.jira.md"]
    assert "still" in (repo / ISSUES_DIR / "source-freshness--jira.jira.md").read_text()
    assert "updated" in (repo / ISSUES_DIR / "source-freshness--jira.jira.md").read_text()


def test_a_problem_that_clears_is_marked_resolved_not_deleted(repo):
    reconcile(tracker(repo), [problem()], ["doctor"])
    (change,) = reconcile(tracker(repo), [], ["doctor"])
    assert change.action == "resolved"
    found = tracker(repo).find("source-freshness/jira.jira")
    assert found.state == RESOLVED and tracker(repo).open_issues() == []
    assert "resolved" in (repo / ISSUES_DIR / "source-freshness--jira.jira.md").read_text()
    assert git(repo, "status", "--porcelain") == ""


def test_a_resolved_issue_is_not_resolved_again_and_a_returning_problem_reopens_it(repo):
    reconcile(tracker(repo), [problem()], ["doctor"])
    reconcile(tracker(repo), [], ["doctor"])
    assert reconcile(tracker(repo), [], ["doctor"]) == []  # nothing open, nothing to do
    (change,) = reconcile(tracker(repo), [problem()], ["doctor"])
    assert change.action == "reopened" and issue_files(repo) == ["source-freshness--jira.jira.md"]
    assert tracker(repo).find("source-freshness/jira.jira").state == OPEN


def test_one_problem_does_not_close_or_touch_another(repo):
    reconcile(tracker(repo), [problem("a"), problem("b")], ["doctor"])
    changes = reconcile(tracker(repo), [problem("b")], ["doctor"])
    assert {c.key: c.action for c in changes} == {"source-freshness/a": "resolved", "source-freshness/b": "unchanged"}


def test_a_family_that_was_not_read_keeps_its_issues_open(repo):
    reconcile(tracker(repo), [problem()], ["doctor"])
    assert reconcile(tracker(repo), [], ["ingest", "monitor"]) == []  # doctor not in scope: no evidence it cleared
    assert tracker(repo).find("source-freshness/jira.jira").state == OPEN


def test_two_problems_with_one_key_are_refused(repo):
    with pytest.raises(ValueError, match="share the key"):
        reconcile(tracker(repo), [problem(), problem()], ["doctor"])


# --- the commit ---------------------------------------------------------------------------


def test_the_commit_takes_only_the_issue_files_and_not_what_else_is_staged(repo):
    (repo / "other.txt").write_text("staged by someone else\n")
    git(repo, "add", "other.txt")
    reconcile(tracker(repo), [problem()], ["doctor"])
    assert git(repo, "show", "--name-only", "--format=", "HEAD").split() == [f"{ISSUES_DIR}/source-freshness--jira.jira.md"]
    assert "other.txt" in git(repo, "diff", "--cached", "--name-only")


def test_a_file_an_earlier_run_could_not_commit_is_committed_by_the_next(repo):
    t = tracker(repo)
    t.commit = False
    reconcile(t, [problem()], ["doctor"])
    assert "??" in git(repo, "status", "--porcelain")
    reconcile(tracker(repo), [problem()], ["doctor"])  # unchanged, but it commits what is left
    assert git(repo, "status", "--porcelain") == "" and len(commits(repo)) == 2


def test_a_directory_that_is_not_a_repository_fails_the_run_loudly(tmp_path):
    with pytest.raises(TrackerError, match="git repository"):
        reconcile(tracker(tmp_path), [problem()], ["doctor"])


def test_a_file_holding_another_key_is_a_collision(repo):
    reconcile(tracker(repo), [problem()], ["doctor"])
    path = repo / ISSUES_DIR / "source-freshness--jira.jira.md"
    path.write_text(path.read_text().replace("key: source-freshness/jira.jira", "key: source-freshness/other"))
    with pytest.raises(TrackerError, match="share a file name"):
        tracker(repo).find("source-freshness/jira.jira")


# --- the step ---------------------------------------------------------------------------------


def payload(problems, scope=("doctor", "monitor"), tracker=None):
    return {"tracker": tracker or {"kind": "local"}, "observed_at": "2026-10-25T12:00:00Z", "problems": [p.as_dict() for p in problems],
            "scope": list(scope), "skipped": []}


def test_the_draft_issue_step_writes_the_issue_and_a_replay_changes_nothing(repo):
    ctx = {"instance_dir": str(repo)}
    out = draft_issue(payload=payload([problem()]), context=ctx)
    again = draft_issue(payload=payload([problem()]), context=ctx)
    assert out["status"] == "succeeded" and "1 created" in out["result"]
    assert "1 unchanged" in again["result"] and len(commits(repo)) == 2
    assert "2026-10-25T12:00:00Z" in (repo / ISSUES_DIR / "source-freshness--jira.jira.md").read_text()


def test_the_draft_issue_step_refuses_a_forge_tracker(repo):
    with pytest.raises(TrackerError, match="live tool"):
        draft_issue(payload=payload([], tracker={"kind": "github", "repo": "o/r", "token_env": "X_TOKEN"}), context={"instance_dir": str(repo)})


def test_text_comparison_ignores_line_endings_a_forge_may_add():
    from archi.health.tracker import Issue

    p = problem()
    body = render_body(p).replace("\n", "\r\n")
    assert not needs_update(Issue(p.key, p.family, p.title, body, OPEN), p)
    assert needs_update(Issue(p.key, p.family, p.title, body, RESOLVED), p)
    assert needs_update(Issue(p.key, p.family, "other title", body, OPEN), p)


# --- review fixes -----------------------------------------------------------------------------------------------------


def test_a_failed_commit_rolls_the_written_files_back_and_leaves_nothing_behind(repo):
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\necho nope >&2\nexit 1\n")
    hook.chmod(0o755)
    with pytest.raises(TrackerError, match="git commit failed"):
        reconcile(tracker(repo), [problem()], ["doctor"])
    assert issue_files(repo) == [] and git(repo, "status", "--porcelain") == ""
    hook.unlink()
    reconcile(tracker(repo), [problem()], ["doctor"])  # the next run starts clean and succeeds
    assert issue_files(repo) == ["source-freshness--jira.jira.md"]


def test_a_rollback_restores_an_issue_that_was_updated_in_the_failed_run(repo):
    reconcile(tracker(repo), [problem(detail="one")], ["doctor"])
    before = (repo / ISSUES_DIR / "source-freshness--jira.jira.md").read_text()
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    with pytest.raises(TrackerError):
        reconcile(tracker(repo), [problem(detail="two")], ["doctor"])
    assert (repo / ISSUES_DIR / "source-freshness--jira.jira.md").read_text() == before
    assert git(repo, "status", "--porcelain") == ""


def test_an_issue_a_person_closed_stays_closed_and_is_noted_once(repo):
    reconcile(tracker(repo), [problem()], ["doctor"])
    path = repo / ISSUES_DIR / "source-freshness--jira.jira.md"
    path.write_text(path.read_text().replace("state: open", "state: resolved"))  # a person closes it
    git(repo, "commit", "-qam", "person closes the issue")
    (first,) = reconcile(tracker(repo), [problem()], ["doctor"])
    (second,) = reconcile(tracker(repo), [problem()], ["doctor"])
    assert first.action == "noted" and second.action == "unchanged"
    found = tracker(repo).find("source-freshness/jira.jira")
    assert found.state == RESOLVED and not found.resolved_by_us and found.noted
    assert path.read_text().count("still_present") == 1 and len(commits(repo)) == 4


def test_an_issue_the_automation_resolved_is_reopened_but_one_a_person_closed_after_that_is_not(repo):
    reconcile(tracker(repo), [problem()], ["doctor"])
    reconcile(tracker(repo), [], ["doctor"])
    assert reconcile(tracker(repo), [problem()], ["doctor"])[0].action == "reopened"
    path = repo / ISSUES_DIR / "source-freshness--jira.jira.md"
    path.write_text(path.read_text().replace("state: open", "state: resolved"))
    git(repo, "commit", "-qam", "person closes it again")
    assert reconcile(tracker(repo), [problem()], ["doctor"])[0].action == "noted"


def test_two_keys_that_differ_only_in_case_are_two_issues_with_two_files(repo):
    from archi.health.problems import make_key

    a, b = make_key("source-freshness", "Foo"), make_key("source-freshness", "foo")
    assert a != b and a == a.lower() and b == b.lower()
    pa = Problem(key=a, family="doctor", title="a", detail="a")
    pb = Problem(key=b, family="doctor", title="b", detail="b")
    reconcile(tracker(repo), [pa, pb], ["doctor"])
    assert len(issue_files(repo)) == 2
    with pytest.raises(Exception):
        Problem(key="source-freshness/Foo", family="doctor", title="t", detail="d")  # upper case is not a key
