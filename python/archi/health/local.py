"""The local issue tracker: one Markdown file per problem in the instance repository.

    <instance>/.archi/issues/<check>--<subject>.md

A file starts with a YAML front matter block (`key`, `family`, `title`, `state`, `severity`, `events`)
and holds the issue text below it. Nothing leaves the machine and no credential is needed, so this is
the default backend. `finish()` commits the changed files in one commit, with a path-limited
`git commit`, so it never sweeps in anything else that is staged. okg publishes only from a committed
instance repository, and a file left uncommitted would block that, so `finish()` also commits files
an earlier run wrote but could not commit.

The file name comes from the key (`archi.health.problems.slug`); a file whose front matter names another
key is a collision and stops the run.
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import yaml

from archi.health.problems import Problem, slug
from archi.health.tracker import OPEN, RESOLVED, Issue, TrackerError, render_body

ISSUES_DIR = ".archi/issues"
COMMITTER = ("archi-instance-health", "archi-instance-health@localhost")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class LocalTracker:
    def __init__(self, instance_dir: Path | str, *, now: Callable[[], datetime] = _now, commit: bool = True):
        self.root = Path(instance_dir)
        self.dir = self.root / ISSUES_DIR
        self.now = now
        self.commit = commit
        self._log: list[str] = []
        self._before: dict[Path, Optional[str]] = {}  # file -> its text before this run first wrote it (None: new)

    # --- files ---------------------------------------------------------------

    def path(self, key: str) -> Path:
        return self.dir / f"{slug(key)}.md"

    def _read(self, path: Path) -> tuple[dict, str]:
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---\n") or "\n---\n" not in text[4:]:
            raise TrackerError(f"{path}: not an issue file (no front matter)")
        head, _, body = text[4:].partition("\n---\n")
        try:
            meta = yaml.safe_load(head) or {}
        except yaml.YAMLError as exc:
            raise TrackerError(f"{path}: front matter is not YAML: {exc}") from exc
        return meta, body.lstrip("\n")

    def _issue(self, path: Path) -> Issue:
        meta, body = self._read(path)
        for field in ("key", "family", "title", "state"):
            if field not in meta:
                raise TrackerError(f"{path}: front matter has no {field!r}")
        events = list(meta.get("events") or [])
        last = events[-1] if events else {}
        by_us = meta["state"] != RESOLVED or (last.get("event") == "resolved" and last.get("by") == "archi")
        noted = last.get("event") == "still_present"
        return Issue(meta["key"], meta["family"], meta["title"], body, meta["state"],
                     ref=str(path.relative_to(self.root)), resolved_by_us=by_us, noted=noted)

    def _write(self, path: Path, meta: dict, body: str) -> None:
        if path not in self._before:
            self._before[path] = path.read_text(encoding="utf-8") if path.is_file() else None
        path.parent.mkdir(parents=True, exist_ok=True)
        head = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).rstrip("\n")
        path.write_text(f"---\n{head}\n---\n\n{body}", encoding="utf-8")

    # --- the tracker ---------------------------------------------------------

    def open_issues(self) -> list[Issue]:
        if not self.dir.is_dir():
            return []
        return [i for i in (self._issue(p) for p in sorted(self.dir.glob("*.md"))) if i.state == OPEN]

    def find(self, key: str) -> Optional[Issue]:
        path = self.path(key)
        if not path.is_file():
            return None
        issue = self._issue(path)
        if issue.key != key:
            raise TrackerError(f"{path} holds the issue for {issue.key!r}, not {key!r}: two keys share a file name")
        return issue

    def create(self, problem: Problem) -> Issue:
        at = _stamp(self.now())
        meta = self._meta(problem, OPEN, [{"at": at, "event": "opened"}])
        self._write(self.path(problem.key), meta, render_body(problem))
        self._log.append(f"open {problem.key}")
        return self.find(problem.key)

    def update(self, issue: Issue, problem: Problem) -> Issue:
        path = self.path(problem.key)
        meta, _ = self._read(path)
        events = list(meta.get("events") or [])
        if issue.state != OPEN:
            events.append({"at": _stamp(self.now()), "event": "reopened"})
        elif issue.title != problem.title or issue.body.strip() != render_body(problem).strip():
            events.append({"at": _stamp(self.now()), "event": "updated"})
        self._write(path, self._meta(problem, OPEN, events), render_body(problem))
        self._log.append(("reopen " if issue.state != OPEN else "update ") + problem.key)
        return self.find(problem.key)

    def resolve(self, issue: Issue, note: str) -> Issue:
        path = self.path(issue.key)
        meta, body = self._read(path)
        meta["state"] = RESOLVED
        meta["events"] = list(meta.get("events") or []) + [{"at": _stamp(self.now()), "event": "resolved", "by": "archi", "note": note}]
        self._write(path, meta, body)
        self._log.append(f"resolve {issue.key}")
        return self.find(issue.key)

    def note_still_present(self, issue: Issue, text: str) -> Issue:
        path = self.path(issue.key)
        meta, body = self._read(path)
        meta["events"] = list(meta.get("events") or []) + [
            {"at": _stamp(self.now()), "event": "still_present", "note": text}]
        self._write(path, meta, body)
        self._log.append(f"note {issue.key}")
        return self.find(issue.key)

    @staticmethod
    def _meta(problem: Problem, state: str, events: list) -> dict:
        return {"key": problem.key, "family": problem.family, "title": problem.title, "state": state,
                "severity": problem.severity, "events": events}

    # --- the commit ----------------------------------------------------------

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(self.root), *args], capture_output=True, text=True)

    def _roll_back(self) -> None:
        """Put back every file this run wrote and unstage the issues directory, so a failed commit leaves
        no half-written issue (and the next run starts from the committed state)."""
        for path, before in self._before.items():
            if before is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(before, encoding="utf-8")
        self._git("reset", "-q", "--", ISSUES_DIR)
        self._before, self._log = {}, []

    def finish(self) -> None:
        """Commit every uncommitted change under `.archi/issues/`, in one commit limited to that path.
        If git fails, the files this run wrote are put back and the run fails."""
        if not self.commit or not self.dir.exists():
            return
        try:
            self._commit()
        except TrackerError:
            self._roll_back()
            raise
        self._before = {}

    def _commit(self) -> None:
        status = self._git("status", "--porcelain", "--untracked-files=all", "--", ISSUES_DIR)
        if status.returncode != 0:
            raise TrackerError(f"{self.root} is not a usable git repository: {status.stderr.strip()[-300:]}")
        changed = [line for line in status.stdout.splitlines() if line.strip()]
        if not changed:
            return
        what = "; ".join(self._log) if self._log else f"commit {len(changed)} issue file(s) left uncommitted"
        message = f"health: {what}"
        self._log = []
        added = self._git("add", "--", ISSUES_DIR)
        if added.returncode != 0:
            raise TrackerError(f"git add failed: {added.stderr.strip()[-300:]}")
        name, email = COMMITTER
        done = self._git("-c", f"user.name={name}", "-c", f"user.email={email}", "commit", "-q", "-m", message,
                         "--", ISSUES_DIR)
        if done.returncode != 0:
            raise TrackerError(f"git commit failed: {(done.stderr or done.stdout).strip()[-300:]}")
