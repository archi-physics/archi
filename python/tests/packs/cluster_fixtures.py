"""Recorded inputs and a fake runner for the cluster-manager packs (imported by module name).

The files under `tests/fixtures/cluster/` are recorded shapes: ServiceNow Table
API rows, Redmine `issues.json` pages, JIRA REST v2 `search` pages, and the text `sinfo`, `scontrol`,
`sacct`, `squeue`, `condor_status` and `condor_q` print. Nothing here starts a
process: `FakeRunner` answers a command from a table and refuses one it was
not given.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from archi.tools._live import RunResult

CLUSTER = Path(__file__).resolve().parents[1] / "fixtures" / "cluster"


def text(name: str) -> str:
    return (CLUSTER / name).read_text(encoding="utf-8")


def load(name: str) -> Any:
    return json.loads(text(name))


Matcher = Callable[[Sequence[str]], bool]


def starts(*prefix: str) -> Matcher:
    return lambda argv: tuple(argv[: len(prefix)]) == prefix


def has(*words: str) -> Matcher:
    return lambda argv: all(w in argv or any(w in a for a in argv) for w in words)


@dataclass
class FakeRunner:
    """Answer commands from rules; record every call. An unmatched command fails the test."""

    rules: list[tuple[Matcher, Any]] = field(default_factory=list)
    calls: list[list[str]] = field(default_factory=list)

    def on(self, matcher: Matcher, stdout: str = "", returncode: int = 0, stderr: str = "") -> "FakeRunner":
        self.rules.append((matcher, RunResult(returncode, stdout, stderr)))
        return self

    def on_call(self, matcher: Matcher, fn: Callable[[Sequence[str]], RunResult]) -> "FakeRunner":
        self.rules.append((matcher, fn))
        return self

    def __call__(self, argv: Sequence[str], *, timeout: float) -> RunResult:
        assert timeout > 0
        self.calls.append(list(argv))
        for matcher, answer in self.rules:
            if matcher(argv):
                return answer(argv) if callable(answer) else answer
        raise AssertionError(f"unexpected command: {list(argv)}")

    def commands(self, program: str) -> list[list[str]]:
        return [c for c in self.calls if c[0] == program]


def slurm_export_runner() -> FakeRunner:
    """The four commands `archi.downloaders.slurm` runs, answered from the recorded output."""
    return (
        FakeRunner()
        .on(lambda a: a[0] == "sinfo" and "-N" not in a, text("sinfo_partitions.out"))
        .on(lambda a: a[0] == "sinfo" and "-N" in a, text("sinfo_nodes.out"))
        .on(starts("scontrol", "show", "reservation"), text("scontrol_reservations.out"))
        .on(starts("sacct"), text("sacct.out"))
    )


def condor_export_runner() -> FakeRunner:
    return (
        FakeRunner()
        .on(lambda a: a[0] == "condor_status" and "-schedd" in a, text("condor_status_schedd.out"))
        .on(lambda a: a[0] == "condor_status", text("condor_status.out"))
    )


def write_slurm_cache(data: Path) -> dict:
    """Run the Slurm exporter on the recorded output and write its cache under `data/slurm`."""
    from archi.downloaders import slurm

    assert slurm.main(["--cluster", "submit", "--output-dir", str(data / "slurm")], runner=slurm_export_runner()) == 0
    return json.loads((data / "slurm" / "records.json").read_text(encoding="utf-8"))


def write_condor_cache(data: Path) -> dict:
    from archi.downloaders import htcondor

    assert htcondor.main(["--pool", "submit-pool", "--output-dir", str(data / "htcondor")], runner=condor_export_runner()) == 0
    return json.loads((data / "htcondor" / "records.json").read_text(encoding="utf-8"))


@dataclass
class FakeResponse:
    status_code: int = 200
    body: Any = None
    text: str = ""

    def json(self) -> Any:
        return self.body


@dataclass
class FakeSession:
    """A `requests.Session` stand-in that serves recorded responses and records requests."""

    gets: list[Callable[[str, dict], Optional[FakeResponse]]] = field(default_factory=list)
    posts: list[Callable[[str, dict], Optional[FakeResponse]]] = field(default_factory=list)
    requests: list[tuple[str, str, dict]] = field(default_factory=list)

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.requests.append(("GET", url, kwargs))
        for handler in self.gets:
            answer = handler(url, kwargs)
            if answer is not None:
                return answer
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.requests.append(("POST", url, kwargs))
        for handler in self.posts:
            answer = handler(url, kwargs)
            if answer is not None:
                return answer
        raise AssertionError(f"unexpected POST {url}")


def write_servicenow_cache(data: Path) -> list:
    records = load("servicenow_records.json")
    out = data / "servicenow"
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.json").write_text(json.dumps(records), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({"record_count": len(records)}), encoding="utf-8")
    return records


def write_tickets_cache(data: Path) -> list:
    issues = [{**i, "system": "redmine"} for page in ("redmine_issues_page1.json", "redmine_issues_page2.json")
              for i in load(page)["issues"]]
    out = data / "tickets"
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.json").write_text(json.dumps(issues), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({"record_count": len(issues)}), encoding="utf-8")
    return issues


JIRA = "https://jira.example.org"


def jira_session(base: str = JIRA) -> FakeSession:
    """Serve the recorded JIRA search pages: two issues, then one (``total`` 3), whatever page size is asked."""

    def search(url, kwargs):
        if url != f"{base}/rest/api/2/search":
            return None
        return FakeResponse(200, load("jira_search_page1.json" if kwargs["params"]["startAt"] == 0 else "jira_search_page2.json"))

    return FakeSession(gets=[search])


def write_jira_tickets_cache(data: Path) -> list:
    """Run the tickets downloader's JIRA reader on the recorded pages and write its cache under `data/tickets`."""
    from archi.downloaders import tickets

    tickets.download(jira_session(), {"JIRA_API_TOKEN": "token-fixture"}, data / "tickets",
                     jira_url=JIRA, jira_projects=["OPS"])
    return json.loads((data / "tickets" / "records.json").read_text(encoding="utf-8"))


def write_node_health_cache(data: Path) -> list:
    reports = load("node_health.json")
    out = data / "node-health"
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.json").write_text(json.dumps(reports), encoding="utf-8")
    return reports
