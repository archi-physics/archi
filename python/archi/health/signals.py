"""The signals the instance-health automation reads, and the problems they yield.

    doctor             `archi doctor --json`: every check that fails (a warning only if asked), including
                       source-freshness, which yields one problem per late source
    ingest             `okg sync frontier --json`: a source whose newest run failed or whose frontier is blocked
    automation-errors  the ops records (`archi.ops.shadow`'s reader seam): an automation whose recent
                       decisions are mostly errors
    eval               the score records `archi eval --context` writes under `<instance>/eval/reports/*.json`.
                       With fewer than two there is no trend, so this signal is skipped, and the skip is reported.

Each collector turns plain data into `Problem`s and reads nothing itself; `collect` runs the readers.
A reader that fails is not hidden: it becomes the problem `monitor/<signal>` ("could not read it"), and
that signal's own issues are left alone for the run (reading nothing is not evidence of health).
Every reader is a parameter, so tests give it data and the lab gives it the real thing.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from archi.health.config import HealthConfig
from archi.health.problems import Problem, make_key

#: ops decision verdicts that mean the run failed (not a cap or a person's "no")
ERROR_VERDICTS = frozenset({"error", "invalid_proposal", "refused"})
EVAL_DIR = "eval/reports"


# --- doctor ---------------------------------------------------------------------


def doctor_problems(report: Mapping[str, Any], *, ignore: Iterable[str] = (), include_warnings: bool = False
                    ) -> list[Problem]:
    """A problem per failing check in an `archi doctor --json` report.

    `source-freshness` is split by source (`source-freshness/<source>`), from its `data.sources` rows, so each
    stale source has its own issue. Any other check is `<check id>/instance`.
    """
    skip = set(ignore)
    wanted = {"fail", "warn"} if include_warnings else {"fail"}
    out: list[Problem] = []
    for check in report.get("checks") or []:
        check_id, status = check.get("id"), check.get("status")
        if not check_id or check_id in skip or status not in wanted:
            continue
        severity = "error" if status == "fail" else "warning"
        rows = ((check.get("data") or {}).get("sources") if check_id == "source-freshness" else None)
        if rows:
            for row in rows:
                if row.get("ok"):
                    continue
                out.append(_late_source(check_id, row, severity))
            continue
        out.append(Problem(
            key=make_key(check_id, "instance"), family="doctor", severity=severity,
            title=f"archi doctor: {check_id} {'failed' if status == 'fail' else 'warns'}",
            detail=_doctor_detail(check),
        ))
    return out


def _late_source(check_id: str, row: Mapping[str, Any], severity: str) -> Problem:
    source = row.get("source") or "unknown"
    last = row.get("last_reconcile_at")
    # no age here: it changes every hour, and the issue is rewritten whenever its text changes
    lines = [
        f"Source `{source}` is over its freshness limit (`max_staleness: {row.get('max_staleness')}`).",
        "",
        f"- Last completed reconcile: {last or 'none recorded'}",
    ]
    if not last and row.get("reason"):
        lines.append(f"- {row['reason']}")
    lines += ["", "Fix: `okg ingest --deployment <instance name>` (run in the instance directory), or relax the "
              "limit in the pack's `freshness:` and re-render. Source: `archi doctor --json`, check "
              f"`{check_id}`."]
    return Problem(make_key(check_id, source), "doctor", f"Source {source} is stale", "\n".join(lines), severity)


def _doctor_detail(check: Mapping[str, Any]) -> str:
    lines = [f"`archi doctor` check `{check['id']}` is {check['status']}.", "", f"> {check.get('message', '')}"]
    if check.get("remediation"):
        lines += ["", f"Fix: {check['remediation']}"]
    if check.get("footgun"):
        lines += ["", f"Footgun ledger: {check['footgun']} (docs/docs/operating/footguns.md)."]
    return "\n".join(lines)


# --- ingest ---------------------------------------------------------------------


def frontier_problems(report: Mapping[str, Any]) -> list[Problem]:
    """A problem per source whose frontier is `blocked` in `okg sync frontier --json`."""
    out = []
    for row in report.get("rows") or []:
        if row.get("frontier_state") != "blocked":
            continue
        source = row.get("registry_source_name") or row.get("source_name") or "unknown"
        lines = [f"okg reports source `{source}` blocked: it cannot move its data into the published graph.", ""]
        for label, field_name in (("Read visibility", "read_visibility_state"), ("Last attempt", "last_attempt_status"),
                                  ("Last attempt run", "last_attempt_run_id"), ("Last error", "last_error")):
            if row.get(field_name):
                lines.append(f"- {label}: {str(row[field_name])[:400]}")
        lines += ["", f"Look: `okg sync frontier --deployment <instance name> --source {source} --json`, then fix "
                  "the cause and re-run the ingest."]
        out.append(Problem(make_key("ingest", source), "ingest", f"Ingest of {source} is blocked", "\n".join(lines)))
    return out


# --- automation error rates -------------------------------------------------------


def automation_error_problems(
    source: Callable[[str], list],
    now: datetime,
    *,
    window: timedelta = timedelta(hours=24),
    min_decisions: int = 3,
    max_rate: float = 0.5,
) -> list[Problem]:
    """A problem per automation whose decisions in the window are mostly errors.

    `source(subtype)` is the reader seam of `archi.ops.shadow` (rows of `{"node_id", "attrs"}`). Placeholder
    decisions and decisions with no usable time are not counted. Fewer than `min_decisions` in the window say
    too little to call a rate.
    """
    from archi.ops.shadow import parse_time

    counts: dict[str, list[int]] = {}
    for row in source("ops_decision"):
        attrs = row.get("attrs") or {}
        name, moment = attrs.get("automation"), parse_time(attrs.get("decided_at"))
        if not name or moment is None or not now - window <= moment <= now:
            continue
        tally = counts.setdefault(name, [0, 0])
        tally[0] += 1
        tally[1] += attrs.get("verdict") in ERROR_VERDICTS
    out = []
    hours = window.total_seconds() / 3600
    for name in sorted(counts):
        total, errors = counts[name]
        if total < min_decisions or errors / total < max_rate:
            continue
        out.append(Problem(
            key=make_key("automation-errors", name), family="automation-errors",
            title=f"Automation {name} is failing",
            detail=(f"{errors} of the last {total} decisions of automation `{name}` in the past {hours:g} hours "
                    f"were errors (threshold: {max_rate:.0%} of at least {min_decisions}).\n\n"
                    f"Look: `archi ops status`, and the `ops_decision` records with `automation: {name}` and a "
                    "verdict of `error`, `invalid_proposal` or `refused`."),
        ))
    return out


# --- eval -----------------------------------------------------------------------


def eval_problems(instance_dir: Path | str, *, min_drop: float = 0.05, history: int = 3
                  ) -> tuple[list[Problem], Optional[str]]:
    """(problems, why it was skipped). Reads `<instance>/eval/reports/*.json`.

    A report is `{"finished_at": <ISO time>, "score": <0..1>, ...}`, the record `archi eval --context` writes
    (`archi.evaluation.context.record_score`; the score is the graph arm's recall@k). The latest score is a regression when it is more than `min_drop`
    below the mean of the `history` reports before it. With no report, or one, there is no trend: skipped.
    """
    directory = Path(instance_dir) / EVAL_DIR
    reports = sorted(directory.glob("*.json")) if directory.is_dir() else []
    if not reports:
        return [], f"no eval report exists ({EVAL_DIR}/*.json); eval trends are skipped"
    scored = []
    for path in reports:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            scored.append((str(data.get("finished_at") or path.name), float(data["score"]), path.name))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}: not an eval report with a numeric score ({type(exc).__name__}: {exc})") from exc
    scored.sort()
    if len(scored) < 2:
        return [], "only one eval report exists; a trend needs two"
    latest, earlier = scored[-1], scored[-1 - history:-1]
    baseline = statistics.fmean(s for _, s, _ in earlier)
    if baseline - latest[1] <= min_drop:
        return [], None
    return [Problem(
        key=make_key("eval", "score"), family="eval", title="Eval score regressed",
        detail=(f"The latest eval report ({latest[2]}) scores {latest[1]:.3f}, more than {min_drop:g} below the "
                f"mean {baseline:.3f} of the {len(earlier)} report(s) before it.\n\nLook: the eval reports in "
                f"`{EVAL_DIR}/`."),
    )], None


# --- readers (the real ones) -------------------------------------------------------


def run_doctor(instance_dir: Path | str, *, timeout: int = 300) -> dict[str, Any]:
    """`archi doctor --json` for the instance. Exit 1 (a failed check) is a report; any other failure raises."""
    code = "import sys; from archi.bundles.cli import main; sys.exit(main(sys.argv[1:]))"
    done = subprocess.run([sys.executable, "-c", code, "doctor", "--instance", str(instance_dir), "--json"],
                          capture_output=True, text=True, timeout=timeout)
    try:
        report = json.loads(done.stdout)
    except ValueError:
        raise RuntimeError(f"archi doctor --json gave no JSON (exit {done.returncode}): "
                           f"{(done.stderr or done.stdout)[-500:].strip()}") from None
    if done.returncode not in (0, 1) or report.get("error"):
        raise RuntimeError(f"archi doctor could not run (exit {done.returncode}): {report.get('error') or done.stderr[-300:]}")
    return report


def run_frontier(deployment: str, dsn: str, *, timeout: int = 120) -> dict[str, Any]:
    """`okg sync frontier --json`: per-source frontier rows (okg 0.2.0)."""
    from archi.ops.record import _okg

    done = subprocess.run([_okg(), "sync", "frontier", "--deployment", deployment, "--dsn", dsn, "--json"],
                          capture_output=True, text=True, timeout=timeout)
    try:
        data = json.loads(done.stdout)
    except ValueError:
        raise RuntimeError(f"okg sync frontier gave no JSON (exit {done.returncode}): "
                           f"{(done.stderr or done.stdout)[-500:].strip()}") from None
    report = data.get("data") if isinstance(data.get("data"), dict) else data
    if done.returncode != 0 or "rows" not in report:
        raise RuntimeError(f"okg sync frontier failed (exit {done.returncode}): "
                           f"{json.dumps(data)[:400]}")
    return report


# --- the whole observation ------------------------------------------------------------


@dataclass
class Observation:
    problems: list[Problem] = field(default_factory=list)
    #: the families read successfully in this run: only their issues may be closed
    scope: list[str] = field(default_factory=list)
    #: `{"signal", "reason"}` for each signal not applicable (no eval report)
    skipped: list[dict[str, str]] = field(default_factory=list)


def collect(
    config: HealthConfig,
    *,
    instance_dir: Path | str,
    now: datetime,
    read_doctor: Callable[[], Mapping[str, Any]],
    read_frontier: Callable[[], Mapping[str, Any]],
    read_ops_records: Callable[[str], list],
) -> Observation:
    """Run the readers and gather the problems. A reader that raises becomes `monitor/<signal>`."""
    obs = Observation()
    failed: dict[str, str] = {}

    def attempt(signal: str, family: str, work: Callable[[], list[Problem]]) -> None:
        try:
            obs.problems += work()
            obs.scope.append(family)
        except Exception as exc:  # noqa: BLE001 - reported as a problem, not swallowed
            failed[signal] = f"{type(exc).__name__}: {exc}"[:500]

    attempt("doctor", "doctor", lambda: doctor_problems(
        read_doctor(), ignore=config.ignore_checks, include_warnings=config.include_warnings))
    attempt("ingest", "ingest", lambda: frontier_problems(read_frontier()))
    attempt("automation-errors", "automation-errors", lambda: automation_error_problems(
        read_ops_records, now, window=timedelta(hours=config.window_hours),
        min_decisions=config.min_decisions, max_rate=config.max_error_rate))

    def eval_signal() -> list[Problem]:
        found, why = eval_problems(instance_dir, min_drop=config.eval_min_drop)
        if why:  # nothing to judge: skipped, and its issues (if any) are not closed on that account
            obs.skipped.append({"signal": "eval", "reason": why})
        return found

    attempt("eval", "eval", eval_signal)
    if any(s["signal"] == "eval" for s in obs.skipped) and "eval" in obs.scope:
        obs.scope.remove("eval")
    for signal, reason in sorted(failed.items()):
        obs.problems.append(Problem(
            key=make_key("monitor", signal), family="monitor", severity="error",
            title=f"instance-health could not read {signal}",
            detail=(f"The instance-health automation could not read the `{signal}` signal, so it cannot say "
                    f"whether that part of the instance is healthy.\n\n> {reason}\n\nIts existing issues are "
                    "left as they are until it can read the signal again."),
        ))
    obs.scope.append("monitor")
    obs.problems.sort(key=lambda p: p.key)
    return obs
