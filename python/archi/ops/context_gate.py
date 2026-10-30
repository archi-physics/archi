"""Context gates (ADR 0002 section 6.3, PACT r3-context-quality).

An automation declares the context it needs: sources, a maximum staleness,
a minimum context-eval score and the invariants that must hold. Before a run,
the runtime evaluates the gate against the current generation's facts. A pass
lets the run act. A failure names each failing check, and the run drops to
shadow mode carrying those names, so the operator sees why.

    spec = GateSpec.from_mapping({
        "sources": ["cms-catalogs.cric", "monit-live.monit_sam"],
        "max_staleness": "36h",            # optional; default is each source's own SLO
        "min_context_eval_score": 0.8,     # optional
        "max_age_days": 7,                 # optional, with the score; default 14
        "context_eval_bundle": "cms-comp-ops",  # optional; default the instance's only bundle
        "context_eval_k": 10,              # optional: the run must have used this k
        "context_eval_min_questions": 40,  # optional: and scored at least this many questions
        "invariants": ["cms_catalogs_coverage", "cms_catalogs_link_site_config_site"],
    })
    result = evaluate_gate(spec, facts_from_doctor(doctor_report, invariants=..., context_eval_score=...))
    result.ok, result.failing

The evaluator reads facts; it fetches nothing. `archi doctor --json` supplies
the source ages (its `source-freshness` check). Invariant results and the
context-eval score come from the caller: okg records invariant outcomes per
generation, and `archi eval --context` writes a report per run under
`<data root>/eval/context/<bundle>/<UTC stamp>/report.json`, whose graph-arm
recall@k is the score. `read_context_eval` takes the newest run of one bundle
and refuses it, with the reason, when it cannot be read, is not that bundle's,
did not execute cleanly, lacks its question-file hash or count, or is older than
`max_age_days` (default 14). A missing score or invariant result is a failure,
never a pass: a gate that cannot see a fact has not shown it holds.

Check names: `freshness:<source>`, `context_eval`, `invariant:<name>`, `published`.

`published: true` asks only whether the graph has a published generation to read. It is the gate of an
automation that reports on the instance's health: a freshness check there would hold the monitor in
shadow exactly when the thing it reports is stale.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from archi.bundles.duration import format_age, parse_duration

#: `archi eval --context` writes each run to `<data root>/eval/context/<bundle>/<UTC stamp>/report.json`.
CONTEXT_EVAL_SUBDIR = ("eval", "context")
CONTEXT_EVAL_REPORT = "report.json"
CONTEXT_REPORT_SCHEMA = "archi-context-eval-report/1"
#: How old a context-eval run may be before the gate stops trusting it (gate key `max_age_days`).
DEFAULT_CONTEXT_EVAL_MAX_AGE_DAYS = 14.0
#: A bundle name as it may appear in a path: no separators, no `..`, no leading dot.
BUNDLE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
#: `<YYYYMMDDTHHMMSSZ>` or `<YYYYMMDDTHHMMSSZ>-<n>`, the run directories `archi eval --context` makes
RUN_DIR = re.compile(r"^(\d{8}T\d{6}Z)(?:-(\d+))?$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
#: The graph arm a run must have used for the gate to take its score: `search` with the server's
#: defaults (no arguments besides the question and k). A question file may override it, but such a
#: run is informational only.
DEFAULT_GRAPH_TOOL = "search"
#: The largest `max_age_days` a gate may set (ten years).
MAX_AGE_DAYS_LIMIT = 3650


def bundle_path_name(name: Any) -> str:
    """`name` if it is safe as one path component; ValueError otherwise."""
    if not isinstance(name, str) or not BUNDLE_NAME.match(name) or ".." in name:
        raise ValueError(f"bundle name {name!r} cannot be used in a path (letters, digits, '.', '_' and '-' "
                         "only, no '/' and no '..')")
    return name


def context_eval_dir(data_root: Path | str, bundle: str) -> Path:
    """`<data root>/eval/context/<bundle>`: one directory per run below it."""
    return Path(data_root).joinpath(*CONTEXT_EVAL_SUBDIR, bundle_path_name(bundle))


def run_dirs(directory: Path) -> list[Path]:
    """The run directories under `directory`, oldest first (by their stamp, then the collision suffix)."""
    found = []
    for path in directory.iterdir() if directory.is_dir() else []:
        match = RUN_DIR.match(path.name)
        if match and path.is_dir():
            found.append(((match.group(1), int(match.group(2) or 1)), path))
    return [path for _, path in sorted(found)]


@dataclass(frozen=True)
class ContextEvalReading:
    """What the gate may take from the newest context-eval run of a bundle.

    `score` is set only when the run is usable; otherwise `problem` says why not."""

    score: Optional[float] = None
    problem: Optional[str] = None
    path: Optional[str] = None
    finished_at: Optional[str] = None
    generation: Optional[str] = None


def read_context_eval(data_root: Path | str, bundle: str, *, max_age_days: float = DEFAULT_CONTEXT_EVAL_MAX_AGE_DAYS,
                      now: Optional[datetime] = None, k: Optional[int] = None,
                      min_questions: Optional[int] = None) -> ContextEvalReading:
    """The newest `archi eval --context` run of `bundle` under the data root. Never raises.

    Only the newest run is read: when it is unusable the gate fails, and an older run is not used
    instead. It is unusable when it cannot be read, is not a context-eval report of this bundle, did
    not execute cleanly (an MCP call of the graph arm failed), ran a graph arm other than the default,
    lacks its question-file hash or question count, finished more than `max_age_days` ago or more than
    five minutes in the future, or (when the gate pins them) used another `k` or fewer than
    `min_questions` scored questions.
    """
    try:
        return _read_context_eval(data_root, bundle, max_age_days=max_age_days, now=now, k=k,
                                  min_questions=min_questions)
    except Exception as exc:  # noqa: BLE001 - the gate must get a named failure, never an exception
        return ContextEvalReading(problem=f"the context eval of bundle {bundle} could not be checked "
                                          f"({type(exc).__name__}: {exc})")


#: How far in the future a run's finished_at may lie (clock skew) before it is refused.
FUTURE_TOLERANCE = timedelta(minutes=5)


def _read_context_eval(data_root, bundle, *, max_age_days, now, k, min_questions) -> ContextEvalReading:
    try:
        directory = context_eval_dir(data_root, bundle)
    except ValueError as exc:
        return ContextEvalReading(problem=str(exc))
    try:
        runs = run_dirs(directory)
    except OSError as exc:
        return ContextEvalReading(problem=f"cannot list the context evals of bundle {bundle} under {directory} "
                                          f"({type(exc).__name__}: {exc})")
    if not runs:
        return ContextEvalReading(problem=f"no context eval of bundle {bundle} is recorded under {directory} "
                                          f"(run `archi eval --context`)")
    path = runs[-1] / CONTEXT_EVAL_REPORT
    where = str(path)

    def bad(why: str) -> ContextEvalReading:
        return ContextEvalReading(problem=f"the newest context eval of bundle {bundle} ({where}) {why}", path=where)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return bad(f"cannot be read ({type(exc).__name__}: {exc})")
    if not isinstance(data, Mapping):
        return bad("is not a JSON object")
    if data.get("schema") != CONTEXT_REPORT_SCHEMA:
        return bad(f"has schema {data.get('schema')!r}, not {CONTEXT_REPORT_SCHEMA}")
    if data.get("bundle") != bundle:
        return bad(f"is a run of bundle {data.get('bundle')!r}")
    if data.get("status") not in ("pass", "fail"):
        return bad(f"did not execute (status {data.get('status')!r}: {data.get('error') or 'see the report'})")
    errors = data.get("errors")
    if isinstance(errors, bool) or not isinstance(errors, int) or errors < 0:
        return bad("has no error count")
    arms = data.get("arms")
    graph = arms.get("graph") if isinstance(arms, Mapping) else None
    if not isinstance(graph, Mapping) or graph.get("errors") != 0:
        return bad("does not show a graph arm whose every call executed")
    if graph.get("tool") != DEFAULT_GRAPH_TOOL or graph.get("arguments") != {}:
        return bad("ran a graph arm the question file overrode; such a run is informational only")
    if k is not None and data.get("k") != k:
        return bad(f"used k={data.get('k')!r}, not the k={k} the gate pins (context_eval_k)")
    score = data.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
        return bad(f"has no score from 0 to 1 (got {score!r})")
    if not isinstance(data.get("questions_sha256"), str) or not _SHA256.match(data["questions_sha256"]):
        return bad("does not record the hash of its question file")
    count, rows = data.get("question_count"), data.get("questions")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1 or not isinstance(rows, list) \
            or count != len(rows):
        shown = len(rows) if isinstance(rows, list) else "no"
        return bad(f"records {count!r} scored questions but has {shown} question rows")
    if min_questions is not None and count < min_questions:
        return bad(f"scored {count} questions, fewer than the {min_questions} the gate requires "
                   "(context_eval_min_questions)")
    finished = _parse_time(data.get("finished_at"))
    if finished is None:
        return bad(f"has no usable finished_at ({data.get('finished_at')!r})")
    now = now or datetime.now(timezone.utc)
    if finished - now > FUTURE_TOLERANCE:
        return bad(f"says it finished at {data['finished_at']}, in the future; a clock or the report is wrong")
    age = now - finished
    if age > timedelta(days=max_age_days):
        return bad(f"finished {format_age(int(age.total_seconds()))} ago, more than the {max_age_days:g} days "
                   "the gate accepts (max_age_days); run `archi eval --context` again")
    generation = data.get("generation")
    return ContextEvalReading(float(score), None, where, data["finished_at"],
                              None if generation is None else str(generation))


def _parse_time(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def context_eval_reading(spec: "GateSpec", instance_dir: Path | str, *, now: Optional[datetime] = None
                         ) -> ContextEvalReading:
    """The reading a gate with `min_context_eval_score` gets on this instance. Never raises.

    The bundle is the gate's `context_eval_bundle`, else the instance's only bundle; an instance with
    several bundles and no `context_eval_bundle` fails the check by name."""
    try:
        from archi.stack.instance import Instance

        instance = Instance.load(Path(instance_dir))
    except Exception as exc:  # noqa: BLE001 - a gate that cannot read the instance fails, by name
        return ContextEvalReading(problem=f"cannot read the instance to find its context evals ({exc})")
    bundle = spec.context_eval_bundle
    if bundle is None:
        if len(instance.bundles) != 1:
            return ContextEvalReading(problem=(
                f"the instance lists {len(instance.bundles)} bundles ({', '.join(instance.bundles) or 'none'}); "
                "name the one whose context eval this gate trusts with context_eval_bundle"))
        bundle = instance.bundles[0]
    elif bundle not in instance.bundles:
        return ContextEvalReading(problem=f"context_eval_bundle {bundle!r} is not a bundle of this instance")
    max_age = spec.max_age_days if spec.max_age_days is not None else DEFAULT_CONTEXT_EVAL_MAX_AGE_DAYS
    return read_context_eval(instance.data_root, bundle, max_age_days=max_age, now=now, k=spec.context_eval_k,
                             min_questions=spec.context_eval_min_questions)

# The results that mean an invariant holds. Compared by type as well as value,
# so 1 (equal to True) and unhashable values never count as a pass.
PASSING = frozenset({"pass", "passed", "ok"})


def _holds(result: Any) -> bool:
    return result is True or (isinstance(result, str) and result in PASSING)
SPEC_KEYS = frozenset({"sources", "max_staleness", "min_context_eval_score", "invariants", "published",
                       "max_age_days", "context_eval_bundle", "context_eval_k", "context_eval_min_questions"})
#: keys that only mean something together with min_context_eval_score
CONTEXT_EVAL_KEYS = ("max_age_days", "context_eval_bundle", "context_eval_k", "context_eval_min_questions")


class GateSpecError(ValueError):
    """The gate spec is malformed."""


@dataclass(frozen=True)
class GateSpec:
    sources: tuple[str, ...] = ()
    #: seconds; None means each source is held to its own pack's max_staleness
    max_staleness: Optional[int] = None
    min_context_eval_score: Optional[float] = None
    invariants: tuple[str, ...] = ()
    #: require a published generation (and nothing about how old its sources are)
    published: bool = False
    #: the oldest context-eval run (days) the score is taken from; None: DEFAULT_CONTEXT_EVAL_MAX_AGE_DAYS
    max_age_days: Optional[float] = None
    #: the bundle whose context eval is trusted; None: the instance's only bundle
    context_eval_bundle: Optional[str] = None
    #: the k the run must have used; None: any
    context_eval_k: Optional[int] = None
    #: the fewest scored questions the run may have had; None: any
    context_eval_min_questions: Optional[int] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GateSpec":
        unknown = sorted(set(data) - SPEC_KEYS)
        if unknown:
            raise GateSpecError(f"unknown gate keys {unknown}; allowed: {sorted(SPEC_KEYS)}")
        sources = _names(data, "sources")
        invariants = _names(data, "invariants")
        staleness = None
        if data.get("max_staleness") is not None:
            try:
                staleness = parse_duration(data["max_staleness"])
            except ValueError as exc:
                raise GateSpecError(f"max_staleness: {exc}") from exc
        score = data.get("min_context_eval_score")
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
                raise GateSpecError(f"min_context_eval_score must be a number from 0 to 1, got {score!r}")
            score = float(score)
        published = data.get("published", False)
        if not isinstance(published, bool):
            raise GateSpecError(f"published must be true or false, got {published!r}")
        max_age = data.get("max_age_days")
        if max_age is not None:
            if isinstance(max_age, bool) or not isinstance(max_age, (int, float)) or not math.isfinite(max_age) \
                    or not 0 < max_age <= MAX_AGE_DAYS_LIMIT:
                raise GateSpecError(f"max_age_days must be a number of days above 0 and at most "
                                    f"{MAX_AGE_DAYS_LIMIT}, got {max_age!r}")
            max_age = float(max_age)
        bundle = data.get("context_eval_bundle")
        if bundle is not None:
            try:
                bundle_path_name(bundle)
            except ValueError as exc:
                raise GateSpecError(f"context_eval_bundle: {exc}") from exc
        k = _count(data, "context_eval_k", 1, 100)
        min_questions = _count(data, "context_eval_min_questions", 1, None)
        given = [key for key in CONTEXT_EVAL_KEYS if data.get(key) is not None]
        if given and score is None:
            raise GateSpecError(f"{', '.join(given)} apply only with min_context_eval_score")
        return cls(sources, staleness, score, invariants, published, max_age, bundle, k, min_questions)


def _count(data: Mapping[str, Any], key: str, low: int, high: Optional[int]) -> Optional[int]:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < low or (high is not None and value > high):
        bound = f"from {low} to {high}" if high is not None else f"of at least {low}"
        raise GateSpecError(f"{key} must be an integer {bound}, got {value!r}")
    return value


def _names(data: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key, ())
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) and v for v in value):
        raise GateSpecError(f"{key} must be a list of names")
    return tuple(dict.fromkeys(value))


@dataclass(frozen=True)
class GateFacts:
    """What is known about the current generation."""

    generation: Optional[str] = None
    #: `archi doctor --json` source-freshness rows: {source, age_seconds, max_staleness_seconds, ...}
    freshness: tuple[Mapping[str, Any], ...] = ()
    #: invariant name -> True/False or "pass"/"fail"/"warn"/"error"
    invariants: Mapping[str, Any] = field(default_factory=dict)
    context_eval_score: Optional[float] = None
    #: does the graph have a published generation? None: not asked
    published: Optional[bool] = None
    #: why no context-eval score can be used (unreadable, too old, ...); fails the check with this reason
    context_eval_problem: Optional[str] = None


@dataclass(frozen=True)
class GateCheck:
    name: str
    ok: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass(frozen=True)
class GateResult:
    generation: Optional[str]
    checks: tuple[GateCheck, ...]

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def status(self) -> str:
        return "pass" if self.ok else "fail"

    @property
    def failing(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.checks if not c.ok)

    @property
    def message(self) -> str:
        if self.ok:
            return f"context gate passed ({len(self.checks)} checks)"
        return "context gate failed: " + "; ".join(f"{c.name}: {c.detail}" for c in self.checks if not c.ok)

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "ok": self.ok, "generation": self.generation,
                "failing": list(self.failing), "message": self.message,
                "checks": [c.as_dict() for c in self.checks]}


def facts_from_doctor(
    report: Mapping[str, Any],
    *,
    invariants: Optional[Mapping[str, Any]] = None,
    context_eval_score: Optional[float] = None,
    generation: Optional[str] = None,
) -> GateFacts:
    """Facts from an `archi doctor --json` report plus what the caller knows.

    The generation defaults to the one the report's `published` check names.
    """
    by_id = {c.get("id"): c for c in report.get("checks", ())}
    rows = tuple(((by_id.get("source-freshness") or {}).get("data") or {}).get("sources") or ())
    if generation is None:
        message = (by_id.get("published") or {}).get("message") or ""
        generation = message.removeprefix("published ") or None
    return GateFacts(generation, rows, dict(invariants or {}), context_eval_score)


def _freshness_check(source: str, row: Optional[Mapping[str, Any]], override: Optional[int]) -> GateCheck:
    name = f"freshness:{source}"
    if row is None:
        return GateCheck(name, False, "no age reported for this source (doctor's source-freshness check did not list it)")
    age = row.get("age_seconds")
    if age is None:
        return GateCheck(name, False, row.get("reason") or "no completed run recorded")
    limit = override if override is not None else row.get("max_staleness_seconds")
    if limit is None:
        return GateCheck(name, False, "no max_staleness to hold it to")
    if age > limit:
        return GateCheck(name, False, f"last run {format_age(age)} ago, over the {format_age(limit)} limit")
    return GateCheck(name, True, f"{format_age(age)} old, within {format_age(limit)}")


def evaluate_gate(spec: GateSpec, facts: GateFacts) -> GateResult:
    """Pass, or fail with every failing check named."""
    checks: list[GateCheck] = []
    rows = {r.get("source"): r for r in facts.freshness}
    if spec.published:
        if facts.published:
            checks.append(GateCheck("published", True, "the graph has a published generation"))
        else:
            checks.append(GateCheck("published", False, "the graph has no published generation to read"))
    for source in spec.sources:
        checks.append(_freshness_check(source, rows.get(source), spec.max_staleness))

    if spec.min_context_eval_score is not None:
        score = facts.context_eval_score
        if facts.context_eval_problem:
            checks.append(GateCheck("context_eval", False, facts.context_eval_problem))
        elif score is None:
            checks.append(GateCheck("context_eval", False,
                                    "no context-eval score recorded (run `archi eval --context`)"))
        elif score < spec.min_context_eval_score:
            checks.append(GateCheck("context_eval", False,
                                    f"score {score:g} is below the minimum {spec.min_context_eval_score:g}"))
        else:
            checks.append(GateCheck("context_eval", True,
                                    f"score {score:g} meets the minimum {spec.min_context_eval_score:g}"))

    for name in spec.invariants:
        label = f"invariant:{name}"
        if name not in facts.invariants:
            checks.append(GateCheck(label, False, "no result recorded for this generation"))
        elif _holds(facts.invariants[name]):
            checks.append(GateCheck(label, True, "holds"))
        else:
            checks.append(GateCheck(label, False, f"does not hold ({facts.invariants[name]})"))
    return GateResult(facts.generation, tuple(checks))
