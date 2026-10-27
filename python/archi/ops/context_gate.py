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
        "invariants": ["cms_catalogs_coverage", "cms_catalogs_link_site_config_site"],
    })
    result = evaluate_gate(spec, facts_from_doctor(doctor_report, invariants=..., context_eval_score=...))
    result.ok, result.failing

The evaluator reads facts; it fetches nothing. `archi doctor --json` supplies
the source ages (its `source-freshness` check). Invariant results and the
context-eval score come from the caller: okg records invariant outcomes per
generation, and `archi eval --context` records the score under the instance
(`eval/reports/context-*.json`), which `read_context_eval_score` returns: the
graph arm's recall@k from the latest run. A missing score or invariant result
is a failure, never a pass: a gate that cannot see a fact has not shown it holds.

Check names: `freshness:<source>`, `context_eval`, `invariant:<name>`, `published`.

`published: true` asks only whether the graph has a published generation to read. It is the gate of an
automation that reports on the instance's health: a freshness check there would hold the monitor in
shadow exactly when the thing it reports is stale.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from archi.bundles.duration import format_age, parse_duration

#: Where `archi eval --context` records its score, under the instance directory. The
#: instance-health monitor reads the same directory for its eval trend.
CONTEXT_EVAL_DIR = "eval/reports"
CONTEXT_EVAL_GLOB = "context-*.json"
CONTEXT_EVAL_SCHEMA = "archi-context-eval-score/1"


class ContextEvalRecordError(ValueError):
    """A context-eval score record cannot be read."""


def read_context_eval_score(instance_dir: Path | str) -> Optional[float]:
    """The score of the latest `archi eval --context` run on this instance, or None if none ran.

    "Latest" is the record with the greatest `finished_at`. A record that is not JSON, or has no
    numeric score from 0 to 1, raises: a broken record is not the same as no record.
    """
    directory = Path(instance_dir) / CONTEXT_EVAL_DIR
    if not directory.is_dir():
        return None
    latest: Optional[tuple[str, float]] = None
    for path in sorted(directory.glob(CONTEXT_EVAL_GLOB)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            score, finished = data["score"], str(data["finished_at"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ContextEvalRecordError(f"{path}: not a context-eval score record ({type(exc).__name__}: {exc})") from exc
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise ContextEvalRecordError(f"{path}: score must be a number from 0 to 1, got {score!r}")
        if latest is None or finished >= latest[0]:
            latest = (finished, float(score))
    return None if latest is None else latest[1]

# The results that mean an invariant holds. Compared by type as well as value,
# so 1 (equal to True) and unhashable values never count as a pass.
PASSING = frozenset({"pass", "passed", "ok"})


def _holds(result: Any) -> bool:
    return result is True or (isinstance(result, str) and result in PASSING)
SPEC_KEYS = frozenset({"sources", "max_staleness", "min_context_eval_score", "invariants", "published"})


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
        return cls(sources, staleness, score, invariants, published)


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
        if score is None:
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
