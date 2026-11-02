"""Score an automation's shadow proposals against what people did later
(PACT r3-ops-runtime, req.shadow; ADR 0002 sections 7.3 and 7.4).

In shadow mode an automation proposes but nothing runs. This module reads the
ops records of one automation over a window and says how its proposals
compare with the people who handled the same incidents. It never writes.

Reading. Each ops subtype is read once, whole (5 `okg filter` calls through
`archi.ops.record.read_nodes`, pinned to the published generation), and
joined in memory. The seam is a `source(subtype) -> rows` function with rows
shaped `{"node_id", "subtype", "attrs"}`; `okg_source` is the live one and
tests pass a fake.

Unit of scoring: the incident. An incident with several shadow proposals from
the automation counts once, through its first proposal in the window (by
`proposed_at`, then record id). `proposals` reports the per-proposal count as
context.

Scope. A proposal is in the window when its `proposed_at` lies in
[since, until] (inclusive). In the window it is scored when its `mode` is
`shadow`; other modes are counted in `ignored_non_shadow`. `as_of` is `until`
when given, else now (UTC). Records after `as_of` are not read for scoring.
Times may carry any UTC offset; a time without one is UTC.

Dropped records are counted in `dropped`, never silently skipped: a proposal
with no `record_id`, no usable `proposed_at`, no `incident_id`, or whose
incident row is missing; and person records with no usable time
(`undated_human_records`).

Human response. A record a person (`actor_kind` = `person`) wrote on the
incident:
  - an `ops_action` (time `executed_at`) on any proposal of the incident, or
    named by an outcome's `action_id`;
  - an `ops_decision` (time `decided_at`) on any proposal of the incident;
  - a person-authored `ops_proposal` (time `proposed_at`).
A person `ops_outcome` is not a first response: it shows a ticket was
closed, not what a person did. It feeds time-to-resolution and
`no_action_needed` only.

Horizon. `horizon_hours` (default 72) after the first proposal's time t0. In
horizon means t0 <= time <= t0 + horizon.

Classes, tested in this order (each incident gets exactly one):
  preempted    a person action happened before t0. The person was first, so
               the proposal cannot be judged against a "later" action. Out of
               every rate.
  agree        an in-horizon person action has the proposal's `action_class`
               and equals its payload on the key fields: all payload fields,
               or `key_fields[action_class]` if given (extra human-side
               fields are ignored). A proposal whose payload is empty or
               unparseable agrees on action class only; those are also
               counted in `agree_class_only` (included in `agree`).
  disagree     an in-horizon person action of another class or payload, or a
               person rejected the proposal (decision, in horizon).
  approved     a person approved the proposal and nothing above applies. In
               shadow mode an approval can be an operator clicking OK, so it
               is not agreement. Counted on its own and scored.
  pending      the horizon ends after `as_of` and nothing above applies. Out
               of every rate and out of the automation's response time.
  false        the horizon has ended, nothing above applies, and a person
               said no action was needed: the latest person outcome checked
               at or after t0 (and by `as_of`) has result `no_action_needed`,
               or the incident row is `false_alarm`, last written by a person
               at or after t0 (`actor_kind` person, `resolved_at` >= t0). A
               resolved incident does not count. Checked before disagree and
               approved when the horizon has ended.
  late         no in-horizon response, but a person response (action,
               decision or person proposal) came after the horizon.
  undecidable  the rest: nothing was done in horizon and nothing says whether
               action was needed. We do not guess.
(While the horizon is open, a disagree or approved response is final; with no
such response the incident is pending. `false` waits for the horizon to end.)

Metrics.
  incidents = scored units; scored = incidents - pending - preempted.
  agreement_rate = agree / (agree + disagree). Approved is left out.
  answered = agree + disagree + false.
  response_coverage = answered / scored.
  false_proposal_rate = false / (scored - undecidable); None when that is 0.
  false_proposal_rate_upper = (false + undecidable) / scored: counts every
    undecidable incident as false. Readiness checks this one.
  time_to_first_response, seconds. `automation`: t0 minus the incident's
    `detected_at`, for every incident except pending ones. `human`: the
    earliest human response at or after `detected_at`, minus `detected_at`;
    a response after the horizon (late) is left out. `median_lead_seconds` is
    the median of human minus automation over incidents with both (negative:
    the person was first). Incidents with no `detected_at`, or t0 before it,
    are left out of both and counted in `excluded`.
  time_to_resolution_human: seconds from `detected_at` to the earliest person
    outcome with result `resolved`.
  Each time summary has n and median. p90 (nearest rank) is given only when
  n >= 10; otherwise it is null and `p90_note` says so.

Known gap. Today nothing writes `no_action_needed`, and nothing records what
an operator did on a ticket without a proposal (`link-tickets` writes only
`resolved` or `unknown` outcomes). Live false-proposal rates will therefore
be mostly undecidable until ticket-triage records operator actions.

    python -m archi.ops.shadow --deployment D --automation A [--since T] [--until T]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Optional, Sequence

from archi.ops.record import RecordError, read_nodes

#: Reads every row of one subtype: subtype -> [{"node_id", "subtype", "attrs"}, ...]
Source = Callable[[str], list]

DEFAULT_HORIZON_HOURS = 72.0
MIN_N_FOR_P90 = 10
NO_ACTION_STATUS = "false_alarm"
NO_ACTION_RESULT = "no_action_needed"
CLASSES = ("agree", "disagree", "false", "approved", "preempted", "late", "undecidable", "pending")
SUBTYPES = ("ops_proposal", "ops_incident", "ops_decision", "ops_action", "ops_outcome")


def okg_source(deployment: str, *, dsn: Optional[str] = None, okg: Optional[str] = None) -> Source:
    """Live source: each subtype is read once, by `okg filter`, then kept."""
    cache: dict[str, list] = {}

    def read(subtype: str) -> list:
        if subtype not in cache:
            cache[subtype] = read_nodes(deployment, subtype, dsn=dsn, okg=okg)
        return cache[subtype]

    return read


def parse_time(value: Any) -> Optional[datetime]:
    """An ISO 8601 time as UTC; None when missing or unparseable."""
    if not value:
        return None
    text = str(value).strip()
    try:
        moment = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def _attrs(row: Mapping[str, Any]) -> dict[str, Any]:
    return dict(row.get("attrs") or {})


def _payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(q * len(ordered))) - 1]


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _ratio(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def _summary(values: Sequence[float]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(values), "median": _median(values)}
    if len(values) >= MIN_N_FOR_P90:
        out["p90"] = _percentile(values, 0.9)
    else:
        out["p90"] = None
        out["p90_note"] = f"p90 needs n >= {MIN_N_FOR_P90}"
    return out


def _wanted(proposal: Mapping[str, Any], key_fields: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    wanted = _payload(proposal.get("payload"))
    fields = key_fields.get(proposal.get("action_class") or "")
    if fields is not None:
        wanted = {k: wanted[k] for k in fields if k in wanted}
    return wanted


def _matches(proposal: Mapping[str, Any], action: Mapping[str, Any],
             key_fields: Mapping[str, Sequence[str]]) -> bool:
    if action.get("action_class") != proposal.get("action_class"):
        return False
    got = _payload(action.get("payload"))
    return all(k in got and got[k] == v for k, v in _wanted(proposal, key_fields).items())


def classify(proposal: Mapping[str, Any], t0: datetime, responses: list, outcomes: list,
             incident: Mapping[str, Any], *, as_of: datetime, horizon: timedelta,
             key_fields: Mapping[str, Sequence[str]]) -> str:
    """The class of one incident's first proposal. `responses` and `outcomes`
    are already limited to records at or before `as_of`."""
    end = t0 + horizon
    me = proposal["record_id"]
    if any(r["kind"] == "action" and r["at"] < t0 for r in responses):
        return "preempted"
    inside = [r for r in responses if t0 <= r["at"] <= end]
    if any(r["kind"] == "action" and _matches(proposal, r["attrs"], key_fields) for r in inside):
        return "agree"
    differing = any(r["kind"] == "action"
                    or (r["kind"] == "decision" and r["attrs"].get("proposal_id") == me
                        and r["attrs"].get("verdict") == "rejected") for r in inside)
    approved = any(r["kind"] == "decision" and r["attrs"].get("proposal_id") == me
                   and r["attrs"].get("verdict") == "approved" for r in inside)
    if end > as_of:
        return "disagree" if differing else "approved" if approved else "pending"
    if _no_action_needed(incident, outcomes, t0):
        return "false"
    if differing:
        return "disagree"
    if approved:
        return "approved"
    if any(r["at"] > end for r in responses):
        return "late"
    return "undecidable"


def _no_action_needed(incident: Mapping[str, Any], outcomes: list, t0: datetime) -> bool:
    mine = [(parse_time(o.get("checked_at")), o) for o in outcomes if o.get("actor_kind") == "person"]
    mine = [(at, o) for at, o in mine if at is not None and at >= t0]
    if mine and max(mine, key=lambda pair: pair[0])[1].get("result") == NO_ACTION_RESULT:
        return True
    written = parse_time(incident.get("resolved_at"))
    return (incident.get("status") == NO_ACTION_STATUS and incident.get("actor_kind") == "person"
            and written is not None and written >= t0)


def score_automation(
    source: Source,
    automation: str,
    *,
    since: Optional[str] = None,
    until: Optional[str] = None,
    horizon_hours: float = DEFAULT_HORIZON_HOURS,
    key_fields: Optional[Mapping[str, Sequence[str]]] = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """The shadow report for `automation` over [since, until] (JSON-ready)."""
    lo, hi = parse_time(since), parse_time(until)
    if (since and lo is None) or (until and hi is None):
        raise RecordError("since and until must be ISO 8601 timestamps")
    as_of = hi or (now.astimezone(timezone.utc) if now else datetime.now(timezone.utc))
    horizon = timedelta(hours=horizon_hours)
    key_fields = key_fields or {}

    rows = {subtype: [_attrs(r) for r in source(subtype)] for subtype in SUBTYPES}
    dropped = {"proposal_no_record_id": 0, "proposal_bad_proposed_at": 0, "proposal_no_incident_id": 0,
               "proposal_incident_missing": 0, "undated_human_records": 0}
    incidents = {a["record_id"]: a for a in rows["ops_incident"] if a.get("record_id")}
    proposals_by_incident: dict[str, list] = {}
    for a in rows["ops_proposal"]:
        if a.get("incident_id") and a.get("record_id"):
            proposals_by_incident.setdefault(a["incident_id"], []).append(a)
    actions_by_proposal: dict[str, list] = {}
    actions_by_id: dict[str, dict] = {}
    for a in rows["ops_action"]:
        actions_by_proposal.setdefault(a.get("proposal_id"), []).append(a)
        actions_by_id[a.get("record_id")] = a
    decisions_by_proposal: dict[str, list] = {}
    for a in rows["ops_decision"]:
        decisions_by_proposal.setdefault(a.get("proposal_id"), []).append(a)
    outcomes_by_incident: dict[str, list] = {}
    for a in rows["ops_outcome"]:
        outcomes_by_incident.setdefault(a.get("incident_id"), []).append(a)

    # The automation's shadow proposals in the window, first per incident.
    ignored = 0
    kept = 0
    first: dict[str, tuple] = {}
    for a in rows["ops_proposal"]:
        if a.get("automation") != automation:
            continue
        if not a.get("record_id"):
            dropped["proposal_no_record_id"] += 1
            continue
        moment = parse_time(a.get("proposed_at"))
        if moment is None:
            dropped["proposal_bad_proposed_at"] += 1
            continue
        if (lo and moment < lo) or moment > as_of:
            continue
        if a.get("mode") != "shadow":
            ignored += 1
            continue
        if not a.get("incident_id"):
            dropped["proposal_no_incident_id"] += 1
            continue
        if a["incident_id"] not in incidents:
            dropped["proposal_incident_missing"] += 1
            continue
        kept += 1
        key = (moment, a["record_id"])
        if a["incident_id"] not in first or key < first[a["incident_id"]][0]:
            first[a["incident_id"]] = (key, a)

    counts = {name: 0 for name in CLASSES}
    class_only = 0
    details = []
    auto_ttfr, human_ttfr, lead, ttr, excluded = [], [], [], [], 0
    for iid, ((t0, _), proposal) in sorted(first.items(), key=lambda kv: kv[1][0]):
        incident = incidents[iid]
        outcomes = [o for o in outcomes_by_incident.get(iid, [])
                    if (parse_time(o.get("checked_at")) or as_of) <= as_of]
        responses, seen = [], set()

        def respond(kind: str, at_value: Any, attrs: dict) -> None:
            at = parse_time(at_value)
            if at is None:
                dropped["undated_human_records"] += 1
            elif at <= as_of:
                responses.append({"kind": kind, "at": at, "attrs": attrs})

        def add_action(a: dict) -> None:
            if a.get("actor_kind") == "person" and a.get("record_id") not in seen:
                seen.add(a.get("record_id"))
                respond("action", a.get("executed_at"), a)

        for q in proposals_by_incident.get(iid, []):
            if q.get("actor_kind") == "person":
                respond("proposal", q.get("proposed_at"), q)
            for a in actions_by_proposal.get(q["record_id"], []):
                add_action(a)
            for d in decisions_by_proposal.get(q["record_id"], []):
                if d.get("actor_kind") == "person":
                    respond("decision", d.get("decided_at"), d)
        for o in outcomes:
            if o.get("action_id") in actions_by_id:
                add_action(actions_by_id[o["action_id"]])

        label = classify(proposal, t0, responses, outcomes, incident, as_of=as_of,
                         horizon=horizon, key_fields=key_fields)
        counts[label] += 1
        if label == "agree" and not _wanted(proposal, key_fields):
            class_only += 1
        details.append({"incident_id": iid, "proposal_id": proposal["record_id"],
                        "action_class": proposal.get("action_class"), "proposed_at": _iso(t0),
                        "class": label})

        detected = parse_time(incident.get("detected_at"))
        if detected is None or t0 < detected:
            excluded += 1
            continue
        if label != "pending":
            auto_ttfr.append((t0 - detected).total_seconds())
        times = [r["at"] for r in responses if r["at"] >= detected]
        if times and min(times) <= t0 + horizon and label != "pending":
            human_ttfr.append((min(times) - detected).total_seconds())
            lead.append((min(times) - t0).total_seconds())
        resolved = [at for o in outcomes if o.get("actor_kind") == "person" and o.get("result") == "resolved"
                    for at in [parse_time(o.get("checked_at"))] if at is not None and at >= detected]
        if resolved:
            ttr.append((min(resolved) - detected).total_seconds())

    total = len(first)
    scored = total - counts["pending"] - counts["preempted"]
    answered = counts["agree"] + counts["disagree"] + counts["false"]
    return {
        "automation": automation,
        "window": {"since": since, "until": until, "as_of": _iso(as_of)},
        "horizon_hours": horizon_hours,
        "incidents": total,
        "proposals": kept,
        "ignored_non_shadow": ignored,
        "dropped": dropped,
        "scored": scored,
        "counts": counts,
        "agree_class_only": class_only,
        "answered": answered,
        "agreement_rate": _ratio(counts["agree"], counts["agree"] + counts["disagree"]),
        "response_coverage": _ratio(answered, scored),
        "false_proposal_rate": _ratio(counts["false"], scored - counts["undecidable"]),
        "false_proposal_rate_upper": _ratio(counts["false"] + counts["undecidable"], scored),
        "time_to_first_response": {
            "automation": _summary(auto_ttfr),
            "human": _summary(human_ttfr),
            "median_lead_seconds": _median(lead),
            "excluded": excluded,
        },
        "time_to_resolution_human": _summary(ttr),
        "details": details,
    }


def promotion_readiness(report: Mapping[str, Any], *, min_proposals: int = 30, min_answered: int = 20,
                        min_response_coverage: float = 0.5, min_agreement: float = 0.8,
                        max_false_rate: float = 0.1) -> dict[str, Any]:
    """Is the report good enough to ask an operator to promote? Advice only.

    Promotion itself is a PACT decision recorded in the instance repo.
    Checks scored incidents (`min_proposals`), answered incidents (agree +
    disagree + false), response coverage, agreement_rate, and
    false_proposal_rate_upper (undecidable counted as false). A missing rate
    fails.
    """
    reasons = []
    scored = report.get("scored", 0)
    if scored < min_proposals:
        reasons.append(f"only {scored} scored incidents; need {min_proposals}")
    answered = report.get("answered", 0)
    if answered < min_answered:
        reasons.append(f"only {answered} answered incidents (agree, disagree or false); need {min_answered}")
    coverage = report.get("response_coverage")
    if coverage is None:
        reasons.append("no response coverage: nothing was scored")
    elif coverage < min_response_coverage:
        reasons.append(f"response coverage {coverage:.2f} is below {min_response_coverage:.2f}")
    agreement = report.get("agreement_rate")
    if agreement is None:
        reasons.append("no agreement rate: no person answered a proposal in horizon")
    elif agreement < min_agreement:
        reasons.append(f"agreement {agreement:.2f} is below {min_agreement:.2f}")
    false_rate = report.get("false_proposal_rate_upper")
    if false_rate is None:
        reasons.append("no false-proposal rate: nothing was scored")
    elif false_rate > max_false_rate:
        reasons.append(f"false-proposal rate (undecidable counted as false) {false_rate:.2f} "
                       f"is above {max_false_rate:.2f}")
    return {"ready": not reasons, "reasons": reasons}


def _fmt(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _hours(seconds: Any) -> str:
    return "n/a" if seconds is None else f"{seconds / 3600:.1f} h"


def _row(label: str, s: Mapping[str, Any]) -> str:
    p90 = _hours(s["p90"]) if s["p90"] is not None else f"n/a (n < {MIN_N_FOR_P90})"
    return f"| {label} | {s['n']} | {_hours(s['median'])} | {p90} |"


def render_markdown(report: Mapping[str, Any]) -> str:
    c, t, w = report["counts"], report["time_to_first_response"], report["window"]
    dropped = {k: v for k, v in report["dropped"].items() if v}
    lines = [
        f"# Shadow report: {report['automation']}",
        "",
        f"Window {_fmt(w['since'])} to {_fmt(w['until'])}, as of {w['as_of']}; "
        f"horizon {report['horizon_hours']:g} h. Scored per incident, through its first proposal.",
        "",
        f"- Incidents: {report['incidents']} ({report['scored']} scored, {c['pending']} pending, "
        f"{c['preempted']} preempted by a person); {report['proposals']} proposals, "
        f"{report['ignored_non_shadow']} non-shadow ignored",
        f"- Agreement: {_fmt(report['agreement_rate'])} "
        f"({c['agree']} agree, of which {report['agree_class_only']} on action class only; "
        f"{c['disagree']} disagree). Approved without agreement: {c['approved']}",
        f"- Response coverage: {_fmt(report['response_coverage'])} "
        f"({report['answered']} answered: agree, disagree or false)",
        f"- False proposals: {c['false']} (rate {_fmt(report['false_proposal_rate'])}; "
        f"upper bound {_fmt(report['false_proposal_rate_upper'])})",
        f"- Late human response: {c['late']}; undecidable: {c['undecidable']}",
        f"- Dropped records: {dropped if dropped else 'none'}",
        "",
        "| Time | n | median | p90 |",
        "|---|---|---|---|",
        _row("first response, automation", t["automation"]),
        _row("first response, human", t["human"]),
        _row("resolution, human", report["time_to_resolution_human"]),
        "",
        f"Median lead of automation over human: {_hours(t['median_lead_seconds'])} "
        f"({t['excluded']} incidents excluded). Late responses are left out of the human figures.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m archi.ops.shadow",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--automation", required=True)
    parser.add_argument("--since")
    parser.add_argument("--until")
    parser.add_argument("--horizon-hours", type=float, default=DEFAULT_HORIZON_HOURS)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--dsn")
    args = parser.parse_args(argv)
    try:
        report = score_automation(okg_source(args.deployment, dsn=args.dsn), args.automation,
                                  since=args.since, until=args.until,
                                  horizon_hours=args.horizon_hours)
    except RecordError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2) if args.format == "json" else render_markdown(report), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
