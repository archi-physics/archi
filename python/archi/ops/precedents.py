"""Retrieve precedents from the operational memory (PACT r3-ops-record).

A precedent is a past incident on the same entity, with what was proposed,
what was decided, what the mediator ran and how it turned out. The next
triage reads them before proposing anything (ADR 0002 section 6.2: "this
transfer error last time was fixed by...").

Lookup is by attribute, through okg's read surface (`okg filter`, pinned to
the published generation): incidents with `entity` = the entity, then the
proposals and outcomes naming each incident, then the decisions and actions
naming each proposal. The ops-record playbook (`ops_precedents.md`) teaches an
agent the same walk over the MCP tools.

    python -m archi.ops.precedents --deployment D --entity T2_US_MIT
    python -m archi.ops.precedents --deployment D --incident inc-20261003-t2usmit-srm
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any, Callable, Mapping, Optional

from archi.ops.record import RecordError, _okg

#: Reads rows of one subtype whose attribute equals a value:
#: (subtype, attr, value) -> [{"node_id", "subtype", "attrs"}, ...]
Reader = Callable[[str, str, str], list]


def okg_reader(deployment: str, *, dsn: Optional[str] = None, okg: Optional[str] = None) -> Reader:
    def read(subtype: str, attr: str, value: str) -> list:
        cmd = [okg or _okg(), "filter", "--deployment", deployment, "--type", subtype,
               "--where", json.dumps({"type": subtype, f"attrs.{attr}": value}),
               "--all", "--jsonl"]
        if dsn:
            cmd += ["--dsn", dsn]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RecordError(f"okg filter {subtype} {attr}={value} failed: "
                              f"{(result.stderr or result.stdout)[-2000:]}")
        return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]

    return read


def _attrs(row: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in (row.get("attrs") or {}).items()
            if k not in ("text", "nomos_source_lineage")}


def find_precedents(
    read: Reader,
    *,
    entity: Optional[str] = None,
    incident: Optional[str] = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Past incidents on `entity` (or on the entity of incident `incident`).

    The incident itself is excluded. Precedents that share its
    `error_signature` come first, then the most recent; each carries its
    proposals (with their decisions and actions) and its outcomes.
    """
    signature = None
    if incident is not None:
        rows = read("ops_incident", "record_id", incident)
        if not rows:
            raise RecordError(f"no ops_incident with record_id {incident!r} in the published graph")
        current = _attrs(rows[0])
        entity = entity or current["entity"]
        signature = current.get("error_signature")
    if not entity:
        raise RecordError("name an entity or an incident")

    candidates = [r for r in read("ops_incident", "entity", entity)
                  if _attrs(r).get("record_id") != incident]
    candidates.sort(key=lambda r: _attrs(r).get("detected_at") or "", reverse=True)
    if signature:
        candidates.sort(key=lambda r: _attrs(r).get("error_signature") != signature)

    precedents = []
    for row in candidates[:limit]:
        attrs = _attrs(row)
        rid = attrs["record_id"]
        proposals = []
        for prow in read("ops_proposal", "incident_id", rid):
            pattrs = _attrs(prow)
            proposals.append({
                "node_id": prow["node_id"],
                **pattrs,
                "decisions": [{"node_id": d["node_id"], **_attrs(d)}
                              for d in read("ops_decision", "proposal_id", pattrs["record_id"])],
                "actions": [{"node_id": a["node_id"], **_attrs(a)}
                            for a in read("ops_action", "proposal_id", pattrs["record_id"])],
            })
        # Oldest first, so the last outcome is the latest word on the incident
        # (a later ticket resolution can follow an outcome check).
        outcomes = sorted(({"node_id": o["node_id"], **_attrs(o)}
                           for o in read("ops_outcome", "incident_id", rid)),
                          key=lambda o: o.get("checked_at") or "")
        precedents.append({
            "node_id": row["node_id"],
            **attrs,
            "same_error_signature": bool(signature) and attrs.get("error_signature") == signature,
            "proposals": proposals,
            "outcomes": outcomes,
        })
    return {"entity": entity, "incident": incident, "precedents": precedents}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m archi.ops.precedents",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--deployment", required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--entity")
    target.add_argument("--incident", help="record id of the incident being triaged")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--dsn")
    args = parser.parse_args(argv)
    try:
        found = find_precedents(okg_reader(args.deployment, dsn=args.dsn), entity=args.entity,
                                incident=args.incident, limit=args.limit)
    except RecordError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(found, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
