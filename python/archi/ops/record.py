"""Write the operational memory through okg admission (PACT r3-ops-record).

An automation records each step of its loop (ADR 0002 section 7.1) as an ops
record: an incident, the proposal it made, the decision on it, the action a
mediator ran, and the outcome checked later. This module turns those records
into an okg admission payload and submits it as a declared `workflow`
producer, never by direct SQL:

    records = OpsRecords()
    records.add("incident", record_id="inc-1", title="...", entity="T2_US_MIT", ...)
    records.add("proposal", record_id="prop-1", incident_id="inc-1", evidence=[...], ...)
    submit(records, deployment="myteam")

`submit` runs `okg admission submit` with a producer policy whose output
signature names exactly the subtypes and edges in the payload, so admission
checks every node against the catalog and every edge against the pack's
narrowings. The policy asks for `fast_track`: the records describe what the
automation did, so there is nothing for a reviewer to accept. The records are
in the graph once okg publishes the next generation. Use one producer id for
every automation on an instance (the default, `workflow:archi-ops`; the
`automation` attribute names which one): a record carried into a later
payload is then re-asserted by the producer that wrote it.

okg admits an edge only when both of its ends are in the payload (okg 0.2.0;
the draft upstream ask is pact/changes/r3-ops-record/okg-asks.md, ask 1). So a record that links to one
admitted earlier, such as an outcome checked hours after its incident, travels
with that record (`OpsRecords.carry`), and references to nodes other packs own
(tickets, cited evidence, the entity that broke) are attributes.

Ticket resolutions link to incidents with `resolution_outcomes`: an incident
names the tickets tracking it (`ticket_keys`), and once a ticket connector
reports one of them resolved, the person's resolution becomes the incident's
`ops_outcome` (check_method `ticket_resolution`, `ticket_key` set).

The field and edge vocabulary is read from the pack's schema slice
(`bundles/packs/ops-record/schemas/`), so this module and the catalog cannot
disagree.

Command line (for a workflow step or an operator):

    python -m archi.ops.record submit --deployment D --records run.jsonl --producer-id P
    python -m archi.ops.record link-tickets --deployment D [--ticket-subtype jira_issue]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml

from archi.paths import packs_dir

PACK = "ops-record"
PRODUCER_KIND = "workflow"
INTENT = "ops_record"
#: One producer for every automation on an instance; `automation` says which.
PRODUCER_ID = "workflow:archi-ops"

#: Record kind -> okg subtype.
KINDS = {
    "incident": "ops_incident",
    "proposal": "ops_proposal",
    "decision": "ops_decision",
    "action": "ops_action",
    "outcome": "ops_outcome",
}

#: Ticket subtype -> node id of the ticket with key {key}, per connector.
#: archi.sources.jira emits `jira:<KEY>`.
TICKET_NODE_IDS = {"jira_issue": "jira:{key}"}

#: Ticket resolutions that mean the problem was fixed. Any other non-empty
#: resolution (Won't Fix, Duplicate, Cannot Reproduce, ...) is recorded with
#: result `unknown`: the ticket closed, but nobody says it worked.
FIXED_RESOLUTIONS = frozenset({"fixed", "done", "resolved", "completed"})


class RecordError(ValueError):
    """A record that the ops-record schema would refuse."""


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


@dataclass(frozen=True)
class Vocabulary:
    """Attributes per subtype and the edge triples the slice declares."""

    attrs: Mapping[str, frozenset[str]]
    required: Mapping[str, frozenset[str]]
    multivalued: Mapping[str, frozenset[str]]
    edges: frozenset[tuple[str, str, str]]


@lru_cache(maxsize=None)
def vocabulary(schemas: Optional[Path] = None) -> Vocabulary:
    schemas = schemas or packs_dir() / PACK / "schemas"
    slice_ = yaml.safe_load((schemas / "ops_record.yaml").read_text(encoding="utf-8"))
    attrs, required, multivalued = {}, {}, {}
    for name, body in slice_["classes"].items():
        subtype = _snake(name)
        fields = body.get("attributes") or {}
        attrs[subtype] = frozenset(fields)
        required[subtype] = frozenset(k for k, v in fields.items() if (v or {}).get("required"))
        multivalued[subtype] = frozenset(k for k, v in fields.items() if (v or {}).get("multivalued"))
    bridge = yaml.safe_load((schemas / "bridges" / "ops_record.yaml").read_text(encoding="utf-8"))
    edges = set()
    for body in bridge["classes"]["EdgeNarrowings"]["attributes"].values():
        ann = body["annotations"]
        edges.add((_snake(ann["src_subtypes"]), ann["edge_archetype"], _snake(ann["dst_subtypes"])))
    return Vocabulary(attrs, required, multivalued, frozenset(edges))


def node_id(kind_or_subtype: str, record_id: str) -> str:
    """The graph node id of an ops record: `<subtype>:<record_id>`."""
    subtype = KINDS.get(kind_or_subtype, kind_or_subtype)
    return f"{subtype}:{record_id}"


def ticket_node_id(ticket_subtype: str, key: str) -> str:
    try:
        return TICKET_NODE_IDS[ticket_subtype].format(key=key)
    except KeyError:
        raise RecordError(
            f"no node-id rule for ticket subtype {ticket_subtype!r}; "
            f"known: {sorted(TICKET_NODE_IDS)}"
        ) from None


@dataclass
class OpsRecords:
    """A batch of ops records and the edges between them: one admission."""

    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[dict[str, Any]] = field(default_factory=list)
    vocab: Vocabulary = field(default_factory=vocabulary)

    def add(self, kind: str, *, evidence: Iterable[Any] = (), **attrs: Any) -> str:
        """Add one record; return its node id.

        `attrs` are the subtype's schema attributes. Links between records are
        given by record id (`incident_id`, `proposal_id`, `decision_id`,
        `action_id`) and become edges. `evidence` (proposal only) lists the
        okg node ids the proposal cites, bare or as `{"node_id": ...}`; they
        go into `evidence_ids`, and a cited ops incident (a precedent) also
        gets a `cites` edge.

        A record linked to one admitted earlier must travel with it: pass the
        earlier record to `carry` (okg admits an edge only when both ends are
        in the payload).
        """
        if kind not in KINDS:
            raise RecordError(f"unknown record kind {kind!r}; expected one of {sorted(KINDS)}")
        subtype = KINDS[kind]
        attrs = {k: v for k, v in attrs.items() if v is not None}

        evidence_ids = [item["node_id"] if isinstance(item, Mapping) else str(item) for item in evidence]
        if evidence_ids:
            if kind != "proposal":
                raise RecordError("only a proposal cites evidence")
            attrs["evidence_ids"] = list(dict.fromkeys(list(attrs.get("evidence_ids", ())) + evidence_ids))

        unknown = set(attrs) - self.vocab.attrs[subtype]
        if unknown:
            raise RecordError(f"{subtype}: unknown attributes {sorted(unknown)}")
        missing = sorted(k for k in self.vocab.required[subtype] if attrs.get(k) in (None, "", []))
        if missing:
            raise RecordError(f"{subtype}: missing required {missing}")
        for key in self.vocab.multivalued[subtype]:
            if key in attrs and isinstance(attrs[key], str):
                attrs[key] = [attrs[key]]
        attrs = {k: (v if isinstance(v, list) else str(v)) for k, v in attrs.items()}

        nid = node_id(subtype, attrs["record_id"])
        self.nodes[nid] = {"node_id": nid, "subtype": subtype, "attrs": attrs, "op": "I"}

        links = {
            "proposal": [("incident_id", "addresses", "ops_incident")],
            "decision": [("proposal_id", "decides", "ops_proposal")],
            "action": [
                ("proposal_id", "executes", "ops_proposal"),
                ("decision_id", "authorized_by", "ops_decision"),
            ],
            "outcome": [
                ("incident_id", "outcome_of", "ops_incident"),
                ("action_id", "evaluates", "ops_action"),
            ],
        }
        for key, edge_type, dst_subtype in links.get(kind, ()):
            if attrs.get(key):
                self._edge(nid, subtype, edge_type, node_id(dst_subtype, attrs[key]), dst_subtype)
        for cited in attrs.get("evidence_ids", ()):
            if cited.startswith("ops_incident:"):
                self._edge(nid, subtype, "cites", cited, "ops_incident")
        return nid

    def carry(self, row: Mapping[str, Any]) -> str:
        """Add an already-admitted ops record (an `okg filter --jsonl` row) as it is,
        so a new record's edge to it can be admitted. A record already in the
        batch wins."""
        nid, subtype = row["node_id"], row["subtype"]
        if subtype not in KINDS.values():
            raise RecordError(f"can only carry ops records, not {subtype!r}")
        attrs = {k: v for k, v in (row.get("attrs") or {}).items() if k in self.vocab.attrs[subtype]}
        attrs.pop("text", None)  # okg stamps it from the searchable fields
        self.nodes.setdefault(nid, {"node_id": nid, "subtype": subtype, "attrs": attrs, "op": "I"})
        return nid

    def _edge(self, src, src_subtype, edge_type, dst, dst_subtype):
        if (src_subtype, edge_type, dst_subtype) not in self.vocab.edges:
            raise RecordError(
                f"the ops-record slice declares no {src_subtype} {edge_type} {dst_subtype} edge"
            )
        self.edges.append(
            {
                "src": src,
                "dst": dst,
                "edge_type": edge_type,
                "src_subtype": src_subtype,
                "dst_subtype": dst_subtype,
                "provenance": "explicit",
                "confidence": 1.0,
            }
        )

    def dangling(self) -> list[str]:
        """Edge endpoints not in the batch; `submit` refuses a batch with any."""
        return sorted({e["dst"] for e in self.edges} - set(self.nodes))

    def payload(self) -> dict[str, Any]:
        return {"nodes": list(self.nodes.values()), "edges": list(self.edges)}

    def output_signature(self) -> dict[str, Any]:
        """Exactly the subtypes and edge triples this batch writes."""
        subtypes = sorted({n["subtype"] for n in self.nodes.values()})
        triples = sorted({(e["src_subtype"], e["edge_type"], e["dst_subtype"]) for e in self.edges})
        return {
            "nodes": [{"subtype": s} for s in subtypes],
            "edges": [{"src_subtype": s, "edge_type": t, "dst_subtype": d} for s, t, d in triples],
        }

    def authority(self, producer_id: str) -> dict[str, Any]:
        """The producer policy `okg admission submit` admits this batch under."""
        return {
            "producer_policy": {
                "producer_id": producer_id,
                "producer_kind": PRODUCER_KIND,
                "trust_label": "fast-track-trusted",
                "admission_mode": "fast_track",
                "output_signature": self.output_signature(),
            }
        }


def load_records(path: Path) -> OpsRecords:
    """Read a JSONL file of records, one `{"kind": ..., <attrs>}` per line."""
    records = OpsRecords()
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        kind = row.pop("kind", None)
        try:
            records.add(kind, **row)
        except (RecordError, TypeError) as exc:
            raise RecordError(f"{path}:{lineno}: {exc}") from None
    return records


def _okg() -> str:
    """The okg installed with this Python, else the first on PATH."""
    beside = Path(sys.executable).with_name("okg")
    return str(beside) if beside.is_file() else (shutil.which("okg") or "okg")


def submit(
    records: OpsRecords,
    *,
    deployment: str,
    producer_id: str = PRODUCER_ID,
    dsn: Optional[str] = None,
    okg: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    max_wait_seconds: Optional[float] = None,
) -> dict[str, Any]:
    """Submit `records` through okg admission; return okg's JSON reply.

    `idempotency_key` makes a replay of the same call (a recovered workflow step)
    write once: okg answers the second submit with `inserted: false`.

    Raises RecordError when the batch is empty or okg refuses it.
    """
    if not records.nodes:
        raise RecordError("no records to submit")
    if records.dangling():
        raise RecordError(
            f"edges to records not in this batch: {records.dangling()}; "
            "carry() the earlier records with it"
        )
    with tempfile.TemporaryDirectory() as tmp:
        payload_file = Path(tmp) / "payload.json"
        payload_file.write_text(json.dumps(records.payload()), encoding="utf-8")
        cmd = [
            okg or _okg(), "admission", "submit", "--json",
            "--deployment", deployment,
            "--producer-id", producer_id,
            "--producer-kind", PRODUCER_KIND,
            "--intent", INTENT,
            "--payload-file", str(payload_file),
            "--authority-json", json.dumps(records.authority(producer_id)),
        ]
        if dsn:
            cmd += ["--dsn", dsn]
        if idempotency_key:
            cmd += ["--idempotency-key", idempotency_key]
        if max_wait_seconds is not None:
            cmd += ["--max-wait-seconds", str(max_wait_seconds)]
        result = subprocess.run(cmd, capture_output=True, text=True)
    reply = _json_or_none(result.stdout)
    if result.returncode != 0 or reply is None:
        raise RecordError(
            f"okg admission submit failed (exit {result.returncode}): "
            f"{(result.stderr or result.stdout)[-2000:]}"
        )
    return reply


def _json_or_none(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Ticket resolutions
# ---------------------------------------------------------------------------


def resolution_outcomes(
    incidents: Iterable[Mapping[str, Any]],
    tickets: Iterable[Mapping[str, Any]],
    *,
    ticket_subtype: str = "jira_issue",
    outcomes: Iterable[Mapping[str, Any]] = (),
) -> OpsRecords:
    """Outcomes for incidents whose tracking ticket a person has resolved.

    `incidents`, `tickets` and `outcomes` are graph rows (`node_id`, `attrs`)
    as `okg filter --jsonl` prints them. For each incident that names a
    ticket in `ticket_keys`, and each of those tickets with a resolution,
    this adds an `ops_outcome` (record id `<incident>.ticket.<key>`) with the
    ticket key and the person's resolution, linked to the incident, unless
    that outcome is already in `outcomes`. When the resolution is a fix the
    incident is written again with status `resolved`.
    """
    by_node = {t["node_id"]: t for t in tickets}
    done = {o.get("attrs", {}).get("record_id") for o in outcomes}
    records = OpsRecords()
    for incident in incidents:
        attrs = dict(incident.get("attrs") or {})
        for key in attrs.get("ticket_keys") or ():
            ticket = by_node.get(ticket_node_id(ticket_subtype, key))
            if ticket is None:
                continue
            tattrs = ticket.get("attrs") or {}
            resolution = (tattrs.get("resolution") or "").strip()
            if not resolution:
                continue
            record_id = f"{attrs['record_id']}.ticket.{key}"
            if record_id in done:
                continue
            fixed = resolution.lower() in FIXED_RESOLUTIONS
            records.add(
                "outcome",
                record_id=record_id,
                incident_id=attrs["record_id"],
                result="resolved" if fixed else "unknown",
                resolution=f"{key} resolved as {resolution}: {tattrs.get('summary') or ''}".strip(),
                check_method="ticket_resolution",
                ticket_key=key,
                checked_at=tattrs.get("updated"),
                actor_kind="person",
                actor=tattrs.get("assignee") or tattrs.get("reporter") or "unknown",
            )
            # The outcome's edge needs its incident in the same payload: the
            # incident again, resolved when the ticket says it was fixed.
            if fixed and attrs.get("status") != "resolved":
                attrs.update(status="resolved", resolved_at=tattrs.get("updated"))
                known = records.vocab.attrs["ops_incident"] - {"text"}
                records.add("incident", **{k: v for k, v in attrs.items() if k in known})
            else:
                records.carry({**incident, "subtype": "ops_incident"})
    return records


def read_nodes(deployment: str, subtype: str, *, dsn: Optional[str] = None,
               okg: Optional[str] = None) -> list[dict[str, Any]]:
    """Every published node of `subtype`, through `okg filter` (generation-pinned)."""
    cmd = [okg or _okg(), "filter", "--deployment", deployment, "--type", subtype, "--all", "--jsonl"]
    if dsn:
        cmd += ["--dsn", dsn]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RecordError(f"okg filter --type {subtype} failed: {(result.stderr or result.stdout)[-2000:]}")
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def link_tickets(deployment: str, *, producer_id: str = PRODUCER_ID, ticket_subtype: str = "jira_issue",
                 dsn: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Read incidents, tickets and outcomes; admit the missing resolution outcomes."""
    records = resolution_outcomes(
        read_nodes(deployment, "ops_incident", dsn=dsn),
        read_nodes(deployment, ticket_subtype, dsn=dsn),
        ticket_subtype=ticket_subtype,
        outcomes=read_nodes(deployment, "ops_outcome", dsn=dsn),
    )
    if not records.nodes:
        return None
    return submit(records, deployment=deployment, producer_id=producer_id, dsn=dsn)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m archi.ops.record", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("submit", help="admit a JSONL file of ops records")
    p.add_argument("--records", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true", help="print the payload and policy; submit nothing")
    q = sub.add_parser("link-tickets", help="admit outcomes for incidents whose ticket was resolved")
    q.add_argument("--ticket-subtype", default="jira_issue", choices=sorted(TICKET_NODE_IDS))
    for s in (p, q):
        s.add_argument("--deployment", required=True)
        s.add_argument("--producer-id", default=PRODUCER_ID)
        s.add_argument("--dsn")
    args = parser.parse_args(argv)
    try:
        if args.command == "submit":
            records = load_records(args.records)
            if args.dry_run:
                print(json.dumps({"payload": records.payload(),
                                  "authority": records.authority(args.producer_id)}, indent=2))
                return 0
            reply = submit(records, deployment=args.deployment, producer_id=args.producer_id, dsn=args.dsn)
        else:
            reply = link_tickets(args.deployment, producer_id=args.producer_id,
                                 ticket_subtype=args.ticket_subtype, dsn=args.dsn)
            if reply is None:
                print(json.dumps({"status": "nothing_to_link"}))
                return 0
    except RecordError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(reply, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
