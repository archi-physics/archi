# okg asks from PACT r3-ops-record (drafts for Jason to file)

Status: drafts, not filed. Target repo: `mitdbg/akmon`. Found on a scratch
instance running okg 0.2.0 (release wheel) while writing ops records through
`okg admission submit`. After filing, open one `upstream-ask` issue on
`archi-physics/archi` per ask that links it (ADR 0002 §9).

Paths below are inside the okg 0.2.0 wheel (`okg/`).

---

## Ask 1. Admit an edge whose other end is already in the graph

**Title:** Generic admission: resolve edge endpoints outside the payload from the graph in the fact-batch validator

**Problem.** A `workflow` producer submits `{"nodes": [...], "edges": [...]}`
through `okg admission submit` with a producer policy whose output signature
declares `ops_incident tracked_in jira_issue`, and the catalog has that
narrowing. The edge's destination, `jira:OPS-11`, is in the published graph,
written by the jira source. The edge sets `dst_subtype: jira_issue`.

Admission refuses it, along with the whole payload:

```
producer_edge_endpoint_missing: edge record references an endpoint whose subtype
is not resolvable in the nodes declared in this payload: dst node 'jira:OPS-11'
```

Admission runs two checks, and they disagree:

- `substrate/runtime/admission_workflows.py::_validate_review_edge_narrowings`
  checks catalog narrowings. It looks up an endpoint that is missing from the
  payload in the graph, and it passes.
- `substrate/fact_batch.py::validate_chunk_authority` (line 1037) runs next,
  during `_apply_facts_in_bounded_chunks`. It turns each fact back into a dict
  and calls `evaluate_payload_signature`. That call resolves endpoints only
  from the nodes in the same chunk. `EdgeFact` has no `src_subtype` or
  `dst_subtype` field, so the subtype the payload declared is lost, and the
  edge is refused.

**Consequence.** A non-source producer can only link nodes it writes in the
same payload. Every automation needs edges like "this proposal cites that
ticket", "this incident affects that site" and "this outcome, checked six
hours later, belongs to that incident". archi works around it: such references
are attributes, and a record linked to an earlier one re-submits the earlier
record in the same payload (`archi.ops.record.OpsRecords.carry`).

**Ask.** In `validate_chunk_authority`, resolve an edge endpoint that is not
in the chunk the same way the narrowing check does: from the graph, or from
the declared subtype after confirming it against the graph.

**Acceptance.**
- A `fast_track` payload holding one node and one edge from that node to a
  node another source already published is admitted. The edge is in the next
  generation.
- An edge to a node id in neither the payload nor the graph is still refused
  with `producer_edge_endpoint_missing`.
- An edge whose declared `dst_subtype` contradicts the graph is refused.

**What archi deletes when this lands.** `OpsRecords.carry` and the
dangling-edge refusal. The ops-record bridge gains optional `tracked_in` /
`cites` / `affects` narrowings to tickets, chunks, pages and sites.

---

## Ask 2. An admission-routed write in the deployment SDK

**Title:** Deployment SDK: let a workflow step submit facts through admission

**Problem.** A deployment workflow may import only `okg.deployment`. The
SDK's writers, `emit_node_fact` and `emit_edge_fact`
(`deployment/_internal/mutations.py`, lines 118 and 252), insert into
`okg.node_facts` / `edge_facts` and apply to Chronos directly. They create no
admission envelope and run no signature or narrowing check. The admission path
(`substrate/runtime/admin.py::client_enqueue_admission_submit`, line 1196) is
in `okg.substrate`, which the workflow boundary lint flags. ADR 0002 §6.2
requires automations to write "through admission like any other producer",
so an archi automation (PACT r3-ops-runtime) has to shell out to
`okg admission submit`.

**Ask.** Add `okg.deployment.admit(payload, *, producer_id, intent, policy)`
(any name will do). It runs the same submit as `okg admission submit`, as a
DBOS step that is idempotent on replay, and returns the envelope.

**Acceptance.**
- A deployment workflow that calls it gets an admission envelope with the
  workflow's `workflow_ref`, and its facts appear in the next generation.
- A payload outside the declared signature is refused inside the workflow
  with the same verdict as the CLI.
- Replaying the workflow step does not create a second envelope.

**What archi deletes when this lands.** The subprocess call in
`archi.ops.record.submit`.

---

## Ask 3. Put list attributes in search text

**Title:** okg_searchable_text_fields: include multivalued attributes

**Problem.** `ops_incident` declares
`okg_searchable_text_fields: record_id,ticket_keys,title,...`, where
`ticket_keys` is `multivalued: true`. The `attrs.text` that okg stamps omits
the list. A search for `OPS-11` returns the jira issue but not the incident
that names it. okg filters also cannot match one element of a list, so a list
attribute cannot be found either way.

**Ask.** When a searchable text field holds a list of strings, stamp its
elements, joined by spaces or newlines, into `attrs.text`. Allow it as an
identifier field too.

**Acceptance.** With the declaration above, `search("OPS-11")` returns the
incident, and an identifier search ranks it first.

**What archi changes when this lands.** It puts `ticket_keys` back in the
incident's searchable and identifier fields.
