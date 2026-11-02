# The context layer

The context layer is the knowledge graph plus the checks that say whether
you can trust it. Automations act on what it says, so "solid" has to be
measured and checked by machine. It is never just asserted.

## What "solid" means

| Property | What it means | How it is checked | Status in this branch |
|---|---|---|---|
| Current | Each source is recent enough. | Each pack declares `max_staleness` for each source. OKG records when each source last finished a run. `archi doctor` compares the two. | Working |
| Complete | The graph holds what each pack should emit. | Each pack owns coverage invariants. A failing `error` invariant blocks the publish. | Working |
| Linked | Identifiers (sites, datasets, runs, releases, tickets) connect records across sources. | Each pack can declare link coverage targets. | Working |
| Cited | Every answer and every proposal names the records it used. | OKG reads are pinned to one published generation. A proposal must list evidence ids. | Working |
| Measured | Operator questions with known right evidence are scored. | Context evals (`archi eval --context`). | Working; each instance keeps its own question set |
| Clean | No credentials or personal data in the graph. | Redaction at ingest, connector tests. | Working in the connectors |

A **generation** is one published snapshot of the graph. An **invariant** is
a stored query that must return no rows. A **publish** makes a new
generation visible to readers.

## Freshness

A pack that has sources lists a freshness limit for each one:

```yaml
freshness:
  indico.indico_live: {max_staleness: 24h}
```

The unit is `s`, `m`, `h`, `d` or `w`. The value is a judgment: how old the
source may get before an answer from it is probably wrong. `archi doctor`
reads each source's last completed run from OKG and reports it late when the
age passes the limit, or when OKG lists no run at all. See
[Context quality](reference/pack-format.md#context-quality) for the JSON it
prints.

## Link coverage

A pack that carries identifiers other packs refer to declares how well they
must link. For example, "at least 90% of storage endpoints sit under a
site". Archi turns each target into an invariant named
`<pack>_link_<name>`. A miss is a warning by default. Set `severity: error`
where a miss should block the publish. The library pages list each pack's
targets, for example [`cms-catalogs`](library/packs/cms-catalogs.md).

## The context gate

An automation says which context it needs: sources, a maximum staleness, a
minimum context-eval score (with `max_age_days`, default 14,
`context_eval_bundle` when the instance has more than one bundle, and the
optional pins `context_eval_k` and `context_eval_min_questions`) and the
invariants that must hold. Before each
run the runtime checks that gate against the current generation. If any check
fails, the run drops to **shadow** mode: it proposes and records, and never
acts. The proposal names the failing check, so the operator sees why. Failing
checks are named `freshness:<source>`, `context_eval`, `invariant:<name>`
and `published`. A missing score or result counts as a failure.

An automation that watches the instance itself asks only for `published`. It
has to keep running while a source is stale.

## Context evals

A context eval is a set of operator questions, each with the evidence that
should answer it (its gold node ids). `archi eval --context` scores retrieval
(was the right evidence found) separately from answer quality: it calls the
instance's MCP search tool directly, with no model, and reports recall@k of
the gold ids for the graph search and for a lexical baseline (the server's
lexical index over chunks and attributes, without roll-up to records). The
graph search is lexical too, with concept expansion and record roll-up on, so
the difference measures those, not graph links. Each
run is kept under `<data root>/eval/context/<bundle>/<UTC stamp>/`. The graph
search's recall@k is the score: the context gate reads the newest run of its
bundle for `min_context_eval_score`, and fails the check, naming the reason,
when there is none, or it cannot be read, did not execute cleanly, overrode
the graph arm, is older than `max_age_days` (default 14) or dated in the
future, or misses the gate's optional `context_eval_k` and
`context_eval_min_questions` pins. The `instance-health` automation does not
read context evals. Question sets are private to the instance and
never committed here; name one at run time with `--questions` or
`ARCHI_CONTEXT_EVAL_QUESTIONS`. See [Operating](operating/evals.md#context-evals-archi-eval-context).

## ops-record: the operational memory

The `ops-record` pack stores the record of operations in the graph, next to
the docs and tickets it cites:

- **incident:** what broke, where, when
- **proposal:** what an automation suggested, and on which evidence
- **decision:** who or what approved or rejected it, and when
- **action:** what the mediator ran, and what it returned
- **outcome:** whether it worked, checked later

Automations write these records through OKG admission, the same path any
producer uses. Past resolutions then become context ("this transfer error
last time was fixed by..."). The record also serves as the test set for the
next automation. See the [`ops-record` page](library/packs/ops-record.md).
