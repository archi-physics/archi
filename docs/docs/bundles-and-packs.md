# Bundles and packs

## Packs

A **pack** is the smallest installable capability. There are three kinds:

- **context** packs bring sources: one family of documents, tickets,
  catalogs or code
- **tool** packs bring live tools: functions an agent can call over MCP
- **automation** packs bring one automation (see
  [the operations layer](operations-layer.md))

A pack is a directory with a `pack.yaml` and the files it names: schemas,
source templates, init questions, invariants, playbooks. It owns its
invariants and freshness limits, so removing the pack removes them too. A
pack that is not selected leaves no trace in the instance.

## Bundles

A **bundle** is a named, versioned selection of packs, plus a chat preset.
It lists default packs (installed unless you remove them) and optional packs
(installed when you ask). A bundle can extend others and inherits their
defaults. A bundle also sets the highest autonomy level for any automation it
selects. It never sets the level itself.

| Bundle | Purpose |
|---|---|
| `cern-team` | A generic CERN team knowledge instance: documentation, meetings, wiki, documents, tickets and code. No automations. |
| `cms-physics` | A CMS physics knowledge instance. Extends `cern-team`. |
| `cms-comp-ops` | A CMS computing operations instance: CMS catalogs, site status, CMSSW releases and the operational memory. Extends `cern-team`. Optional: `monit-live`, `dqm`, `hypernews`, `jira`, `ticket-triage`. |
| `cluster-manager` | A knowledge instance for people who run a batch cluster: documentation, Redmine tickets and operational memory, with optional ServiceNow, Slurm, HTCondor, node health and code. It depends on no CERN pack. Its Slurm pack ships live tools whose actions (drain or resume a node, requeue a job) run only after an approved escalation. The optional `ticket-triage` automation drafts a cited triage note for each new or updated ticket. See the [bundle README](https://github.com/archi-physics/archi/blob/archi_v3/bundles/cluster-manager/README.md). |

The
[library](library/index.md) has one generated page per bundle and per pack,
so it matches what ships.

## Choosing packs

```bash
archi init myteam --bundle cern-team --with git-code,jira --without indico
archi bundles list
archi bundles show cern-team
```

The result is the closure of what the packs require, plus the defaults of
the selected bundles, plus `--with`, minus `--without`. Several bundles can
go into one instance by repeating `--bundle`. The rendered profile and an
`archi.lock` (bundle and pack versions plus content digests) are committed
into the instance repository.

Two properties always hold:

- A bundle with nothing configured publishes. Each bundle has at least one
  default context pack that needs no credential, or a required question.
- A pack that is not selected does not appear in the rendered profile.

`archi add <pack>` adds a pack to a running instance and publishes again.
`archi diff` shows how the instance has drifted from its lock.

## More

- [Library](library/index.md): every bundle and pack in this release.
- [Pack and bundle format](reference/pack-format.md): every key in
  `pack.yaml` and `bundle.yaml`, and the composition rules.
- [Contributing a pack](contributing-a-pack.md).
