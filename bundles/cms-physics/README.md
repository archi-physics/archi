# cms-physics bundle

A CMS physics knowledge instance (ADR 0002 section 5). It extends
[`cern-team`](../cern-team/) and adds the physics packs by default:

| Pack | What it brings |
|---|---|
| `twiki` | TWiki topics from an EOS dump. Install with `--twiki-physics-filter true` to keep physics topics only. |
| `cds` | CMS physics analysis summaries and notes from CERN Document Server |
| `cmssw-releases` | CMSSW releases |
| `analysis-notes` | Analysis notes from a mirror of `gitlab.cern.ch/tdr/notes` |
| `cms-analyses` | PAG and POG hubs, physics entities, and the rules linking analysis ids across the above |

## Install

```bash
archi render --bundle cms-physics --instance .
okg install --profile cms-physics --deployment-name <slug> \
  --postgres-dsn "$OKG_DSN" --archi-data-root "$PWD/archi-data/data" \
  --twiki-eos-root <TWiki dump> --analysis-notes-mirror-root <note mirror>
```

The bundle sets two answers you can override with their flags:
`--twiki-physics-filter` is `true`, and `--analysis-notes-min-notes` is 2500
(calibrated on a real mirror build; a smaller mirror fails
preflight).

**Wire the cross-link enricher.** Packs cannot yet install an enricher, and
without it the publish succeeds but the CDS-to-note, citation-to-CDS,
TWiki-to-group and PAG-to-POG edges do not exist. Paste this into the
deployment's manifest, commit, and publish:

```yaml
enrichers:
  - class: archi.enrichment.physics_cross_links.PhysicsCrossLinker
```

If it is missing, the warn invariant `cms_analyses_enricher_edges` fires. See
the [cms-analyses README](../packs/cms-analyses/README.md).

## What needs credentials

The TWiki dump lives on CERN EOS and the note mirror is a clone of a CERN
GitLab group, so this bundle installs green only where both exist. The
[analysis-notes README](../packs/analysis-notes/README.md) has the mirror
commands. The bundle has no smoke eval yet, so it has no install proof of its
own; that proof needs both inputs.
