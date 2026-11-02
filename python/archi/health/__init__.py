"""Instance health: the automation that watches an archi instance (ADR 0002 section 7.5).

    signals.py   what it reads (doctor, okg ingest state, automation error rates, eval reports) -> Problems
    problems.py  a Problem and its stable key (<check>/<subject>)
    tracker.py   the IssueTracker interface and `reconcile` (one issue per key, no duplicates)
    local.py     the default tracker: Markdown files in the instance repository
    forge.py     GitLab and GitHub trackers (REST; run only from the live tool)
    config.py    the `ops.yaml` block that configures it
    proposer.py  the algorithmic proposer the ops runtime calls (`proposer: {kind: instance-health}`)
    steps.py     the `draft_issue` workflow step (local tracker, runs at L1)

The pack is `bundles/packs/instance-health/`; the forge live tool is `archi.tools.instance_health`.
"""
