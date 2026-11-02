# Skills

These are reviewed instruction files served to agents at runtime (`SKILL.md`
format, per ADR 0001): retrieval planning, traversal recipes, evidence and
answer discipline, CMS domain skills (sites, Rucio, Condor monitoring,
incidents, migrations, policy arbitration), and two benchmark-mode contracts.
They were ported from the CMS deployment's skill set in `okg-deployments`
(`cms/skills/` at `main@f33a9c4`), de-site-ified: instance-specific hostnames,
paths, credentials, and deployment names were removed while the CMS domain
content — which is the product — was kept intact. An instance wires them in
through its `deployment.yaml`: `skills_dir` points at the installed copy of
this directory, `skill_triggers` maps question intents to skills (the shipped
default mapping is `skill-triggers.yaml` alongside this file). Content pinning
via `skill_bundle_hash` is **not** done by the bundle install, contrary to what
this file used to claim: the hash is computed by okg from the instance's
`skills_dir` when an operator runs `okg deployment release`, which additionally
requires a `release.channel` in the instance manifest. It is an instance
release-lifecycle step, not a distribution one. Agents read skills;
they have no write path to them — any runtime skill-write tool is rejected by
construction (ADR 0001, invariant 11).
