# Consumer-owned immutable installation

Archi owns the frozen snapshot format, cache readers, private configuration and
release provenance. The producer uses public `okg.distributions` package APIs;
the framework consumes a selected inert configuration manifest. This migration
is under review: the generic entry and final framework CI pin are prerequisites
for supported installation. Do not use a historical framework wheel with these
new adapters and claim that the new package was validated.

No command-line entry point ships yet. Call the three steps in order:

1. `archi.install.snapshots.read_snapshot` verifies an exact five-file
   `archi.frozen-snapshot/v1` manifest and its input digests.
2. `archi.install.configuration.prepare_configuration` composes the cache,
   consumer schemas, skills, invariants and all five explicit source policies.
   Each policy must declare every field the framework's policy audit reads,
   with redaction, `store_raw: false` and no live calls.
   `docs/cern-team-private-source-policy.example.yaml` is a reviewed example
   for a private instance; it is never a default.
3. `archi.install.distribution.build_prepared_package` packages those bytes.

Three frozen readers are selected: CMSSW `release_new`, documentation
`scope_complete`, Jira `reconcile`. Optional repository entries remain
unselected; no live authority is added. The Jira and documentation adapters
refuse to run when their cached record identifiers differ from the declared
allowlist, and every adapter rechecks cache digests before each run. The
generic installer audits the prepared policy before writing or publishing, so
a package build is not a successful policy audit or install.

`build_prepared_package` takes those bytes, the exact target, the framework
wheel digest and an externally authenticated consumer build receipt. It refuses
a configuration prepared for a different target, instance name or module set,
any source selection other than the three frozen readers, and cache bytes that
differ from their pins. The receipt binds `archi.install-release/v1`, package,
version, repository, source revision and wheel digest. The producer verifies
wheel bytes and package/version METADATA; it cannot infer an authenticated Git
revision from arbitrary wheel metadata. The receipt stays outside the wheel to
avoid a self-hash cycle.

Replacement uses a new target, environment and database. Keep the previous
launcher, environment, configuration, pseudonym key and database intact until
explicit cutover. Ordinary install/update is not a release upgrade and must
refuse changed release identity on an existing target. Scoped update/search are
same-release operations; failed updates or health checks remain failures even
when a published generation is still readable.

The external-agent protocol helper now lives in `archi.compat.agent_pipe` and
remains standard-library-only. A deployment still supplies its own Pipe class,
agent construction and endpoint wiring. Consumer tests compare the helper's
tool/posture semantics with the unchanged public `okg.chat` authorities. The
Open WebUI cutover benchmark report and its raw transcripts belong in this
repository at `pact/changes/archi-agent-pipe-bridge/reports/path-a-vs-path-b.json`
and `pact/changes/archi-agent-pipe-bridge/evidence/transcripts/`, where
`python/tests/test_pipe_bridge_benchmark_report.py` validates them. The
framework no longer carries that check.
