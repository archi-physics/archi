# Consumer-owned immutable installation

Archi owns the frozen snapshot format, cache readers, private configuration and
release provenance. The producer uses public `okg.distributions` package APIs;
the framework consumes a selected inert configuration manifest. This migration
is under review: the generic entry and final framework CI pin are prerequisites
for supported installation. Do not use a historical framework wheel with these
new adapters and claim that the new package was validated.

`archi.install.snapshots.read_snapshot` verifies an exact five-file manifest and
its input digests. `prepare_configuration` composes the cache, consumer schemas,
skills, invariant and all five explicit source policies. Three frozen readers
are selected: CMSSW `release_new`, documentation `scope_complete`, Jira
`reconcile`. Optional repository entries remain unselected; no live authority is
added. `docs/cern-team-private-source-policy.example.yaml` is a reviewed
example of the explicit policy for a private instance; it is never a default. The generic installer must audit the prepared policy before writing or
publishing. A package build is not a successful policy audit or install.

`build_prepared_package` takes those bytes, exact target, framework wheel digest
and an externally authenticated consumer build receipt. The receipt binds
`archi.install-release/v1`, package, version, repository, source revision and
wheel digest. The producer verifies wheel bytes and package/version METADATA;
it cannot infer an authenticated Git revision from arbitrary wheel metadata.
The receipt stays outside the wheel to avoid a self-hash cycle.

Replacement uses a new target, environment and database. Keep the previous
launcher, environment, configuration, pseudonym key and database intact until
explicit cutover. Ordinary install/update is not a release upgrade and must
refuse changed release identity on an existing target. Scoped update/search are
same-release operations; failed updates or health checks remain failures even
when a published generation is still readable.

The external-agent protocol helper now lives in `archi.compat.agent_pipe` and
remains standard-library-only. A deployment still supplies its own Pipe class,
agent construction and endpoint wiring. Consumer tests compare the helper's
tool/posture semantics with the unchanged public `okg.chat` authorities.
