Issue: mitdbg/okg#1906. This is the consumer slice of a two-repository code migration.

Ownership inventory: current archi_v3 is e1f65bf95436fead32fcf0750c169e9fd7e0abd6 (frozen input PR642 merged). The active install-gaps PR629 and evaluation PR619 retain their files; this task adds a separate install package, wrappers and tests, rather than changing their source reader algorithms. The existing cern-team bundle already owns profiles, schemas, skills and frozen source examples. Current framework dev was inspected at 3e09c5392413328d7572c700be0e1f9030a87ce0. Framework installer entry work remains with PR1894/1886 until its coordinated handoff; source replay and shared infrastructure remain separately owned.

Archi implementation: python/archi/install/snapshots.py validates immutable input receipts and derives its reader caches; adapters.py wraps the three existing readers through public okg.deployment without changing run/completion semantics; distribution.py produces registered typed assets and an immutable package with a neutral prepared-configuration manifest using public okg.distributions. No command-line entry point ships yet: an operator calls read_snapshot, prepare_configuration and build_prepared_package, and the builder refuses a configuration prepared for a different target, name, module set or source selection. Consumer templates own deployment/source configuration and private posture. No private host renderer, installer, admission or release-state imports are allowed. Existing live defaults remain untouched.

Release identities are external build/release inputs, not a wheel containing its own impossible self hash. Producer validates explicit actual wheel digests, expected package metadata and source revision supplied in its release receipt, and records selected public contract compatibility. Original historical wheel and corrected snapshot-only wheel remain historical artifacts; the new wrapper-bearing consumer wheel is a new release, never silently treated as either old artifact.

Compatibility and transition: the current installer binds framework wheel, consumer wheel, configuration, lock and translation exactly. Existing launcher and environment pins remain untouched. Ordinary reapply or scoped update is not a release upgrade; a mismatched new release must refuse before modification. Supported transition is a new target, environment and separately owned database populated from explicitly selected authorized inputs, verification of its new generation/readers, then explicit operator cutover. Keep the old installation available for pinned reads/rollback. Same-database in-place migration is not provided by the existing contract and is not invented here. Missing historical input identity remains unproven.

The generic prepared-configuration prototype is only a candidate minimal seam: inert registered release-policy asset, complete mapped bytes, exact framework/consumer/root/name/selection/plan-lock binding, no callback. Entry integration and removal of the old framework branch require owner handoff and independent review. Consumer preparation can be implemented and tested independently now; no working installed replacement is claimed until actual generic entry and artifact proof execute.

Runtime proof will retain bulky inventories and raw logs as artifacts with hashes, not source-PR dumps. Concise test and design receipts belong here. Actual authentic snapshot identity, scope exclusions, first/read results, failed unchanged behavior and transition refusals stay distinct.

Additional inventoried consumer runtime: the host chat/agent_pipe_bridge.py is a standard-library-only external agent/message compatibility module with two test callers and no current consumer runtime references. Move its consumer-specific protocol adapter and acceptance tests to archi.compat.agent_pipe without a new chat architecture; preserve tool authorization and failure behavior. The live source-admin forbidden-surface guard is framework safety enforcement and remains unchanged in effect. Historical PACT/license references remain explicit history. This added move requires independent design review before implementation.

External release receipts are trusted authenticated build inputs: wheel hashes and METADATA are verified from actual bytes, while a source Git revision cannot be inferred from arbitrary wheel metadata. Record that provenance separately; never upgrade a supplied revision string into independently proven build identity. Refusal proof distinguishes launcher environment rejection from install release-identity rejection and verifies the original retained reader after each observed refusal.

Hosted verification must update the consumer CI framework pin and alignment
record together once the reviewed framework commit exists. Current CI resolves
an older commit through the authorized private fork; publication of the new
commit there requires its owner's coordination. Local checks do not substitute
for hosted execution, and the new tests must not be skipped at the old pin.

The supported-old compatibility fixture is a new installation of the retained
released wheel pair using its ordinary public catalogue fetch (full scope,
60-second network timeout). Capture actual fetched bytes and independently
calculate its inventory afterward. This does not reproduce historical inputs.
Do not rerun the old source to simulate a frozen update. Cross-release rejection
must preserve the old receipt/key/config and actual pinned reader; the new
release separately proves the current scoped update/search entry points.
