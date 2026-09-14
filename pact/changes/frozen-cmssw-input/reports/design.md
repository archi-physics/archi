# Frozen CMSSW consumer binding

Base is `archi_v3` at `0aada49aedc4d0bbd25bdb3fd06c6b29baa03a35`, matching the recovered consumer wheel source. PRs target `archi_v3`. Existing PR629 edits docsite and cache documentation; this change does not take over those paths. The existing W2 PACT names catalogs broadly; this PACT owns only the CMSSW frozen-input correction requested through OKG1795.

Add optional `map_cache_digest="sha256:<64 lowercase hex>"` to CMSSWReleaseSource. A pin requires map_cache_path (or its existing cache_path alias) and fetch=False. The pin selects an explicit frozen path; no missing-file or digest-error branch may reach urllib. Unpinned behavior stays compatible.

A dedicated frozen read obtains bytes once, verifies their raw SHA256, decodes/parses that buffer, and calculates the existing path+NUL+bytes+NUL content_hash from that buffer. Preflight and run call it independently, so preflight does not authorize a later changed file. Source revision records the raw map_cache_digest separately from content_hash. Probe configuration records the frozen binding. Existing parsing, skipped-row handling, limits and completed-scope rules remain in use.

Ship a separate frozen registry example rather than changing the live CERN profile. The installer owner must bind the chosen cache and digest through the separately owned packaged-profile contract; this change does not edit OKG installer/runtime entrypoints. Update the alignment page to describe the new consumer contract without changing its existing CI pin or claiming installer support prematurely.

Verification covers network traps, missing/mismatched input, malformed/conflicting configuration, exact emitted IDs/edges, same-buffer revision under a deliberate post-read file mutation, probe identity, and existing live/cache regression tests. Build the normal consumer wheel and verify the contract from the installed artifact. Test fixture rows prove code behavior, not the historical real-deployment baseline.

The authorized handoff currently lacks the historical CMSSW releases.map/cache. The retained packaged entry also fixes fetch:true and accepts only the four doc/Jira snapshot files. Those facts remain explicit dependencies; no live capture is labeled frozen and no synthetic fixture replaces real baseline acceptance. A real publish additionally awaits the separately owned PostgreSQL resource-identity guard resolution.
