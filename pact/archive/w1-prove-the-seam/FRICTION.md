# W1 seam proof — friction log

**Result: the seam works.** A pip-installed `archi` wheel (`archi-3.0.0a0-py3-none-any.whl`,
one source + one schema, zero dependencies) served a real source to an out-of-tree OKG
deployment: registry says `module: archi.sources.cmssw`, the source fetched the public
cms-bot `releases.map` over the network, ingest ran clean, a generation published, and both
MCP `search` and `query` returned the facts generation-pinned with provenance
`source: archi.cmssw-releases, trust: trusted`. `okg deployment lint` reports only
info-level findings (producer write envelope missing — same as OKG's own scaffold fixture).

**Environment:** a dev host · okg `dev` @ `21c5b8c3e` (editable install, uv venv,
python 3.12) · postgres = `okg-pg17:local` (timescaledb-ha pg17 + pg_textsearch) rootless
podman · deployment at `<scratch>/w1-scratch` ·
wheel at `w1-seam/dist/`.

**Working bring-up sequence** (the order matters — see friction 3):

```bash
uv venv --python 3.12 <okg-venv>
VIRTUAL_ENV=... uv pip install -e <okg-clone>          # + pip install <archi wheel>
podman build --from docker.io/timescale/timescaledb-ha:pg17-all@sha256:7fb5b82... \
  --network=host -t okg-pg17:local <okg>/ops/pg        # unqualified FROM fails rootless (friction 0)
podman run ... okg-pg17:local postgres -c shared_preload_libraries=timescaledb,pg_cron,pg_textsearch ...
export OKG_DSN=postgresql://postgres:okg@127.0.0.1:<port>/okg_w1_scratch
export OKG_DEPLOYMENTS_DIR=<scratch>
okg deployment new w1-scratch --deployments-root $OKG_DEPLOYMENTS_DIR
# add nomos.rollout deferred block (friction 1); register sources; copy schema in
okg migrate --deployment w1-scratch
psql: ALTER ROLE okg_mcp/app_rw/app_ro ... PASSWORD (friction 5)
okg catalog ownership claim --deployment w1-scratch    # BEFORE load, after any manifest edit
okg catalog load --deployment w1-scratch --apply
okg ingest --deployment w1-scratch
okg mcp-serve --deployment w1-scratch                  # stdio; drive with any MCP client
```

## Friction found (each → upstream ask or W2 design input)

0. **Rootless podman + unqualified base image.** `podman build` on AlmaLinux fails on the
   Dockerfile's own pinned `FROM` with "short-name resolution enforced but cannot prompt
   without a TTY" — must pass `--from docker.io/...` explicitly, exactly as
   okg-deployments' bring-up script already does. Doc-worthy, not a bug.

1. **The scaffold fails its own first ingest on Nomos.** `okg deployment new` emits no
   `nomos:` block; first `okg ingest` fails with *"source privacy policy for
   'bootstrap-fixture' failed Nomos audit"*. Lint only flags it as a *warning*
   (`deployment.nomos.deployment_posture_missing`) while ingest treats it as fatal. Fix
   that worked: `nomos.rollout: {status: deferred, owner: {...}, review_by: ...}` (copied
   from `deployments/consistency-fixture`). → Ask: scaffold should emit a deferred rollout
   block, or the ingest error should carry that fix hint.

2. **Path-vs-name deployment resolution is inconsistent.** `okg deployment new/lint`
   accept paths; `okg ingest --deployment <path>` resolved runtime config by *name* against
   the okg checkout's `deployments/` and failed until `OKG_DEPLOYMENTS_DIR` was exported.

3. **Manifest edits wedge the catalog-ownership claim, and the repair verb is buried.**
   Any `deployment.yaml`/`source_registry.yaml` edit after `catalog load --apply` makes
   *both* publish and the next `catalog load` refuse with `catalog_ownership_mismatch`;
   meanwhile the accepted-but-unpublished envelope blocks re-runs with
   `prior_emission_unpublished`. The fix — `okg catalog ownership claim` — appears in no
   fix_hint on this path. Correct loop: **edit → claim → load --apply → ingest**.
   The guard itself is good behavior; the discoverability is the gap.

4. **`--json` output is not pure JSON.** `okg ingest --json` prints a bare warning line
   ("deployment declares no sync_root; ...") to stdout ahead of the JSON document, breaking
   strict parsers. Warnings belong on stderr.

5. **Least-privilege MCP roles exist but can't log in on the migrate path.** `okg migrate`
   creates `okg_mcp`/`app_rw`/`app_ro` without login passwords; `okg mcp-serve` derives
   role DSNs using the substrate dev passwords in `local_database_roles.py`
   (`okg_mcp`/`okg_rw`/`okg_ro`) and the pool fails with a generic *"MCP pool acquisition
   failed after 5s"* (auth failure only visible in server-side warnings). Manual
   `ALTER ROLE ... LOGIN PASSWORD` per that file fixed it. The academic-pi initdb
   provisions these; the plain scaffold+migrate path does not.

6. **W2 design input, not a bug:** a network catalog behind a fetch-managed cache must use
   a `mutable_api`-style probe. The `content_hash` probe only hashes the cache file, so it
   cannot see upstream change once a cache exists (fine for W1's single run).

Items 1–5 fold into §7 ask 5's reproduction (and asks of their own where noted); item 6
is a W2 porting rule.
