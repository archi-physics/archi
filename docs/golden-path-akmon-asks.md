# AKMON asks from the golden path

Status: **filed 2026-10-25** on `mitdbg/akmon`, each tracked by an `upstream-ask`
issue here: ask 1 [mitdbg/akmon#3](https://github.com/mitdbg/akmon/issues/3), #798; ask 4 [mitdbg/akmon#4](https://github.com/mitdbg/akmon/issues/4), #799; ask 5 [mitdbg/akmon#5](https://github.com/mitdbg/akmon/issues/5), #800; ask 6 [mitdbg/akmon#6](https://github.com/mitdbg/akmon/issues/6), #801; ask A [mitdbg/akmon#7](https://github.com/mitdbg/akmon/issues/7), #802; ask B [mitdbg/akmon#8](https://github.com/mitdbg/akmon/issues/8), #803; ask C [mitdbg/akmon#9](https://github.com/mitdbg/akmon/issues/9), #804; ask D [mitdbg/akmon#10](https://github.com/mitdbg/akmon/issues/10), #805; ask F [mitdbg/akmon#11](https://github.com/mitdbg/akmon/issues/11), #806; ask G [mitdbg/akmon#12](https://github.com/mitdbg/akmon/issues/12), #807; ask H [mitdbg/akmon#13](https://github.com/mitdbg/akmon/issues/13), #808. Ask E was withdrawn (no change needed).

Asks 1, 4, 5 and 6 are the ones ADR 0002 §9 names. A–H are traps found or
confirmed while building `archi init` / `up` / `doctor` on okg v0.2.0.
r3-bundle-format drafted asks 2 (profile entry points) and 3 (composition)
separately.

Each ask names the archi code that goes away when it lands (ADR 0002 D7).

---

## Ask 1. okg wheels on PyPI

**Problem.** okg is installed from a GitHub release URL
(`https://github.com/mitdbg/akmon/releases/download/v0.2.0/okg-0.2.0-py3-none-any.whl`).
Nothing resolves `okg>=0.2` from an index, so a distribution cannot declare
okg as a dependency, and `uv tool install archi-kb` cannot bring okg with it.
The older requirement, that okg be `pip install -e` from a clone so it could
stamp its git revision, is gone: archi's golden path installs the v0.2.0
release wheel non-editable and publishes (PACT r3-golden-path evidence).

**Ask.** Publish okg (same name, or `akmon` with an `okg` module) to PyPI with
trusted publishing, one release per tag, and keep the wheel installable
without a checkout.

**Acceptance.** `pip install okg==0.2.*` from PyPI, in a fresh venv with no
clone, runs `okg install` through to a published generation.

**archi deletes.** The wheel URL in `.github/workflows/*.yml` and in the MCP
image (`archi/stack/compose.py`, `OKG_WHEEL`); archi-kb can depend on okg.

## Ask 4. Role passwords during migrate, not after

**Problem.** `okg migrate` creates `okg_mcp`, `app_rw`, `app_ro` and
`catalog_admin` as `NOLOGIN`. Until `okg db role-passwords --apply` runs, the
catalog load can fail on authentication and every MCP read fails with
`mcp_role_cannot_login`. The step is separate, easy to miss, and refused
outright on a non-loopback server. Its secrets file is keyed by loopback port
under `~/.config/okg/role-secrets/`, so two deployments that reuse a port
overwrite each other's passwords.

**Ask.** Let `okg install` / `okg migrate` set the role passwords as part of
the migration (an opt-in flag is fine), and record them in a file the
deployment names (for example under the deployment's `.okg/`), not one keyed
by port. For a password-authenticated server okg does not own, accept the
role DSNs as input instead of refusing.

**Acceptance.** A fresh `okg install --profile <p> --postgres-dsn ...`
followed directly by `okg mcp-serve` serves reads, with no
`role-passwords` step; two deployments on the same port keep separate
passwords.

**archi deletes.** The `okg db role-passwords` call and the
`OKG_ROLE_SECRETS_FILE` override in `archi init`, and the copy of the okg_mcp
password into `.env`.

## Ask 5. `okg chat-instance up` starts the MCP server too

**Problem.** The chat site and its graph tools are two processes. `okg
chat-instance up` starts Open WebUI; `okg mcp-serve` must run separately,
on `chat.mcp.port`, bound where the container can reach it but nothing else
can. `chat-instance status` then reports the tools endpoint dead, truthfully.
`okg chat setup` starts mcp-serve as a detached child, which does not survive
a reboot.

**Ask.** One command that brings up the site and its MCP server together and
supervises both, for example a `--with-mcp` on `chat-instance up` or a compose
file okg renders. Alternatively (and enough for archi): let `okg chat setup`
adopt an instance and an MCP endpoint it did not start (see ask A).

**Acceptance.** After one command, `okg chat-instance status` reports the site
and the tools endpoint both up, and they restart together.

**archi deletes.** Nothing while archi runs compose; with ask A, the admin
sign-in in `archi/stack/webui.py`.

## Ask 6. Publish a fresh install without a manual commit

**Problem.** A deployment directory inside a git repository must have its
configuration committed before okg publishes. `okg install` writes that
configuration and then publishes in the same run, so the install's own last
step fails with `repository provenance inputs are not committed`, listing the
files it just wrote. Outside a git repository, `release.kind: standalone`
works, but an instance that is versioned in git cannot use it.

**Ask.** Either let `okg install` commit the files it scaffolds (opt-in, with
a stated message), or treat files the current run wrote as the provenance
input and record them as such, so the first publish succeeds.

**Acceptance.** `git init deployments && okg install --profile <p>
--deployments-root deployments ...` ends with a published generation and exit
code 0.

**archi deletes.** The `--no-publish` / commit / `okg provision` sequence in
`archi init` becomes one `okg install`.

---

## Ask A. `chat-instance up` refuses every published image on v0.2.0

**Problem.** `okg.substrate.chat.instance.PINNED_IMAGE_REF` is
`your-registry.invalid/okg-openwebui-compat@sha256:2561...`, a placeholder
registry, and `TESTED_IMAGES` holds only that entry. On the public v0.2.0
wheel, `okg chat-instance up` and `okg chat setup` therefore refuse every
image that exists (`chat_image_not_tested`), including upstream
`ghcr.io/open-webui/open-webui:v0.11.4`. The module's own comment says that
every okg result is plain text now, which the upstream image reads;
archi's golden path runs that image under compose, and `okg chat sync` (which
checks `/api/version` = 0.11.4) accepts it.

**Ask.** Publish the compat image to a real registry and pin that, or admit
upstream v0.11.4 in `TESTED_IMAGES` now that results are text. Separately,
let `okg chat setup` work against an instance it did not launch (a URL plus
an admin credential source), so a compose-run or Kubernetes-run site can use
setup's sign-up, provider and proof steps.

**Acceptance.** `okg chat-instance up --deployment <d>` with no `--image`
starts a container on a stock v0.2.x install; `okg chat setup --instance-url
http://127.0.0.1:8080` completes against an externally started site.

**archi deletes.** `archi/stack/webui.py` (admin sign-in, ollama URL).

## Ask B. Make the preset the site's default model

**Problem.** `chat sync` sets the site default from `chat.models.default`,
which is also the preset's base model. A new chat therefore opens on the raw
model, with no graph tools and no system prompt, and answers fluently from
what it already knows.

**Ask.** Default new chats to the preset `chat sync` created (or add a
manifest key for the site default that is separate from the preset's base).

**Acceptance.** On a freshly synced site, a new chat's selected model is the
deployment's preset.

**archi deletes.** The "pick the preset" line `archi up` prints.

## Ask C. A source that reads a predecessor's checkout fails its first ingest

**Problem.** With the code-symbols source (`code_structure`) enabled, the first
`okg ingest` after install fails for that source, because it runs before the
repository source has cloned the checkout it reads. Running the same ingest a
second time passes.

**Ask.** Order a source after its `predecessor_source` has materialised its
checkout in the same run (or retry it once after the predecessor finishes).

**Acceptance.** Install with `github_repo` + `code_structure`, one `okg
ingest`, exit 0, symbols present.

**archi deletes.** Nothing yet; archi does not enable code symbols by default.

## Ask D. Tell "failed" from "staged, not published", and recover without a teardown

**Problem.** A reinstall can exit 1 with `install failed at stage ingest` and
a cut-off tail while its data was staged correctly. Conversely, a run that
fails partway leaves `staged source work was not published; refusing to start
new source writes`, and neither a re-run nor `okg up` clears it; the only
recovery is dropping the database.

**Ask.** A distinct exit code (and a status field) for "staged, publish
pending"; and an explicit `okg run --once --apply`-style resume or a
`--discard-staged` to clear staged work.

**Acceptance.** After an injected publish failure, one documented command
either publishes or discards the staged work, and the next ingest runs.

**archi deletes.** The "start over" advice in `archi doctor`'s `published`
remediation.

## Ask E. Prove that removing a source removes its records

**Withdrawn (2026-10-25): no change needed.** `remove-source --apply` was run on a
scratch instance and did retract the source's records (footgun FG-39).

**Problem.** The quickstart found that a source removed from the registry
leaves its records in the graph until the database is recreated. v0.2.0 lists
`okg deployment remove-source --apply` as "retract its records, publish, and
remove its registry entry"; archi has not run it.

**Ask.** Confirm that `remove-source --apply` retracts every record the
source wrote (or fix it), and document it as the way to drop a source.

**Acceptance.** Add a source, publish, `remove-source --apply`, and `okg
filter` finds none of its records in the new generation.

**archi deletes.** Nothing; archi gains an `archi remove <pack>` on top.

## Ask F. `okg mcp-serve` needs git to serve reads

**Problem.** In a container without `git`, `okg mcp-serve --deployment <d>`
exits with `repository provenance is unavailable` (`rev-parse
--show-toplevel`), even though it only reads. On a mounted repository owned by
another user, git also needs `safe.directory`.

**Ask.** Do not resolve repository provenance in the read path, or degrade to
"provenance unknown" for a read-only server.

**Acceptance.** `okg mcp-serve` in `python:3.12-slim` with only okg installed
serves `tools/list`.

**archi deletes.** git and the `safe.directory` line in the MCP image
(`archi/stack/compose.py`, `DOCKERFILE`).

## Ask G. The chat health banner reads "unavailable" on a healthy instance

**Problem.** After `okg chat sync`, a freshly published cern-team instance
shows "OKG graph health: unavailable (error). Affected: not itemized." The
banner projects the dogfood status `trust_level`, which a deployment without
the dogfood runtime never reports as ready, so a healthy graph reads as broken.

**Ask.** Compute the banner from a signal every deployment has (latest
published generation, source freshness), or omit it when the deployment
declares no runtime.

**Acceptance.** A fresh install with a published generation shows no error
banner after `chat sync`.

**archi deletes.** Nothing; archi does not touch banners.

## Ask H. Let a source declare the hosts it fetches

**Problem.** Indico talk images are fetched by the MCP `inspect` tool only
when `deployment.yaml` carries an `agent_surface` block allowing the host. A
source that needs fetches has no way to bring that allowance with it, so
enabling it is a hand edit of the manifest, and a pack cannot carry it.

**Ask.** Let a source registry entry (or a profile part) declare its fetch
hosts, merged into the deployment's allow-list at install.

**Acceptance.** Enabling `indico_live` with `attachment_images: true` through
the profile alone makes `inspect(kind="image")` work, with no manifest edit.

**archi deletes.** The `agent_surface` instructions in the Indico pack.
