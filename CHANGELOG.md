# Changelog

All notable changes to `archi-kb` (import name `archi`). Numbers like #786
are pull requests on `archi-physics/archi`. The v2 line (`v2.x` tags, branch
`v2` after the cutover) is not covered here.

## 3.0.0

Archi 3.0 is a new product, not an upgrade of v2. It automates operations
for scientific computing on two layers built on OKG: a context layer that
holds the facility's own knowledge, kept current, linked, cited and measured,
and an operations layer whose automations propose actions that a person
approves, and record every outcome. Any MCP agent can read the context layer.
There is no in-place conversion from v2. See
[Moving from v2 to 3.0](docs/docs/migration-from-v2.md).

This entry covers milestones M2 to M7 of
[ADR 0002](docs/adr/0002-archi-3.0-release.md), from pull request #779 on.
(M0 and M1 landed earlier, in the archi-okg repository and the first
merge into this one.)

The version is `3.0.0`. The PyPI distribution is `archi-kb`; the import
name is `archi`.

### What an operator gets

**One command from an empty machine to a cited answer** (M3)

- `archi init`, `archi up`, `archi down` and `archi doctor`. `init` makes the
  instance as a git repository, generates every secret into `.env`, and
  publishes a first graph. `up` starts one compose project (podman or docker)
  with the graph database, the chat database, the MCP server and Open WebUI.
  `doctor` runs the okg checks and Archi's own, and prints a fix line for
  each failure. (#786)
- The graph now survives `archi down` followed by `archi up`. It used to live
  only inside the container. podman-compose is supported. Instances made
  before this fix must be re-initialised. (#834)
- `cern-team` publishes on a bare machine with no credential. Its defaults are
  `docsite` (two public documentation sites) and `indico` (one public
  event). (#835)
- A nightly build runs the whole path on an empty runner, with a down/up
  durability check. It prints the elapsed time to the run summary; it does not
  fail past 15 minutes. (#786, #837)
- Every footgun found on the way is listed in the footgun ledger as absorbed
  by Archi or filed with AKMON. (#809)

**Bundles and packs: choose what goes in the graph** (M2)

- A pack is one installable capability: a context pack (sources), a tool pack
  (live tools) or an automation pack. A bundle is a named, versioned
  selection of packs. Several bundles can share one instance. (#779)
- `archi render` composes bundles into one okg profile. (#780)
- Packs and bundles are found through the `archi.packs` entry point, so a pack
  can live in another pip package. No `OKG_PROFILES_DIR` step. (#781)
- `archi.lock`, `archi add <pack>` and `archi diff`. `archi add` edits the
  registry, claims ownership of the source, commits, and publishes again.
  `archi diff` reports drift from the lock. (#782)
- The connectors, schemas, invariants and playbooks are split into
  context and tool packs. `cern-team` is a bundle of them. (#789)
- `cms-physics`, with the `analysis-notes` and `cms-analyses` packs. It links
  analysis ids across CDS records, TWiki pages and analysis notes. (#838)
- `cms-comp-ops`: sites, storage, datasets, workflows, site status, CMSSW
  releases and past incidents, with the comp-ops playbooks moved into their
  packs. (#849)
- `cluster-manager`, for any cluster operator at any site: documentation,
  tickets and operational memory by default; ServiceNow, Slurm, HTCondor,
  node health and code as options. The Slurm pack's live tools act only after
  an approved escalation. (#850)
- The `tickets` pack reads Redmine, any JIRA Server or Data Center, or both,
  with nothing CERN-specific. (#855)

**A context layer that says how current and complete it is** (M2)

- Each pack declares how stale each source may be, how many records must link
  across sources, and the checks it owns. `archi doctor` reports them.
  Automations can refuse to run live when the context fails its gate. (#796)
- The `ops-record` pack stores incidents, proposals, decisions, actions and
  outcomes as graph records, admitted like any other producer. Ticket
  resolutions link to them, and a precedent playbook lets an agent ask what
  was done last time. (#785, #787, #790)

**An operations layer** (M4)

- Automation packs, autonomy levels L0 to L2, and validation of an automation
  before it installs. An L3 setting is refused. (#811)
- The loop signal, context, proposal, decision, action, outcome, with
  `archi ops status`, `pending`, `approve`, `reject`, `promote`, `halt` and
  `resume`. Shadow scoring reports how often a shadow automation agreed with
  the person who handled the case. (#810, #833)
- `instance-health`, the first automation. It watches the instance itself and
  keeps one issue per problem, with no duplicates. It starts at L1 and may go to L2. (#839)
- `ticket-triage`, the reference automation, an optional pack of `cluster-manager`
  and `cms-comp-ops`. A new or updated Redmine, JIRA or ServiceNow ticket gets a
  cited triage draft (category, likely component, related tickets, precedents,
  documentation) that is never sent (L1). At L2 it proposes an assignment or a
  status change, which a live tool makes only after a person approves, and only
  on the server the ticket was read from. SubMIT switches to it from its v2 Redmine
  bot at the 3.0 cutover, with no trial run first (board d717). (#746,
  `r3-auto-ticket-triage`)

**Evals: a checked claim that a bundle answers, and how well it retrieves** (M5, M2)

- `archi eval --bundle <name>` runs a bundle's smoke set against the
  instance's MCP endpoint and writes `report.md` and `report.json`. A question
  passes when the answer cites a record of the right source or type and holds
  the required strings; there is no LLM judge. Smoke sets ship for
  `cern-team`, `cms-physics` and `cluster-manager`. (#851, #857)
- `archi eval --context` scores retrieval on its own, with no model: recall@k
  and MRR for the default search against a lexical baseline. The question set
  is named at run time, so private question sets stay out of the repository.
  The automation context gate reads the newest clean run of its bundle. (#858)
- `archi eval --context --agents` adds a raw-document baseline: the same
  model answers each question in two arms, one with bash over the raw files
  the graph was built from (in a bubblewrap sandbox with no network), the other with the
  instance's MCP tools. Both are scored on the gold records they read and,
  with `--judge`, on their answers. `archi eval --raw-corpus` builds the file
  corpus. Every model call goes to a spend ledger, and `--cap` is required.
  (#698)
- `instance-health` reads the eval reports and raises an issue when a bundle's
  latest run could not run, falls below its threshold, or drops. (#853)

**Bring your own agent** (M6)

- `archi connect` writes the MCP configuration for Claude Code, codex,
  openclaw, Open WebUI and generic MCP over HTTP. It keeps the bearer token
  out of git unless you ask. External MCP servers and external packs are
  supported. (#840)

**Documentation** (M7)

- The documentation site is rewritten for 3.0. v2 pages sit under "v2
  (legacy)". The bundle, pack and command reference pages are generated from
  the code and checked by a test. (#841, #831, #832)
- The README, the package description and the docs home describe Archi as
  operations automation for scientific computing. (#860)
- The upgrade guide for v2 operators, the cutover runbook, release notes and
  this changelog. (This release.)

**Maintenance**

- Regenerated `uv.lock` (#784). Recorded that archi-okg is archived (#783).
  PACT evidence and approvals: #788, #791, #794, #795, #797, #836, #852, #856.

### Known limits

Some of what the plan calls for is not in 3.0.0. Plan around these.

Open requirements (each is a PACT requirement still unproven at this tag):

- **The context-eval acceptance run.** `archi eval --context` ships and is
  tested on fixtures, but the run on the comp-ops instance with the operator
  question set has not happened. (`r3-context-quality`, requirement
  `context-eval`.)
- **Instances rebuilt from the bundles.** The lab `cms-physics` instance,
  the live comp-ops instance (#726) and SubMIT (#740) have not been rebuilt
  from their bundles yet. (`r3-bundle-cms-physics`, `r3-bundle-cms-comp-ops`,
  `r3-bundle-cluster-manager`.)
- **PyPI pre-release proof.** `archi-kb` has not been published to PyPI before
  this tag. The clean-venv install check is part of the release. (`r3-release-home`,
  `r3-release`.)

Also not in 3.0.0:

- **`ticket-triage`: no shadow trial; SubMIT resolution emails pause.** It
  is back in 3.0 (board d701) and replaces SubMIT's v2 Redmine bot at the
  cutover, with no shadow trial first (board d717). The v2 service that
  posted the bot's answers also sent the resolution email, and it stops at
  the cutover, so SubMIT users get no resolution emails from the cutover
  until 3.1 (board d719). Mail intake stays on v2 until 3.1 (board d704).
  (`r3-auto-ticket-triage`, `r3-release`.)

- **Smoke sets on a local model.** The smoke-set requirement is met on GPT-6
  Luna, which the maintainers accepted for this release (board d366): the sets
  pass their threshold (7 of 10) there. No local model has reached it. The
  nightly run reports a small CPU model's score and does not gate. A CI gate on
  bundle changes waits for the #619 eval framework. `cms-comp-ops` has no
  smoke set in 3.0. (`r3-bundle-evals`.)
- **Adding an automation pack to an existing instance.** `archi add` does not
  turn on okg's runtime, so an automation added that way does not run. Select
  automations at `archi init` (`--with instance-health`). No bundle includes
  `instance-health` by default.

- The v2 Piazza bot, the grader, the A/B tools, the playbook store, document
  upload in the chat UI, and Helm deployment. See
  [what has no equivalent yet](docs/docs/migration-from-v2.md#what-has-no-equivalent-yet).
- Autonomy level L3. Langfuse (deferred).
- `okg` is installed from an AKMON GitHub release wheel, not from PyPI
  (AKMON ask 1). `uv tool install archi-kb` alone does not bring it.

### AKMON asks

These are asks filed against `mitdbg/akmon` for things Archi works around.
Archi was built and tested against okg 0.2.0. None is a release blocker, and
none has landed in an okg release.

- Packaging and setup: wheels on PyPI (#3), role passwords set during migrate
  (#4), `chat-instance up` also starts the MCP server (#5), publish a fresh
  install without a manual commit (#6), plus traps found on the way (#7 to #13).
  Detail: [golden-path-akmon-asks.md](docs/golden-path-akmon-asks.md).
- Profiles: discovery by entry point (#1), composition and optional sources
  (#2).
- Operations layer: expiry on admission envelopes (#14), approve with edits
  (#15), a mounted authenticated proposal API (#16), per-actor cost
  attribution (#17), a first-class kill switch (#18), a retraction-size guard
  (#19), and bugs found while building the loop (#20 to #27). Until these
  land, automations run co-located as deployment workflows, edits are
  handled as reject-and-resubmit, and `archi ops halt` wraps pause and cancel.
  Detail: [ops-runtime-akmon-asks.md](docs/ops-runtime-akmon-asks.md).
- One more ask, O15 (read when and by whom a PACT change was approved), is a
  draft and not filed.

### Upgrading

There is no upgrade from v2, only a move. Read
[Moving from v2 to 3.0](docs/docs/migration-from-v2.md). Owners of v2
deployments: the cutover waits on you. See the
[cutover runbook](docs/docs/operating/cutover.md).
