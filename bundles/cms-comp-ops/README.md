# cms-comp-ops bundle

A CMS computing operations knowledge instance (ADR 0002 section 5). It
extends [`cern-team`](../cern-team/), which supplies documentation, meetings
and the generic playbooks, and adds the CMS computing catalogs on top.

| Pack | Default | What it brings |
|---|---|---|
| `cms-catalogs` | yes | Sites, storage and compute endpoints, SITECONF, DBS datasets, WMStats workflows, GOCDB downtimes, CondDB global tags. Carries the `rucio_command`, `rucio_dataset_hosting`, `site_profile` and `incident_root_cause` playbooks. |
| `cmssst` | yes | Site status from the CMS Site Support Team |
| `cmssw-releases` | yes | CMSSW releases |
| `ops-record` | yes | The operational memory: incidents, proposals, decisions, actions and outcomes. |
| `monit-live` | optional | MONIT snapshots (SAM, HTCondor, Rucio) and the live MONIT query tools. Carries the `monitoring_condor_snapshot` playbook. |
| `dqm` | optional | DQM data-quality certifications and the runs they cover |
| `hypernews` | optional | HyperNews forum threads |
| `jira` | optional | JIRA issues and their participants |
| `git-code` | optional | One Git repository's tracked files and history. Selecting it also brings `code-text` (below). |
| `code-text` | with `git-code` | The text of every text file in that repository (code, Markdown, YAML, shell scripts, configuration), searchable and readable; files holding a literal secret are left out. Comes on whenever `git-code` is selected; opt out with `--without code-text`. |

The ticket-triage automation pack is not in this release, so the bundle does
not list it yet.

## Install

```bash
archi init myops --bundle cms-comp-ops
archi init myops --bundle cms-comp-ops --with monit-live,jira   # optional packs
archi init myops --bundle cms-comp-ops --with git-code --git-code-repo-url URL   # a repository, with its file text
```

What an install gets for repositories:

- Without `git-code` (the default): no repository content.
- With `git-code`: the repository's files and history, and, as its
  companion, `code-text`: the text of every text file, so a question about a
  deployment's YAML or a script's flags can be answered from the graph.
- With `git-code` and `--without code-text`: files and history only, as
  before this bundle named the companion.
- On an installed instance, `archi add git-code` brings `code-text` too.

The bundle fixes no install answers. `gitlab_url` (cms-catalogs) defaults to
`https://gitlab.cern.ch`, and `releases_map_url` (cmssw-releases) has its own
default.

## What each pack needs

Credentials are read from the environment by name. Never put a value in a
manifest or answer file.

| Pack | Credential or input |
|---|---|
| `cms-catalogs` | `CERN_GITLAB_TOKEN` for SITECONF. The CRIC, CRIC core, DBS, WMStats, GOCDB and CondDB caches are files under `archi_data_root`; fetching CRIC, DBS and WMStats needs a CERN grid certificate or SSO. |
| `cmssst` | none; the cmssst downloader fills its cache from public data |
| `cmssw-releases` | none |
| `ops-record` | none; records arrive through okg admission |
| `git-code`, `code-text` | none for a public repository; a private one is cloned with the operator's own git credentials. code-text's source checks every file on every run and leaves out any file holding a literal secret; the run log names the files it left out. It covers the files' text only: git-code's history keeps per-file diffs, which okg redacts by its own patterns. |
| `monit-live` | `MONIT_GRAFANA_TOKEN` (snapshots and live tools). Requires `cms-catalogs`. |
| `dqm` | none by name; the certification files are behind CERN SSO and the cache is placed by hand |
| `hypernews` | `HYPERNEWS_COOKIE_FILE`, a CERN SSO cookie jar |
| `jira` | `CERN_JIRA_TOKEN` (optional; public projects read anonymously) |

A bare install is not credential-free: `cms-catalogs` is a default and its
SITECONF source reads GitLab with `CERN_GITLAB_TOKEN`. Run `archi init
myops --bundle cms-comp-ops --without cms-catalogs` to start without it.

## Playbooks

| Playbook | Pack |
|---|---|
| `rucio_command`, `rucio_dataset_hosting`, `site_profile`, `incident_root_cause` | `cms-catalogs` |
| `monitoring_condor_snapshot` | `monit-live` |

Each pack links the playbook from `skills/` and carries its trigger, copied
from `skills/skill-triggers.yaml`.

## Validation instance

`gitlab.cern.ch/archi/cms-compops` is the standing validation target (ADR
0001 D10). Rebuilding it from this bundle is PACT
`r3-bundle-cms-comp-ops` requirement `instance`, and needs live access; the
pull request description lists what it needs.
