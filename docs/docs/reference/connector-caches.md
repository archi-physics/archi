# Connector cache contracts

Most Archi connectors do not fetch anything. They read a **local cache
file** that some other process — a downloader script, an operator, a
scheduled job — is responsible for refreshing. This page is the contract
between those two halves: for each cache-backed connector, what file it
reads, what shape that file must have, and which keys the parser looks
at.

It exists because a live bring-up of the comp-ops instance found that
eight connectors had no refresh tooling anywhere: the runbook assumed
downloader scripts that were never written, and only the JIRA and static
docs downloaders exist (now `archi.downloaders.jira` and
`archi.downloaders.static_docs`, moved from okg-deployments). Whoever writes the
missing ones — us or an operations team who already fetch this data —
needs to know exactly what to produce. That is this page.

## How caches are read

- Paths in a registry entry's `params` are **relative to the deployment
  directory** (or to `ARCHI_DATA_ROOT`), e.g. `data/cric/sites.json`.
- A cache file that is **configured but missing** makes the connector
  fail loudly. It never silently ingests nothing — an empty scope claim
  would retract every record the connector previously published.
- Records that are malformed (not an object, or missing the identity
  key) are **skipped and counted**, and any skip makes the run withhold
  its completed-scope claim. So a downloader that emits half-broken
  records degrades coverage; it does not destroy data.
- Field names are often accepted in two spellings — the upstream API's
  own name and a normalized one (for example DBS `dataset` or
  `dataset_name`). Where that is true it is noted; either works.

## The eight without a downloader

### `cric` — CRIC topology
Files, all JSON objects keyed by name:

| file | keyed by | value keys read |
|---|---|---|
| `data/cric/sites.json` | site name | `facility`, `tier_level`, `country`, `timezone`, `state`, `is_monitored`, `sitedb_title` |
| `data/cric/storage_units.json` | unit name | `endpoint`, `flavour`, `type`, `site` |
| `data/cric/compute_units.json` | unit name | `corepower`, `pledged_cms`, `potential_max`, `promised`, `state` |
| `data/cric/facilities.json` | facility name | `name`, `country`, `tier_level` |
| `data/cric/responsibilities.json` | — | an object carrying a `result` list; each entry names a person and their role for a site |

A missing `tier_level` defaults to 0, which presents a site as Tier-0 —
downloaders should emit it explicitly.

`responsibilities.json` must be the full payload including its `result`
key. A payload without `result` is treated as corrupt and fails the run
rather than being read as "nobody is responsible for anything".

### `cric_core` — CRIC core services
| file | keyed by | value keys read |
|---|---|---|
| `data/cric-core/services.json` | service name | `endpoint`, `type`, `state`, `rcsite` |
| `data/cric-core/rcsites.json` | site name | `country`, `infrastructure`, `tier_level`, `vos` |
| `data/cric-core/federations.json` | federation name | `accounting_name`, `country`, `infrastructure`, `pledges`, `rcsites`, `tier_level`, `vos` |

Service `endpoint` values may carry a scheme (`https://host:port/path`)
or be a bare host; both parse.

### `conddb_global_tags` — condition global tags
`data/conddb-global-tags/records.json`, a JSON **list** of objects:
`name` (or `tag_name`), `description`, `release`, `scenario`,
`snapshot_time` (or `created_at`). The `release` value is matched
against the CMSSW release set when `cmssw_records_path` is configured,
so it should be a release label like `CMSSW_14_0_2`.

### `dbs_datasets` — dataset catalog
`data/dbs-datasets/records.json`, a JSON **list**. Keys, upstream name or
normalized: `dataset` / `dataset_name`, `primary_ds_name` /
`primary_dataset`, `processed_ds_name` / `processed_dataset`,
`data_tier_name` / `data_tier`, `dataset_access_type`,
`physics_group_name` / `physics_group`, `creation_date`, `nevents` /
`total_events`, `nfiles` / `total_files`, `dataset_size` /
`total_size_bytes`. Dataset paths are case-sensitive.

### `dqm` — data-quality certifications
`data/dqm/records.json`, a JSON **list**: `cert_name`, `filename`,
`run_range` (a two-element range the connector expands against runs),
`datasets`, `num_lumi_sections`.

### `gocdb_downtimes` — grid downtimes
`data/gocdb-downtimes/records.json`, a JSON **list**: `downtime_id`
(must be numeric — non-numeric entries are skipped and counted),
`primary_key`, `hostname`, `hosted_by`, `service_type`, `severity`,
`classification`, `description`, `start_date`, `end_date`. The connector
matches `hostname` against CRIC sites and services, so this cache is
most useful refreshed alongside the two CRIC ones.

### `indico` — meetings and their attachments
`data/indico/records.json`, a JSON **list** of event objects: `id`,
`title`, `type`, `category`, `categoryId`, `startDate`, `endDate`,
`description`, `chairs`, `folders`, plus two derived fields the
downloader must produce — `_contributions_text` (flattened agenda text)
and `_pdf_texts` (extracted text per attached PDF). Only PDFs whose text
was extracted become documents in the graph; other attachments appear as
URLs only.

### `wmstats_workflows` — production workflows
`data/wmstats-workflows/records.json`, a JSON **list** of request docs,
upstream or normalized names: `RequestName` / `request_name` /
`workflow_name`, `RequestType` / `request_type`, `RequestStatus` /
`status`, `RequestPriority` / `priority`, `Campaign` / `campaign`,
`CMSSWVersion` / `cmssw_version`, `PrepID` / `prep_id`, `InputDataset` /
`input_dataset`, `OutputDatasets` / `output_datasets`, `RequestDate` /
`created_at`, `updated_at`.

## The ones that do have tooling

| connector | cache | refreshed by |
|---|---|---|
| `jira` | `data/jira/records.json` (+ `meta.json`) | `python -m archi.downloaders.jira`. Needs the `jira` Python package in the operator venv (`pip install jira`); without it the downloader stops with a message naming the package. The token (`CERN_JIRA_TOKEN`) is optional: unset, the download is anonymous and sees only public projects such as `ADAPTYST`; a project that is not public stops the run with a plain message. The parsed record count is cross-checked against `meta.json`'s `record_count`; a mismatch withholds the scope claim. |
| `docsite`, `gitlab_docs` | `data/docsite/records.json`, `data/gitlab-docs/records.json` | `python -m archi.downloaders.static_docs` (targets in the packaged `source-download-manifest.yaml`) |
| `cmssw_releases` | `data/cmssw-releases/releases.map` | the connector itself, from the public cms-bot map — no auth, no downloader |
| `cmssst_site_status` | `data/cmssst-site-status/records.json`: a JSON **list**, one object per CMS site, keys `site`, `time` (snapshot epoch), `alert`, `msg`, `ggus` (`[ticket id, opened epoch]` pairs) and `metrics` (metric name → time range → list of status-letter strings; the reader reads each metric's `today`). | `python -m archi.downloaders.cmssst`, from the public CMS SST site status data (`siteStatus/data/summary.js` and `<SITE>.json`) — no auth. All or nothing: one failed site fetch writes no cache. |
| `cmsweb_docs`, `hypernews` | crawled live through an SSO cookie jar | `python -m archi.downloaders.sso_login`, which is **interactive and needs a TOTP code** — inherently an operator step |
| `twiki_eos` | an EOS snapshot path | whatever populates that EOS area |
| `analysis_notes` | a directory of `<PAG>-YY-NNN/` and `AN-YY-NNN/` LaTeX trees | a clone of `gitlab.cern.ch/tdr/notes`, which needs a CERN account and a `CERN_GITLAB_TOKEN`; the commands are in `bundles/packs/analysis-notes/README.md` |
| `monit_*` | none — queried live | a MONIT/Grafana token |

Upgrade note for `cmssw_releases`: release records now carry `production_architecture`. An instance installed earlier keeps its own copy of `schemas/operations.yaml`; copy the new file over it (or reinstall) so the field is declared and searchable. After copying, commit the deployment config, then run `okg catalog ownership claim --deployment .` and `okg catalog load --apply` from the deployment directory before the next ingest: without the load, ingest refuses with `catalog_applied_version_absent`. Then re-ingest `cmssw_releases` and publish; about 2,365 release records update.

Upgrade note for `twiki_eos`: TWiki pages whose topic name holds a CMS analysis id (HIG29991, B2G28992Review) now carry `analysis_id` (HIG-29-991) and the page text gains "HIG-29-991 CMS-HIG-29-991", so a search for the hyphenated id finds them. An earlier install keeps its own copy of `schemas/sources.yaml`; copy the new file over it (or reinstall), commit, run `okg catalog ownership claim --deployment .` and `okg catalog load --apply`, then re-ingest `twiki_eos` and publish. On the CMS dump, 4,282 of 43,431 page records change (new attribute and text); no chunk changes, so no chunk is re-embedded.

## Cluster-manager caches

The `cluster-manager` bundle's packs read these caches under `ARCHI_DATA_ROOT`
(for example `servicenow/records.json`). Each downloader writes `records.json` and
`meta.json` atomically; `meta.json`'s `record_count` is cross-checked against what the
reader parses, and a mismatch withholds the scope claim. A cache that is missing, empty
or the wrong JSON shape fails the run loudly, as above. Nothing is fetched at ingest.

| pack | cache | refreshed by |
|---|---|---|
| `servicenow` | `servicenow/records.json`, a JSON **list** of Table API rows, each with a `table` (`incident`, `sc_request` or `kb_knowledge`) and a `number`. Reads `short_description`, `description` (`text` for an article), `state`, `priority`, `category`, `assignment_group`, `assigned_to`, `opened_by` / `caller_id` / `requested_for`, `author`, `cmdb_ci`, `opened_at`, `sys_updated_on`, `resolved_at`, `close_notes`. A reference may be a string or a `{"display_value": ...}` object. | `python -m archi.downloaders.servicenow --instance-url <url> --auth oauth\|basic`. OAuth reads `SERVICENOW_CLIENT_ID` and `SERVICENOW_CLIENT_SECRET`; basic auth reads `SERVICENOW_USERNAME` and `SERVICENOW_PASSWORD`. Read-only GETs; no value is written or printed. |
| `tickets` | `tickets/records.json`, a JSON **list**: the flat shape (`system`, `key`, `subject`, `description`, `project`, `tracker`, `status`, `priority`, `author`, `assignee`, `created`, `updated`, `closed`, `notes`, optional `url`) or a Redmine REST issue (`id`, `subject`, `project`, `author`, `assigned_to`, `created_on`, `journals`). The node id is `tracker_ticket:<system>:<key>`, so a JIRA issue moved to another project (a new key) becomes a new node and the old one is retracted at the next complete download. | `python -m archi.downloaders.tickets --redmine-url <url>` and/or `--jira-url <url> --jira-project <KEY>` (or `--jql`); both URLs write one cache. `REDMINE_API_KEY` is optional: unset, only public projects are visible. A JIRA download needs `JIRA_API_TOKEN` (a personal access token, sent as a bearer header to the REST v2 search API) and an `https://` URL; it pages in key order, writes people as display names, replaces `[~login]` mentions, and skips comments restricted to a group or role unless `--include-restricted-comments` is given. Email addresses in text are removed when the cache is read. `python -m archi.downloaders.redmine --base-url <url>` still writes a Redmine-only cache. |
| `slurm` | `slurm/records.json`, a JSON **object**: `cluster`, `collected_at`, and lists `partitions`, `nodes`, `reservations`, `jobs` (field list in `archi/sources/slurm.py`). No partitions and no nodes is a failed export. | `python -m archi.downloaders.slurm --cluster <name>` on a host with the Slurm client: it runs `sinfo`, `scontrol show reservation` and `sacct`, all read-only, and needs no credential from archi. |
| `htcondor` | `htcondor/records.json`, a JSON **object**: `pool`, `collected_at`, `totals`, `machines`, `schedds`. No machines is a failed export. | `python -m archi.downloaders.htcondor --pool <name>`: two read-only `condor_status` queries. |
| `node-health` | `node-health/records.json`, a JSON **list** of per-node reports: `node`, `cluster`, `observed_at`, optional `status`, and `checks` (`name`, `status` of `ok`, `warn` or `fail`, `detail`). | the site's own health exporter; nothing in archi runs a check. |

## Reference caches are different

`docsite` and `gitlab_docs` can also read *reference* caches to emit
cross-links (`document_chunk references site` / `cmssw_release` /
`jira_issue` / `infrastructure_service`). Those point at other
connectors' caches and are optional — but each one configured must
exist, or the run fails rather than quietly dropping that edge kind.

For releases specifically there are two accepted shapes: a
`records.json` list of `{label: ...}` objects (`releases_path`), or the
cms-bot map that `cmssw_releases` already fetches
(`releases_map_path: data/cmssw-releases/releases.map`). Use the map on
a live instance — nothing produces a releases `records.json`.

## Shipping caches without credentials

`python -m archi.snapshot` packs existing caches into checksummed, redacted
per-group archives with a lock file, so an install can fetch data without
CERN access. See [cache snapshots](https://github.com/archi-physics/archi/blob/archi_v3/docs/cache-snapshot.md).
