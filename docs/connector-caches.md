# Connector cache contracts

Most Archi connectors do not fetch anything. They read a **local cache
file** that some other process — a downloader script, an operator, a
scheduled job — is responsible for refreshing. This page is the contract
between those two halves: for each cache-backed connector, what file it
reads, what shape that file must have, and which keys the parser looks
at.

It exists because a live bring-up of the comp-ops instance found that
eight connectors had no refresh tooling anywhere: the runbook assumed
downloader scripts that were never written, and only `download_jira.py`
and `download_static_docs.py` exist in any repo. Whoever writes the
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
| `jira` | `data/jira/records.json` (+ `meta.json`) | `download_jira.py`. Needs the `jira` Python package installed in the operator venv — absent, the token alone does not help. The parsed record count is cross-checked against `meta.json`'s `record_count`; a mismatch withholds the scope claim. |
| `docsite`, `gitlab_docs` | `data/docsite/records.json`, `data/gitlab-docs/records.json` | `download_static_docs.py` |
| `cmssw_releases` | `data/cmssw-releases/releases.map` | the connector itself, from the public cms-bot map — no auth, no downloader |
| `cmsweb_docs`, `hypernews` | crawled live through an SSO cookie jar | `sso-login.py`, which is **interactive and needs a TOTP code** — inherently an operator step |
| `twiki_eos` | an EOS snapshot path | whatever populates that EOS area |
| `monit_*` | none — queried live | a MONIT/Grafana token |

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
