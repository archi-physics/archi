# Evaluation guide

Archi's evaluator measures whether an agent answer satisfies a fixed set of
atomic answer obligations, called **gold atoms**. Use it to test an agent
against a curated question-and-answer dataset, repeat each question to measure
stability, review failures, and compare agent configurations or prompts.

This guide covers the complete evaluation workflow:

- defining a strict JSON or JSONL dataset;
- defining the evaluator models that extract and judge gold atoms;
- selecting the Archi agent configuration and Markdown agent spec to test;
- running the workflow from the CLI or the browser console;
- understanding run states, scores, artifacts, and failures.

Evaluation is separate from the legacy `archi evaluate` benchmarking
workflow. The legacy workflow measures RAGAS and source-retrieval metrics; see
[Benchmarking](benchmarking.md). `archi eval qa` evaluates complete agent
answers against explicit obligations and preserves reproducible run artifacts.

## What the evaluator does

Every evaluation has three phases:

1. **Prepare** validates the Dataset V2 input and fixes the gold atoms. Live
   rows are resolved through their evaluator-only MCP oracle. When an eligible
   row does not supply atoms, the profile's atom extractor derives them from the
   canonical or resolved answer. The CLI uses those inferred atoms immediately;
   the console can instead present them for operator review and save an approved,
   immutable child dataset.
2. **Run** asks the selected Archi agent every prepared question for the
   requested number of independent attempts. The agent sees the question, but
   never the canonical answer or gold atoms.
3. **Score** asks an evaluator model to classify each gold atom as `entailed`,
   `not_mentioned`, or `contradicted` by each complete agent answer. It then
   writes per-attempt results, aggregate metrics, and a Markdown report.

You can run all three phases with one CLI command, run them separately for
inspection between phases, or use the browser console.

## Choose an interface

| Interface                                      | Best for                                                                                           | Important behavior                                                                          |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| [CLI Guide](#cli-guide)                         | Automation, CI, local experiments, and exact control over workspaces                               | `archi eval qa` accepts file paths directly and can run all phases or one phase at a time |
| [Browser Console Guide](#browser-console-guide) | Importing shared datasets/profiles, manually reviewing atoms, launching runs, and browsing history | `/evaluations` uses persistent catalogs and background jobs in the chat service           |

Both interfaces use the same validation, preparation, agent runtime, scoring,
and artifact formats.

## CLI Guide

Use the CLI for automation, CI, local experiments, or direct control over
workspace paths and individual phases.

### Inputs and prerequisites

| Input                  | Required              | Format                       | Purpose                                                                                                     |
| ---------------------- | --------------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Dataset                | Yes                   | UTF-8`.json` or `.jsonl` | Questions, canonical answers or live oracle recipes, metadata, and optional supplied atoms                  |
| Agent config           | Yes for the run phase | Local`.yaml` or `.yml`   | Complete Archi runtime configuration for the agent being tested                                             |
| Agent spec             | Yes for the run phase | Local`.md`                 | Agent name, enabled tools, and system prompt                                                                |
| Evaluator profile      | No                    | Local`.yaml` or `.yml`   | Models used for atom extraction and answer comparison; omitting it selects the built-in profile             |
| Evaluator MCP registry | For live items        | Local`.yaml` or `.yml`   | Evaluator-only aliases, transports, and environment-backed authentication used by Dataset V2 oracle recipes |
| Output directory       | Yes                   | Directory path               | New or explicitly overwritten workspace where all run artifacts are stored                                  |

The output directory is not a config file and does not need to exist in
advance. Before running:

1. [Install Archi](install.md) and confirm `archi eval qa --help` works.
2. Make the tested agent's model provider and selected tool dependencies
   reachable from the CLI process.
3. Provide credentials for the tested agent and evaluator models. See
   [Models &amp; Providers](models_providers.md).
4. For Dataset V2 live items, make the evaluator MCP servers reachable and
   pass their registry with `--mcp-config`.
5. Choose an empty output directory, or deliberately use `--overwrite` as
   described in [Rerunning and integrity protection](#rerunning-and-integrity-protection).

#### Dataset format

The dataset is strict, non-empty, UTF-8 encoded, and must declare
`qa-dataset-v2`. Unknown fields are rejected.

- `.json` contains a `qa-dataset-v2` envelope.
- `.jsonl` starts with `{"schema_version":"qa-dataset-v2"}` and then contains
  one item per non-blank line.

##### Row fields

Every row has these common fields:

| Field              | Required | Type    | Rules and meaning                                                           |
| ------------------ | -------- | ------- | --------------------------------------------------------------------------- |
| `question`       | Yes      | string  | Non-empty question sent to the tested agent                                 |
| `time_sensitive` | Yes      | boolean | Distinguishes static items from live oracle-backed items                    |
| `category`       | No       | string  | Non-empty grouping metadata retained in preparation results                 |
| `answer_mode`    | No       | string  | One of`direct_answer`, `needs_information`, `escalate`, or `refuse` |
| `answer_source`  | No       | string  | Non-empty free-form provenance label retained as metadata                   |

The remaining fields depend on whether the item is static or live:

| Field              | Static item     | Unresolved live item | Rules and meaning                                                                                                                                         |
| ------------------ | --------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `id`             | Required        | Required             | Non-empty unique item ID                                                                                                                                  |
| `answer`         | Required string | Must be omitted      | Hidden canonical answer used for atom extraction. The console alone may persist a trusted materialized live child whose answer is a non-empty JSON object |
| `expected_atoms` | Optional        | Must be omitted      | Complete, pre-reviewed atom set. If absent, preparation invokes the atom extractor                                                                        |
| `oracle`         | Not allowed     | Required             | Strict MCP recipe used to resolve the live answer; see[Dataset V2 live items](#dataset-v2-live-items)                                                      |

`question` and `answer` line endings are normalized. NUL characters are
rejected. IDs must be unique across the entire dataset.
Metadata fields do not currently change scoring.

##### Minimal dataset with automatic atom extraction

Save this as `questions.json`:

```json
{
  "schema_version": "qa-dataset-v2",
  "items": [{
    "id": "storage-quota",
    "question": "How much storage quota remains?",
    "answer": "The account has 2.8 TB remaining.",
    "time_sensitive": false,
    "category": "storage",
    "answer_mode": "direct_answer",
    "answer_source": "operations-handbook"
  }]
}
```

Because `expected_atoms` is absent, `prepare` passes the question and canonical
answer to the atom extractor, which produces required and optional atoms. The
CLI continues with that fixed inferred set. In the browser console, **Generate
Atoms** opens the candidates for review; saving creates a new immutable dataset
version without changing the imported parent.

##### Dataset with supplied gold atoms

Use supplied atoms when domain reviewers already know the exact obligations or
when you do not want an LLM to infer them:

```json
{
  "schema_version": "qa-dataset-v2",
  "items": [{
    "id": "storage-quota",
    "question": "How much storage quota remains?",
    "answer": "The account has 2.8 TB remaining and should request an increase before it reaches 500 GB.",
    "time_sensitive": false,
    "category": "storage",
    "answer_mode": "direct_answer",
    "answer_source": "operations-handbook",
    "expected_atoms": [
      {
        "id": "remaining-capacity",
        "text": "The account has 2.8 TB remaining.",
        "required": true
      },
      {
        "id": "increase-threshold",
        "text": "An increase should be requested before the remaining quota reaches 500 GB.",
        "required": false
      }
    ]
  }]
}
```

Each atom must contain exactly:

| Atom field   | Type    | Rule                                          |
| ------------ | ------- | --------------------------------------------- |
| `id`       | string  | Non-empty and unique within that row          |
| `text`     | string  | Non-empty, independently judgeable obligation |
| `required` | boolean | Whether omission makes the answer fail        |

`expected_atoms` must be non-empty and contain at least one required atom.
Supplying it bypasses atom extraction for that row, but preparation still
validates and snapshots the row. A dataset may mix rows with supplied and
inferred atoms.

Write atoms so each one tests one claim. Avoid combining independent values,
conditions, or instructions into a single atom. Mark an atom optional only
when an answer can omit it and still correctly answer the question.

##### Equivalent JSONL

The same minimal dataset in `questions.jsonl` uses a version header followed by
one complete item per line:

```json
{"schema_version":"qa-dataset-v2"}
{"id":"storage-quota","question":"How much storage quota remains?","answer":"The account has 2.8 TB remaining.","time_sensitive":false,"category":"storage","answer_mode":"direct_answer","answer_source":"operations-handbook"}
```

Do not wrap JSONL items in an array and do not add trailing commas.

##### Dataset V2 live items

Every item requires an explicit stable `id`. Static items keep a
string `answer` and no `oracle`. Unresolved live items use
`time_sensitive: true`, an inline `oracle`, and contain neither `answer` nor
`expected_atoms`:

```json
{
  "schema_version": "qa-dataset-v2",
  "items": [{
    "id": "service-capacity",
    "question": "What is the current service capacity?",
    "time_sensitive": true,
    "oracle": {
      "kind": "mcp",
      "calls": [{
        "id": "capacity",
        "server": "operations-readonly",
        "tool": "get_capacity",
        "arguments": {"service": "primary"},
        "answer_fields": {"available": "/available"},
        "metadata_fields": {"revision": "/revision"}
      }]
    }
  }]
}
```

The evaluator resolves calls in order, canonicalizes the selected answer data,
and hashes it. Metadata is recorded for provenance but does not participate in
equality or atom extraction. External imports may not claim to be materialized
live children by including an answer or atoms; only an approved internal
catalog child is trusted for launch.

##### Evaluator MCP registry

Recipes name only a server alias and tool. Deployment operators own the strict
registry that maps aliases to `stdio` or `streamable_http` connections. Secrets
are environment-variable names, never inline values. The source filename is
conventional: `mcp_config_path` may point to any YAML filename, and Archi stages
the validated snapshot as `qa_evaluation_mcp.yaml`.

Create a registry such as `configs/qa_evaluation_mcp.yaml`:

```yaml
schema_version: qa-evaluation-mcp-v1
servers:
  operations-readonly:
    transport: streamable_http
    url: http://operations-mcp:8000/mcp
    timeout_seconds: 300
    authentication:
      mode: bearer
      token_env: EVALUATION_MCP_TOKEN

  operations-local:
    transport: stdio
    command: /opt/operations/.venv/bin/python
    args:
      - /opt/operations/server.py
    timeout_seconds: 300
    authentication:
      mode: inherited_environment
```

Set the fields as follows:

- `schema_version` must be exactly `qa-evaluation-mcp-v1`.
- Each key under `servers` is a server alias. A Dataset V2
  `oracle.calls[].server` value must exactly match one of these aliases.
- Set `transport: streamable_http` for an HTTP MCP server. Its `url` is required
  and must be an absolute `http://` or `https://` URL without embedded
  credentials.
- Set `transport: stdio` to start an MCP subprocess. `command` is the required
  executable and `args` is an optional list of command arguments. The executable
  and every referenced file must exist in the environment running the
  evaluation.
- `timeout_seconds` is optional and must be a positive integer. It controls
  initialization, tool discovery, and tool-call timeouts for that server. The
  default is 120 seconds.
- `authentication` is required for every server. For HTTP, set it to one of:
  `mode: none`; `mode: bearer` with `token_env`; `mode: basic` with
  `username_env` and `password_env`; or `mode: oauth_client_credentials` with
  `token_url`, `client_id_env`, `client_secret_env`, and optional string-list
  `scopes`. These fields name environment variables; do not put secret values
  in this YAML. For `stdio`, authentication must be
  `mode: inherited_environment`.

The schema is strict: duplicate or unknown keys are rejected. A server is
initialized lazily on first use, calls are not retried automatically, and each
Dataset V2 `oracle.calls[].tool` must be exposed by the selected server.

The registry is structurally separate from the tested agent's top-level
`mcp_servers` configuration. Reusing an alias in both files does not connect
them, and evaluator servers are not inferred from agent servers. This separation
keeps oracle access and credentials outside the agent under test.

Path interpretation differs by interface:

- The CLI's `--mcp-config` option reads the file from the machine running the
  CLI. Its HTTP URLs, subprocess commands, and arguments are then used by that
  same CLI process.
- The browser console reads the registry staged from
  `services.chat_app.evaluations.mcp_config_path`, but executes calls from the
  deployed chatbot container or pod. Its URLs must therefore be reachable from
  that container or pod, its environment must contain the named secret
  variables, and `stdio` commands and argument paths must exist inside that
  runtime. In particular, `localhost` means the chatbot container or pod, not
  the deployment host.

See [Chat-app evaluation configuration](#chat-app-evaluation-configuration)
for the browser-console path and staging rules.

#### Evaluator profile format

The evaluator profile selects two structured-output model calls:

- `qa.atoms_extractor` creates gold atoms for rows that omit
  `expected_atoms`;
- `qa.evaluator` compares each complete agent answer with the fixed atoms.

Save a custom profile as `evaluator.yaml`:

```yaml
version: 1
qa:
  atoms_extractor:
    provider: openai
    model: gpt-5.5
    timeout: 180
  evaluator:
    provider: openai
    model: gpt-5.5
    timeout: 180
```

Set the fields as follows:

- `version` is required and must be `1`.
- `qa.atoms_extractor` selects the model that converts canonical answers into
  gold atoms when a dataset row does not already provide `expected_atoms`.
- `qa.evaluator` selects the model that compares each completed agent answer
  with those gold atoms.
- Each model block requires a non-empty Archi `provider` name and a non-empty
  provider `model` name.
- `timeout` is optional in each model block. When set, it must be a finite
  number greater than zero and is interpreted in seconds.

No other profile or descriptor fields are accepted. Evaluator temperature is
fixed at zero and cannot be configured in the profile. Choose models that
support structured JSON output and `temperature=0`.

If the profile is omitted, Archi uses:

```yaml
version: 1
qa:
  atoms_extractor:
    provider: openai
    model: gpt-5.6-terra
  evaluator:
    provider: openai
    model: gpt-5.6-terra
```

For the CLI, pass the file with
`--evaluator-profile evaluator.yaml` on the composite command or `prepare`.
Archi stores the resolved profile in the workspace. If you pass a profile again
to `score`, it must resolve to exactly the stored profile so that the judge
cannot change midway through a run.

For the browser Console, open **Profiles**, import the `.yaml` or `.yml` file,
and select it when generating atoms or launching the evaluation. Select the
built-in profile instead when no custom evaluator configuration is needed.

##### Local evaluator models with Ollama

Use Archi's `local` provider name, not `ollama`:

```yaml
version: 1
qa:
  atoms_extractor:
    provider: local
    model: qwen3:8b
    timeout: 300
  evaluator:
    provider: local
    model: qwen3:8b
    timeout: 300
```

Pull the model first:

```bash
ollama pull qwen3:8b
```

Set `OLLAMA_HOST` to an endpoint reachable from the process or chatbot
container:

```dotenv
# Archi runs directly on the host
OLLAMA_HOST=http://localhost:11434

# Common Docker host endpoint; use the address supported by your installation
OLLAMA_HOST=http://host.docker.internal:11434

# Common Podman host endpoint
OLLAMA_HOST=http://host.containers.internal:11434
```

The evaluator profile does not read
`services.chat_app.providers.local.base_url`; the evaluator's local provider
uses `OLLAMA_HOST`. The generated Compose service forwards this variable to the
chatbot container. In Helm, set it in the chatbot pod environment or a
referenced Secret.

The selected local model must reliably follow both structured-output schemas.
The profile cannot currently configure an OpenAI-compatible local endpoint or
provider mode.

#### Agent config format

The **agent config** is the Archi deployment YAML that defines the agent being
tested. In this example, it is `deployments/comp_ops_config.yaml`; it is not a
second evaluation-specific file. It contains the agent class, provider, model,
agent MCP servers, vector store, and other runtime settings. The run phase
requires at least these non-empty fields:

```yaml
services:
  chat_app:
    agent_class: CMSCompOpsAgent
    default_provider: openai
    default_model: gpt-5.5
```

This is only the minimum shape accepted before runtime construction. Use the
full config required by the selected agent class and tools, including provider,
vector-store, data-manager, or MCP settings. The `agent_class` must name a
pipeline exported by `src.archi.pipelines`.

The CLI accepts an existing local `.yaml` or `.yml` file through
`--agent-config`. During the run, Archi snapshots the resolved file as
`agent_config.resolved.yaml`.

The browser Console needs no separate agent-config file. It evaluates the agent
already defined by the running deployment's YAML—for example,
`deployments/comp_ops_config.yaml`. See
[Chat-app evaluation configuration](#chat-app-evaluation-configuration) for the
Console-specific fields in that file.

All Console-launched evaluations use that deployment configuration. To compare
agents with different deployment configurations, use separate deployments or
the CLI with separate workspaces. Selecting an agent spec in the Console changes
the Markdown prompt and enabled-tool declaration; it does not replace the
deployment configuration.

#### Agent spec format

The agent spec is a local Markdown file with YAML frontmatter followed by a
non-empty system prompt:

```markdown
---
name: Operations Evaluation Agent
tools:
  - search_vectorstore_hybrid
---

You answer operational questions from the configured knowledge base.
Use the available search tool before making factual claims.
If the evidence does not answer the question, say so.
```

`name` must be a non-empty string. `tools` must be a non-empty list of
non-empty tool names. The body after the closing `---` must be non-empty.
Optional frontmatter such as `ab_only` may be present if supported by the
normal agent-spec loader.

Make sure every selected tool is supported by the chosen agent class and fully
configured. In particular:

- `search_vectorstore_hybrid` makes the evaluation runtime connect to the
  configured vector store;
- `mcp` requires at least one MCP tool to load successfully, otherwise each
  attempt fails before model invocation.

The CLI requires an existing local `.md` file and snapshots it as
`agent_spec.resolved.md`. The console lists `.md` files from
`services.chat_app.agents_dir`, or `/root/archi/agents` when that setting is
empty.

### Usage

#### Run all phases

The composite command prepares, runs, and scores in one operation:

```bash
archi eval qa \
  --dataset questions.json \
  --agent-config agent.yaml \
  --agent-spec agent.md \
  --evaluator-profile evaluator.yaml \
  --mcp-config qa_evaluation_mcp.yaml \
  --output-dir evaluation-run/ \
  --attempts 4 \
  --run-workers 4 \
  --score-workers 8
```

`--attempts` defaults to `1` and must be positive. Four attempts means each
prepared question is independently asked four times. More attempts provide a
better view of stability but increase agent and evaluator calls linearly.

`--run-workers` and `--score-workers` default to `1` and accept values from `1`
through `16`. They control concurrency independently: the run phase must finish
all attempts before the score phase begins. Each worker owns one runtime, and
artifacts remain in canonical question and attempt order even when calls finish
out of order. Start low and raise each value only within your provider's rate
limits and the deployment's available memory.

For Dataset V2 live rows, the composite and `prepare` commands accept
`--skip-live` to omit every live question intentionally. The composite,
`prepare`, and `run` commands accept `--mcp-config`; `score` deliberately does
not. A normal run resolves every live item before agent work and once again
after the global agent-attempt barrier. Only items whose two observations match
the approved baseline are scored.

Omit `--evaluator-profile` to use the built-in profile:

```bash
archi eval qa \
  --dataset questions.json \
  --agent-config agent.yaml \
  --agent-spec agent.md \
  --output-dir evaluation-run/
```

A successful composite command ends with a `scored` manifest and creates
`evaluation-run/report.md`.

#### Run and inspect one phase at a time

Use the staged workflow to review atoms before running the agent, or answers
before paying for evaluator calls.

##### 1. Prepare

```bash
archi eval qa prepare questions.json \
  --evaluator-profile evaluator.yaml \
  --mcp-config qa_evaluation_mcp.yaml \
  --output-dir evaluation-run/
```

Inspect:

```bash
less evaluation-run/preparation.jsonl
```

Preparation writes exactly one terminal record per input item and a manifest
with status `prepared`. Prepared records contain fixed atoms. Failed rows and
live rows omitted with `--skip-live` contain no runnable output. Run eligibility
and lifecycle counts come from this same artifact. Preparation does not invoke
the tested agent. If no row is prepared, the subsequent run refuses to start.

##### 2. Run the agent

```bash
archi eval qa run evaluation-run/ \
  --agent-config agent.yaml \
  --agent-spec agent.md \
  --mcp-config qa_evaluation_mcp.yaml \
  --attempts 4 \
  --run-workers 4
```

Inspect:

```bash
less evaluation-run/answers.jsonl
```

The command reads questions from the prepared workspace; it does not take the
original dataset again. Each run worker owns and reuses a separate selected
pipeline while every attempt still receives fresh invocation state. When all
attempt slots are terminal, the manifest becomes
`run_completed`. Each terminal answer row also records non-negative
`duration_ms` measured only around the tested-agent execution. Its
`tool_calls` array records each observed tool's ordinal, name, success, error,
or incomplete status, complete query, complete response or error when observed,
and duration when available. Content is stored without truncation. A call that
starts without a matching terminal callback remains visible as `incomplete` and
omits unavailable response and duration fields.

##### 3. Score

```bash
archi eval qa score evaluation-run/ --score-workers 8
```

You may pass `--evaluator-profile evaluator.yaml`, but it must match the
profile already snapshotted during preparation:

```bash
archi eval qa score evaluation-run/ \
  --evaluator-profile evaluator.yaml
```

Scoring does not invoke the tested agent again. It writes the judgments,
summary, report, and a `scored` manifest.

### Outputs

The CLI writes one self-contained workspace under `--output-dir`. Its main
outputs are separated by phase:

| Phase   | Primary outputs                                                                                                       |
| ------- | --------------------------------------------------------------------------------------------------------------------- |
| Prepare | Input snapshot, resolved evaluator profile,`preparation.jsonl`, and `manifest.json`                               |
| Run     | Resolved agent config/spec,`answers.jsonl`, `live_checks.jsonl`, and the updated manifest                         |
| Score   | `evaluation_results.jsonl`, machine-readable `summary.json`, human-readable `report.md`, and the final manifest |

Start with `report.md` for a human review and `summary.json` for automation.
See [Run workspace artifacts](#run-workspace-artifacts) for the complete file
contract and [Rerunning and integrity protection](#rerunning-and-integrity-protection)
before reusing an existing output directory.

## Browser Console Guide

Use the browser console for easy metrics visualization, manual atom
review, background execution, retries, and run-history visualization.

### Inputs and prerequisites

| Input or prerequisite  | Required       | How the console receives it                                                                      |
| ---------------------- | -------------- | ------------------------------------------------------------------------------------------------ |
| Running chat service   | Yes            | Deployment with`services.chat_app.evaluations.enabled: true`                                   |
| Dataset                | Yes            | `.json` or `.jsonl` upload in **Datasets**, limited to 25 MiB                          |
| Evaluator profile      | No             | Built-in profile or`.yaml`/`.yml` upload in **Profiles**, limited to 25 MiB            |
| Agent config           | Yes            | The running deployment YAML; the Console uses it automatically and requires no separate path     |
| Agent spec             | Yes            | Deployment-controlled Markdown file in`services.chat_app.agents_dir`; select it in the Console |
| Evaluator MCP registry | For live items | Deployment-controlled file staged from`mcp_config_path`                                        |

#### Supported input formats

- Datasets use the same strict `.json` or `.jsonl` Dataset V2 contract
  described in [Dataset format](#dataset-format).
- Imported evaluator profiles must use `.yaml` or `.yml` and follow
  [Evaluator profile format](#evaluator-profile-format). The built-in profile
  requires no upload.
- Agent YAML and Markdown specs follow [Agent config format](#agent-config-format)
  and [Agent spec format](#agent-spec-format), but are selected from files
  controlled by the deployment rather than uploaded through the browser.
- Dataset V2 live items require the registry described in
  [Evaluator MCP registry](#evaluator-mcp-registry).

#### Chat-app evaluation configuration

The evaluation console is opt-in. Enable it explicitly in the deployment
configuration. For example, if the repository contains
`deployments/comp_ops_config.yaml`, `configs/agents/`, and
`configs/qa_evaluation_mcp.yaml`, write:

```yaml
services:
  chat_app:
    # Host source directory; relative paths resolve from this deployment YAML.
    agents_dir: ../configs/agents
    evaluations:
      enabled: true
      # Host source file; required only for Dataset V2 live oracle items.
      mcp_config_path: ../configs/qa_evaluation_mcp.yaml
```

Here, `../configs/qa_evaluation_mcp.yaml` is specifically required only for live items so if the dataset is strictly static it can be ignored. In this example, it resolves from the directory containing
`deployments/comp_ops_config.yaml`, so it points to
`configs/qa_evaluation_mcp.yaml`. An absolute host path is also accepted, but a
repository-relative path is usually more portable. Any folder is accepted as long as the yaml content for the mcp config is correctly written as explained.

Both `archi create` and the chat runtime treat an omitted evaluation block,
an omitted `enabled` field, and `enabled: false` as disabled. The runtime
registers `/evaluations` and its APIs only when `enabled` is explicitly `true`.

Set the fields as follows:

- `services.chat_app.agents_dir` points to the host directory containing the
  Markdown agent specs. Archi stages those files, and the Console lists them
  when you choose the agent spec for a run. An absolute host path is accepted.
  A relative path is resolved from the deployment YAML when that path exists.
- `services.chat_app.evaluations.enabled` must be `true` to register the
  `/evaluations` page and its APIs.
- `services.chat_app.evaluations.mcp_config_path` points to the evaluator MCP
  registry described in [Evaluator MCP registry](#evaluator-mcp-registry). Set
  it when Dataset V2 contains live oracle items. It accepts an absolute host
  path or a path relative to the deployment YAML. Static-only datasets do not
  require this field.

The optional `mcp_config_path` is a host source path, not a path inside the
chatbot container. A relative value is resolved against the YAML file containing
this deployment configuration. An explicitly configured file must exist, be
readable UTF-8, and satisfy the strict `qa-evaluation-mcp-v1` schema or
deployment generation fails. Omit the field for static-only evaluation. If a
live item is attempted without a registry, the item fails with `Evaluator MCP registry is not configured.`

The generated Docker Compose deployment persists the root at
`./data/evaluations`. The root contains:

```text
evaluations/
├── datasets/   # immutable imported and reviewed datasets
├── profiles/   # immutable imported evaluator profiles
├── drafts/     # atom-review drafts
├── jobs/       # persisted background-job records
└── runs/       # evaluation workspaces and reports
```

Do not expose this root publicly. Dataset snapshots, prepared items,
judgments, and evaluator rationales can reveal canonical answers.

#### Permissions

Authenticated deployments use:

- `evaluations:view` to open the console and read catalogs, jobs, run details,
  and reports;
- `evaluations:run` to launch, cancel, continue, or retry an evaluation;
- `evaluations:manage` to import datasets and profiles, generate or review
  atoms, refresh live snapshots, and save reviewed datasets.

The wildcard administrator role grants all three. Add them explicitly to
custom roles as needed. When authentication is disabled, evaluation routes
retain the normal unrestricted local-development behavior.

### Usage

#### Import and prepare a dataset

1. Open `/evaluations`.
2. In **Datasets**, import a `.json` or `.jsonl` dataset and give it a display
   name. Importing identical bytes reuses the existing catalog entry.
3. In **Profiles**, use the built-in profile or import a `.yaml`/`.yml`
   evaluator profile.
4. Select the dataset:
   - If the complete dataset has zero supplied atoms, choose **Generate
     Atoms** and select a profile. The background provider job generates atoms
     for eligible rows.
   - If the dataset already has one or more atoms, choose **Review Atoms**.
     Existing atoms are preserved and eligible rows without atoms are shown
     empty for manual completion.
5. Review every eligible row. Atom IDs and text must be non-empty and unique
   per item, and every item must have at least one required atom.
6. Save under a new dataset name.

Saving never mutates the imported parent. It creates an immutable child dataset
with reviewed `expected_atoms` and records the parent dataset ID.

For Dataset V2, **Generate Atoms** resolves live rows and shows their read-only
resolved answer, recipe summary, metadata, and call evidence beside editable
atoms. The unchecked **Create a static-only dataset** option omits live
questions. A complete approved child exposes **Refresh live snapshot**; a
static-only child exposes **Add live questions from parent**. Both create a new
review draft and publish a new immutable sibling after approval.

The console intentionally does not auto-generate atoms for a partially
annotated dataset. Review the existing atoms and manually fill its missing
rows, or import a dataset with zero atoms and generate all of them.

#### How live questions in a child dataset are checked

An approved child dataset contains the live answer that was resolved and
reviewed when that child was created. That stored answer is the **approved
baseline** for the run. “Stale” is not based on the child's age or a time-to-live
value; it means that a new oracle observation no longer matches that approved
baseline, or that the oracle cannot currently produce an answer.

When you start a Console evaluation on a child containing live questions,
Archi performs these checks:

1. Preparation reads each materialized live answer from the selected child and
   computes its baseline SHA-256 hash from canonical JSON. JSON object key order
   does not affect the hash. Oracle metadata is retained as provenance but is
   not part of the answer hash.
2. Before starting **any** agent attempt, Archi executes every prepared live
   question's `oracle.calls` recipe again through `qa_evaluation_mcp.yaml`. It
   selects the configured `answer_fields`, builds the current canonical answer,
   and hashes it in the same way as the baseline.
3. The question is currently valid only when the oracle resolves successfully
   and the current answer hash exactly matches the approved baseline hash. A
   different hash produces `answer_changed`; a connection, authentication,
   timeout, tool, or response failure produces `oracle_failed`.
4. If every live question still matches, agent attempts begin. Static questions
   do not make oracle calls and proceed normally.
5. If any live question is changed or unavailable, the whole Console launch
   pauses as `attention_required` before all agent attempts, including static
   ones. The UI reports **No agent attempts have started**, and
   `live_checks.jsonl` records the observations, hashes, metadata, call evidence,
   and failure reason.

At `attention_required`, choose one of these actions:

- **Refresh live snapshot** cancels the paused evaluation and creates a review
  draft from the child's definition parent. Archi resolves the parent's live
  recipes again, labels them as changed, unchanged, or unavailable, and
  generates atoms from the current answers. Review and save the draft as a new
  immutable sibling dataset, then launch a new evaluation using that sibling.
- **Continue with valid questions** repeats the complete pre-run check. Live
  questions that now match the approved baseline are included. Questions that
  still differ or remain unavailable are excluded from agent execution and get
  terminal `live_validation_failed` results. Static questions remain included.
  If a live question that was previously valid has become invalid, the run
  pauses again. Continue is unavailable when no static or matching live question
  can run.
- **Cancel** closes the paused evaluation without running the agent.

After all admitted agent attempts finish, Archi resolves each admitted live
question once more and compares that post-run answer with the same approved
baseline. A post-run change or oracle failure does not pause—the attempts have
already happened—but their results become `live_validation_failed` and are not
sent to the judge or included in quality metrics. Successful pre-run and
post-run observations are required for a live attempt to be scored.

#### Launch and inspect a run

1. Open **New evaluation**.
2. Enter a name.
3. Select the exact dataset, evaluator profile, and agent spec. A reviewed
   dataset uses its supplied atoms. An unreviewed dataset is also valid, but
   preparation infers atoms for eligible rows that do not supply them, without
   a manual review checkpoint.
4. Choose a positive attempt count.
5. Choose **Run workers** and **Evaluation workers** from `1` through `16`.
   Run workers control simultaneous tested-agent calls. Evaluation workers
   control simultaneous judge calls after the complete run phase. Higher values
   increase concurrent provider requests and runtime memory.
6. Select **Start evaluation**.
7. Watch the background job or leave the page; the run continues in the chat
   service.
8. A live run first shows **Checking live answers…**. If a value changed or
   cannot be resolved, the persisted job enters `attention_required` and states
   **No agent attempts have started.** Refresh the live snapshot, continue with
   only currently valid questions, or cancel. Continue repeats the complete
   pre-check before starting any agent call.
9. If the launch was accidental or takes too long, select **Cancel evaluation**
   in the active-job banner and confirm. The local evaluation worker stops and
   the run remains visible as `canceled`.
10. Open **Runs** to inspect status, answers, judgments, metrics, and the
    report. The underlying run API and workspace also preserve the manifest,
    preparation records, and other raw artifacts listed below.

Only one provider-consuming atom-generation or evaluation job runs at a time.
A conflicting launch returns HTTP 409. `attention_required` releases that lock
and survives restart; stale queued or running jobs become `interrupted`.

Canceling stops the local evaluation process and its local child processes. It
cannot recall a request a remote model provider has already accepted or undo an
external tool side effect. Canceled runs have no score, trend point, report, or
retry action.

The Runs page is reconstructed from persisted artifacts. A malformed,
unsupported, missing, or hash-mismatched workspace appears as an isolated
`invalid` entry instead of breaking the history list.

#### Retry technical failures

The console exposes retry actions only for provider or runtime failures:

- an open generated atom draft with `preparation_failed` rows can retry those
  rows in place without regenerating successful candidates or modifying the
  imported parent dataset;
- a scored run with `execution_failed`, `evaluation_failed`, or
  `live_validation_failed` attempts can
  create a complete successor run. Execution failures rerun Archi and the
  comparator, while evaluation failures reuse the verified terminal answer and
  rerun only the comparator.

Successful scored attempts are carried forward unchanged. The parent run
remains immutable, the successor records its direct parent and retry selection,
and both runs remain visible in history. Evaluation retries inherit the parent's
run and score worker counts. Scored attempts that merely fail the quality
threshold are not retryable.

A live-validation retry always resolves the item again, never reuses a prior
answer for that item, runs fresh agent work only after a matching pre-check, and
requires a matching post-check before comparison.

Atom retries require `evaluations:manage`; evaluation retries require
`evaluations:run`. A draft or run without retryable technical failures creates
neither a job nor a new artifact.

### Outputs

The console exposes both rendered results and their persisted evidence:

| Output                                                       | Where to use it                                           |
| ------------------------------------------------------------ | --------------------------------------------------------- |
| Run status and metrics                                       | **Runs** list and run detail                        |
| Human-readable report                                        | **Open report** from a completed run                |
| Per-attempt answers, atom judgments, latency, and tool calls | Expand a question and attempt in run detail               |
| Historical comparisons                                       | Attempt-latency, pass-rate, and technical-failure charts  |
| Immutable evidence                                           | The run workspace under the configured evaluation root    |
| Reusable inputs and work state                               | Dataset/profile catalogs, atom drafts, and persisted jobs |

#### Compare history trends

The evaluation homepage requests one bounded history window from the server for
the run table and all graphs. Choose the last 7, 30, 90, or 365 days; 90 days is
the default and there is no unbounded option. The server computes an
explicit UTC cutoff and excludes older timestamped runs before verifying their
summary or answer artifacts. Runs without an authoritative timestamp remain in
the table because their age cannot be established safely.

Within that window, the homepage charts valid, fully scored runs with an
authoritative timestamp and metric value. Prepared or execution-complete runs
remain in the run table while work continues, but do not appear in the trend
dataset selector or graphs. One shared dataset selector controls all three graphs:

- **Attempt latency** plots the average, best, and worst tested-agent latency
  across the attempts recorded by each run.
- **Pass rate** plots `passed_attempts / quality_accounted_attempts`.
- **Technical failure rate** plots `(execution_failed + evaluation_failed) / (scored + execution_failed + evaluation_failed)`. It is not the inverse of
  pass rate; a scored attempt may fail its quality threshold without being a
  technical failure.

All datasets are selected initially. Clear dataset checkboxes to compare a
subset, or choose **Show all datasets** to restore the complete history. Hover
or focus a dot for the dataset, run, exact value, denominator, timestamp, and
retry relationship. Click the dot, or focus it and press Enter or Space, to
open that run's detail page.

Retry successors appear as their own complete persisted runs and retain their
lineage in the tooltip. CLI-created runs use their immutable input snapshot as
the dataset identity and show only the input filename when available; the
console does not expose its host path.

Older or partial artifacts can lack timestamps, lifecycle counts, or latency.
The graphs omit the unavailable point, leave a gap in its series line, and
report the incomplete coverage; they never infer a value or replace missing
data with zero. For runs inside the selected window, history aggregation streams
answer artifacts rather than loading complete answer files into memory.

#### Inspect per-question latency

Run detail displays tested-agent latency per question before the aggregate
quality metrics. Each question provides an attempt selector. The selected
attempt's vertical bar stacks summed tool-call latency and remaining agent time;
the full height is the authoritative attempt `duration_ms`. Changing the
attempt animates the bar to its new height and composition. The tool label
reports the raw sum of tool-call durations. If concurrent calls make that sum
greater than total wall-clock latency, the colored tool segment is capped at the
full bar and remaining agent time is shown as zero.

If any recorded call lacks an authoritative duration, the chart still shows the
sum of calls that were timed but labels the remaining attempt time as
unattributed. It does not misclassify untimed tool execution as other agent
work.

Historical runs without per-attempt timings show an explicit unavailable state.
Historical attempts that have total latency but predate tool timings show the
total and mark the tool portion unavailable. The console does not infer latency
from phase timestamps.

#### Inspect per-attempt tool calls

Expand a question, then expand one of its attempts. The nested **Tool calls**
section lists every recorded call in execution order. Expand an individual call
to read its complete query and response or error. JSON-shaped content is
pretty-printed, long content remains readable, and duration appears only when
the artifact contains an authoritative `duration_ms` value.

An attempt with no recorded calls says so explicitly. Historical timing-only
calls remain listed with their available name, status, and duration, while the
console states that their query and response details were not captured. Run
detail returns the selected run's complete trace without truncation or
pagination.

## Detailed result reference

The CLI and browser console produce the same scoring records and workspace
formats. Use this appendix when interpreting results, inspecting raw artifacts,
or troubleshooting lifecycle behavior after following either guide above.

### Understand states

There are two independent state machines.

#### Run manifest states

| Manifest status        | Exact meaning                                                                                                                                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prepared`           | Dataset validation and preparation finished and preparation artifacts were committed. Agent attempts are not yet committed. Individual input rows may still be skipped or have preparation failures.     |
| `attention_required` | The console's pre-run live check found changed or unavailable live answers. No agent attempt has started; refresh the approved snapshot, continue with valid questions, or cancel.                       |
| `run_completed`      | The requested attempt slots for every prepared item are terminal as`answer_ready` or `execution_failed`, and `answers.jsonl` plus resolved agent inputs were committed. Scoring has not completed. |
| `scored`             | Scoring finished and`evaluation_results.jsonl`, `summary.json`, and `report.md` were committed. Some individual attempts may still have execution or evaluation failures.                          |

The manifest remains `prepared` while the agent loop is in progress because
`answers.jsonl` is written atomically. Likewise, `run_completed` means the
agent phase is complete, not that the overall evaluation passed.

#### Console job states

| Job status             | Meaning                                                                      |
| ---------------------- | ---------------------------------------------------------------------------- |
| `queued`             | Accepted by the console and waiting for its worker                           |
| `running`            | Background work is executing                                                 |
| `attention_required` | A live pre-check paused before agent execution and awaits an operator action |
| `cancel_requested`   | Cancellation was accepted and worker termination is in progress              |
| `canceled`           | The evaluation worker stopped and canceled history was persisted             |
| `completed`          | The complete requested operation returned successfully                       |
| `failed`             | The operation raised an error; inspect the job's error field                 |
| `interrupted`        | The service restarted before a non-terminal job completed                    |

For a console evaluation, the job normally remains `running` while the
workspace progresses through `prepared` and `run_completed`. The job becomes
`completed` only after the composite workflow returns a `scored` run.

### Understand scoring

For each atom, the evaluator returns:

- `entailed`: the answer communicates the expected meaning, worth `1`;
- `not_mentioned`: the answer neither supports nor contradicts it, worth `0`;
- `contradicted`: the answer makes an incompatible claim, worth `-1`.

`unjudgeable` is part of the evaluator response schema but is rejected for
scoring and records that attempt as `evaluation_failed`.

For one successfully judged attempt:

- **atom score** is the mean atom value, floored at zero;
- **required-atom recall** is the fraction of required atoms entailed;
- **passed** is true only when every required atom is entailed.

Optional atoms affect atom score, but not pass/fail. A response can therefore
pass while omitting optional details.

Execution failures count as failed quality attempts. Evaluation failures remain
visible but are excluded from quality denominators because the answer could
not be judged reliably. Review lifecycle counts alongside pass rates so
evaluator failures are not mistaken for good quality.

### Run workspace artifacts

The current workspace schema is `qa-v2`.

| File                                  | Written in            | Contents                                                                                                                                                                                             |
| ------------------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `input.snapshot.json` or `.jsonl` | Prepare               | Exact input bytes used by the run                                                                                                                                                                    |
| `evaluator_profile.resolved.yaml`   | Prepare               | Fixed evaluator profile                                                                                                                                                                              |
| `preparation.jsonl`                 | Prepare               | One terminal record per input item, containing either runnable normalized data and fixed atoms, a skip, or a preparation failure                                                                     |
| `agent_config.resolved.yaml`        | Run                   | Exact tested Archi config                                                                                                                                                                            |
| `agent_spec.resolved.md`            | Run                   | Exact tested agent spec and prompt                                                                                                                                                                   |
| `answers.jsonl`                     | Run                   | One terminal`answer_ready` or `execution_failed` row per attempt slot, including tested-agent `duration_ms` and complete ordered tool-call query/response/error records with optional duration |
| `live_checks.jsonl`                 | Run                   | Ordered pre-run and post-run oracle observations, normalized answers, hashes, metadata, bounded call evidence, or item-scoped live failures                                                          |
| `evaluation_results.jsonl`          | Score                 | Answers, atom judgments, rationales, metrics, or terminal failures                                                                                                                                   |
| `summary.json`                      | Score                 | Machine-readable aggregate and per-item metrics plus provenance hashes                                                                                                                               |
| `report.md`                         | Score                 | Human-readable result summary                                                                                                                                                                        |
| `manifest.json`                     | Every completed phase | Schema/run version, state, phase timestamps/counts, phase worker counts, agent metadata, and artifact SHA-256 hashes                                                                                 |
| `console_metadata.json`             | Console only          | Display name, selected catalog IDs/spec, and launch worker counts                                                                                                                                    |

The workspace is the reproducibility record. Keep it intact when comparing
runs, and archive it with any external version identifiers you need. The
current artifacts record tested-agent and tool-call latency but do not record
source-control commits, release gates, token usage, model prompts, evaluator
prompts, or reasoning traces. Tool queries and responses are complete.

### Rerunning and integrity protection

Completed-phase artifacts are hashed in `manifest.json`. A later phase verifies
its inputs and fails closed if an artifact is edited, missing, or replaced.
Review files without modifying them.

Existing evaluator-owned files are not overwritten by default:

- `prepare --overwrite` replaces preparation and invalidates run and score
  artifacts;
- `run --overwrite` replaces run artifacts and invalidates score artifacts;
- `score --overwrite` replaces only results, summary, and report;
- composite `archi eval qa --overwrite` rebuilds the complete workspace.

Use a new output directory when comparing agents, prompts, providers, models,
attempt counts, datasets, or evaluator profiles. Reusing and overwriting one
directory destroys the previous comparison point.

### Failure and lifecycle records

Preparation is item-scoped:

- `prepared`: the row has a valid fixed atom set;
- `preparation_failed`: oracle resolution, atom extraction, or validation failed
  for that row;
- `skipped_live`: a Dataset V2 live row was intentionally omitted with
  `--skip-live` or static-only generation.

Agent and evaluator work is attempt-scoped:

- `answer_ready`: the agent produced a usable terminal string;
- `execution_failed`: agent construction, tool loading, invocation, or terminal
  answer validation failed;
- `scored`: comparison succeeded;
- `evaluation_failed`: the evaluator response was invalid, incomplete,
  unjudgeable, or otherwise failed.
- `live_validation_failed`: the pre-run or post-run live observation was
  unavailable or did not match the approved baseline. It is excluded from both
  quality and technical-failure denominators.

A composite operation may reach `scored` even when individual rows or attempts
failed, because those failures are preserved as evidence. Decide acceptance
using the lifecycle counts and quality metrics, not the top-level state alone.

### Cost, concurrency, and data handling

For `P` prepared items and `N` attempts:

- the agent is invoked up to `P × N` times;
- the comparator is invoked once for each `answer_ready` attempt;
- the atom extractor is invoked once for each eligible row without supplied
  atoms.

Start with one or two representative items and one attempt to validate
connectivity, structured output, tool loading, and artifact permissions. Then
increase dataset size or attempt count.

Canonical answers and atoms are hidden from the tested agent, but they are
stored in the workspace. Evaluator prompts and rationales also contain or may
reveal them. Restrict access to datasets, the evaluation root, run artifacts,
logs, and reports according to the sensitivity of the evaluation set.

## Troubleshooting

### The run remains `prepared`

Preparation finished, but the agent phase has not atomically committed all
attempts. Check the console job state and chatbot logs. Slow provider or tool
calls can keep this state for the duration of the agent loop.

### The run remains `run_completed`

All agent attempts are terminal, but scoring has not committed. Check
evaluator credentials, structured-output support, timeouts, and chatbot logs.

### `run requires at least one prepared item`

All rows were time-sensitive or failed atom preparation. Inspect
`preparation.jsonl`, fix the rows/profile, and prepare a new workspace
or deliberately rerun preparation with `--overwrite`.

### `agent spec selected 'mcp', but no MCP tools were loaded`

The spec enables `mcp`, but normal pipeline construction loaded no MCP tools.
Fix the agent config, server reachability, mounts, credentials, or spec.

### Profile mismatch during score

Do not supply a different profile to `score`. Use the profile that was used
during preparation, omit the option to use the snapshotted profile, or prepare
a new workspace.

### Hash mismatch or missing artifact

The workspace changed after a phase completed or is incomplete. Restore the
original artifact from a trusted copy, or rerun the appropriate phase with
`--overwrite`. Do not edit the manifest to bypass integrity checks.

### The console says another job is active

Atom generation and evaluation share a single-flight worker. Wait for the
active job to finish, or cancel it from the active-job banner when it is an
evaluation. A `cancel_requested` job remains active until its process exits. If
the service restarted, refresh the catalog and verify that the old job was
marked `interrupted`.

For exact command flags, see [`archi eval qa` in the CLI
reference](cli_reference.md#archi-eval-qa). For deployment-level paths and
authentication settings, see [Configuration](configuration.md).
