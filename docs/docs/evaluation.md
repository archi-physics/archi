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

1. **Prepare** validates the dataset, skips time-sensitive items (will be supported in the future), and generates the
   gold atoms given an answer entry. When a row does not supply its own atoms, an evaluator model
   extracts them from the canonical answer.
2. **Run** asks the selected Archi agent every prepared question for the
   requested number of attempts. The agent sees the question, but never the
   canonical answer or gold atoms.
3. **Score** asks an evaluator model to classify each gold atom as `entailed`,
   `not mentioned`, or `contradicted` by each complete agent answer. It then writes
   per-attempt results, aggregate metrics, and a Markdown report.

You can run all three phases with one CLI command, run them separately for
inspection between phases, or use the browser console.

## Choose an interface

| Interface                        | Best for                                                                                           | Important behavior                                                        |
| -------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `archi eval qa`                | Automation, CI, local experiments, and exact control over workspaces                               | Accepts file paths directly and can run all phases or one phase at a time |
| `/evaluations` browser console | Importing shared datasets/profiles, manually reviewing atoms, launching runs, and browsing history | Uses persistent catalogs and background jobs in the chat service          |

Both interfaces use the same validation, preparation, agent runtime, scoring,
and artifact formats.

## Inputs and prerequisites

### Files used by the CLI

| Input             | Required              | Format                       | Purpose                                                                                         |
| ----------------- | --------------------- | ---------------------------- | ----------------------------------------------------------------------------------------------- |
| Dataset           | Yes                   | UTF-8`.json` or `.jsonl` | Questions, canonical answers, metadata, and optional supplied atoms                             |
| Agent config      | Yes for the run phase | Local`.yaml` or `.yml`   | Complete Archi runtime configuration for the agent being tested                                 |
| Agent spec        | Yes for the run phase | Local`.md`                 | Agent name, enabled tools, and system prompt                                                    |
| Evaluator profile | No                    | YAML                         | Models used for atom extraction and answer comparison; omitting it selects the built-in profile |
| Output directory  | Yes                   | Directory path               | New or explicitly overwritten workspace where all run artifacts are stored                      |

The dataset and agent files are inputs. The output directory is not a config
file and does not need to exist in advance.

Before running:

1. [Install Archi](install.md) and confirm `archi eval qa --help` works.
2. Make the agent's model provider and every selected tool dependency
   reachable from the process running the command. For example, an agent using
   vector search needs its configured vector store; an agent using MCP needs
   its configured MCP servers.
3. Provide the credentials required by both the tested agent model and the
   evaluator models. See [Models &amp; Providers](models_providers.md).
4. Choose an empty output directory, or deliberately use `--overwrite` as
   described in [Rerunning and integrity protection](#rerunning-and-integrity-protection).



### Inputs used by the browser console

The console needs:

- a running chat service with the evaluation console enabled;
- a persistent evaluation root;
- a valid agent config at `agent_config_path`;
- one or more valid Markdown agent specs in `services.chat_app.agents_dir`;
- an imported dataset;
- either the built-in evaluator profile or an imported evaluator profile.

The console accepts dataset and profile uploads up to 25 MiB. It does not
upload agent configs or specs: those are deployment-controlled files.

## Dataset format

The dataset is strict: unknown fields are rejected. It must be non-empty and
UTF-8 encoded.

- A `.json` file contains one JSON array of row objects.
- A `.jsonl` file contains one row object per non-blank line.

### Row fields

| Field              | Required | Type    | Rules and meaning                                                                                                                                                                                                         |
| ------------------ | -------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `question`       | Yes      | string  | Non-empty question sent to the tested agent                                                                                                                                                                               |
| `answer`         | Yes      | string  | Non-empty canonical answer used only to create or support the gold atoms                                                                                                                                                  |
| `time_sensitive` | Yes      | boolean | `true` skips the row before atom extraction or agent execution at the moment                                                                                                                                            |
| `id`             | No       | string  | Non-empty unique item ID; if omitted, Archi derives a stable ID from`question` and `answer`                                                                                                                           |
| `category`       | No       | string  | Non-empty grouping metadata retained in preparation results                                                                                                                                                               |
| `answer_mode`    | No       | string  | One of`direct_answer`, `needs_information`, `escalate`, or `refuse`                                                                                                                                               |
| `answer_source`  | No       | string  | Non-empty free-form provenance label retained as metadata                                                                                                                                                                 |
| `expected_atoms` | No       | array   | Complete, pre-reviewed gold-atom set; omitting it will cause the pipeline to perform automatic extraction using the`prepare` subcommand in the CLI. The browser console will display a "Generate Atoms" button instead. |

`question` and `answer` line endings are normalized. NUL characters are
rejected. Explicit and derived IDs must be unique across the entire dataset.
Metadata fields do not currently change scoring.

### Minimal dataset with automatic atom extraction

Save this as `questions.json`:

```json
[
  {
    "id": "storage-quota",
    "question": "How much storage quota remains?",
    "answer": "The account has 2.8 TB remaining.",
    "time_sensitive": false,
    "category": "storage",
    "answer_mode": "direct_answer",
    "answer_source": "operations-handbook"
  }
]
```

Because `expected_atoms` is absent, the atom extractor is used in the pipeline via the `prepare` subcommand. It receives the question
and canonical answer during preparation and breaks down the answer in required and optional atoms. A user is able to review the generated atoms in the browser console and generate different version of the dataset every time new atoms are generated or edited.

### Dataset with supplied gold atoms

Use supplied atoms when domain reviewers already know the exact obligations or
when you do not want an LLM to infer them:

```json
[
  {
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
  }
]
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

### Equivalent JSONL

The same minimal row in `questions.jsonl` is one complete JSON object on one
line:

```json
{"id":"storage-quota","question":"How much storage quota remains?","answer":"The account has 2.8 TB remaining.","time_sensitive":false,"category":"storage","answer_mode":"direct_answer","answer_source":"operations-handbook"}
```

Do not wrap JSONL rows in an array and do not add trailing commas.

## Evaluator profile format

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

The schema is strict:

| Path                   | Required               | Type    | Rule                                       |
| ---------------------- | ---------------------- | ------- | ------------------------------------------ |
| `version`            | Yes                    | integer | Must be`1`                               |
| `qa.atoms_extractor` | Yes                    | mapping | Atom-extraction model descriptor           |
| `qa.evaluator`       | Yes                    | mapping | Answer-comparison model descriptor         |
| `provider`           | Yes in each descriptor | string  | Non-empty Archi provider name              |
| `model`              | Yes in each descriptor | string  | Non-empty provider model name              |
| `timeout`            | No                     | number  | Finite value greater than zero, in seconds |

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

The CLI loads the profile supplied during preparation and stores the resolved
copy in the workspace. A profile supplied again during scoring must resolve to
exactly the same profile. This prevents scoring with a different judge by
accident. The console requires imported profile filenames to end in `.yaml` or
`.yml`.

### Local evaluator models with Ollama

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

## Agent config format

The agent config is a normal Archi YAML config, not an evaluation-specific
config. The run phase validates that it has these non-empty fields:

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

The CLI accepts only an existing local `.yaml` or `.yml` file. During the run,
Archi snapshots the resolved file as `agent_config.resolved.yaml`.

For console evaluations, set the deployment's existing resolved config:

```yaml
services:
  chat_app:
    evaluations:
      agent_config_path: /root/archi/configs/config.yaml
```

All console-launched evaluations use that config. To compare different runtime
configs, use separate deployments or the CLI with separate workspaces.

## Agent spec format

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

## Run a complete evaluation from the CLI

The composite command prepares, runs, and scores in one operation:

```bash
archi eval qa \
  --dataset questions.json \
  --agent-config agent.yaml \
  --agent-spec agent.md \
  --evaluator-profile evaluator.yaml \
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

## Run and inspect one phase at a time

Use the staged workflow to review atoms before running the agent, or answers
before paying for evaluator calls.

### 1. Prepare

```bash
archi eval qa prepare questions.json \
  --evaluator-profile evaluator.yaml \
  --output-dir evaluation-run/
```

Inspect:

```bash
less evaluation-run/preparation.jsonl
```

Preparation writes exactly one terminal record per input item and a manifest
with status `prepared`. Prepared records contain fixed atoms; failed and
time-sensitive records contain no runnable output. Run eligibility and
lifecycle counts come from this same artifact. Preparation does not invoke the
tested agent. If every eligible item fails preparation or is time-sensitive,
the subsequent run refuses to start because there are no prepared items.

### 2. Run the agent

```bash
archi eval qa run evaluation-run/ \
  --agent-config agent.yaml \
  --agent-spec agent.md \
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

### 3. Score

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

## Configure and use the browser console

### Enable and persist it

The evaluation console is opt-in. Enable it explicitly in the deployment
configuration:

```yaml
services:
  chat_app:
    agents_dir: /root/archi/agents
    evaluations:
      enabled: true
      root: /root/archi/evaluations
      agent_config_path: /root/archi/configs/config.yaml
```

Both `archi create` and the chat runtime treat an omitted evaluation block,
an omitted `enabled` field, and `enabled: false` as disabled. The runtime
registers `/evaluations` and its APIs only when `enabled` is explicitly `true`.

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

### Permissions

Authenticated deployments use:

- `evaluations:view` to open the console and read catalogs, jobs, run details,
  and reports;
- `evaluations:run` to launch an evaluation;
- `evaluations:manage` to import datasets and profiles, generate or review
  atoms, and save reviewed datasets.

The wildcard administrator role grants all three. Add them explicitly to
custom roles as needed. When authentication is disabled, evaluation routes
retain the normal unrestricted local-development behavior.

### Import and prepare a dataset

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

The console intentionally does not auto-generate atoms for a partially
annotated dataset. Review the existing atoms and manually fill its missing
rows, or import a dataset with zero atoms and generate all of them.

### Launch and inspect a run

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
8. If the launch was accidental or takes too long, select **Cancel evaluation**
   in the active-job banner and confirm. The local evaluation worker stops and
   the run remains visible as `canceled`.
9. Open **Runs** to inspect status, answers, judgments, metrics, and the
   report. The underlying run API and workspace also preserve the manifest,
   preparation records, and other raw artifacts listed below.

Only one atom-generation or evaluation job runs at a time in the v0 job
manager. A conflicting launch returns HTTP 409. If the chat service restarts,
a previously queued or running job is marked `interrupted`; it is not resumed
automatically.

Canceling stops the local evaluation process and its local child processes. It
cannot recall a request a remote model provider has already accepted or undo an
external tool side effect. Canceled runs have no score, trend point, report, or
retry action.

The Runs page is reconstructed from persisted artifacts. A malformed,
unsupported, missing, or hash-mismatched workspace appears as an isolated
`invalid` entry instead of breaking the history list.

### Retry technical failures

The console exposes retry actions only for provider or runtime failures:

- an open generated atom draft with `preparation_failed` rows can retry those
  rows in place without regenerating successful candidates or modifying the
  imported parent dataset;
- a scored run with `execution_failed` or `evaluation_failed` attempts can
  create a complete successor run. Execution failures rerun Archi and the
  comparator, while evaluation failures reuse the verified terminal answer and
  rerun only the comparator.

Successful scored attempts are carried forward unchanged. The parent run
remains immutable, the successor records its direct parent and retry selection,
and both runs remain visible in history. Evaluation retries inherit the parent's
run and score worker counts. Scored attempts that merely fail the quality
threshold are not retryable.

Atom retries require `evaluations:manage`; evaluation retries require
`evaluations:run`. A draft or run without retryable technical failures creates
neither a job nor a new artifact.

### Compare history trends

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
- **Technical failure rate** plots `(execution_failed + evaluation_failed) /
  (scored + execution_failed + evaluation_failed)`. It is not the inverse of
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

### Inspect per-question latency

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

### Inspect per-attempt tool calls

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

## Understand states

There are two independent state machines.

### Run manifest states

| Manifest status   | Exact meaning                                                                                                                                                                                            |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prepared`      | Dataset validation and preparation finished and preparation artifacts were committed. Agent attempts are not yet committed. Individual input rows may still be skipped or have preparation failures.     |
| `run_completed` | The requested attempt slots for every prepared item are terminal as`answer_ready` or `execution_failed`, and `answers.jsonl` plus resolved agent inputs were committed. Scoring has not completed. |
| `scored`        | Scoring finished and`evaluation_results.jsonl`, `summary.json`, and `report.md` were committed. Some individual attempts may still have execution or evaluation failures.                          |

The manifest remains `prepared` while the agent loop is in progress because
`answers.jsonl` is written atomically. Likewise, `run_completed` means the
agent phase is complete, not that the overall evaluation passed.

### Console job states

| Job status          | Meaning                                                        |
| ------------------- | -------------------------------------------------------------- |
| `queued`            | Accepted by the console and waiting for its worker             |
| `running`           | Background work is executing                                   |
| `cancel_requested`  | Cancellation was accepted and worker termination is in progress |
| `canceled`          | The evaluation worker stopped and canceled history was persisted |
| `completed`         | The complete requested operation returned successfully         |
| `failed`            | The operation raised an error; inspect the job's error field   |
| `interrupted`       | The service restarted before a non-terminal job completed      |

For a console evaluation, the job normally remains `running` while the
workspace progresses through `prepared` and `run_completed`. The job becomes
`completed` only after the composite workflow returns a `scored` run.

## Understand scoring

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


## Run workspace artifacts

The current workspace schema is `qa-v1`. It introduced the canonical
`preparation.jsonl` artifact. Current run, score, and retry workflows require
`qa-v1`. The history console reads intact `qa-v0` workspaces through a
read-only adapter that joins their separate `prepared_items.jsonl` and
`preparation_results.jsonl` artifacts in memory. It never rewrites those
historical workspaces, and they cannot be retried.

| File                                  | Written in            | Contents                                                                                        |
| ------------------------------------- | --------------------- | ----------------------------------------------------------------------------------------------- |
| `input.snapshot.json` or `.jsonl` | Prepare               | Exact input bytes used by the run                                                               |
| `evaluator_profile.resolved.yaml`   | Prepare               | Fixed evaluator profile                                                                         |
| `preparation.jsonl`                 | Prepare               | One terminal record per input item, containing either runnable normalized data and fixed atoms, a skip, or a preparation failure |
| `agent_config.resolved.yaml`        | Run                   | Exact tested Archi config                                                                       |
| `agent_spec.resolved.md`            | Run                   | Exact tested agent spec and prompt                                                              |
| `answers.jsonl`                     | Run                   | One terminal `answer_ready` or `execution_failed` row per attempt slot, including tested-agent `duration_ms` and complete ordered tool-call query/response/error records with optional duration |
| `evaluation_results.jsonl`          | Score                 | Answers, atom judgments, rationales, metrics, or terminal failures                              |
| `summary.json`                      | Score                 | Machine-readable aggregate and per-item metrics plus provenance hashes                          |
| `report.md`                         | Score                 | Human-readable result summary                                                                   |
| `manifest.json`                     | Every completed phase | Schema/run version, state, phase timestamps/counts, phase worker counts, agent metadata, and artifact SHA-256 hashes |
| `console_metadata.json`             | Console only          | Display name, selected catalog IDs/spec, and launch worker counts                                |

The workspace is the reproducibility record. Keep it intact when comparing
runs, and archive it with any external version identifiers you need. The
current artifacts record tested-agent and tool-call latency but do not record
source-control commits, release gates, token usage, model prompts, evaluator
prompts, or reasoning traces. Tool queries and responses are complete.

## Rerunning and integrity protection

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

## Failure and lifecycle records

Preparation is item-scoped:

- `prepared`: the row has a valid fixed atom set;
- `skipped_time_sensitive`: the row was deliberately excluded;
- `preparation_failed`: atom extraction or validation failed for that row.

Agent and evaluator work is attempt-scoped:

- `answer_ready`: the agent produced a usable terminal string;
- `execution_failed`: agent construction, tool loading, invocation, or terminal
  answer validation failed;
- `scored`: comparison succeeded;
- `evaluation_failed`: the evaluator response was invalid, incomplete,
  unjudgeable, or otherwise failed.

A composite operation may reach `scored` even when individual rows or attempts
failed, because those failures are preserved as evidence. Decide acceptance
using the lifecycle counts and quality metrics, not the top-level state alone.

## Cost, concurrency, and data handling

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
