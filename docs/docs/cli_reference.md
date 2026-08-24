# CLI Reference

The Archi CLI provides commands to create, manage, and monitor deployments.

## Installation

The CLI is installed automatically with `pip install -e .` from the repository root. Verify with:

```bash
which archi
```

---

## Commands

### `archi create`

Create a new Archi deployment.

```bash
archi create --name <name> --config <config.yaml> --env-file <secrets.env> --services <services> [OPTIONS]
```

**Required options:**

| Option | Description |
|--------|-------------|
| `--name`, `-n` | Name of the deployment |
| `--config`, `-c` | Path to YAML configuration file (repeatable for multiple files) |

**Recommended options:**

| Option | Description |
|--------|-------------|
| `--env-file`, `-e` | Path to the secrets `.env` file |
| `--services`, `-s` | Comma-separated list of services to enable (e.g., `chatbot,uploader`) |

**Optional flags:**

| Option | Description | Default |
|--------|-------------|---------|
| `--config-dir`, `-cd` | Directory containing configuration files | — |
| `--podman`, `-p` | Use Podman instead of Docker | Docker |
| `--gpu-ids` | GPU configuration: `all` or comma-separated IDs (e.g., `0,1`) | None |
| `--tag`, `-t` | Image tag for built containers | `2000` |
| `--hostmode` | Use host network mode for all services | Off |
| `--verbosity`, `-v` | Logging verbosity level (0=quiet, 4=debug) | `3` |
| `--force`, `-f` | Overwrite existing deployment if it exists | Off |
| `--dry`, `--dry-run` | Validate and show what would be created without deploying | Off |

**Examples:**

```bash
# Basic deployment with Ollama
archi create -n my-archi -c config.yaml -e .secrets.env \
  --services chatbot --podman

# Full deployment with GPU and multiple services
archi create -n prod-archi -c config.yaml -e .secrets.env \
  --services chatbot,uploader,grafana \
  --gpu-ids all

# Dry run to validate configuration
archi create -n test -c config.yaml -e .secrets.env \
  --services chatbot --dry-run
```

**Notes:**

- The CLI checks that host ports are free before deploying. If a port is in use, adjust `services.*.external_port` in your config.
- The first deployment builds container images from scratch (may take several minutes). Subsequent deployments reuse images.
- Use `-v 4` for debug-level logging when troubleshooting.

---

### `archi delete`

Delete an existing deployment.

```bash
archi delete --name <name> [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--name`, `-n` | Name of the deployment to delete |
| `--rmi` | Also remove container images |
| `--rmv` | Also remove volumes |
| `--keep-files` | Keep deployment files on disk |
| `--list` | List all deployments |

**Examples:**

```bash
# Delete deployment and clean up everything
archi delete -n my-archi --rmi --rmv

# Delete but keep data volumes
archi delete -n my-archi --rmi
```

---

### `archi restart`

Restart a specific service in an existing deployment without restarting the entire stack.

```bash
archi restart --name <name> --service <service> [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--name`, `-n` | Name of the existing deployment | Required |
| `--service`, `-s` | Service to restart | `chatbot` |
| `--config`, `-c` | Updated configuration file(s) | — |
| `--config-dir`, `-cd` | Directory containing configuration files | — |
| `--env-file`, `-e` | Updated secrets file | — |
| `--no-build` | Restart without rebuilding the container image | Off |
| `--with-deps` | Also restart dependent services | Off |
| `--podman`, `-p` | Use Podman instead of Docker | Docker |
| `--verbosity`, `-v` | Logging verbosity (0-4) | `3` |

**Examples:**

```bash
# Quick config update (no rebuild needed)
archi restart -n my-archi --service chatbot --no-build

# Rebuild after code changes
archi restart -n my-archi --service chatbot -c updated_config.yaml

# Re-scrape data sources
archi restart -n my-archi --service data_manager

# Restart with updated secrets
archi restart -n my-archi --service chatbot -e new_secrets.env --no-build
```

---

### `archi list-services`

List all available services and data sources with descriptions.

```bash
archi list-services
```

---

### `archi list-deployments`

List all existing deployments.

```bash
archi list-deployments
```

---

### `archi evaluate`

Launch the benchmarking runtime to evaluate configurations against a set of questions and answers.

```bash
archi evaluate --name <name> --env-file <secrets.env> --config <config.yaml> [OPTIONS]
```

Supports the same flags as `create` (`--podman`, `--gpu-ids`, `--tag`, `--hostmode`, `--verbosity`, `--force`). Configuration files should define the `services.benchmarking` section.

**Example:**

```bash
archi evaluate -n benchmark \
  -c examples/benchmarking/benchmark_configs/example_conf.yaml \
  -e .secrets.env --gpu-ids all
```

See [Benchmarking](benchmarking.md) for full details on query format and evaluation modes.

---

### `archi eval qa`

Run source-neutral, atom-based evaluation without changing the legacy
`archi evaluate` workflow. The composite command validates and prepares the
dataset, runs the selected agent independently for every attempt, then scores
each usable answer:

For prerequisites, complete input schemas, agent and evaluator examples,
console usage, state semantics, scoring, and troubleshooting, see the
[Evaluation Guide](evaluation.md).

```bash
archi eval qa \
  --dataset questions.jsonl \
  --agent-config agent.yaml \
  --agent-spec agent.md \
  --output-dir evaluation-run/ \
  --attempts 4 \
  --run-workers 4 \
  --score-workers 8
```

Here, `evaluation-run/` is a directory used as the evaluation workspace. You
may choose any directory name or path. The command creates it if necessary and
stores all prepared data, Archi answers, scoring results, and reports inside
it.

`--run-workers` and `--score-workers` independently control concurrent agent
attempts and evaluator comparisons. Both default to `1` and accept `1` through
`16`. Scoring starts only after every agent attempt finishes. Higher values can
reduce wall-clock time for provider-bound work, but increase concurrent provider
requests and keep up to that many agent or evaluator runtimes in memory.

The YAML must define `services.chat_app.agent_class`, `default_provider`, and
`default_model`. The Markdown frontmatter selects tools and its body is the
agent system prompt.

A minimal Dataset V2 JSON document is:

```json
{
  "schema_version": "qa-dataset-v2",
  "items": [{
    "id": "quota-remaining",
    "question": "How much quota remains?",
    "answer": "The account has 2.8 TB remaining.",
    "time_sensitive": false
  }]
}
```

The dataset uses an explicit `qa-dataset-v2` JSON envelope or JSONL version
header. Every item requires an `id`, `question`, and `time_sensitive` value.
Static items require a string `answer`; live items require an oracle and omit
the unresolved answer. Optional fields include `category`, `answer_mode`,
`answer_source`, and static-item `expected_atoms`. `answer_source` is a
non-empty free-form string, so original Golden Set v2 values are accepted
directly. Canonical answers and gold atoms are stored in the evaluation
workspace and are never passed to the tested agent.

For live-state questions, use Dataset V2 and an evaluator-only MCP registry:

```json
{
  "schema_version": "qa-dataset-v2",
  "items": [{
    "id": "current-quota",
    "question": "How much quota remains?",
    "time_sensitive": true,
    "oracle": {
      "kind": "mcp",
      "calls": [{
        "id": "quota",
        "server": "operations-readonly",
        "tool": "get_quota",
        "arguments": {},
        "answer_fields": {"remaining": "/remaining"}
      }]
    }
  }]
}
```

Pass `--mcp-config qa_evaluation_mcp.yaml` to the composite, `prepare`, and `run`
commands. The registry maps the recipe alias to a deployment-owned MCP
transport and environment-backed credentials; it is never added to the tested
agent configuration. `--skip-live` on the composite or `prepare` command omits
live rows without making MCP or model calls. `score` accepts neither option.

`--mcp-config` is a direct filesystem path for the CLI process. The browser
console instead uses `services.chat_app.evaluations.mcp_config_path` in the
deployment YAML; that value is a host source path, resolved relative to the
deployment YAML and staged into the generated deployment's dedicated
`evaluation_config/` directory. It is not an in-container path and is separate
from the tested agent's top-level `mcp_servers` configuration.

The same workflow can be run as three separate stages. This is useful when you
want to inspect the generated atoms before running Archi or inspect Archi's
answers before scoring them.

#### 1. Prepare the dataset

```bash
archi eval qa prepare questions.jsonl \
  --evaluator-profile evaluator.yaml \
  --mcp-config qa_evaluation_mcp.yaml \
  --output-dir evaluation-run/
```

`prepare` validates the entire input dataset and creates the fixed gold atoms
that later stages use as the scoring reference. For each eligible row that
does not already contain `expected_atoms`, the atoms extractor selected by
`qa.atoms_extractor` in `evaluator.yaml` receives the row's `question` and
`answer`. It decomposes the canonical answer into independent,
judgeable obligations, marks each one as required or optional, and assigns it
an ID. If `--evaluator-profile` is omitted, the built-in evaluator profile is
used.

The command snapshots the input dataset and writes exactly one terminal record
per input item to `preparation.jsonl`. Prepared records contain the normalized
question, hidden canonical or resolved answer, fixed atoms, and atom source;
other records contain an intentional `--skip-live` omission or an item-scoped
oracle/atom-extraction failure. The
same artifact determines both run eligibility and preparation lifecycle
counts. The command also stores the resolved evaluator profile in the run
workspace. It does not run Archi or generate tested-agent answers.

#### 2. Run Archi

```bash
archi eval qa run evaluation-run/ \
  --agent-config agent.yaml \
  --agent-spec agent.md \
  --mcp-config qa_evaluation_mcp.yaml \
  --attempts 4 \
  --run-workers 4
```

The positional `evaluation-run/` argument is the prepared workspace directory created
by `prepare`. `run` reads the questions from that directory, so it does not take
the original dataset path again. It asks every prepared question once per
requested attempt. Each run worker owns and reuses a separate selected Archi
runtime, so mutable agent state is never shared by concurrent attempts. The
canonical answer and gold atoms are not passed to Archi.
The command stores the resolved agent configuration and spec, then writes each
verbatim terminal answer—or an `execution_failed` record—to `answers.jsonl`.
Each row includes total tested-agent latency and an ordered record for every
observed tool call: ordinal, name, status, complete query, complete response or
error when observed, and duration in milliseconds when available. Historical
rows with timing-only records remain supported by the evaluation console.

#### 3. Score the answers

```bash
archi eval qa score evaluation-run/ --score-workers 8
```

The positional `evaluation-run/` argument is the same workspace directory used by the
previous stages. `score` reads `preparation.jsonl` and `answers.jsonl` from it.
The evaluator selected by `qa.evaluator` compares each complete Archi
answer directly with every fixed gold atom and classifies the atom as
`entailed`, `not_mentioned`, or `contradicted`. Entailed atoms contribute `1`,
unmentioned atoms `0`, and contradicted atoms `-1`; their mean, floored at zero,
is the attempt's atom score. Required-atom recall is reported separately, and
an attempt passes only when every required atom is entailed. Optional atoms
affect the atom score but do not determine whether the attempt passes.

The command writes the per-atom judgments and per-attempt scores to
`evaluation_results.jsonl`, aggregate metrics to `summary.json`, and a readable
summary to `report.md`. It does not invoke Archi again.

#### Understanding the output

Current commands write workspace schema `qa-v2`. Input datasets use
`qa-dataset-v2`, including strict live recipes, materialized answers, and
`live_checks.jsonl` evidence.

At the end of a successful evaluation, the main result for a person to inspect
is `report.md` in the selected output directory:

```bash
less evaluation-run/report.md
```

The report shows the evaluated agent, attempts per item, overall and per-item
pass rates, mean atom score, required-atom recall, lifecycle counts, and the
pass rate of each gold atom. It is a summary rather than a full transcript. For
more detailed analysis or automation, the same workspace contains:

| File | Contents |
|------|----------|
| `summary.json` | Machine-readable aggregate and per-item metrics, lifecycle counts, atom pass rates, and configuration provenance hashes. |
| `evaluation_results.jsonl` | One terminal scoring record per attempt: the Archi answer, each atom's outcome and evaluator rationale, numeric scores and pass result, or an evaluation/execution failure. |
| `answers.jsonl` | The verbatim answer produced by Archi for every attempt, or its terminal execution error, plus total attempt latency and complete ordered tool-call query/response or error evidence with optional per-call latency before evaluator scoring. |
| `preparation.jsonl` | Exactly one terminal record per input item: normalized prepared questions with canonical or resolved answers and fixed gold atoms, intentional live omissions, or oracle/atom-extraction failures. |
| `live_checks.jsonl` | Ordered pre-run and post-run Dataset V2 oracle observations, answer hashes, metadata, call evidence, and item-scoped live failures. Static runs contain an empty file. |
| `input.snapshot.json` or `input.snapshot.jsonl` | An exact snapshot of the input dataset used for the evaluation. |
| `agent_config.resolved.yaml` and `agent_spec.resolved.md` | The resolved Archi configuration and exact agent spec used to generate the answers. |
| `evaluator_profile.resolved.yaml` | The resolved atoms-extractor and scoring-evaluator configuration. |
| `manifest.json` | The run ID, phase states, attempt count, artifact names, versions, and SHA-256 hashes used to detect changes to completed-phase inputs. |

The snapshot and prepared records in `preparation.jsonl` contain the hidden canonical answers or
gold atoms, and evaluator rationales may reveal the same information. Keep the
output directory private when those evaluation references are sensitive.

#### Supplying your own atoms

You can define the complete gold-atom set directly in a static dataset item:

```json
{
  "schema_version": "qa-dataset-v2",
  "items": [{
    "id": "quota-question",
    "question": "How much quota remains?",
    "answer": "The account has 2.8 TB remaining.",
    "time_sensitive": false,
    "expected_atoms": [
      {
        "id": "quota-remaining",
        "text": "The account has 2.8 TB remaining.",
        "required": true
      }
    ]
  }]
}
```

Each atom must have a unique, non-empty `id`, non-empty `text`, and a boolean
`required` value, and every row must contain at least one required atom.
Supplying `expected_atoms` replaces automatic atom generation for that row:
`prepare` validates and copies the supplied atoms without calling the atoms
extractor. The composite `archi eval qa` command detects this automatically as
well, so no separate atom-generation command is needed. It still performs the
preparation phase to validate and snapshot the dataset before running and
scoring it. In the staged workflow, `run` still requires that prepared
workspace, so `archi eval qa prepare` cannot be omitted entirely.

Across all three stages, `manifest.json` records phase state and SHA-256 hashes.
Editing an artifact from a completed phase makes the next phase fail closed.

`run` uses the selected Archi pipeline through its normal production interface.
The evaluation runner observes the pipeline's tool callbacks and stores complete
ordered query, response, error, status, and available duration evidence in
`answers.jsonl`. It does not add tool-schema preflight, secret redaction,
automatic retries, or evaluation-specific changes to agent behavior. Treat the
workspace as sensitive because tool evidence is not truncated or redacted.
When the selected agent spec enables `mcp` but normal pipeline construction
loads no MCP tools, that attempt is recorded as `execution_failed` before the
model is invoked.

Existing evaluator-owned outputs require `--overwrite`. Preparation overwrite
invalidates run and score artifacts, run overwrite invalidates score artifacts,
and score overwrite replaces only score/report artifacts. Execution failures
count as failed quality attempts. Evaluation failures remain visible but are
excluded from quality denominators. Lifecycle summaries include every defined
state, including states whose count is zero.

An evaluator profile is optional. The built-in profile is equivalent to:

```yaml
version: 1
qa:
  atoms_extractor: {provider: openai, model: gpt-5.6-terra}
  evaluator: {provider: openai, model: gpt-5.6-terra}
```

Use `--evaluator-profile profile.yaml` on the composite command or `prepare`.
If supplied again to `score`, its resolved content must match the prepared
profile. Evaluator temperature is fixed at zero; selected evaluator models must
accept `temperature=0`, otherwise the `archi eval qa` command fails.

---

### `archi install`

Create and optionally install a Helm chart for an Archi deployment.

```bash
archi install --name <name> --config <config.yaml> --templates-dir <dir> --env-file <secrets.env> --services <services> [OPTIONS]
```

Supports `--gpu-ids` with the same values as `create`. For explicit IDs such as `0,1`, the generated chart sets `NVIDIA_VISIBLE_DEVICES` and requests the matching `nvidia.com/gpu` count. For `all`, the chart sets `NVIDIA_VISIBLE_DEVICES=all`; adjust `gpu.count` in `values.yaml` if your Kubernetes cluster requires an explicit GPU limit.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ARCHI_DIR` | Override the deployment directory (default: `~/.archi`) |
| `OLLAMA_HOST` | Ollama server address (default: `http://localhost:11434`) |

---

## Troubleshooting

### Port Conflicts

If a port is already in use, the CLI will report an error. Adjust `services.*.external_port` in your config:

```yaml
services:
  chat_app:
    external_port: 7862  # default: 7861
  grafana:
    external_port: 3001  # default: 3000
```

### GPU Issues

GPU access requires NVIDIA drivers and the NVIDIA Container Toolkit.

**Podman:**
```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list
```

**Docker:**
```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

### Verbose Logging

Add `-v 4` to any command for debug-level output:

```bash
archi create [...] -v 4
```

### Multiple Deployments

Multiple deployments can run on the same machine. Container networks are separate, but be careful with external port assignments. See [Advanced Setup](advanced_setup_deploy.md#running-multiple-deployments-on-the-same-machine).
