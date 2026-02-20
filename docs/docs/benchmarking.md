# Benchmarking

Archi provides benchmarking functionality via the `archi evaluate` CLI command to measure retrieval and response quality.

## Evaluation Modes

Two modes are supported (can be used together):

### SOURCES Mode

Checks if retrieved documents contain the correct sources by comparing metadata fields.

- Default match field: `file_name` (configurable per-query)
- Override with `sources_match_field` in the queries file

### RAGAS Mode

Uses the [Ragas](https://docs.ragas.io/en/stable/concepts/metrics/) evaluator for four metrics:

- **Answer relevancy**: How relevant the answer is to the question
- **Faithfulness**: Whether the answer is grounded in the retrieved context
- **Context precision**: How relevant the retrieved documents are
- **Context relevancy**: How much of the retrieved context is useful

---

## Preparing the Queries File

Provide questions, expected answers, and correct sources in JSON format:

```json
[
  {
    "question": "Does Jorian Benke work with the PPC?",
    "sources": [
      "https://ppc.mit.edu/blog/2025/07/14/welcome-our-first-ever-in-house-masters-student/",
      "CMSPROD-42"
    ],
    "answer": "Yes, Jorian works with the PPC and her topic is Lorentz invariance.",
    "source_match_field": ["url", "ticket_id"]
  }
]
```

| Field | Required | Description |
|-------|----------|-------------|
| `question` | Yes | The question to ask |
| `sources` | Yes | List of source identifiers (URLs, ticket IDs, etc.) |
| `answer` | Yes | Expected answer (used for RAGAS evaluation) |
| `source_match_field` | No | Metadata fields to match sources against (defaults to config value) |

See `examples/benchmarking/queries.json` for a complete example.

---

## Configuration

```yaml
services:
  benchmarking:
    queries_path: examples/benchmarking/queries.json
    out_dir: bench_out
    modes:
      - "RAGAS"
      - "SOURCES"
    timeout: 180
    batch_size: 10
    mode_settings:
      sources:
        default_match_field: ["file_name"]
      ragas_settings:
        provider: OpenAI
        evaluation_model_settings:
          model_name: gpt-4o
        embedding_model: OpenAI
```

| Key | Default | Description |
|-----|---------|-------------|
| `queries_path` | — | Path to the queries JSON file |
| `out_dir` | — | Output directory for results (must exist) |
| `modes` | — | List of evaluation modes (`RAGAS`, `SOURCES`) |
| `timeout` | `180` | Max seconds per QA pair for RAGAS evaluation |
| `batch_size` | Ragas default | Number of QA pairs to evaluate at once |

### RAGAS Settings

| Key | Description |
|-----|-------------|
| `provider` | One of: `OpenAI`, `HuggingFace`, `Ollama`, `Anthropic` |
| `evaluation_model_settings.model_name` | LangChain model name for evaluation |
| `evaluation_model_settings.base_url` | For Ollama: address of the running server |
| `embedding_model` | `OpenAI` or `HuggingFace` |

---

## Running

Evaluate one or more configurations:

```bash
# Single config file
archi evaluate -n benchmark -c config.yaml -e .secrets.env

# Directory of configs (for comparing hyperparameters)
archi evaluate -n benchmark -cd configs/ -e .secrets.env

# With GPU support
archi evaluate -n benchmark -c config.yaml -e .secrets.env --gpu-ids all
```

Make sure the `out_dir` exists before running.

---

## Results

Results are saved in a timestamped subdirectory of `out_dir` (e.g., `bench_out/2042-10-01_12-00-00/`).

To analyze results, see `scripts/benchmarking/` which contains:

- Plotting functions
- An IPython notebook with usage examples (`benchmark_handler.ipynb`)
