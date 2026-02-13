# vLLM Provider

Run open-weight models on your own GPUs using [vLLM](https://docs.vllm.ai/) as an inference backend. Archi deploys vLLM as a **sidecar container** alongside the chatbot — no external server management required.

## Why vLLM?

| | vLLM | Ollama | API providers |
|---|---|---|---|
| **Throughput** | High (PagedAttention, continuous batching) | Moderate | N/A (cloud) |
| **Multi-GPU** | Tensor-parallel across GPUs | Single GPU | N/A |
| **Tool calling** | Supported (with parser flag) | Model-dependent | Supported |
| **Cost** | Hardware only | Hardware only | Per-token |
| **Privacy** | Data stays on-premises | Data stays on-premises | Data leaves your network |

vLLM is the best fit when you need high-throughput local inference, multi-GPU support, or full data privacy with tool-calling capabilities.

## Prerequisites

- NVIDIA GPUs with sufficient VRAM for your chosen model
- NVIDIA drivers and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed
- Container runtime configured for GPU access (see [Advanced Setup](advanced_setup_deploy.md#running-llms-locally-on-your-gpus))

## Quick Start

### 1. Configure your deployment

In your config YAML, reference models with the `vllm/` provider prefix:

```yaml
archi:
  pipeline_map:
    CMSCompOpsAgent:
      models:
        required:
          agent_model: vllm/Qwen/Qwen3-8B

services:
  vllm:
    model: Qwen/Qwen3-8B          # HuggingFace model ID
    tool_parser: hermes            # tool-call parser (optional)
```

> **Model naming**: vLLM uses HuggingFace model IDs (e.g. `Qwen/Qwen3-8B`), not Ollama-style names (e.g. `Qwen/Qwen3:8B`). Make sure the model ID matches what is available on HuggingFace Hub.

### 2. Deploy

```bash
archi create -n my-deployment \
  -c config.yaml \
  -e .env \
  --services chatbot,vllm-server \
  --gpu-ids all
```

The CLI will:

1. Add the `vllm-server` sidecar to Docker Compose
2. Wire `VLLM_BASE_URL` into the chatbot container
3. Set the chatbot to wait for vLLM's health check before starting

### 3. Verify

Once the deployment is up, check the vLLM server:

```bash
curl http://localhost:8000/v1/models
```

You should see your model listed in the response.

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Docker Compose stack                │
│                                                  │
│  ┌──────────┐    ┌────────────┐    ┌──────────┐ │
│  │ chatbot  │───>│ vllm-server│    │ postgres │ │
│  │ (Flask)  │    │ (sidecar)  │    │          │ │
│  └──────────┘    └────────────┘    └──────────┘ │
│      :7861           :8000             :5432     │
│                    GPU access                    │
└─────────────────────────────────────────────────┘
```

The vLLM server runs as a **sidecar** — a companion container in the same Compose stack. It:

- Exposes an OpenAI-compatible `/v1` API on port 8000
- Receives requests from the chatbot over the Docker network
- Loads the model onto GPU at startup and serves it continuously
- Reports health via `/v1/models` (chatbot waits for this before starting)

The chatbot talks to vLLM using the same `ChatOpenAI` LangChain class it would use for the OpenAI API. From the pipeline's perspective, vLLM looks identical to a remote OpenAI endpoint.

## Configuration Reference

### Config YAML

#### Model references

Anywhere a model is referenced in `pipeline_map`, use the `vllm/` prefix:

```yaml
archi:
  pipeline_map:
    CMSCompOpsAgent:
      models:
        required:
          agent_model: vllm/Qwen/Qwen3-8B
```

The part after `vllm/` must match the HuggingFace model ID that vLLM is serving.

#### vLLM service settings

```yaml
services:
  vllm:
    model: Qwen/Qwen3-8B      # Model to load (required)
    tool_parser: hermes         # Tool-call parser backend (optional)
```

| Setting | Default | Description |
|---|---|---|
| `model` | `Qwen/Qwen2.5-7B-Instruct-1M` | HuggingFace model ID to serve |
| `tool_parser` | `hermes` | Parser for structured tool calls. Common values: `hermes` (Qwen, Hermes models), `mistral`, `llama3_json` |

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VLLM_BASE_URL` | `http://vllm-server:8000/v1` | Override the vLLM server URL (auto-set by the CLI) |

You generally don't need to set `VLLM_BASE_URL` manually — the CLI injects it into the chatbot container. It is useful if you are running vLLM on a separate host.

### Host Networking

When deploying with `--hostmode`, the vLLM server uses `network_mode: host` and all services communicate via `localhost`. Without host mode, services communicate via Docker DNS (`vllm-server:8000`).

## Tool Calling

vLLM supports function/tool calling for ReAct agents, but requires explicit server flags. Archi configures these automatically:

- `--enable-auto-tool-choice` — enables the tool calling pathway
- `--tool-call-parser <parser>` — selects the parser for the model family

The `tool_parser` setting should match your model's chat template:

| Model family | Parser |
|---|---|
| Qwen (Qwen2.5, Qwen3) | `hermes` |
| Mistral / Mixtral | `mistral` |
| Llama 3 | `llama3_json` |

If tool calling is not needed for your use case, these flags are harmless and can be left at defaults.

## Smoke Testing

To run smoke tests against a vLLM deployment:

```bash
export SMOKE_PROVIDER=vllm
export SMOKE_VLLM_BASE_URL=http://localhost:8000/v1
export SMOKE_VLLM_MODEL=Qwen/Qwen3-8B
scripts/dev/run_smoke_preview.sh my-deployment
```

This runs preflight checks (verifies vLLM is reachable) followed by a basic chat completion test through the chatbot endpoint.

## Troubleshooting

### vLLM server not starting

**Symptom**: Container exits immediately or stays in a restart loop.

**Check logs**:
```bash
docker logs vllm-server-<deployment-name>
```

Common causes:

- **Insufficient VRAM**: The model doesn't fit in GPU memory. Try a smaller model or use `--gpu-ids` to add more GPUs.
- **Missing NVIDIA runtime**: Ensure the NVIDIA Container Toolkit is installed and configured.
- **/dev/shm too small**: vLLM warns at startup if shared memory is below 1 GB. The container uses `ipc: host` by default, but if that is restricted, increase `shm_size`.

### Chatbot can't reach vLLM

**Symptom**: `ConnectionError: Name or service not known` or `Connection refused`.

- Verify both containers are on the same Docker network (default when not using `--hostmode`).
- Check that `VLLM_BASE_URL` in the chatbot container resolves correctly:
  ```bash
  docker exec <chatbot-container> curl http://vllm-server:8000/v1/models
  ```
- If using `--hostmode`, ensure `VLLM_BASE_URL` uses `localhost` instead of `vllm-server`.

### Model not found (404)

**Symptom**: `Error: model 'Qwen/Qwen3:8B' does not exist`.

vLLM uses HuggingFace model IDs, not Ollama-style names. Check:

- Config uses dashes, not colons: `vllm/Qwen/Qwen3-8B` (not `Qwen/Qwen3:8B`)
- The model ID matches exactly what vLLM is serving (`curl localhost:8000/v1/models`)

### Tool calling returns 400

**Symptom**: `400 Bad Request: "auto" tool choice requires --enable-auto-tool-choice`.

This means the vLLM server wasn't started with tool calling flags. If you are deploying through the CLI, this is handled automatically. If running vLLM manually, add:

```bash
--enable-auto-tool-choice --tool-call-parser hermes
```

### Slow first response

The first request after startup may be slow (30-60s) while vLLM compiles CUDA kernels and warms up. Subsequent requests will be significantly faster. The chatbot's `depends_on` health check ensures it doesn't send requests before vLLM is ready, but the health check only confirms the server is listening — not that the first compilation is complete.
