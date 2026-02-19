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

#### vLLM provider settings

The vLLM provider is configured under `services.chat_app.providers.vllm` in your config YAML. At minimum you need `enabled` and `default_model`:

```yaml
services:
  chat_app:
    providers:
      vllm:
        enabled: true
        base_url: http://localhost:8000/v1
        default_model: "Qwen/Qwen3-8B"
        tool_call_parser: hermes          # optional, default: hermes
        models:
          - "Qwen/Qwen3-8B"
```

| Setting | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Enable the vLLM provider |
| `base_url` | string | `http://localhost:8000/v1` | vLLM server OpenAI-compatible endpoint |
| `default_model` | string | `Qwen/Qwen2.5-7B-Instruct-1M` | HuggingFace model ID to serve |
| `tool_call_parser` | string | `hermes` | Parser for structured tool calls (`hermes`, `mistral`, `llama3_json`) |
| `models` | list | — | Available model IDs for the UI model selector |

#### vLLM Server Tuning

When archi manages the vLLM sidecar container (deployed via `--services vllm-server`), you can configure server launch arguments alongside the provider settings above. Each key is translated to a vLLM CLI flag at container startup. All keys are optional — when omitted, vLLM's own defaults apply.

> **Note**: These keys only affect the managed vLLM sidecar container. If you are pointing `base_url` at an external vLLM server, configure that server directly instead.

| Key | Type | Default | When to change |
|---|---|---|---|
| `gpu_memory_utilization` | float | `0.9` | Model barely fits in VRAM, or you want to reserve GPU memory for other processes |
| `max_model_len` | int | model default | Reduce context window to lower memory usage, or increase it for long-document workloads |
| `tensor_parallel_size` | int | `1` | Shard a large model across multiple GPUs |
| `dtype` | string | `auto` | Force a specific weight precision (`float16`, `bfloat16`) instead of auto-detection |
| `quantization` | string | none | Run quantized model weights (`awq`, `gptq`, `fp8`) to reduce memory |
| `enforce_eager` | bool | `false` | Disable CUDA graph compilation to save memory at the cost of throughput |
| `max_num_seqs` | int | `256` | Limit concurrent sequences to reduce memory pressure under high load |
| `enable_prefix_caching` | bool | `true` | Disable KV cache prefix sharing if it causes issues with your model |

##### Complete config example

A single-GPU deployment with memory tuning:

```yaml
services:
  chat_app:
    providers:
      vllm:
        enabled: true
        base_url: http://localhost:8000/v1
        default_model: "Qwen/Qwen3-8B"
        tool_call_parser: hermes
        models:
          - "Qwen/Qwen3-8B"
        gpu_memory_utilization: 0.85
        max_model_len: 8192
```

##### `engine_args` passthrough

For any vLLM flag not covered by a named key above, use the `engine_args` map. Each entry is passed as `--<key> <value>` to the vLLM server. Keys use kebab-case matching vLLM's CLI flags. For boolean flags that take no argument (e.g. `--trust-remote-code`), use an empty string as the value. Do not duplicate flags that already have a named key above.

```yaml
services:
  chat_app:
    providers:
      vllm:
        engine_args:
          swap-space: 8        # CPU swap space per GPU in GiB (default: 4)
          seed: 42
          trust-remote-code: "" # bare flag (no value) — use "" for boolean flags
```

##### Multi-GPU example

Sharding a 30B model across 4 GPUs:

```yaml
services:
  chat_app:
    providers:
      vllm:
        enabled: true
        base_url: http://localhost:8000/v1
        default_model: "Qwen/Qwen3-30B-A3B-Instruct"
        tool_call_parser: hermes
        models:
          - "Qwen/Qwen3-30B-A3B-Instruct"
        gpu_memory_utilization: 0.92
        tensor_parallel_size: 4
        max_model_len: 16384
        dtype: bfloat16
        engine_args:
          swap-space: 8
```

Deploy with all GPUs:

```bash
archi create -n my-deployment \
  -c config.yaml \
  --services chatbot,vllm-server \
  --gpu-ids 0,1,2,3
```

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

- **Insufficient VRAM**: The model doesn't fit in GPU memory. Options:
    - Lower `gpu_memory_utilization` (e.g. `0.7`) to leave headroom for other processes
    - Set `max_model_len` to a smaller value (e.g. `4096`) to reduce KV cache memory
    - Add `quantization: awq` or `quantization: gptq` if the model has quantized weights available
    - Set `enforce_eager: true` to disable CUDA graphs (saves memory, reduces throughput)
    - Increase `tensor_parallel_size` and add more GPUs via `--gpu-ids`
    - Try a smaller model
- **Missing NVIDIA runtime**: Ensure the NVIDIA Container Toolkit is installed and configured.
- **/dev/shm too small**: vLLM warns at startup if shared memory is below 1 GB. The container uses `ipc: host` by default, but if that is restricted, increase `shm_size`.
- **Invalid engine argument**: If the vLLM log shows `unrecognized arguments`, check for typos in `engine_args` keys (must be kebab-case, e.g. `swap-space` not `swap_space`) or boolean flags that need an empty-string value (`""`).

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

The first request after startup may be slow (30-60s) while vLLM compiles CUDA kernels and warms up. Subsequent requests will be significantly faster. The chatbot's `depends_on` health check ensures it doesn't send requests before vLLM is ready, but the health check only confirms the server is listening — not that the first compilation is complete. If startup compilation time is a problem, set `enforce_eager: true` to skip CUDA graph compilation (at the cost of lower throughput).
