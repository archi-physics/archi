# Design: vLLM Provider Integration

## Architecture
- **Client:** `chatbot-main-gpu-agent` container.
- **Server:** `vllm-server` container (host-based or separate container).
- **Protocol:** HTTP/REST (OpenAI Schema).

## Technical Decisions

### Decision: Inherit from OpenAIProvider
Since vLLM is OpenAI-compatible, the `VLLMProvider` should inherit from Archi's `OpenAIProvider` (or `BaseProvider`) to reuse JSON mapping logic, but override the endpoint resolution to handle the Docker internal network.

### Decision: V100 Stability
Inject `NCCL_P2P_DISABLE=1` into the provider's connection logic if not already handled by the environment to ensure stable communication with older NVLink/PCIe topologies on V100s.

### Decision: Critical Docker Performance Tuning
To achieve bare-metal parity when running vLLM inside a container, the following runtime configurations MUST be enforced. These prevent the common "Docker Tax" on LLM inference.

#### 1. Shared Memory Access (`--ipc=host`)
vLLM utilizes NCCL for multi-GPU communication and PagedAttention for memory management. 
- **Requirement:** Containers must be started with `--ipc=host`.
- **Reason:** Docker’s default 64MB shm-size causes immediate crashes during Tensor Parallelism initialization. Using the host's IPC namespace provides the necessary memory bandwidth for inter-GPU coordination.

#### 2. Network Latency Optimization (`--network=host`)
- **Requirement:** Use `--network=host` for the `vllm-server` container where feasible.
- **Reason:** Bypasses the Docker bridge (docker0) and user-land proxy (docker-proxy), reducing request/response overhead by 0.5–2ms per call—critical for high-concurrency streaming applications.

#### 3. GPU Passthrough and Memory Locking
- **Requirement:** Ensure `--gpus all` and `--ulimit memlock=-1 --ulimit stack=67108864` are set.
- **Reason:** vLLM pre-allocates up to 90% of VRAM (default). Memlocking prevents the OS from swapping out these critical buffers, ensuring consistent P99 latencies.
