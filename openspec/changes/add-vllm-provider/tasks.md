Phase 1: Provider (thin client layer)
[x] 1. Add ProviderType.VLLM to enum: Add VLLM = "vllm" to ProviderType in src/archi/providers/base.py.

[x] 2. Create VLLMProvider class: New file src/archi/providers/vllm_provider.py. Inherit from BaseProvider (not OpenAIProvider — avoids coupling to OpenAI's default model list and API key logic). Default base_url http://localhost:8000/v1, api_key defaults to "not-needed". get_chat_model() returns ChatOpenAI with correct base_url. list_models() hits /v1/models for dynamic discovery. validate_connection() health-checks /v1/models.

[x] 3. Register provider: Update src/archi/providers/__init__.py — add to _ensure_providers_registered(), repoint "vllm" alias from ProviderType.LOCAL to ProviderType.VLLM in name_map.

[x] 4. Config schema support: Support archi.providers.vllm section in YAML (fields: enabled, base_url, default_model, models). Wire into _build_provider_config_from_payload() in src/interfaces/chat_app/app.py.

Phase 2: Infrastructure (server-side)
[x] 5. Register vllm-server in ServiceRegistry: New ServiceDefinition in src/cli/service_registry.py. GPU-dependent, port 8000 default, no volume required (model weights bind-mounted or cached).

[x] 6. Docker Compose template for vllm-server: Base image vllm/vllm-openai or custom from base-pytorch-image. Server command: python -m vllm.entrypoints.openai.api_server --model <model>. Environment: NCCL_P2P_DISABLE=1 (V100 stability). Runtime: ipc: host, ulimits (memlock: -1, stack: 67108864), GPU passthrough via deploy.resources.reservations.devices.

[x] 7. Inter-container networking: If vllm-server uses network_mode: host, chatbot must reach it via host IP not Docker DNS. Expose VLLM_BASE_URL env var to chatbot container. VLLMProvider reads base_url from config or VLLM_BASE_URL env fallback.

[x] 8. CLI integration: Wire vllm-server into archi create --services. Leverage existing --gpu-ids flag for GPU passthrough. Support model name configuration (which model the server loads).

Phase 3: Validation
[x] 9. Unit tests for VLLMProvider: Mock /v1/models response for list_models(). Verify ChatOpenAI instantiation with correct base_url and api_key. Verify validate_connection() success/failure paths.

[x] 10. Startup health check: Compose healthcheck or entrypoint script for vllm-server. Log warning if /dev/shm < 1GB. Chatbot depends_on vllm-server with condition: service_healthy.

[x] 11. Smoke test: Extend existing smoke test infrastructure. Verify end-to-end: deploy → ingest → query via vLLM provider.
