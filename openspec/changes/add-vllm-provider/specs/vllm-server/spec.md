## ADDED Requirements

### Requirement: vllm-server registered as a deployable service
The system SHALL register a `vllm-server` service in the `ServiceRegistry` that can be enabled via `archi create --services`.

#### Scenario: User deploys with vllm-server
- **WHEN** `archi create --name my-bot --services chatbot,vllm-server --gpu-ids all` is run
- **THEN** the deployment directory SHALL contain a docker-compose service block for `vllm-server` with GPU passthrough

#### Scenario: vllm-server not requested
- **WHEN** `archi create` is run without `vllm-server` in --services
- **THEN** no vllm-server service block SHALL be generated

### Requirement: vllm-server container runs with required runtime config
The generated docker-compose service for `vllm-server` SHALL include `ipc: host`, `ulimits` (memlock unlimited, stack 67108864), and GPU device reservations.

#### Scenario: Compose file generated with runtime config
- **WHEN** the deployment includes `vllm-server`
- **THEN** the docker-compose YAML for vllm-server SHALL contain `ipc: host`, `ulimits.memlock.soft: -1`, `ulimits.memlock.hard: -1`, `ulimits.stack: 67108864`, and `deploy.resources.reservations.devices` with GPU capabilities

### Requirement: V100 stability via NCCL environment variable
The vllm-server container SHALL set `NCCL_P2P_DISABLE=1` in its environment to ensure stability on V100 GPU topologies.

#### Scenario: NCCL flag present in container environment
- **WHEN** vllm-server is deployed
- **THEN** the container environment SHALL include `NCCL_P2P_DISABLE=1`

### Requirement: vllm-server supports host networking mode
The vllm-server compose service SHALL use `network_mode: host` by default to minimize inference latency.

#### Scenario: Host networking enabled
- **WHEN** vllm-server is deployed with default settings
- **THEN** the compose service SHALL include `network_mode: host`

#### Scenario: Chatbot resolves vllm-server via host
- **WHEN** vllm-server uses host networking and chatbot uses bridge networking
- **THEN** the chatbot container SHALL receive a `VLLM_BASE_URL` environment variable pointing to the host IP and vLLM port

### Requirement: vllm-server startup health check
The vllm-server compose service SHALL include a healthcheck that verifies the `/v1/models` endpoint is responding before dependent services start.

#### Scenario: Healthy startup
- **WHEN** vllm-server finishes loading the model and `/v1/models` returns HTTP 200
- **THEN** the healthcheck SHALL pass and dependent services (chatbot) SHALL start

#### Scenario: Slow model load
- **WHEN** vllm-server takes longer than the healthcheck interval to load
- **THEN** the healthcheck SHALL retry until the model is loaded or the timeout is reached

### Requirement: Shared memory size warning
The vllm-server startup SHALL log a warning if `/dev/shm` is smaller than 1GB.

#### Scenario: Insufficient shared memory
- **WHEN** vllm-server starts and `/dev/shm` is less than 1GB
- **THEN** a warning SHALL be logged indicating that `ipc: host` or a larger `shm_size` is required for stable multi-GPU inference
