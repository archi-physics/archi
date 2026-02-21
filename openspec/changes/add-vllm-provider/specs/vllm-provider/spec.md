## ADDED Requirements

### Requirement: VLLMProvider registered as a first-class provider type
The system SHALL register a `VLLM` provider type in the `ProviderType` enum and provider registry, distinct from the `LOCAL` provider. Pipeline configs SHALL reference vLLM models as `vllm/<model-name>`.

#### Scenario: Pipeline resolves vLLM model reference
- **WHEN** a pipeline config specifies `vllm/Qwen/Qwen2.5-7B-Instruct-1M` as a model
- **THEN** `BasePipeline._parse_provider_model()` splits it into provider `"vllm"` and model `"Qwen/Qwen2.5-7B-Instruct-1M"`, and `get_model()` returns a `ChatOpenAI` instance from `VLLMProvider`

#### Scenario: Provider name alias resolves to VLLM type
- **WHEN** `get_provider_by_name("vllm")` is called
- **THEN** it SHALL return a `VLLMProvider` instance (not `LocalProvider`)

### Requirement: VLLMProvider returns ChatOpenAI with correct defaults
The `VLLMProvider.get_chat_model()` SHALL return a `ChatOpenAI` instance configured with `base_url` defaulting to `http://localhost:8000/v1` and `api_key` defaulting to `"not-needed"`.

#### Scenario: Default base URL used when none configured
- **WHEN** `VLLMProvider` is instantiated with no `base_url` in config
- **THEN** `get_chat_model("my-model")` returns a `ChatOpenAI` with `base_url="http://localhost:8000/v1"`

#### Scenario: Custom base URL from config
- **WHEN** `VLLMProvider` is instantiated with `base_url="http://vllm-host:9000/v1"` in config
- **THEN** `get_chat_model("my-model")` returns a `ChatOpenAI` with that base URL

#### Scenario: Environment variable overrides config
- **WHEN** `VLLM_BASE_URL` environment variable is set
- **THEN** `VLLMProvider` SHALL use that value as base URL, overriding the config default

### Requirement: VLLMProvider discovers models dynamically
The `VLLMProvider.list_models()` SHALL query the vLLM server's `/v1/models` endpoint and return discovered models as `ModelInfo` objects.

#### Scenario: Server is reachable with loaded models
- **WHEN** `list_models()` is called and the vLLM server responds with a model list
- **THEN** each model is returned as a `ModelInfo` with `id`, `name`, and `display_name` populated from the response

#### Scenario: Server is unreachable
- **WHEN** `list_models()` is called and the vLLM server does not respond
- **THEN** it SHALL return the statically configured model list from `ProviderConfig.models`, or an empty list if none configured

### Requirement: VLLMProvider validates server connection
The `VLLMProvider.validate_connection()` SHALL check the vLLM server's health by hitting the `/v1/models` endpoint.

#### Scenario: Server is healthy
- **WHEN** `validate_connection()` is called and `/v1/models` returns HTTP 200
- **THEN** it SHALL return `True`

#### Scenario: Server is down
- **WHEN** `validate_connection()` is called and the request fails or times out
- **THEN** it SHALL return `False`

### Requirement: YAML config section for vLLM provider
The system SHALL support an `archi.providers.vllm` section in deployment YAML configs with fields: `enabled`, `base_url`, `default_model`, `models`.

#### Scenario: Config loaded from YAML
- **WHEN** a deployment config contains `archi.providers.vllm` with `enabled: true` and `base_url: http://gpu-node:8000/v1`
- **THEN** `_build_provider_config_from_payload()` SHALL construct a `ProviderConfig` with `provider_type=ProviderType.VLLM` and the specified fields
