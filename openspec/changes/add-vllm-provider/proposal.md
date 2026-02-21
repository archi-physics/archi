# Proposal: Add vLLM Provider
## Intent
Enable high-throughput local inference on NVIDIA V100 GPUs using the vLLM engine. This provides an OpenAI-compatible alternative to the current Ollama and external API providers.

## Scope
- New provider class `VLLMProvider` in `src/archi/providers/`.
- Integration into the provider factory.
- Support for streaming and non-streaming chat completions.
- V100-specific configuration (NCCL flags).

## Constraints
- MUST use OpenAI-compatible API format.
- MUST support the `base_url` parameter for remote container access.
- MUST handle V100-specific environment variables for stability.