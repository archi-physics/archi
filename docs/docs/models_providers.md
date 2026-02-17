# Models & Providers

Archi uses a **provider-based architecture** for LLM access. Each provider wraps a specific LLM service and exposes a unified interface for model listing, connection validation, and chat model creation.

## Provider Architecture

All providers extend the `BaseProvider` abstract class and are registered in a global provider registry. The system supports five provider types:

| Provider | Type | API Key Env Var | Default Model | LangChain Backend |
|----------|------|----------------|---------------|-------------------|
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-4o` | `ChatOpenAI` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` | `ChatAnthropic` |
| Google Gemini | `gemini` | `GOOGLE_API_KEY` | `gemini-2.0-flash` | `ChatGoogleGenerativeAI` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | `anthropic/claude-3.5-sonnet` | `ChatOpenAI` (custom base URL) |
| Local (Ollama/vLLM) | `local` | N/A | Dynamic (fetched from server) | `ChatOllama` or `ChatOpenAI` |

### Key Concepts

- **`ProviderType`**: An enum of supported provider names (`OPENAI`, `ANTHROPIC`, `GEMINI`, `OPENROUTER`, `LOCAL`).
- **`ProviderConfig`**: A dataclass holding provider settings — type, API key, base URL, enabled state, models list, and extra kwargs.
- **`ModelInfo`**: Describes a model's capabilities — context window, tool support, streaming support, vision support, and max output tokens.
- **Provider Registry**: Providers are lazily registered at first use. Factory functions (`get_provider`, `get_model`) handle instantiation and caching.

---

## Configuring Providers

Providers are configured per-service in your deployment's configuration file. Each service can specify a default provider and model, plus provider-specific settings.

## Quick Start by Provider

Use this flow if you want to start with a provider other than Ollama:

1. Start from `examples/deployments/basic-ollama/config.yaml` and copy it to a new file.
2. Change `services.chat_app.default_provider` and `services.chat_app.default_model`.
3. If you are not using `local`, remove `services.chat_app.providers.local`.
4. Add provider-specific `services.chat_app.providers.<provider>` settings only when needed (for example `local` mode/base URL).
5. Put required secrets in your `.env` file.
6. Run `archi create --name my-archi --config <your-config>.yaml --podman --env-file .secrets.env --services chatbot`.

Minimal provider snippets for `services.chat_app`:

### OpenAI

```yaml
services:
  chat_app:
    default_provider: openai
    default_model: gpt-4o
```

Required secret: `OPENAI_API_KEY`

### Anthropic

```yaml
services:
  chat_app:
    default_provider: anthropic
    default_model: claude-sonnet-4-20250514
```

Required secret: `ANTHROPIC_API_KEY`

### Google Gemini

```yaml
services:
  chat_app:
    default_provider: gemini
    default_model: gemini-2.0-flash
    providers:
      gemini:
        enabled: true
```

Required secret: `GOOGLE_API_KEY`

### OpenRouter

OpenRouter uses an OpenAI-compatible API to access models from multiple providers.

```yaml
services:
  chat_app:
    default_provider: openrouter
    default_model: anthropic/claude-3.5-sonnet
```

Required secret: `OPENROUTER_API_KEY`

Optional secrets: `OPENROUTER_SITE_URL`, `OPENROUTER_APP_NAME`

### Local Models (Ollama)

```yaml
services:
  chat_app:
    default_provider: local
    default_model: llama3.2
    providers:
      local:
        base_url: http://localhost:11434
        mode: ollama
        models:
          - llama3.2
```

### Local OpenAI-compatible server (vLLM, LM Studio, etc.)

```yaml
services:
  chat_app:
    default_provider: local
    default_model: my-model
    providers:
      local:
        base_url: http://localhost:8000/v1
        mode: openai_compat
        models:
          - my-model
```

Secret usually not required unless your local server enforces API auth.

The `local` provider supports two modes:

- **`ollama`** (default): Uses `ChatOllama`. Models are dynamically fetched from the Ollama server's `/api/tags` endpoint.
- **`openai_compat`**: Uses `ChatOpenAI` with a custom base URL. Suitable for vLLM, LM Studio, or other OpenAI-compatible servers.

> **Note:** For GPU setup with local models, see [Advanced Setup & Deployment](advanced_setup_deploy.md#running-llms-locally-on-your-gpus).

---

## Embedding Models

Embeddings convert text into numerical vectors for semantic search. Configure these in the `data_manager` section:

### OpenAI Embeddings

```yaml
data_manager:
  embedding_name: OpenAIEmbeddings
  embedding_class_map:
    OpenAIEmbeddings:
      class: OpenAIEmbeddings
      kwargs:
        model: text-embedding-3-small
      similarity_score_reference: 10
```

Requires `OPENAI_API_KEY` in your secrets file.

### HuggingFace Embeddings

```yaml
data_manager:
  embedding_name: HuggingFaceEmbeddings
  embedding_class_map:
    HuggingFaceEmbeddings:
      class: HuggingFaceEmbeddings
      kwargs:
        model_name: sentence-transformers/all-MiniLM-L6-v2
        model_kwargs:
          device: cpu
        encode_kwargs:
          normalize_embeddings: true
      similarity_score_reference: 10
```

Uses HuggingFace models locally. Optionally requires `HUGGINGFACEHUB_API_TOKEN` for private models.

---

## Bring Your Own Key (BYOK)

BYOK allows users to provide their own API keys for LLM providers at runtime, enabling cost attribution, provider flexibility, and privacy.

> **Supported Providers:** BYOK session keys work with all configured provider types (OpenAI, Anthropic, OpenRouter, Gemini, etc.). The Settings UI shows status indicators for each provider.

### Key Hierarchy

API keys are resolved in the following order (highest priority first):

1. **Session Storage**: User-provided keys via the Settings UI (BYOK)
2. **Environment Variables / Docker Secrets**: Admin-configured keys (e.g., `OPENAI_API_KEY` or keys mounted at `/run/secrets/`)

> **Note:** When a user provides a session key, it overrides any environment-level key for that user's requests. Environment keys serve as the default fallback for users who have not configured their own key.

### Using BYOK in the Chat Interface

1. Open the **Settings** modal (gear icon in the header)
2. Expand the **API Keys** section
3. Enter your API key for each provider you want to use
4. Click **Save** to store it in your session
5. Select your preferred **Provider** and **Model** from the dropdowns
6. Start chatting

**Status Indicators:**

| Icon | Meaning |
|------|---------|
| ✓ Env | Key configured via environment variable (cannot be changed) |
| ✓ Session | Key configured via your session |
| ○ | No key configured |

### BYOK API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/providers/keys` | GET | Get status of all provider keys |
| `/api/providers/keys/set` | POST | Set a session API key (validates before storing) |
| `/api/providers/keys/clear` | POST | Clear a session API key |

### Security Considerations

- Keys are never logged or echoed
- Keys are session-scoped and cleared on logout or session expiry
- HTTPS is strongly recommended for production — see [HTTPS Configuration](advanced_setup_deploy.md#https-configuration-for-production)
