# Secrets Management

Secrets are sensitive values — API keys, passwords, and tokens — that Archi needs at runtime but should never be stored in configuration files or version control. This guide covers how secrets work end-to-end.

## Quick Start

Secrets live in a plain `.env` file with `KEY=value` pairs. At minimum you need a database password:

```bash
echo "PG_PASSWORD=my_strong_password" > ~/.secrets.env
```

Pass it to the CLI when creating a deployment:

```bash
archi create --name my-archi --config config.yaml --env-file ~/.secrets.env --services chatbot
```

Add more secrets as needed for your providers and services:

```bash
PG_PASSWORD=my_strong_password
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## How Secrets Work

When you run `archi create` or `archi restart`, secrets flow through several stages:

```
~/.secrets.env                        # 1. You provide a .env file
    ↓
SecretsManager                        # 2. CLI parses and validates
    ↓
~/.archi/archi-{name}/
├── secrets/openai_api_key.txt        # 3. Written as individual files
├── secrets/pg_password.txt
└── .env                              # 4. Also written as combined .env
    ↓
Docker/Podman secrets                 # 5. Mounted at /run/secrets/
    ↓
read_secret("OPENAI_API_KEY")         # 6. Services read at runtime
```

### The `*_FILE` pattern

Inside containers, each secret is available two ways:

1. **File path**: An environment variable like `OPENAI_API_KEY_FILE` points to `/run/secrets/openai_api_key`
2. **Direct value**: The secret value is also set directly via the `.env` file

Archi's `read_secret()` utility checks both, preferring the file-based approach:

```python
from src.utils.env import read_secret

api_key = read_secret("OPENAI_API_KEY")
# 1. Checks OPENAI_API_KEY_FILE env var → reads from /run/secrets/openai_api_key
# 2. Falls back to OPENAI_API_KEY env var
# 3. Returns empty string if neither exists
```

## Secret Reference

### Always Required

| Secret | Description |
|--------|-------------|
| `PG_PASSWORD` | PostgreSQL database password |

### LLM Providers

These are required automatically when your pipeline config references models from the corresponding provider.

| Secret | Provider |
|--------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GOOGLE_API_KEY` | Google Gemini |
| `OPENROUTER_API_KEY` | OpenRouter |
| `HUGGING_FACE_HUB_TOKEN` | HuggingFace (warning only, not enforced) |

### Services

Required when the corresponding service is enabled via `--services`.

| Service | Required Secrets |
|---------|-----------------|
| Grafana | `GRAFANA_PG_PASSWORD` |
| Grader | `ADMIN_PASSWORD` |
| Piazza | `PIAZZA_EMAIL`, `PIAZZA_PASSWORD`, `SLACK_WEBHOOK` |
| Mattermost | `MATTERMOST_WEBHOOK`, `MATTERMOST_CHANNEL_ID_READ`, `MATTERMOST_CHANNEL_ID_WRITE`, `MATTERMOST_PAK` |
| Redmine-Mailer | `IMAP_USER`, `IMAP_PW`, `REDMINE_USER`, `REDMINE_PW`, `SENDER_SERVER`, `SENDER_PORT`, `SENDER_REPLYTO`, `SENDER_USER`, `SENDER_PW` |

### Data Sources

Required when the corresponding source is enabled via `--sources`.

| Source | Required Secrets |
|--------|-----------------|
| SSO | `SSO_USERNAME`, `SSO_PASSWORD` |
| Git | `GIT_USERNAME`, `GIT_TOKEN` |
| JIRA | `JIRA_PAT` |
| Redmine | `REDMINE_USER`, `REDMINE_PW` |

## Validation

The CLI validates secrets at deployment time (`archi create`, `archi restart`, `archi evaluate`). It determines which secrets are required based on:

- **Enabled services** — looked up from the service registry
- **Enabled sources** — looked up from the source registry
- **Configured models** — inferred from `pipeline_map.*.models` in your config
- **Embedding model** — inferred from `data_manager.embedding_name`

If any required secrets are missing, you'll see an error like:

```
Missing required secrets in /home/user/.secrets.env:
  OPENAI_API_KEY, GRAFANA_PG_PASSWORD

Please add these to your .env file in the format:
SECRET_NAME=secret_value
```

> **Tip:** If you're unsure which secrets you need, start with just `PG_PASSWORD` and let the validation tell you what's missing.

## Updating Secrets

To update secrets on a running deployment, edit your `.env` file and restart:

```bash
archi restart --name my-archi --service chatbot --env-file ~/.secrets.env
```

This re-writes the secret files under `~/.archi/archi-{name}/secrets/` and restarts the affected service with the new values.

To update secrets without rebuilding the container image, add `--no-build`:

```bash
archi restart --name my-archi --service chatbot --env-file ~/.secrets.env --no-build
```

## Security Best Practices

- **Never commit** `.env` files to version control. Add `*.env` to your `.gitignore`.
- **Restrict file permissions**: `chmod 600 ~/.secrets.env`
- **Rotate keys regularly**, especially after team member changes.
- **Use least-privilege API keys** — read-only access when possible.
- **Avoid dummy passwords** in production. The CLI provides a fallback `secrets_dummy.env` for local development only.

## Further Reading

- [Quickstart](quickstart.md) — First deployment walkthrough
- [User Guide](user_guide.md) — Per-service configuration details
- [API Reference](api_reference.md) — Full CLI flag documentation
