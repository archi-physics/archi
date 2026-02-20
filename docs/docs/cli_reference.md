# CLI Reference

The Archi CLI provides commands to create, manage, and monitor deployments.

## Installation

The CLI is installed automatically with `pip install -e .` from the repository root. Verify with:

```bash
which archi
```

---

## Commands

### `archi create`

Create a new Archi deployment.

```bash
archi create --name <name> --config <config.yaml> --env-file <secrets.env> --services <services> [OPTIONS]
```

**Required options:**

| Option | Description |
|--------|-------------|
| `--name`, `-n` | Name of the deployment |
| `--config`, `-c` | Path to YAML configuration file (repeatable for multiple files) |

**Recommended options:**

| Option | Description |
|--------|-------------|
| `--env-file`, `-e` | Path to the secrets `.env` file |
| `--services`, `-s` | Comma-separated list of services to enable (e.g., `chatbot,uploader`) |

**Optional flags:**

| Option | Description | Default |
|--------|-------------|---------|
| `--config-dir`, `-cd` | Directory containing configuration files | — |
| `--podman`, `-p` | Use Podman instead of Docker | Docker |
| `--gpu-ids` | GPU configuration: `all` or comma-separated IDs (e.g., `0,1`) | None |
| `--tag`, `-t` | Image tag for built containers | `2000` |
| `--hostmode` | Use host network mode for all services | Off |
| `--verbosity`, `-v` | Logging verbosity level (0=quiet, 4=debug) | `3` |
| `--force`, `-f` | Overwrite existing deployment if it exists | Off |
| `--dry`, `--dry-run` | Validate and show what would be created without deploying | Off |

**Examples:**

```bash
# Basic deployment with Ollama
archi create -n my-archi -c config.yaml -e .secrets.env \
  --services chatbot --podman

# Full deployment with GPU and multiple services
archi create -n prod-archi -c config.yaml -e .secrets.env \
  --services chatbot,uploader,grafana \
  --gpu-ids all

# Dry run to validate configuration
archi create -n test -c config.yaml -e .secrets.env \
  --services chatbot --dry-run
```

**Notes:**

- The CLI checks that host ports are free before deploying. If a port is in use, adjust `services.*.external_port` in your config.
- The first deployment builds container images from scratch (may take several minutes). Subsequent deployments reuse images.
- Use `-v 4` for debug-level logging when troubleshooting.

---

### `archi delete`

Delete an existing deployment.

```bash
archi delete --name <name> [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--name`, `-n` | Name of the deployment to delete |
| `--rmi` | Also remove container images |
| `--rmv` | Also remove volumes |
| `--keep-files` | Keep deployment files on disk |
| `--list` | List all deployments |

**Examples:**

```bash
# Delete deployment and clean up everything
archi delete -n my-archi --rmi --rmv

# Delete but keep data volumes
archi delete -n my-archi --rmi
```

---

### `archi restart`

Restart a specific service in an existing deployment without restarting the entire stack.

```bash
archi restart --name <name> --service <service> [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--name`, `-n` | Name of the existing deployment | Required |
| `--service`, `-s` | Service to restart | `chatbot` |
| `--config`, `-c` | Updated configuration file(s) | — |
| `--config-dir`, `-cd` | Directory containing configuration files | — |
| `--env-file`, `-e` | Updated secrets file | — |
| `--no-build` | Restart without rebuilding the container image | Off |
| `--with-deps` | Also restart dependent services | Off |
| `--podman`, `-p` | Use Podman instead of Docker | Docker |
| `--verbosity`, `-v` | Logging verbosity (0-4) | `3` |

**Examples:**

```bash
# Quick config update (no rebuild needed)
archi restart -n my-archi --service chatbot --no-build

# Rebuild after code changes
archi restart -n my-archi --service chatbot -c updated_config.yaml

# Re-scrape data sources
archi restart -n my-archi --service data_manager

# Restart with updated secrets
archi restart -n my-archi --service chatbot -e new_secrets.env --no-build
```

---

### `archi list-services`

List all available services and data sources with descriptions.

```bash
archi list-services
```

---

### `archi list-deployments`

List all existing deployments.

```bash
archi list-deployments
```

---

### `archi evaluate`

Launch the benchmarking runtime to evaluate configurations against a set of questions and answers.

```bash
archi evaluate --name <name> --env-file <secrets.env> --config <config.yaml> [OPTIONS]
```

Supports the same flags as `create` (`--podman`, `--gpu-ids`, `--tag`, `--hostmode`, `--verbosity`, `--force`). Configuration files should define the `services.benchmarking` section.

**Example:**

```bash
archi evaluate -n benchmark \
  -c examples/benchmarking/benchmark_configs/example_conf.yaml \
  -e .secrets.env --gpu-ids all
```

See [Benchmarking](benchmarking.md) for full details on query format and evaluation modes.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ARCHI_DIR` | Override the deployment directory (default: `~/.archi`) |
| `OLLAMA_HOST` | Ollama server address (default: `http://localhost:11434`) |

---

## Troubleshooting

### Port Conflicts

If a port is already in use, the CLI will report an error. Adjust `services.*.external_port` in your config:

```yaml
services:
  chat_app:
    external_port: 7862  # default: 7861
  grafana:
    external_port: 3001  # default: 3000
```

### GPU Issues

GPU access requires NVIDIA drivers and the NVIDIA Container Toolkit.

**Podman:**
```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list
```

**Docker:**
```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

### Verbose Logging

Add `-v 4` to any command for debug-level output:

```bash
archi create [...] -v 4
```

### Multiple Deployments

Multiple deployments can run on the same machine. Container networks are separate, but be careful with external port assignments. See [Advanced Setup](advanced_setup_deploy.md#running-multiple-deployments-on-the-same-machine).
