# Helm deployment with archi install

> **v2 (legacy).** This page describes the v2 application (the RAG chatbot) on the `main` branch. It is not the 3.0 documentation. v2 keeps running on `main` until the cutover; see [Moving from v2 to 3.0](../migration-from-v2.md).

Archi can render and install a Helm chart from the same configuration and secrets that drive its container-based deployments. The `archi install` command prepares a chart in the directory you provide with `--templates-dir` and then installs it with Helm.

## Prerequisites

Before using Helm deployments, make sure you have:

- Helm v3 installed and available on your `PATH`
- Access to a Kubernetes cluster with a working `kubectl` context
- The Archi CLI installed in your environment
- One Archi configuration file and, if needed, an environment file with secrets

## Basic workflow

1. Prepare your Archi config and secrets.
2. Render and validate the chart in a temporary directory:

```bash
mkdir -p /tmp/archi-helm

archi install \
  --name my-archi \
  --config configs/config.yaml \
  --templates-dir /tmp/archi-helm \
  --env-file .secrets.env \
  --services chatbot \
  --dry-run
```

3. Install the chart for real:

```bash
archi install \
  --name my-archi \
  --config configs/config.yaml \
  --templates-dir /tmp/archi-helm \
  --env-file .secrets.env \
  --services chatbot
```

4. Inspect the release:

```bash
helm status my-archi
kubectl get pods
```

> The Helm release name is derived from `--name`, with underscores replaced by hyphens so it remains valid for Helm.

## Common options

The `archi install` command supports the same core inputs as the container workflow:

- `--config` or `--config-dir`: provide the Archi configuration. The CLI currently expects exactly one config file.
- `--templates-dir`: directory where the generated chart files will be written.
- `--env-file`: file containing deployment secrets.
- `--services`: comma-separated list of services to include.
- `--gpu-ids`: pass GPU settings for supported deployments.
- `--hostmode`: use host networking for the rendered resources.
- `--dry-run`: render and validate the chart without installing it.
- `--reinstall`: force Helm to reinstall or upgrade the existing release.

## What gets generated

The install command writes a Helm chart with files such as:

- `Chart.yaml` and `values.yaml`
- Kubernetes manifests under the chart `templates/` directory
- Secret and PVC resources derived from your Archi config
- Service-specific resources for the enabled services

This makes it easy to inspect the rendered manifests before applying them or to keep a versioned copy of a deployment chart.

## Upgrading and removing a deployment

To replace an existing Helm release:

```bash
archi install \
  --name my-archi \
  --config configs/config.yaml \
  --templates-dir /tmp/archi-helm \
  --env-file .secrets.env \
  --services chatbot \
  --reinstall
```

To remove a deployment:

```bash
helm uninstall my-archi
```

## Troubleshooting

If the deployment fails, check the following:

- Helm is installed and available in your shell: `helm version`
- Your Kubernetes context is correct: `kubectl config current-context`
- The config file is valid and only one file is provided to `--config`
- The secrets file contains the values required for the selected services
- The selected services are supported by your current Archi configuration

For more detail about the CLI options, see the [CLI reference](cli_reference.md).
