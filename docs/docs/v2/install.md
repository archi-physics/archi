# Install

> **v2 (legacy).** This page describes the v2 application (the RAG chatbot) on the `main` branch. It is not the 3.0 documentation. v2 keeps running on `main` until the cutover; see [Moving from v2 to 3.0](../migration-from-v2.md).

## System Requirements

Archi is deployed using a Python-based CLI onto containers. It requires:

- `docker` version 24+ or `podman` version 5.4.0+ (for containers)
- `python 3.10.0+` (for the CLI)

> **Note:** We support either running open-source models locally or connecting to existing APIs. If you plan to run open-source models on your machine's GPUs, see the [Advanced Setup & Deployment](advanced_setup_deploy.md) section.

## Installation

Clone the Archi repository:

```bash
git clone https://github.com/archi-physics/archi.git
```

Check out the latest stable tag (recommended for users; stay on `main` only if you're actively developing):

```bash
cd archi
git checkout $(git describe --tags $(git rev-list --tags --max-count=1))
```

Install Archi (from inside the repository):

```bash
pip install -e .
```

This installs Archi's dependencies and the CLI tool. Verify the installation with:

```bash
which archi
```

The command prints the path to the `archi` executable.

## Helm deployments

Archi also supports rendering and installing a Helm chart through the `archi install` command. This is useful when you want to deploy archi into a Kubernetes cluster instead of running the compose-based workflow locally. The standard command uses the local helm installation to deploy archi using the created chart. With the `dry-run` flag the CLI command only creates the chart and required templates.

For a full walkthrough, see the [Helm deployment guide](helm_deployment.md).

<details>
<summary>Show Full Installation Script</summary>

```bash
# Clone the repository
git clone https://github.com/archi-physics/archi.git
cd archi
export ARCHI_DIR=$(pwd)

# (Optional) Checkout the latest stable tag (recommended for users)
# Skip this if you're developing and want the tip of main.
git checkout $(git describe --tags $(git rev-list --tags --max-count=1))

# (Optional) Create and activate a virtual environment
python3 -m venv archi_venv
source archi_venv/bin/activate

# Install dependencies
cd "$ARCHI_DIR"
pip install -e .

# Verify installation
which archi
```

</details>
