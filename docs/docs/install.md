# Install

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
