#!/usr/bin/env bash
set -euo pipefail

# Env vars used by this runner:
# BASE_URL, DM_BASE_URL, OLLAMA_URL, OLLAMA_MODEL,
# PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE,
# ARCHI_CONFIG_PATH, ARCHI_CONFIG_NAME, ARCHI_PIPELINE_NAME, USE_PODMAN

NAME="${1:-}"
if [[ -z "${NAME}" ]]; then
  echo "Usage: $0 <deployment-name>"
  echo "Requires env vars: ARCHI_CONFIG_PATH, OLLAMA_MODEL, PGHOST, PGUSER, PGPASSWORD, PGDATABASE"
  exit 1
fi

info() { echo "[combined-smoke] $*"; }

BASE_URL="${BASE_URL:-http://localhost:2786}"
DM_BASE_URL="${DM_BASE_URL:-http://localhost:7871}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

export BASE_URL
export DM_BASE_URL
export OLLAMA_URL

info "Running preflight checks..."
python3 tests/smoke/preflight.py

info "Running direct tool probes (chatbot container)..."
tool="docker"
use_podman="${USE_PODMAN:-false}"
if [[ "${use_podman,,}" == "true" ]]; then
  tool="podman"
fi
container_name="chatbot-${NAME}"
if ! "${tool}" inspect "${container_name}" >/dev/null 2>&1; then
  echo "[combined-smoke] ERROR: Missing container ${container_name}" >&2
  exit 1
fi
config_name="${ARCHI_CONFIG_NAME:-}"
if [[ -z "${config_name}" && -n "${ARCHI_CONFIG_PATH:-}" ]]; then
  config_name="$(basename "${ARCHI_CONFIG_PATH}" .yaml)"
fi
if [[ -z "${config_name}" ]]; then
  echo "[combined-smoke] ERROR: ARCHI_CONFIG_NAME is required for container tool checks" >&2
  exit 1
fi
"${tool}" exec -i -w /root/archi \
  -e ARCHI_CONFIG_NAME="${config_name}" \
  -e ARCHI_CONFIG_PATH="/root/archi/configs/${config_name}.yaml" \
  -e DM_BASE_URL="${DM_BASE_URL}" \
  -e OLLAMA_URL="${OLLAMA_URL}" \
  -e OLLAMA_MODEL="${OLLAMA_MODEL}" \
  "${container_name}" \
  python3 - < tests/smoke/tools_smoke.py

info "Running ReAct smoke check..."
python3 tests/smoke/react_smoke.py

info "Combined smoke checks passed for ${NAME}"
