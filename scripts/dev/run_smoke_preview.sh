#!/usr/bin/env bash
set -euo pipefail

info() { echo "[smoke-runner] $*"; }

DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-${1:-}}"
if [[ -z "${DEPLOYMENT_NAME}" ]]; then
  echo "Usage: $0 <deployment-name>"
  exit 1
fi

CONFIG_SRC="${CONFIG_SRC:-tests/pr_preview_config/pr_preview_config.yaml}"
CONFIG_DEST="${CONFIG_DEST:-configs/ci/ci_config.generated.yaml}"
DEFAULT_ENV_FILE=".env_tmp_smoke"
ENV_FILE="${ENV_FILE:-${DEFAULT_ENV_FILE}}"
EXTRA_ENV="${EXTRA_ENV:-}"
SERVICES="${SERVICES:-chatbot}"
HOSTMODE="${HOSTMODE:-true}"
WAIT_URL="${WAIT_URL:-http://localhost:2786/api/health}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-60}"
WAIT_SLEEP="${WAIT_SLEEP:-2}"
BASE_URL="${BASE_URL:-http://localhost:2786}"
USE_PODMAN="${USE_PODMAN:-false}"
SMOKE_CLEANUP="${SMOKE_CLEANUP:-true}"
SMOKE_DUMP_LOGS="${SMOKE_DUMP_LOGS:-true}"
SMOKE_FORCE_CREATE="${SMOKE_FORCE_CREATE:-true}"
SMOKE_OLLAMA_MODEL="${SMOKE_OLLAMA_MODEL:-}"
SMOKE_OLLAMA_URL="${SMOKE_OLLAMA_URL:-}"
SMOKE_OLLAMA_HOST="${SMOKE_OLLAMA_HOST:-}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-900}"
export USE_PODMAN

ARCHI_DIR="${ARCHI_DIR:-${HOME}/.archi}"

ENV_FILE_CREATED=0
CONFIG_DEST_CREATED=0

apply_extra_env() {
  if [[ -z "${EXTRA_ENV}" ]]; then
    return
  fi
  while IFS= read -r line; do
    if [[ -z "${line}" ]]; then
      continue
    fi
    key="${line%%=*}"
    value="${line#*=}"
    if [[ -n "${key}" ]]; then
      export "${key}"="${value}"
    fi
  done <<< "${EXTRA_ENV}"
}

cleanup() {
  exit_code=$?

  if [[ "${exit_code}" -ne 0 && "${SMOKE_DUMP_LOGS,,}" == "true" ]]; then
    tool="docker"
    if [[ "${USE_PODMAN,,}" == "true" ]]; then
      tool="podman"
    fi
    echo "=== Container status ==="
    "${tool}" ps -a || true
    echo "=== Container logs ==="
    "${tool}" ps -a --format '{{.Names}}' | xargs -I{} sh -c 'echo "---- {} ----"; '"${tool}"' logs --tail 50 {} || true'
  fi

  if [[ "${SMOKE_CLEANUP,,}" != "true" ]]; then
    exit "${exit_code}"
  fi

  if [[ -n "${DEPLOYMENT_NAME}" ]]; then
    if [[ "${USE_PODMAN,,}" == "true" ]]; then
      yes | archi delete --name "${DEPLOYMENT_NAME}" --podman --rmi --rmv || true
    else
      yes | archi delete --name "${DEPLOYMENT_NAME}" || true
    fi
  fi

  if [[ "${ENV_FILE_CREATED}" -eq 1 ]]; then
    rm -f "${ENV_FILE}"
  fi
  if [[ "${CONFIG_DEST_CREATED}" -eq 1 ]]; then
    rm -f "${CONFIG_DEST}"
  fi

  exit "${exit_code}"
}

trap cleanup EXIT

apply_extra_env

info "Ensuring smoke scripts are executable..."
chmod +x tests/smoke/*.sh

info "Preparing config..."
mkdir -p "$(dirname "${CONFIG_DEST}")"
if [[ ! -f "${CONFIG_DEST}" ]]; then
  CONFIG_DEST_CREATED=1
fi
cp "${CONFIG_SRC}" "${CONFIG_DEST}"
if [[ -z "${SMOKE_OLLAMA_MODEL}" ]]; then
  SMOKE_OLLAMA_MODEL="$(CONFIG_DEST="${CONFIG_DEST}" python - <<'PY'
import os
import yaml

config_dest = os.environ.get("CONFIG_DEST")
with open(config_dest, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
chat_cfg = (cfg.get("services") or {}).get("chat_app") or {}
local_cfg = (chat_cfg.get("providers") or {}).get("local") or {}
models = local_cfg.get("models") or []
print(chat_cfg.get("default_model") or (models[0] if models else ""))
PY
)"
fi
if [[ -z "${SMOKE_OLLAMA_URL}" ]]; then
  SMOKE_OLLAMA_URL="$(CONFIG_DEST="${CONFIG_DEST}" python - <<'PY'
import os
import yaml

config_dest = os.environ.get("CONFIG_DEST")
with open(config_dest, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
chat_cfg = (cfg.get("services") or {}).get("chat_app") or {}
local_cfg = (chat_cfg.get("providers") or {}).get("local") or {}
print(local_cfg.get("base_url", "http://localhost:11434"))
PY
)"
fi
if [[ -z "${SMOKE_OLLAMA_HOST}" ]]; then
  SMOKE_OLLAMA_HOST="${SMOKE_OLLAMA_URL}"
fi
if [[ -z "${SMOKE_OLLAMA_MODEL}" ]]; then
  echo "Unable to determine Ollama model from ${CONFIG_DEST}. Set SMOKE_OLLAMA_MODEL." >&2
  exit 1
fi
CONFIG_DEST="${CONFIG_DEST}" SMOKE_OLLAMA_MODEL="${SMOKE_OLLAMA_MODEL}" SMOKE_OLLAMA_URL="${SMOKE_OLLAMA_URL}" python - <<'PY'
import os
import yaml

config_dest = os.environ.get("CONFIG_DEST")
smoke_model = os.environ.get("SMOKE_OLLAMA_MODEL")
smoke_url = os.environ.get("SMOKE_OLLAMA_URL")
with open(config_dest, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
services = cfg.setdefault("services", {})
chat_cfg = services.setdefault("chat_app", {})
providers = chat_cfg.setdefault("providers", {})
local_cfg = providers.setdefault("local", {})
models = local_cfg.get("models") or []
if smoke_model:
    if models:
        models[0] = smoke_model
    else:
        models.append(smoke_model)
    local_cfg["models"] = models
    local_cfg["default_model"] = smoke_model
    local_cfg.setdefault("enabled", True)
    chat_cfg["default_provider"] = "local"
    chat_cfg["default_model"] = smoke_model
if smoke_url:
    local_cfg["base_url"] = smoke_url
providers["local"] = local_cfg
chat_cfg["providers"] = providers
services["chat_app"] = chat_cfg
cfg["services"] = services
with open(config_dest, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

AGENTS_DIR="$(CONFIG_DEST="${CONFIG_DEST}" python - <<'PY'
import os
import yaml

config_dest = os.environ.get("CONFIG_DEST")
with open(config_dest, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
chat_cfg = (cfg.get("services") or {}).get("chat_app") or {}
print(chat_cfg.get("agents_dir", ""))
PY
)"

if [[ -z "${AGENTS_DIR}" ]]; then
  echo "agents_dir is missing from ${CONFIG_DEST}" >&2
  exit 1
fi

CONFIG_DEST="${CONFIG_DEST}" AGENTS_DIR="${AGENTS_DIR}" REPO_ROOT="$(pwd)" python - <<'PY'
import os
from pathlib import Path
import importlib.util
import yaml

agents_dir = Path(os.environ.get("AGENTS_DIR", ""))
if not agents_dir.exists() or not agents_dir.is_dir():
    raise SystemExit(f"agents_dir does not exist: {agents_dir}")

repo_root = Path(os.environ.get("REPO_ROOT", ".")).resolve()
agent_spec_path = repo_root / "src" / "archi" / "pipelines" / "agents" / "agent_spec.py"
if not agent_spec_path.exists():
    raise SystemExit(f"agent_spec.py not found at {agent_spec_path}")

module_name = "archi_agent_spec"
spec = importlib.util.spec_from_file_location(module_name, agent_spec_path)
if spec is None or spec.loader is None:
    raise SystemExit("Unable to import agent_spec module")
agent_spec = importlib.util.module_from_spec(spec)
import sys
sys.modules[module_name] = agent_spec
spec.loader.exec_module(agent_spec)
list_agent_files = agent_spec.list_agent_files
load_agent_spec = agent_spec.load_agent_spec

agent_files = list_agent_files(agents_dir)
if not agent_files:
    raise SystemExit(f"agents_dir has no .md files: {agents_dir}")

for path in agent_files:
    load_agent_spec(path)
PY

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama CLI not found; install it or set a different model." >&2
  exit 1
fi
info "Ensuring Ollama model '${SMOKE_OLLAMA_MODEL}' is available..."
OLLAMA_HOST="${SMOKE_OLLAMA_URL}" ollama pull "${SMOKE_OLLAMA_MODEL}"

DEPLOYMENT_DIR="${ARCHI_DIR}/archi-${DEPLOYMENT_NAME}"
if [[ -d "${DEPLOYMENT_DIR}" ]]; then
  if [[ "${SMOKE_FORCE_CREATE,,}" == "true" ]]; then
    info "Existing deployment found; deleting ${DEPLOYMENT_NAME}..."
    if [[ "${USE_PODMAN,,}" == "true" ]]; then
      yes | archi delete --name "${DEPLOYMENT_NAME}" --podman --rmi --rmv || true
    else
      yes | archi delete --name "${DEPLOYMENT_NAME}" || true
    fi
  else
    echo "Deployment ${DEPLOYMENT_NAME} already exists at ${DEPLOYMENT_DIR}." >&2
    echo "Set SMOKE_FORCE_CREATE=true or delete it with: archi delete --name ${DEPLOYMENT_NAME}" >&2
    exit 1
  fi
fi

if [[ "${ENV_FILE}" == "${DEFAULT_ENV_FILE}" ]]; then
  info "Creating .env file with random PG password..."
  ENV_FILE_CREATED=1
  : > "${ENV_FILE}"
  echo "PG_PASSWORD=$(openssl rand -base64 32)" >> "${ENV_FILE}"
  if [[ -n "${SMOKE_OLLAMA_HOST}" ]]; then
    echo "OLLAMA_HOST=${SMOKE_OLLAMA_HOST}" >> "${ENV_FILE}"
  fi
else
  info "Using existing env file ${ENV_FILE}; leaving it unchanged."
fi
if [[ -z "${DM_CATALOG_SEED_FILE:-}" ]]; then
  DM_CATALOG_SEED_FILE="$(pwd)/tests/smoke/seed.txt"
  export DM_CATALOG_SEED_FILE
fi

info "Launching deployment..."
CMD=(archi create --name "${DEPLOYMENT_NAME}" --config "${CONFIG_DEST}" -v 4 --services "${SERVICES}" --env-file "${ENV_FILE}")
if [[ "${USE_PODMAN,,}" == "true" ]]; then
  CMD+=(--podman)
fi
if [[ "${HOSTMODE,,}" == "true" ]]; then
  CMD+=(--hostmode)
fi
if [[ "${SMOKE_FORCE_CREATE,,}" == "true" ]]; then
  CMD+=(--force)
fi
echo "${CMD[@]}"
"${CMD[@]}"

info "Waiting for service readiness..."
attempt=0
while true; do
  if curl -fsS "${WAIT_URL}" >/dev/null 2>&1; then
    info "Service is ready."
    break
  fi
  attempt=$((attempt + 1))
  if (( attempt >= WAIT_ATTEMPTS )); then
    echo "Service did not become ready in time" >&2
    exit 1
  fi
  info "Attempt ${attempt}/${WAIT_ATTEMPTS}..."
  sleep "${WAIT_SLEEP}"
done

CONFIG_NAME="$(CONFIG_DEST="${CONFIG_DEST}" python - <<'PY'
import os
import yaml

config_dest = os.environ.get("CONFIG_DEST")
with open(config_dest, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
print(cfg.get("name", ""))
PY
)"
if [[ -z "${CONFIG_NAME}" ]]; then
  echo "Missing config name in ${CONFIG_DEST}" >&2
  exit 1
fi

RENDERED_DIR="${ARCHI_DIR}/archi-${DEPLOYMENT_NAME}/configs"
RENDERED_CONFIG="${RENDERED_DIR}/${CONFIG_NAME}.yaml"
if [[ ! -f "${RENDERED_CONFIG}" ]]; then
  DEFAULT_RENDERED_CONFIG="${RENDERED_DIR}/config.yaml"
  if [[ -f "${DEFAULT_RENDERED_CONFIG}" ]]; then
    # Older/newer archi versions render a single config as 'config.yaml'
    RENDERED_CONFIG="${DEFAULT_RENDERED_CONFIG}"
  else
    echo "Rendered config not found; checked ${RENDERED_CONFIG} and ${DEFAULT_RENDERED_CONFIG}" >&2
    exit 1
  fi
fi

export ARCHI_CONFIG_PATH="${RENDERED_CONFIG}"
export ARCHI_CONFIG_NAME="$(basename "${RENDERED_CONFIG}" .yaml)"
export ARCHI_PIPELINE_NAME="$(RENDERED_CONFIG="${RENDERED_CONFIG}" python - <<'PY'
import os
import yaml

rendered = os.environ.get("RENDERED_CONFIG")
with open(rendered, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
chat_cfg = (cfg.get("services") or {}).get("chat_app") or {}
print(chat_cfg.get("agent_class", ""))
PY
)"

export PGHOST="$(RENDERED_CONFIG="${RENDERED_CONFIG}" python - <<'PY'
import os
import yaml

rendered = os.environ.get("RENDERED_CONFIG")
with open(rendered, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
print(((cfg.get("services") or {}).get("postgres") or {}).get("host", "localhost"))
PY
)"
export PGPORT="$(RENDERED_CONFIG="${RENDERED_CONFIG}" python - <<'PY'
import os
import yaml

rendered = os.environ.get("RENDERED_CONFIG")
with open(rendered, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
print(((cfg.get("services") or {}).get("postgres") or {}).get("port", 5432))
PY
)"
export PGUSER="$(RENDERED_CONFIG="${RENDERED_CONFIG}" python - <<'PY'
import os
import yaml

rendered = os.environ.get("RENDERED_CONFIG")
with open(rendered, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
print(((cfg.get("services") or {}).get("postgres") or {}).get("user", "archi"))
PY
)"
export PGDATABASE="$(RENDERED_CONFIG="${RENDERED_CONFIG}" python - <<'PY'
import os
import yaml

rendered = os.environ.get("RENDERED_CONFIG")
with open(rendered, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
print(((cfg.get("services") or {}).get("postgres") or {}).get("database", "archi-db"))
PY
)"
export PGPASSWORD="$(ENV_FILE="${ENV_FILE}" python - <<'PY'
import os

path = os.environ.get("ENV_FILE")
if not path:
    print("")
    raise SystemExit(0)
value = ""
with open(path, "r", encoding="utf-8") as handle:
    for line in handle:
        if line.startswith("PG_PASSWORD="):
            value = line.split("=", 1)[1].strip()
            break
print(value)
PY
)"

export DM_BASE_URL="http://localhost:$(RENDERED_CONFIG="${RENDERED_CONFIG}" python - <<'PY'
import os
import yaml

rendered = os.environ.get("RENDERED_CONFIG")
with open(rendered, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
dm = (cfg.get("services") or {}).get("data_manager") or {}
print(dm.get("external_port") or dm.get("port") or 7871)
PY
)"
export OLLAMA_URL="$(RENDERED_CONFIG="${RENDERED_CONFIG}" python - <<'PY'
import os
import yaml

rendered = os.environ.get("RENDERED_CONFIG")
with open(rendered, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
local_cfg = (((cfg.get("services") or {}).get("chat_app") or {}).get("providers") or {}).get("local") or {}
print(local_cfg.get("base_url", "http://localhost:11434"))
PY
)"
export OLLAMA_HOST="${SMOKE_OLLAMA_HOST:-${OLLAMA_URL}}"
export OLLAMA_MODEL="$(RENDERED_CONFIG="${RENDERED_CONFIG}" python - <<'PY'
import os
import yaml

rendered = os.environ.get("RENDERED_CONFIG")
with open(rendered, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
local_cfg = (((cfg.get("services") or {}).get("chat_app") or {}).get("providers") or {}).get("local") or {}
models = local_cfg.get("models") or []
print(models[0] if models else "")
PY
)"
export BASE_URL

info "Running combined smoke checks..."
TIMEOUT_CMD=()
if [[ -n "${SMOKE_TIMEOUT}" && "${SMOKE_TIMEOUT}" != "0" ]]; then
  if command -v timeout >/dev/null 2>&1; then
    # --foreground keeps signal handling aligned with our trap/cleanup
    TIMEOUT_CMD=(timeout --foreground "${SMOKE_TIMEOUT}")
  else
    info "'timeout' not found; proceeding without enforcing SMOKE_TIMEOUT=${SMOKE_TIMEOUT}s"
  fi
fi

if (( ${#TIMEOUT_CMD[@]} )); then
  "${TIMEOUT_CMD[@]}" ./tests/smoke/combined_smoke.sh "${DEPLOYMENT_NAME}"
else
  ./tests/smoke/combined_smoke.sh "${DEPLOYMENT_NAME}"
fi
