#!/usr/bin/env bash
set -euo pipefail

# Common utilities shared by Docker image build/push scripts.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Map of image name -> build context directory.
declare -Ag IMAGE_DIRS=(
  ["a2rchi/a2rchi-python-base"]="$ROOT_DIR/src/cli/templates/dockerfiles/base-python-image"
  ["a2rchi/a2rchi-pytorch-base"]="$ROOT_DIR/src/cli/templates/dockerfiles/base-pytorch-image"
)

resolve_tag() {
  local provided_tag="${1:-}"
  if [[ -n "$provided_tag" ]]; then
    printf '%s' "$provided_tag"
    return 0
  fi

  if [[ -f "$ROOT_DIR/pyproject.toml" ]]; then
    local version
    version=$(awk -F'"' '/^version[[:space:]]*=/{print $2; exit}' "$ROOT_DIR/pyproject.toml" || true)
    if [[ -n "$version" ]]; then
      printf '%s' "$version"
      return 0
    fi
  fi

  return 1
}

ensure_runtime() {
  local runtime="$1"
  if ! command -v "$runtime" >/dev/null 2>&1; then
    echo "Error: container runtime '$runtime' not found in PATH." >&2
    return 1
  fi
}

docker_login() {
  local runtime="$1"

  echo
  echo "Logging in to Docker Hub..."
  if [[ -n "${DOCKERHUB_USERNAME:-}" && -n "${DOCKERHUB_PASSWORD:-${DOCKERHUB_TOKEN:-}}" ]]; then
    local secret
    secret="${DOCKERHUB_PASSWORD:-${DOCKERHUB_TOKEN:-}}"
    echo "Using non-interactive Docker Hub login"
    printf '%s' "$secret" | "$runtime" login docker.io --username "$DOCKERHUB_USERNAME" --password-stdin
  else
    echo "Falling back to interactive docker login"
    "$runtime" login docker.io
  fi
}
