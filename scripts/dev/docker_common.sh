#!/usr/bin/env bash
set -euo pipefail

# Common utilities shared by Docker image build/push scripts.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Map of image name -> build context directory.
declare -Ag IMAGE_DIRS=(
  ["a2rchi/a2rchi-python-base"]="$ROOT_DIR/src/cli/templates/dockerfiles/base-python-image"
  ["a2rchi/a2rchi-pytorch-base"]="$ROOT_DIR/src/cli/templates/dockerfiles/base-pytorch-image"
  ["a2rchi/archi-chat"]="$ROOT_DIR"
  ["a2rchi/archi-data-manager"]="$ROOT_DIR"
  ["a2rchi/archi-postgres"]="$ROOT_DIR"
  ["a2rchi/archi-grafana"]="$ROOT_DIR"
)

declare -Ag IMAGE_DOCKERFILES=(
  ["a2rchi/archi-chat"]="$ROOT_DIR/src/cli/templates/dockerfiles/base-helm-images/Dockerfile-chat-universal"
  ["a2rchi/archi-data-manager"]="$ROOT_DIR/src/cli/templates/dockerfiles/base-helm-images/Dockerfile-data-manager-universal"
  ["a2rchi/archi-postgres"]="$ROOT_DIR/src/cli/templates/dockerfiles/base-helm-images/Dockerfile-postgres-universal"
  ["a2rchi/archi-grafana"]="$ROOT_DIR/src/cli/templates/dockerfiles/base-helm-images/Dockerfile-grafana-universal"
)

# Image groups, in build order. Base images must be built before the Helm
# service images because archi-chat and archi-data-manager build FROM
# a2rchi/a2rchi-python-base:latest.
BASE_IMAGES=(
  "a2rchi/a2rchi-python-base"
  "a2rchi/a2rchi-pytorch-base"
)
SERVICE_IMAGES=(
  "a2rchi/archi-chat"
  "a2rchi/archi-data-manager"
  "a2rchi/archi-postgres"
  "a2rchi/archi-grafana"
)

# Resolve image selection into the global SELECTED_IMAGES array, in build order.
# Usage: select_images <group> [image...]
#   group: all | base | services ("" means all). Explicit images override the group.
select_images() {
  local group="${1:-all}"
  shift || true
  local -a requested=("$@")
  local -a ordered=("${BASE_IMAGES[@]}" "${SERVICE_IMAGES[@]}")
  local image

  SELECTED_IMAGES=()
  if [[ ${#requested[@]} -gt 0 ]]; then
    for image in "${requested[@]}"; do
      if [[ -z "${IMAGE_DIRS[$image]+x}" ]]; then
        echo "Error: unknown image '$image'. Available images:" >&2
        printf '  - %s\n' "${ordered[@]}" >&2
        return 1
      fi
    done
    # Keep build order regardless of the order images were requested in.
    for image in "${ordered[@]}"; do
      if [[ " ${requested[*]} " == *" $image "* ]]; then
        SELECTED_IMAGES+=("$image")
      fi
    done
    return 0
  fi

  case "$group" in
    all|"") SELECTED_IMAGES=("${ordered[@]}") ;;
    base) SELECTED_IMAGES=("${BASE_IMAGES[@]}") ;;
    services) SELECTED_IMAGES=("${SERVICE_IMAGES[@]}") ;;
    *)
      echo "Error: unknown image group '$group' (expected all, base, or services)." >&2
      return 1
      ;;
  esac
}

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
