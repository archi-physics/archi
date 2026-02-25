#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/docker_common.sh"

RUNTIME="${CONTAINER_RUNTIME:-docker}"
TAG_INPUT=""
IMAGE_FILTER=""

usage() {
  cat <<'USAGE'
Usage: build_docker_images.sh [OPTIONS] [TAG]

Builds the base Docker images locally.
  TAG                 Image tag to use; defaults to the project version from pyproject.toml.

Options:
  -i, --image NAME    Build only the specified image (e.g. a2rchi/a2rchi-python-base).
  -h, --help          Show this help message and exit.

Set CONTAINER_RUNTIME=docker|podman to override the container CLI (defaults to docker).
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--image)
      if [[ $# -lt 2 ]]; then
        echo "Error: --image requires a value." >&2
        usage
        exit 1
      fi
      IMAGE_FILTER="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Error: unknown option '$1'." >&2
      usage
      exit 1
      ;;
    *)
      if [[ -n "$TAG_INPUT" ]]; then
        echo "Error: multiple tag values provided: '$TAG_INPUT' and '$1'." >&2
        usage
        exit 1
      fi
      TAG_INPUT="$1"
      shift
      ;;
  esac
done

if [[ $# -gt 0 ]]; then
  echo "Error: unexpected arguments: $*" >&2
  usage
  exit 1
fi

TAG="$(resolve_tag "$TAG_INPUT" || true)"
if [[ -z "$TAG" ]]; then
  echo "Error: unable to determine image tag." >&2
  usage
  exit 1
fi

ensure_runtime "$RUNTIME"

echo "Using container runtime: $RUNTIME"
echo "Image tag: $TAG"

echo "Updating requirements files..."
cat "$ROOT_DIR/requirements/cpu-requirementsHEADER.txt" \
    "$ROOT_DIR/requirements/requirements-base.txt" \
    > "$ROOT_DIR/src/cli/templates/dockerfiles/base-python-image/requirements.txt"
cat "$ROOT_DIR/requirements/gpu-requirementsHEADER.txt" \
    "$ROOT_DIR/requirements/requirements-base.txt" \
    > "$ROOT_DIR/src/cli/templates/dockerfiles/base-pytorch-image/requirements.txt"

declare -a IMAGES_TO_BUILD=()
if [[ -n "$IMAGE_FILTER" ]]; then
  if [[ -n "${IMAGE_DIRS[$IMAGE_FILTER]+x}" ]]; then
    IMAGES_TO_BUILD=("$IMAGE_FILTER")
  else
    echo "Error: unknown image '$IMAGE_FILTER'. Available images:" >&2
    for image in "${!IMAGE_DIRS[@]}"; do
      echo "  - $image" >&2
    done
    exit 1
  fi
else
  IMAGES_TO_BUILD=("${!IMAGE_DIRS[@]}")
fi

for image in "${IMAGES_TO_BUILD[@]}"; do
  build_context="${IMAGE_DIRS[$image]}"
  echo "Building $image:$TAG from $build_context"
  "$RUNTIME" build \
    -t "$image:$TAG" \
    -t "$image:latest" \
    -t "localhost/$image:$TAG" \
    -t "localhost/$image:latest" \
    "$build_context"
  echo "Tagged $image:latest"
done

echo "All images built with tag $TAG."
