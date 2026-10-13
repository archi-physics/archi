#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/docker_common.sh"

RUNTIME="${CONTAINER_RUNTIME:-docker}"
TAG_INPUT=""
IMAGE_GROUP="all"
IMAGE_FILTERS=()

usage() {
  cat <<'USAGE'
Usage: push_docker_images.sh [OPTIONS] [TAG]

Pushes previously built Docker images to Docker Hub.
  TAG                 Image tag to use; defaults to the project version from pyproject.toml.

Options:
  -g, --group GROUP   Push an image group: all (default), base, or services.
  -i, --image NAME    Push only the specified image. Repeat for several; overrides --group.
  -h, --help          Show this help message and exit.

Set PUSH_LATEST=false to push only TAG (defaults to also pushing latest).
Set CONTAINER_RUNTIME=docker|podman to override the container CLI (defaults to docker).
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--image|-g|--group)
      if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires a value." >&2
        usage
        exit 1
      fi
      if [[ "$1" == -i || "$1" == --image ]]; then
        IMAGE_FILTERS+=("$2")
      else
        IMAGE_GROUP="$2"
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
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

if ! select_images "$IMAGE_GROUP" "${IMAGE_FILTERS[@]}"; then
  usage
  exit 1
fi
IMAGES_TO_PUSH=("${SELECTED_IMAGES[@]}")

TAG="$(resolve_tag "$TAG_INPUT" || true)"
if [[ -z "$TAG" ]]; then
  echo "Error: unable to determine image tag." >&2
  usage
  exit 1
fi

ensure_runtime "$RUNTIME"

echo "Using container runtime: $RUNTIME"
echo "Image tag: $TAG"

PUSH_LATEST="${PUSH_LATEST:-true}"
TAGS=("$TAG")
if [[ "${PUSH_LATEST,,}" == "true" ]]; then
  TAGS+=("latest")
fi

for image in "${IMAGES_TO_PUSH[@]}"; do
  for tag in "${TAGS[@]}"; do
    if ! "$RUNTIME" image inspect "$image:$tag" >/dev/null 2>&1; then
      echo "Error: image $image:$tag not found locally. Build it before pushing." >&2
      exit 1
    fi
  done
done

docker_login "$RUNTIME"

for image in "${IMAGES_TO_PUSH[@]}"; do
  for tag in "${TAGS[@]}"; do
    echo "Pushing $image:$tag"
    "$RUNTIME" push "$image:$tag"
  done
done

if [[ "${PUSH_LATEST,,}" == "true" ]]; then
  echo "Pushed ${IMAGES_TO_PUSH[*]} with tags $TAG and latest."
else
  echo "Pushed ${IMAGES_TO_PUSH[*]} with tag $TAG."
fi
