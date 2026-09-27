#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [ -z "${IMAGE:-}" ]; then
  echo "IMAGE is required, e.g. ghcr.io/elyasirankhah/rag_debugger"
  exit 1
fi

TAGS=("${IMAGE}:latest")
if [ -n "${IMAGE_TAG:-}" ]; then
  TAGS+=("${IMAGE}:${IMAGE_TAG}")
fi

TAG_ARGS=()
for tag in "${TAGS[@]}"; do
  TAG_ARGS+=(-t "$tag")
done

docker build "${TAG_ARGS[@]}" .
for tag in "${TAGS[@]}"; do
  docker push "$tag"
done
