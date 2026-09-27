#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

IMAGE_NAME="${IMAGE_NAME:-rag-debugger}"
docker build -t "$IMAGE_NAME" .
