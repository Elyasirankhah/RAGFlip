#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Fail the build on high-severity findings only. No secrets or payloads.
python -m bandit -r rag utils app.py start_server.py \
  --severity-level high \
  --confidence-level medium \
  -q
