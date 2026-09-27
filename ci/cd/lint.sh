#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Catch syntax/undefined-name errors without failing the whole repo on style.
python -m ruff check rag utils app.py start_server.py tests --select E9,F63,F7,F82
