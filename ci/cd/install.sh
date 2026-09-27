#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

python -m pip install --upgrade pip
python -m pip install -r ci/cd/requirements-ci.txt
