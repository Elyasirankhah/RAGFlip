#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

bash ci/cd/install.sh
bash ci/cd/lint.sh
bash ci/cd/test.sh
bash ci/cd/security.sh
