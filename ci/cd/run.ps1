$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root

python -m pip install --upgrade pip
python -m pip install -r ci/cd/requirements-ci.txt
python -m ruff check rag utils app.py start_server.py tests --select E9,F63,F7,F82
python -m pytest
python -m bandit -r rag utils app.py start_server.py --severity-level high --confidence-level medium -q
