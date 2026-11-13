#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[analysis,dev]"
printf '%s\n' "Environment ready. Activate with: source .venv/bin/activate"
