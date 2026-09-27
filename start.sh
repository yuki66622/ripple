#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Create .venv and install requirements.txt first; see README.md."
  exit 1
fi
exec .venv/bin/python -m ripple_live.server --port 5176
