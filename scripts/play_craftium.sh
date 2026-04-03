#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CRAFTIUM_DIR="$PROJECT_ROOT/craftium"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
	echo "Error: expected virtualenv interpreter at $PYTHON_BIN" >&2
	echo "Run 'uv sync' from the repository root first." >&2
	exit 1
fi

if [[ ! -d "$CRAFTIUM_DIR" ]]; then
	echo "Error: Craftium checkout not found at $CRAFTIUM_DIR" >&2
	exit 1
fi

cd "$CRAFTIUM_DIR"
exec "$PYTHON_BIN" play_env.py "$@"