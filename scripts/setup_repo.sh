
#!/usr/bin/env bash
# Set up external repositories required by this project.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TD3_DIR_IS_EXPLICIT="${TD3_DIR+x}"
CRAFTIUM_DIR_IS_EXPLICIT="${CRAFTIUM_DIR+x}"
DEPENDENCIES_DIR="${DEPENDENCIES_DIR:-$PROJECT_ROOT}"
TD3_DIR="${TD3_DIR:-$DEPENDENCIES_DIR/TD3}"
CRAFTIUM_DIR="${CRAFTIUM_DIR:-$DEPENDENCIES_DIR/craftium}"
LEGACY_TD3_DIR="$PROJECT_ROOT/benchmarks/TD3"
LEGACY_CRAFTIUM_DIR="$PROJECT_ROOT/benchmarks/craftium"

require_command() {
	local command_name="$1"

	if ! command -v "$command_name" >/dev/null 2>&1; then
		echo "Error: $command_name is required but was not found in PATH" >&2
		exit 1
	fi
}

require_command git

mkdir -p "$DEPENDENCIES_DIR"


ensure_repo() {
	local name="$1"
	local target_dir="$2"
	shift 2

	echo "============================================================"
	echo "Downloading the $name Repository"
	echo "-------------------------------------------------------------"
	echo "Target directory: $target_dir"
	if [[ -d "$target_dir/.git" ]]; then
		echo "$name already exists in $target_dir; skipping clone"
	elif [[ -e "$target_dir" ]]; then
		echo "Error: $target_dir exists but is not a git repository" >&2
		exit 1
	else
		git clone "$@" "$target_dir"
	fi
	echo "-------------------------------------------------------------"
	echo "Finished downloading the $name Repository"
	echo "============================================================"
}

ensure_repo "TD3" "$TD3_DIR" https://github.com/sfujim/TD3.git
ensure_repo "Craftium" "$CRAFTIUM_DIR" --recurse-submodules https://github.com/mikelma/craftium.git
