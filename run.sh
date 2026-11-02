#!/usr/bin/env bash
set -e

# Resolve Isaac Sim python.sh relative to project root
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="${PROJECT_DIR}/../isaacsim/python.sh"

if [ ! -f "$ISAAC_PYTHON" ]; then
    echo "[ERROR] Could not find Isaac Sim at: $ISAAC_PYTHON"
    exit 1
fi

if [ -z "$1" ]; then
    echo "Usage:   ./run.sh <path_to_script.py> [args...]"
    echo "Example: ./run.sh scripts/run_pick_and_place.py"
    exit 1
fi

exec "$ISAAC_PYTHON" "$@"
