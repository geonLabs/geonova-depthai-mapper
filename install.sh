#!/usr/bin/env sh
set -eu

REPOSITORY_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT="$REPOSITORY_ROOT/code"

if [ ! -f "$PROJECT_ROOT/setup_env.py" ]; then
    echo "Installer not found: $PROJECT_ROOT/setup_env.py" >&2
    exit 1
fi

cd "$PROJECT_ROOT"
exec "$PROJECT_ROOT/scripts/setup_env.sh" --config "$PROJECT_ROOT/configs/setup.yaml" "$@"
