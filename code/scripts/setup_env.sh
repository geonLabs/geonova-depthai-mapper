#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

if command -v python3 >/dev/null 2>&1; then
    BOOTSTRAP_PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    BOOTSTRAP_PYTHON=python
else
    echo "Python 3.8 or newer is required to bootstrap the environment." >&2
    exit 1
fi

exec "$BOOTSTRAP_PYTHON" "$ROOT/setup_env.py" "$@"
