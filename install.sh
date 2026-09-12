#!/usr/bin/env sh
set -eu
REPOSITORY_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${PYTHON:-python3}" "$REPOSITORY_ROOT/tools/install_component.py" --component capture "$@"
