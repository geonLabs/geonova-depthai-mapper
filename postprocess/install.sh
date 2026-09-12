#!/usr/bin/env sh
set -eu
COMPONENT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
INSTALLER="$COMPONENT_ROOT/tools/install_component.py"
if [ ! -f "$INSTALLER" ]; then INSTALLER="$COMPONENT_ROOT/../tools/install_component.py"; fi
exec "${PYTHON:-python3}" "$INSTALLER" --component postprocess "$@"
