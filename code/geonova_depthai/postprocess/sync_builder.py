"""Backward-compatible adapter; implementation moved to a component."""
import importlib
import sys
_implementation = importlib.import_module("geonova_postprocess.sync_builder")
if __name__ == "__main__":
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
