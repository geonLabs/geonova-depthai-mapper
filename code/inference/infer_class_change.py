"""Backward-compatible adapter; implementation moved to a component."""
import importlib
import sys
from pathlib import Path
for _path in (Path(__file__).resolve().parents[2] / "postprocess/src", Path(__file__).resolve().parents[2] / "shared/src"):
    sys.path.insert(0, str(_path))
_implementation = importlib.import_module("geonova_postprocess.infer_class_change")
if __name__ == "__main__":
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
