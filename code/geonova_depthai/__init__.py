"""Compatibility imports for scripts under code/. Prefer component entrypoints."""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
for _path in (_ROOT / "shared/src", _ROOT / "postprocess/src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
__path__.append(str(_ROOT / "capture/src/geonova_depthai"))
