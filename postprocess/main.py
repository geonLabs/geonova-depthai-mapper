#!/usr/bin/env python3
"""Independent offline command router. Never opens camera/GNSS/serial devices."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for source in (ROOT / "src", ROOT.parent / "shared/src"):
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))

COMMANDS = {
    "sync": "sync_builder",
    "yolo": "yolo_seg_shp",
    "linearize": "fence_linearization",
    "inspect": "debug_ui",
    "infer": "infer_class_change",
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    if not args or args[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    command = parser.parse_args(args[:1]).command
    module = importlib.import_module("geonova_postprocess." + COMMANDS[command])
    result = module.main(args[1:])
    return int(result) if result is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
