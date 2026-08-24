#!/usr/bin/env python3
"""Jetson Controller compatible entry point for the field recorder."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parent
CODE_ROOT = REPOSITORY_ROOT / "code"
DEFAULT_CONFIG = REPOSITORY_ROOT / "config.yaml"


def _has_option(arguments: Sequence[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in arguments)


def recorder_arguments(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Return recorder arguments with Controller-managed writable paths last."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    environment = os.environ if environ is None else environ
    if not _has_option(arguments, "--config"):
        arguments[:0] = ["--config", str(DEFAULT_CONFIG)]

    results_dir = environment.get("JETSON_PIPELINE_RESULTS_DIR", "").strip()
    if results_dir:
        # The Controller passes a read-only release config and exposes only this
        # directory as the conventional writable result location. Appending the
        # option makes it win over YAML and any earlier CLI value.
        arguments.extend(("--output-dir", results_dir))
    bridge_dir = environment.get("JETSON_PIPELINE_SENSOR_BRIDGE_DIR", "").strip()
    if not bridge_dir and results_dir:
        bridge_dir = str(Path(results_dir) / "controller-bridge")
    if bridge_dir:
        arguments.extend(("--controller-bridge-dir", bridge_dir))
    return arguments


def _load_recorder_main():
    if not CODE_ROOT.is_dir():
        raise RuntimeError(f"DepthAI source directory is missing: {CODE_ROOT}")
    code_path = str(CODE_ROOT)
    if code_path not in sys.path:
        sys.path.insert(0, code_path)
    from geonova_depthai.capture.raw_event_recorder import main as recorder_main

    return recorder_main


def main(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Load the snapshot's application package and start capture."""

    recorder_main = _load_recorder_main()
    result = recorder_main(recorder_arguments(argv, environ))
    return int(result) if result is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
