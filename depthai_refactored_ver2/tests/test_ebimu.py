#!/usr/bin/env python3
"""Read and validate EBIMU output without starting camera capture."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import serial

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geonova_depthai import runtime  # noqa: E402
from geonova_depthai.capture.defaults import DEFAULTS  # noqa: E402
from geonova_depthai.config_cli import parse_args_with_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device", default=DEFAULTS["external_imu_device"], help="EBIMU serial path or Windows COM port")
    parser.add_argument("--baudrate", type=int, default=DEFAULTS["external_imu_baudrate"], help="EBIMU serial baud rate")
    parser.add_argument("--duration-s", type=float, default=10.0, help="Acquisition duration in seconds")
    parser.add_argument("--min-rate-hz", type=float, default=20.0, help="Minimum parsed sample rate required to pass")
    parser.add_argument("--print-raw", action="store_true", help="Print every received serial line")
    return parse_args_with_yaml(parser)


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    parsed_samples = []
    malformed = 0
    with serial.Serial(args.device, args.baudrate, timeout=0.2) as port:
        while time.monotonic() - started < args.duration_s:
            raw = port.readline()
            if not raw:
                continue
            line = raw.decode("ascii", errors="replace").strip()
            if args.print_raw:
                print(line)
            parsed = runtime.parse_ebimu_line(line)
            if parsed.get("orientation_format"):
                parsed_samples.append(parsed)
            else:
                malformed += 1

    elapsed = max(time.monotonic() - started, 1e-9)
    rate_hz = len(parsed_samples) / elapsed
    orientation_formats = sorted(
        {sample.get("orientation_format") for sample in parsed_samples if sample.get("orientation_format")}
    )
    report = {
        "passed": bool(parsed_samples and rate_hz >= args.min_rate_hz),
        "duration_s": elapsed,
        "valid_samples": len(parsed_samples),
        "malformed_lines": malformed,
        "measured_rate_hz": rate_hz,
        "orientation_formats": orientation_formats,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
