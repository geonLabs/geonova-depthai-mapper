#!/usr/bin/env python3
"""Test GPS NMEA reception and NTRIP correction without camera capture."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geonova_depthai import runtime  # noqa: E402
from geonova_depthai.capture.defaults import DEFAULTS  # noqa: E402
from geonova_depthai.config_cli import SafeDefaultsHelpFormatter, parse_args_with_yaml  # noqa: E402


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def nonnegative_float(value):
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0.")
    return number


def parse_args() -> argparse.Namespace:
    d = DEFAULTS
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=SafeDefaultsHelpFormatter
    )
    parser.add_argument("--gps-device", default=d["gps_device"], help="GPS serial path or Windows COM port")
    parser.add_argument("--gps-baudrate", type=int, default=d["gps_baudrate"], help="GPS serial baud rate")
    parser.add_argument("--duration-s", type=float, default=30.0, help="Test duration in seconds")
    parser.add_argument("--no-ntrip", action="store_true", help="Read GPS without opening the NTRIP correction stream")
    parser.add_argument("--print-raw", action="store_true", help="Print raw NMEA lines")
    parser.add_argument("--rtk-ntrip-host", default=d["rtk_ntrip_host"], help="NTRIP caster hostname")
    parser.add_argument("--rtk-ntrip-port", type=int, default=d["rtk_ntrip_port"], help="NTRIP caster TCP port")
    parser.add_argument("--rtk-ntrip-mountpoint", default=d["rtk_ntrip_mountpoint"], help="NTRIP correction mountpoint")
    parser.add_argument("--rtk-ntrip-auto-mountpoint", type=str2bool, nargs="?", const=True, default=d["rtk_ntrip_auto_mountpoint"], help="Choose the nearest NTRIP mountpoint from the caster source table after GPS position is known")
    parser.add_argument("--rtk-ntrip-mountpoint-format", default=d["rtk_ntrip_mountpoint_format"], help="Mountpoint suffix/format to auto-select")
    parser.add_argument("--rtk-ntrip-mountpoint-candidates", default=d["rtk_ntrip_mountpoint_candidates"], help="Comma-separated fallback mountpoints")
    parser.add_argument("--rtk-ntrip-username", default=d["rtk_ntrip_username"], help="NTRIP username")
    parser.add_argument("--rtk-ntrip-password", default=d["rtk_ntrip_password"], help="NTRIP password")
    parser.add_argument("--rtk-ntrip-gga", default=d["rtk_ntrip_gga"], help="Fixed GGA sentence; empty uses receiver position")
    parser.add_argument("--rtk-ntrip-gga-interval", type=nonnegative_float, default=d["rtk_ntrip_gga_interval"], help="Seconds between GGA updates")
    parser.add_argument("--rtk-ntrip-reconnect-delay", type=nonnegative_float, default=d["rtk_ntrip_reconnect_delay"], help="Seconds before reconnecting")
    parser.add_argument("--rtk-ntrip-position-wait-s", type=nonnegative_float, default=d["rtk_ntrip_position_wait_s"], help="Seconds to wait for an initial GPS position")
    parser.add_argument("--rtk-ntrip-connect-timeout-s", type=nonnegative_float, default=d["rtk_ntrip_connect_timeout_s"], help="Seconds before giving up on an NTRIP TCP/header connection")
    parser.add_argument("--rtk-ntrip-data-timeout-s", type=nonnegative_float, default=d["rtk_ntrip_data_timeout_s"], help="Seconds without RTCM data before switching mountpoints")
    parser.add_argument("--rtk-ntrip-sourcetable-timeout-s", type=nonnegative_float, default=d["rtk_ntrip_sourcetable_timeout_s"], help="Seconds before giving up on source table")
    parser.add_argument("--rtk-ntrip-max-mountpoints", type=int, default=d["rtk_ntrip_max_mountpoints"], help="Maximum auto-selected mountpoints per reconnect cycle")
    parser.add_argument("--rtk-initial-latitude-deg", type=float, default=d["rtk_initial_latitude_deg"], help="Fallback latitude before a GPS fix")
    parser.add_argument("--rtk-initial-longitude-deg", type=float, default=d["rtk_initial_longitude_deg"], help="Fallback longitude before a GPS fix")
    parser.add_argument("--rtk-initial-altitude-m", type=float, default=d["rtk_initial_altitude_m"], help="Fallback altitude in meters")
    return parse_args_with_yaml(parser)


def main() -> None:
    args = parse_args()
    values = dict(DEFAULTS)
    values.update(vars(args))
    values["enable_gps"] = True
    if args.no_ntrip:
        values["rtk_ntrip_host"] = ""
    config = SimpleNamespace(**values)
    reader = runtime.SerialRateLimitedReader(
        "gps",
        config.gps_device,
        config.gps_baudrate,
        config.gps_max_hz,
        parser=runtime.NmeaParserState(),
        rtk_config=runtime.build_rtk_config(config),
    )
    reader.start()
    started = time.monotonic()
    gga_count = 0
    last_gga = None
    try:
        while time.monotonic() - started < args.duration_s:
            for sample in reader.drain():
                if args.print_raw:
                    print(sample.get("raw", ""))
                if sample.get("nmea_type") == "GGA":
                    gga_count += 1
                    last_gga = sample
            if reader.error:
                raise RuntimeError(reader.error)
            time.sleep(0.05)
    finally:
        reader.stop()

    ntrip = reader.rtk_client
    report = {
        "passed": gga_count > 0 and (args.no_ntrip or (ntrip is not None and ntrip.bytes_received > 0)),
        "gga_count": gga_count,
        "last_fix_quality": (last_gga or {}).get("fix_quality"),
        "last_fix_quality_name": (last_gga or {}).get("fix_quality_name"),
        "ntrip_enabled": not args.no_ntrip,
        "ntrip_connected": bool(ntrip and ntrip.connected),
        "ntrip_mountpoint": ntrip.current_mountpoint if ntrip else None,
        "ntrip_candidates": [entry.get("mountpoint") for entry in ntrip.mountpoint_sequence] if ntrip else [],
        "rtcm_bytes": ntrip.bytes_received if ntrip else 0,
        "ntrip_error": ntrip.error if ntrip else None,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
