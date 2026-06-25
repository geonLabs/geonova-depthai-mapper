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
    parser.add_argument("--rtk-ntrip-username", default=d["rtk_ntrip_username"], help="NTRIP username")
    parser.add_argument("--rtk-ntrip-password", default=d["rtk_ntrip_password"], help="NTRIP password")
    parser.add_argument("--rtk-ntrip-gga", default=d["rtk_ntrip_gga"], help="Fixed GGA sentence; empty uses receiver position")
    parser.add_argument("--rtk-ntrip-gga-interval", type=float, default=d["rtk_ntrip_gga_interval"], help="Seconds between GGA updates")
    parser.add_argument("--rtk-ntrip-reconnect-delay", type=float, default=d["rtk_ntrip_reconnect_delay"], help="Seconds before reconnecting")
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
        "rtcm_bytes": ntrip.bytes_received if ntrip else 0,
        "ntrip_error": ntrip.error if ntrip else None,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
