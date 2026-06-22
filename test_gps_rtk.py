#!/usr/bin/env python3

import argparse
import base64
import signal
import socket
import threading
import time
from datetime import datetime

import serial


stop_requested = False


def request_stop(signum, frame):
    global stop_requested
    stop_requested = True


def nonnegative_float(value):
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0.")
    return number


def nmea_coord_to_decimal(raw_value, hemisphere):
    if not raw_value or not hemisphere:
        return ""
    try:
        dot_index = raw_value.index(".")
        degree_digits = dot_index - 2
        degrees = int(raw_value[:degree_digits])
        minutes = float(raw_value[degree_digits:])
        value = degrees + minutes / 60.0
        if hemisphere in ("S", "W"):
            value *= -1.0
        return value
    except (ValueError, IndexError):
        return ""


def parse_nmea_line(line):
    if not line.startswith("$"):
        return {}
    payload = line[1:].split("*", 1)[0]
    parts = payload.split(",")
    if not parts:
        return {}

    kind = parts[0][-3:]
    parsed = {"nmea_type": kind}
    if kind == "GGA" and len(parts) >= 10:
        parsed.update({
            "gps_time_utc": parts[1],
            "latitude_deg": nmea_coord_to_decimal(parts[2], parts[3]),
            "longitude_deg": nmea_coord_to_decimal(parts[4], parts[5]),
            "fix_quality": parts[6],
            "satellites": parts[7],
            "hdop": parts[8],
            "altitude_m": parts[9],
            "differential_age_s": parts[13] if len(parts) > 13 else "",
            "reference_station_id": parts[14] if len(parts) > 14 else "",
        })
    elif kind == "RMC" and len(parts) >= 10:
        parsed.update({
            "gps_time_utc": parts[1],
            "status": parts[2],
            "latitude_deg": nmea_coord_to_decimal(parts[3], parts[4]),
            "longitude_deg": nmea_coord_to_decimal(parts[5], parts[6]),
            "speed_knots": parts[7],
            "course_deg": parts[8],
            "date_utc": parts[9],
        })
    return parsed


def nmea_checksum(sentence_body):
    checksum = 0
    for char in sentence_body:
        checksum ^= ord(char)
    return f"{checksum:02X}"


def build_gga_sentence(latitude_deg, longitude_deg, altitude_m=0.0):
    def coord(value, positive_hemi, negative_hemi, degree_digits):
        hemi = positive_hemi if value >= 0 else negative_hemi
        absolute = abs(float(value))
        degrees = int(absolute)
        minutes = (absolute - degrees) * 60.0
        return f"{degrees:0{degree_digits}d}{minutes:08.5f}", hemi

    lat, ns = coord(latitude_deg, "N", "S", 2)
    lon, ew = coord(longitude_deg, "E", "W", 3)
    utc = datetime.utcnow().strftime("%H%M%S")
    body = f"GPGGA,{utc},{lat},{ns},{lon},{ew},1,08,1.0,{float(altitude_m):.1f},M,0.0,M,,"
    return f"${body}*{nmea_checksum(body)}\r\n"


def fix_quality_name(value):
    return {
        "0": "invalid",
        "1": "GPS",
        "2": "DGPS",
        "4": "RTK fixed",
        "5": "RTK float",
        "6": "estimated",
    }.get(str(value), "unknown")


class NtripCorrectionClient:
    def __init__(self, args, serial_write, stop_event, latest_nmea):
        self.args = args
        self.serial_write = serial_write
        self.stop_event = stop_event
        self.latest_nmea = latest_nmea
        self.thread = None
        self.bytes_received = 0
        self.error = None
        self.connected = False

    def start(self):
        self.thread = threading.Thread(target=self._run, name="ntrip", daemon=True)
        self.thread.start()

    def stop(self):
        if self.thread is not None:
            self.thread.join(timeout=3.0)

    def current_gga(self):
        gga = self.latest_nmea.get("gga")
        if gga:
            return gga if gga.endswith("\r\n") else f"{gga}\r\n"
        if self.args.rtk_ntrip_gga:
            gga = self.args.rtk_ntrip_gga
            return gga if gga.endswith("\r\n") else f"{gga}\r\n"
        if self.args.rtk_initial_latitude_deg is None or self.args.rtk_initial_longitude_deg is None:
            return None
        return build_gga_sentence(
            self.args.rtk_initial_latitude_deg,
            self.args.rtk_initial_longitude_deg,
            self.args.rtk_initial_altitude_m,
        )

    def request(self):
        mountpoint = self.args.rtk_ntrip_mountpoint.lstrip("/")
        lines = [
            f"GET /{mountpoint} HTTP/1.0",
            f"Host: {self.args.rtk_ntrip_host}",
            "User-Agent: NTRIP gps-rtk-test",
            "Ntrip-Version: Ntrip/2.0",
            "Connection: close",
        ]
        if self.args.rtk_ntrip_username or self.args.rtk_ntrip_password:
            credentials = f"{self.args.rtk_ntrip_username}:{self.args.rtk_ntrip_password}"
            token = base64.b64encode(credentials.encode("utf-8")).decode("ascii")
            lines.append(f"Authorization: Basic {token}")
        lines.extend(["", ""])
        return "\r\n".join(lines).encode("ascii")

    def _run(self):
        while not self.stop_event.is_set():
            self.connected = False
            try:
                with socket.create_connection(
                    (self.args.rtk_ntrip_host, self.args.rtk_ntrip_port),
                    timeout=10.0,
                ) as sock:
                    sock.settimeout(1.0)
                    sock.sendall(self.request())
                    response = b""
                    while b"\r\n\r\n" not in response and len(response) < 4096:
                        chunk = sock.recv(1)
                        if not chunk:
                            break
                        response += chunk
                    header, _, remainder = response.partition(b"\r\n\r\n")
                    first_line = header.splitlines()[0].decode("ascii", errors="replace") if header else ""
                    if "200" not in first_line and "ICY 200" not in first_line:
                        raise RuntimeError(f"NTRIP rejected request: {first_line}")

                    self.connected = True
                    print(f"NTRIP connected: {self.args.rtk_ntrip_host}:{self.args.rtk_ntrip_port}/{self.args.rtk_ntrip_mountpoint}")
                    last_gga_time = 0.0
                    gga = self.current_gga()
                    if gga:
                        sock.sendall(gga.encode("ascii", errors="ignore"))
                        last_gga_time = time.monotonic()

                    if remainder:
                        self.serial_write(remainder)
                        self.bytes_received += len(remainder)

                    while not self.stop_event.is_set():
                        now = time.monotonic()
                        if self.args.rtk_ntrip_gga_interval > 0:
                            if now - last_gga_time >= self.args.rtk_ntrip_gga_interval:
                                gga = self.current_gga()
                                if gga:
                                    sock.sendall(gga.encode("ascii", errors="ignore"))
                                    last_gga_time = now
                        try:
                            data = sock.recv(4096)
                        except socket.timeout:
                            continue
                        if not data:
                            break
                        self.serial_write(data)
                        self.bytes_received += len(data)
            except Exception as exc:
                self.error = str(exc)
                if not self.stop_event.is_set():
                    print(f"NTRIP reconnecting after error: {self.error}")
                    self.stop_event.wait(self.args.rtk_ntrip_reconnect_delay)


def parse_args():
    parser = argparse.ArgumentParser(description="Test GPS NMEA and RTK NTRIP correction input without camera capture.")
    parser.add_argument("--gps-device", default="/dev/ttyACM0", help="GPS serial device")
    parser.add_argument("--gps-baudrate", type=int, default=921600, help="GPS serial baudrate")
    parser.add_argument("--duration", type=nonnegative_float, default=0.0, help="Stop after N seconds; 0 runs until Ctrl+C")
    parser.add_argument("--print-raw", action="store_true", help="Print every raw NMEA line")
    parser.add_argument("--no-rtk", action="store_true", help="Read GPS only without NTRIP RTCM injection")
    parser.add_argument("--rtk-ntrip-host", default="www.gnssdata.or.kr", help="NTRIP caster host")
    parser.add_argument("--rtk-ntrip-port", type=int, default=2101, help="NTRIP caster TCP port")
    parser.add_argument("--rtk-ntrip-mountpoint", default="YANJ-RTCM31", help="NTRIP mountpoint name")
    parser.add_argument("--rtk-ntrip-username", default="pjmsm0319@gmail.com", help="NTRIP username")
    parser.add_argument("--rtk-ntrip-password", default="gnss", help="NTRIP password")
    parser.add_argument("--rtk-ntrip-gga", default="", help="Explicit NMEA GGA sentence sent to the NTRIP caster")
    parser.add_argument("--rtk-ntrip-gga-interval", type=nonnegative_float, default=10.0, help="Seconds between GGA messages to caster")
    parser.add_argument("--rtk-ntrip-reconnect-delay", type=nonnegative_float, default=5.0, help="Seconds before reconnect after NTRIP error")
    parser.add_argument("--rtk-initial-latitude-deg", type=float, default=None, help="Initial approximate antenna latitude if GPS has not emitted GGA yet")
    parser.add_argument("--rtk-initial-longitude-deg", type=float, default=None, help="Initial approximate antenna longitude if GPS has not emitted GGA yet")
    parser.add_argument("--rtk-initial-altitude-m", type=float, default=0.0, help="Initial approximate antenna altitude")
    return parser.parse_args()


def main():
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    args = parse_args()
    latest_nmea = {}
    serial_write_lock = threading.Lock()
    stop_event = threading.Event()
    started = time.monotonic()
    last_status = 0.0
    last_fix_quality = None

    print(f"Opening GPS serial: {args.gps_device} @ {args.gps_baudrate}")
    with serial.Serial(args.gps_device, args.gps_baudrate, timeout=0.5) as port:
        def write_corrections(data):
            with serial_write_lock:
                port.write(data)

        ntrip = None
        if not args.no_rtk:
            ntrip = NtripCorrectionClient(args, write_corrections, stop_event, latest_nmea)
            ntrip.start()

        try:
            while not stop_requested:
                if args.duration and time.monotonic() - started >= args.duration:
                    break
                raw = port.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if args.print_raw:
                    print(line)

                parsed = parse_nmea_line(line)
                if parsed.get("nmea_type") == "GGA":
                    latest_nmea["gga"] = line
                    fix_quality = parsed.get("fix_quality", "")
                    if fix_quality != last_fix_quality:
                        print(
                            "GGA fix_quality="
                            f"{fix_quality} ({fix_quality_name(fix_quality)}), "
                            f"sats={parsed.get('satellites', '')}, "
                            f"hdop={parsed.get('hdop', '')}, "
                            f"correction_age={parsed.get('differential_age_s', '') or '-'}s, "
                            f"base={parsed.get('reference_station_id', '') or '-'}, "
                            f"lat={parsed.get('latitude_deg', '')}, "
                            f"lon={parsed.get('longitude_deg', '')}, "
                            f"alt={parsed.get('altitude_m', '')}"
                        )
                        last_fix_quality = fix_quality

                now = time.monotonic()
                if now - last_status >= 5.0:
                    if ntrip is not None:
                        status = "connected" if ntrip.connected else "connecting"
                        print(f"Status: NTRIP {status}, RTCM bytes={ntrip.bytes_received}")
                    last_status = now
        finally:
            stop_event.set()
            if ntrip is not None:
                ntrip.stop()


if __name__ == "__main__":
    main()
