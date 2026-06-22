#!/usr/bin/env python3

import argparse
import base64
import csv
import json
import queue
import signal
import socket
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
import serial


WIDTH = 1280
HEIGHT = 720
MIN_DEPTHAI_MAJOR = 3
stop_requested = False

GPS_FIX_QUALITY_NAMES = {
    "0": "invalid",
    "1": "standalone",
    "2": "DGPS",
    "3": "PPS",
    "4": "RTK fixed",
    "5": "RTK float",
    "6": "estimated",
    "7": "manual",
    "8": "simulation",
}


def request_stop(signum, frame):
    global stop_requested
    stop_requested = True


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Value must be >= 1.")
    return number


def png_compression_level(value):
    number = int(value)
    if number < 0 or number > 9:
        raise argparse.ArgumentTypeError("PNG compression must be between 0 and 9.")
    return number


def jpeg_quality(value):
    number = int(value)
    if number < 1 or number > 100:
        raise argparse.ArgumentTypeError("JPEG quality must be between 1 and 100.")
    return number


def nonnegative_float(value):
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0.")
    return number


def time_to_ns(value):
    if value is None:
        return ""
    if hasattr(value, "total_seconds"):
        return int(value.total_seconds() * 1_000_000_000)
    return ""


def ns_delta_ms(a_ns, b_ns):
    if a_ns == "" or b_ns == "":
        return ""
    return (a_ns - b_ns) / 1_000_000.0


def get_sequence_num(message):
    if hasattr(message, "getSequenceNum"):
        return message.getSequenceNum()
    return ""


def get_device_ts_ns(message):
    if hasattr(message, "getTimestampDevice"):
        return time_to_ns(message.getTimestampDevice())
    if hasattr(message, "getTimestamp"):
        return time_to_ns(message.getTimestamp())
    return ""


def get_group_item(group, name):
    try:
        return group[name]
    except KeyError:
        pass
    raise KeyError(f"Sync group did not contain '{name}'")


def imgframe_type_name(message):
    try:
        frame_type = message.getType()
    except Exception:
        return ""
    return getattr(frame_type, "name", str(frame_type))


def get_color_cv_frame(message):
    data = np.asarray(message.getData(), dtype=np.uint8)
    if data.size >= 2 and data[0] == 0xFF and data[1] == 0xD8:
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Failed to decode MJPEG RGB frame")
        return frame

    frame_type = message.getType()
    width = int(message.getWidth())
    height = int(message.getHeight())
    if frame_type in (dai.ImgFrame.Type.NV12, dai.ImgFrame.Type.NV21):
        stride = int(message.getStride() or width)
        expected_size = stride * height * 3 // 2
        if width > 0 and height > 0 and data.size >= expected_size:
            yuv = data[:expected_size].reshape((height * 3 // 2, stride))
            if stride != width:
                yuv = yuv[:, :width]
            code = cv2.COLOR_YUV2BGR_NV12 if frame_type == dai.ImgFrame.Type.NV12 else cv2.COLOR_YUV2BGR_NV21
            return cv2.cvtColor(yuv, code)

    return message.getCvFrame()


def get_confidence_cv_frame(message):
    data = np.asarray(message.getData(), dtype=np.uint8)
    if data.size >= 2 and data[0] == 0xFF and data[1] == 0xD8:
        frame = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise RuntimeError("Failed to decode MJPEG confidence frame")
        return frame
    return message.getFrame()


def drain_message_queue(queue, buffer, max_buffer=180):
    if queue is None:
        return 0
    drained = 0
    while True:
        message = queue.tryGet()
        if message is None:
            break
        buffer.append(message)
        drained += 1
    if len(buffer) > max_buffer:
        del buffer[:len(buffer) - max_buffer]
    return drained


def discard_message_queue(queue):
    if queue is None:
        return 0
    discarded = 0
    while queue.tryGet() is not None:
        discarded += 1
    return discarded


def take_nearest_message(buffer, target_ts_ns, max_delta_ms):
    if not buffer:
        return None
    if target_ts_ns == "":
        return buffer.pop()

    best_index = None
    best_delta_ns = None
    for index, message in enumerate(buffer):
        ts_ns = get_device_ts_ns(message)
        if ts_ns == "":
            continue
        delta_ns = abs(ts_ns - target_ts_ns)
        if best_delta_ns is None or delta_ns < best_delta_ns:
            best_index = index
            best_delta_ns = delta_ns

    if best_index is None:
        return None
    if best_delta_ns is not None and best_delta_ns > int(max_delta_ms * 1_000_000):
        cutoff_ns = target_ts_ns - int(max_delta_ms * 1_000_000)
        while buffer:
            ts_ns = get_device_ts_ns(buffer[0])
            if ts_ns == "" or ts_ns >= cutoff_ns:
                break
            buffer.pop(0)
        return None

    message = buffer.pop(best_index)
    if best_index > 0:
        del buffer[:best_index]
    return message


def nearest_index_and_delta(buffer, target_ts_ns):
    best_index = None
    best_delta_ns = None
    for index, message in enumerate(buffer):
        ts_ns = get_device_ts_ns(message)
        if ts_ns == "" or target_ts_ns == "":
            continue
        delta_ns = abs(ts_ns - target_ts_ns)
        if best_delta_ns is None or delta_ns < best_delta_ns:
            best_index = index
            best_delta_ns = delta_ns
    return best_index, best_delta_ns


def newest_ts_ns(buffer):
    for message in reversed(buffer):
        ts_ns = get_device_ts_ns(message)
        if ts_ns != "":
            return ts_ns
    return ""


def build_host_synced_group(rgb_buffer, depth_buffer, imu_buffer, threshold_ms):
    threshold_ns = int(threshold_ms * 1_000_000)
    while rgb_buffer:
        rgb_msg = rgb_buffer[0]
        rgb_ts_ns = get_device_ts_ns(rgb_msg)
        if rgb_ts_ns == "":
            rgb_buffer.pop(0)
            continue
        if not depth_buffer or not imu_buffer:
            return None

        depth_index, depth_delta_ns = nearest_index_and_delta(depth_buffer, rgb_ts_ns)
        imu_index, imu_delta_ns = nearest_index_and_delta(imu_buffer, rgb_ts_ns)

        depth_ready = depth_index is not None and depth_delta_ns is not None and depth_delta_ns <= threshold_ns
        imu_ready = imu_index is not None and imu_delta_ns is not None and imu_delta_ns <= threshold_ns
        if depth_ready and imu_ready:
            rgb_buffer.pop(0)
            depth_msg = depth_buffer.pop(depth_index)
            imu_msg = imu_buffer.pop(imu_index)
            return {"rgb": rgb_msg, "depth": depth_msg, "imu": imu_msg}

        newest_depth = newest_ts_ns(depth_buffer)
        newest_imu = newest_ts_ns(imu_buffer)
        depth_has_passed = newest_depth != "" and newest_depth > rgb_ts_ns + threshold_ns
        imu_has_passed = newest_imu != "" and newest_imu > rgb_ts_ns + threshold_ns
        if depth_has_passed or imu_has_passed:
            rgb_buffer.pop(0)
            continue
        return None
    return None


def require_depthai_v3():
    major = int(dai.__version__.split(".", 1)[0])
    if major < MIN_DEPTHAI_MAJOR:
        raise RuntimeError(
            f"This script is DepthAI v3-only. Found depthai=={dai.__version__}; "
            "activate .venv or install depthai>=3."
        )


def enum_by_name(enum_cls, name):
    try:
        return getattr(enum_cls, name)
    except AttributeError as exc:
        valid_names = ", ".join(item for item in dir(enum_cls) if item.isupper())
        raise argparse.ArgumentTypeError(f"Invalid value '{name}'. Valid values: {valid_names}") from exc


def request_camera_output(camera, fps, size=(WIDTH, HEIGHT), frame_type=None):
    capability = dai.ImgFrameCapability()
    capability.size.fixed(size)
    capability.fps.fixed(fps)
    if frame_type is not None:
        capability.type = frame_type
    return camera.requestOutput(capability, True)


def make_file_stem(frame_index, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now()
    return f"{timestamp.strftime('%Y-%m-%d-%H-%M-%S')}-frame{frame_index:07d}"


def now_wall_iso():
    return datetime.now().isoformat(timespec="milliseconds")


def wall_iso_from_unix_ns(unix_ns):
    return datetime.fromtimestamp(unix_ns / 1_000_000_000.0).isoformat(timespec="milliseconds")


def parse_gps_datetime_utc_ns(sample):
    date_text = sample.get("date_utc", "")
    time_text = sample.get("gps_time_utc", "")
    if not date_text or not time_text:
        return None
    for fmt in ("%d%m%y%H%M%S.%f", "%d%m%y%H%M%S"):
        try:
            parsed = datetime.strptime(f"{date_text}{time_text}", fmt).replace(tzinfo=timezone.utc)
            return int(parsed.timestamp() * 1_000_000_000)
        except ValueError:
            continue
    return None


class DeviceHostClockMapper:
    """Map DepthAI device timestamps to host clocks without queue-delay bias."""

    def __init__(self):
        self.device_to_host_monotonic_offset_ns = None

    def stamp(self, device_ts_ns):
        dequeue_monotonic_ns = time.monotonic_ns()
        dequeue_wall_ns = time.time_ns()
        if device_ts_ns in (None, ""):
            return {
                "capture_monotonic_ns": dequeue_monotonic_ns,
                "capture_wall_time": wall_iso_from_unix_ns(dequeue_wall_ns),
                "dequeue_monotonic_ns": dequeue_monotonic_ns,
                "dequeue_wall_time": wall_iso_from_unix_ns(dequeue_wall_ns),
                "queue_lag_ms": 0.0,
            }

        candidate_offset_ns = dequeue_monotonic_ns - int(device_ts_ns)
        if (
            self.device_to_host_monotonic_offset_ns is None
            or candidate_offset_ns < self.device_to_host_monotonic_offset_ns
        ):
            self.device_to_host_monotonic_offset_ns = candidate_offset_ns

        capture_monotonic_ns = int(device_ts_ns) + self.device_to_host_monotonic_offset_ns
        capture_wall_ns = capture_monotonic_ns + (dequeue_wall_ns - dequeue_monotonic_ns)
        return {
            "capture_monotonic_ns": capture_monotonic_ns,
            "capture_wall_time": wall_iso_from_unix_ns(capture_wall_ns),
            "dequeue_monotonic_ns": dequeue_monotonic_ns,
            "dequeue_wall_time": wall_iso_from_unix_ns(dequeue_wall_ns),
            "queue_lag_ms": (dequeue_monotonic_ns - capture_monotonic_ns) / 1_000_000.0,
        }


def gps_fix_details(fix_quality):
    quality = str(fix_quality or "")
    name = GPS_FIX_QUALITY_NAMES.get(quality, "unknown")
    if quality == "4":
        rtk_status = "fixed"
    elif quality == "5":
        rtk_status = "float"
    elif quality == "0":
        rtk_status = "invalid"
    elif quality:
        rtk_status = "not_rtk"
    else:
        rtk_status = "unknown"
    return {
        "fix_quality_name": name,
        "rtk_status": rtk_status,
        "rtk_fixed": quality == "4",
        "rtk_corrected": quality in ("4", "5"),
    }


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

    sentence = parts[0]
    kind = sentence[-3:]
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
            "geoid_separation_m": parts[11] if len(parts) > 11 else "",
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


class NtripCorrectionClient:
    def __init__(self, config, serial_write, stop_event, latest_nmea=None):
        self.config = config
        self.serial_write = serial_write
        self.stop_event = stop_event
        self.latest_nmea = latest_nmea if latest_nmea is not None else {}
        self.thread = None
        self.bytes_received = 0
        self.error = None
        self.connected = False

    def start(self):
        self.thread = threading.Thread(target=self._run, name="gps-ntrip", daemon=True)
        self.thread.start()

    def stop(self):
        if self.thread is not None:
            self.thread.join(timeout=3.0)

    def _fallback_gga(self):
        gga = self.config.get("gga")
        if gga:
            return gga if gga.endswith("\r\n") else f"{gga}\r\n"
        latitude = self.config.get("latitude")
        longitude = self.config.get("longitude")
        if latitude is None or longitude is None:
            return None
        return build_gga_sentence(latitude, longitude, self.config.get("altitude", 0.0))

    def _current_gga(self):
        gga = self.latest_nmea.get("gga")
        if gga:
            return gga if gga.endswith("\r\n") else f"{gga}\r\n"
        return self._fallback_gga()

    def _request(self):
        mountpoint = self.config["mountpoint"].lstrip("/")
        lines = [
            f"GET /{mountpoint} HTTP/1.0",
            f"Host: {self.config['host']}",
            "User-Agent: NTRIP synced-image-recorder",
            "Ntrip-Version: Ntrip/2.0",
            "Connection: close",
        ]
        username = self.config.get("username")
        password = self.config.get("password")
        if username or password:
            token = base64.b64encode(f"{username or ''}:{password or ''}".encode("utf-8")).decode("ascii")
            lines.append(f"Authorization: Basic {token}")
        lines.extend(["", ""])
        return "\r\n".join(lines).encode("ascii")

    def _run(self):
        reconnect_delay = self.config.get("reconnect_delay", 5.0)
        while not self.stop_event.is_set():
            self.connected = False
            try:
                with socket.create_connection((self.config["host"], self.config["port"]), timeout=10.0) as sock:
                    sock.settimeout(1.0)
                    sock.sendall(self._request())
                    response = b""
                    while b"\r\n\r\n" not in response and len(response) < 4096:
                        chunk = sock.recv(1)
                        if not chunk:
                            break
                        response += chunk
                    header, _, remainder = response.partition(b"\r\n\r\n")
                    first_line = header.splitlines()[0].decode("ascii", errors="replace") if header else ""
                    if "200" not in first_line and "ICY 200" not in first_line:
                        raise RuntimeError(f"NTRIP caster rejected request: {first_line}")

                    self.connected = True
                    self.error = None

                    last_gga_time = 0.0
                    gga = self._current_gga()
                    if gga:
                        sock.sendall(gga.encode("ascii", errors="ignore"))
                        last_gga_time = time.monotonic()

                    if remainder:
                        self.serial_write(remainder)
                        self.bytes_received += len(remainder)

                    while not self.stop_event.is_set():
                        now = time.monotonic()
                        if self.config.get("gga_interval", 10.0) > 0:
                            if now - last_gga_time >= self.config.get("gga_interval", 10.0):
                                gga = self._current_gga()
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
                self.connected = False
                self.error = str(exc)
                if not self.stop_event.is_set():
                    print(f"RTK NTRIP reconnecting after error: {self.error}")
                    self.stop_event.wait(reconnect_delay)


class NmeaParserState:
    def __init__(self):
        self.latest = {}

    def __call__(self, line):
        parsed = parse_nmea_line(line)
        if not parsed:
            return {}

        for key in (
            "gps_time_utc",
            "date_utc",
            "latitude_deg",
            "longitude_deg",
            "altitude_m",
            "fix_quality",
            "satellites",
            "hdop",
            "status",
            "speed_knots",
            "course_deg",
            "geoid_separation_m",
            "differential_age_s",
            "reference_station_id",
        ):
            value = parsed.get(key)
            if value not in ("", None):
                self.latest[key] = value

        merged = dict(parsed)
        for key, value in self.latest.items():
            if merged.get(key) in ("", None):
                merged[key] = value
        merged.update(gps_fix_details(merged.get("fix_quality")))
        merged["position_valid"] = bool(
            merged.get("latitude_deg") not in ("", None)
            and merged.get("longitude_deg") not in ("", None)
            and str(merged.get("fix_quality", "")) != "0"
        )
        return merged


def parse_ebimu_line(line):
    if not line.startswith("*"):
        return {}

    parts = line[1:].split(",")
    parsed = {"external_imu_format": "ebimu"}

    try:
        values = [float(part) for part in parts]
    except ValueError:
        return parsed

    # Recommended configure_ebimu.py output:
    # *qz,qy,qx,qw,gx,gy,gz,ax,ay,az,mx,my,mz,timestamp_ms
    if len(values) >= 14:
        parsed.update({
            "orientation_format": "quaternion",
            "q_z": values[0],
            "q_y": values[1],
            "q_x": values[2],
            "q_w": values[3],
            "gyro_x": values[4],
            "gyro_y": values[5],
            "gyro_z": values[6],
            "accel_x": values[7],
            "accel_y": values[8],
            "accel_z": values[9],
            "mag_x": values[10],
            "mag_y": values[11],
            "mag_z": values[12],
            "ebimu_timestamp_ms": int(values[13]),
        })
        return parsed

    # Euler check mode:
    # *roll,pitch,yaw,gx,gy,gz,ax,ay,az,mx,my,mz,timestamp_ms
    if len(values) >= 13:
        parsed.update({
            "orientation_format": "euler",
            "roll_deg": values[0],
            "pitch_deg": values[1],
            "yaw_deg": values[2],
            "gyro_x": values[3],
            "gyro_y": values[4],
            "gyro_z": values[5],
            "accel_x": values[6],
            "accel_y": values[7],
            "accel_z": values[8],
            "mag_x": values[9],
            "mag_y": values[10],
            "mag_z": values[11],
            "ebimu_timestamp_ms": int(values[12]),
        })

    return parsed


def matrix_to_json(matrix):
    return [[float(value) for value in row] for row in matrix]


def adjust_intrinsics_for_saved_transform(intrinsics, args):
    adjusted = matrix_to_json(intrinsics)
    if args.rotate_180:
        adjusted[0][2] = (WIDTH - 1) - adjusted[0][2]
        adjusted[1][2] = (HEIGHT - 1) - adjusted[1][2]
    if args.flip:
        adjusted[1][2] = (HEIGHT - 1) - adjusted[1][2]
    return adjusted


def read_camera_model_metadata(device, args):
    try:
        calibration = device.readCalibration()
        rgb_intrinsics = calibration.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, WIDTH, HEIGHT)
        metadata = {
            "model": "pinhole",
            "source": "DepthAI device calibration",
            "socket": "CAM_A",
            "width": WIDTH,
            "height": HEIGHT,
            "intrinsics_original": matrix_to_json(rgb_intrinsics),
            "intrinsics": adjust_intrinsics_for_saved_transform(rgb_intrinsics, args),
            "image_stream_undistorted": False,
            "coordinate_unprojection": "cv2.undistortPoints with original intrinsics and distortion coefficients",
            "notes": (
                "Saved RGB remains in the camera's distorted pixel geometry. Intrinsics are adjusted "
                "for the saved flip/rotation; position unprojection removes lens distortion per pixel."
            ),
        }
        try:
            metadata["distortion_coefficients"] = [
                float(value)
                for value in calibration.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A)
            ]
            metadata["distortion_model"] = (
                "opencv_rational_thin_prism_tilt_14"
                if len(metadata["distortion_coefficients"]) == 14
                else "opencv"
            )
        except Exception:
            metadata["distortion_coefficients"] = []
            metadata["distortion_model"] = "unavailable"
        try:
            metadata["fov_deg"] = float(calibration.getFov(dai.CameraBoardSocket.CAM_A))
        except Exception:
            metadata["fov_deg"] = None
        return metadata
    except Exception as exc:
        print(f"Camera calibration metadata unavailable: {exc}")
        return {
            "model": "pinhole",
            "source": "unavailable",
            "socket": "CAM_A",
            "width": WIDTH,
            "height": HEIGHT,
            "intrinsics": None,
            "error": str(exc),
        }


class SerialRateLimitedReader:
    def __init__(self, name, device, baudrate, max_hz, parser=None, timeout=0.2, rtk_config=None):
        self.name = name
        self.device = device
        self.baudrate = baudrate
        self.max_hz = max_hz
        self.parser = parser
        self.timeout = timeout
        self.rtk_config = rtk_config
        self.rtk_client = None
        self.write_lock = threading.Lock()
        self.latest_nmea = {}
        self.samples = queue.Queue()
        self.recent = deque(maxlen=20000)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None
        self.sample_count = 0
        self.dropped_count = 0
        self.started = False

    def start(self):
        self.thread = threading.Thread(target=self._run, name=f"{self.name}-serial", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def _run(self):
        min_interval = 1.0 / self.max_hz if self.max_hz > 0 else 0.0
        last_saved_monotonic = 0.0
        try:
            with serial.Serial(self.device, self.baudrate, timeout=self.timeout) as port:
                self.started = True
                if self.rtk_config is not None:
                    def write_corrections(data):
                        with self.write_lock:
                            port.write(data)

                    self.rtk_client = NtripCorrectionClient(
                        self.rtk_config,
                        write_corrections,
                        self.stop_event,
                        latest_nmea=self.latest_nmea,
                    )
                    self.rtk_client.start()
                    print(
                        f"RTK NTRIP enabled: {self.rtk_config['host']}:{self.rtk_config['port']}/"
                        f"{self.rtk_config['mountpoint']}"
                    )
                while not self.stop_event.is_set():
                    raw = port.readline()
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    if self.rtk_config is not None and parse_nmea_line(line).get("nmea_type") == "GGA":
                        self.latest_nmea["gga"] = line
                    parsed = self.parser(line) if self.parser is not None else {}
                    received_monotonic = time.monotonic()
                    # GGA carries the RTK solution state, correction age, and base ID.
                    # Preserve it even when other high-rate NMEA sentences are throttled.
                    preserve_gga = self.name == "gps" and parsed.get("nmea_type") == "GGA"
                    if (
                        min_interval
                        and not preserve_gga
                        and received_monotonic - last_saved_monotonic < min_interval
                    ):
                        self.dropped_count += 1
                        continue
                    last_saved_monotonic = received_monotonic

                    received_wall_ns = time.time_ns()
                    received_monotonic_ns = time.monotonic_ns()
                    sample = {
                        "sample_index": self.sample_count,
                        "source": self.name,
                        "device": self.device,
                        "host_wall_time": wall_iso_from_unix_ns(received_wall_ns),
                        "host_monotonic_ns": received_monotonic_ns,
                        "raw": line,
                    }
                    sample.update(parsed)
                    if self.name == "gps":
                        measurement_wall_ns = parse_gps_datetime_utc_ns(sample)
                        if measurement_wall_ns is not None:
                            receive_latency_ns = received_wall_ns - measurement_wall_ns
                            sample["measurement_wall_time_utc"] = datetime.fromtimestamp(
                                measurement_wall_ns / 1_000_000_000.0,
                                tz=timezone.utc,
                            ).isoformat(timespec="milliseconds")
                            sample["measurement_host_monotonic_ns"] = (
                                received_monotonic_ns - receive_latency_ns
                            )
                            sample["receive_latency_ms"] = receive_latency_ns / 1_000_000.0

                    self.sample_count += 1
                    self.samples.put(sample)
                    with self.lock:
                        self.recent.append(sample)
        except Exception as exc:
            self.error = str(exc)
            print(f"{self.name} serial disabled: {self.error}")
        finally:
            if self.rtk_client is not None:
                self.rtk_client.stop()

    def drain(self):
        drained = []
        while True:
            try:
                drained.append(self.samples.get_nowait())
            except queue.Empty:
                break
        return drained

    def nearest(self, host_monotonic_ns, timestamp_key="host_monotonic_ns", predicate=None):
        with self.lock:
            if not self.recent:
                return None
            candidates = [
                sample
                for sample in self.recent
                if sample.get(timestamp_key) not in (None, "")
                and (predicate is None or predicate(sample))
            ]
            if not candidates:
                return None
            return min(
                candidates,
                key=lambda sample: abs(sample[timestamp_key] - host_monotonic_ns),
            )

    def latest_sample(self):
        with self.lock:
            return self.recent[-1] if self.recent else None

    def status_text(self):
        sample = self.latest_sample() or {}
        quality = sample.get("fix_quality", "")
        details = gps_fix_details(quality)
        text = f"gps_fix={quality or '-'}({details['fix_quality_name']})"
        if self.rtk_client is not None:
            state = "connected" if self.rtk_client.connected else "connecting"
            text += f", ntrip={state}, rtcm_bytes={self.rtk_client.bytes_received}"
        return text


def serial_max_hz_values(args):
    gps_max_hz = args.gps_max_hz if args.gps_max_hz is not None else args.serial_max_hz
    external_imu_max_hz = (
        args.external_imu_max_hz
        if args.external_imu_max_hz is not None
        else args.serial_max_hz
    )
    return gps_max_hz, external_imu_max_hz


def gps_status_text(serial_readers):
    reader = (serial_readers or {}).get("gps")
    return reader.status_text() if reader is not None else "gps=disabled"


def build_rtk_config(args):
    if not args.rtk_ntrip_host:
        return None
    if not args.rtk_ntrip_mountpoint:
        raise ValueError("--rtk-ntrip-mountpoint is required when --rtk-ntrip-host is set")
    return {
        "host": args.rtk_ntrip_host,
        "port": args.rtk_ntrip_port,
        "mountpoint": args.rtk_ntrip_mountpoint,
        "username": args.rtk_ntrip_username,
        "password": args.rtk_ntrip_password,
        "latitude": args.rtk_initial_latitude_deg,
        "longitude": args.rtk_initial_longitude_deg,
        "altitude": args.rtk_initial_altitude_m,
        "gga": args.rtk_ntrip_gga,
        "gga_interval": args.rtk_ntrip_gga_interval,
        "reconnect_delay": args.rtk_ntrip_reconnect_delay,
    }


def create_serial_readers(args):
    readers = {}
    gps_max_hz, external_imu_max_hz = serial_max_hz_values(args)
    if args.enable_gps:
        readers["gps"] = SerialRateLimitedReader(
            "gps",
            args.gps_device,
            args.gps_baudrate,
            gps_max_hz,
            parser=NmeaParserState(),
            rtk_config=build_rtk_config(args),
        )
    if args.enable_external_imu:
        parser = parse_ebimu_line if args.external_imu_format == "ebimu" else None
        readers["external_imu"] = SerialRateLimitedReader(
            "external_imu",
            args.external_imu_device,
            args.external_imu_baudrate,
            external_imu_max_hz,
            parser=parser,
        )
    return readers


def start_serial_readers(readers):
    for reader in readers.values():
        print(f"Starting {reader.name} serial: {reader.device} @ {reader.baudrate}, max {reader.max_hz:g} Hz")
        reader.start()


def stop_serial_readers(readers):
    for reader in readers.values():
        reader.stop()


def usb_speed_name(device):
    try:
        speed = device.getUsbSpeed()
        return getattr(speed, "name", str(speed).split(".")[-1])
    except Exception:
        return "UNKNOWN"


def connect_depthai_device(args):
    attempts = max(int(args.usb3_retries), 0) + 1
    last_speed = "UNKNOWN"
    for attempt in range(1, attempts + 1):
        device = dai.Device(dai.UsbSpeed.SUPER)
        last_speed = usb_speed_name(device)
        if last_speed in ("SUPER", "SUPER_PLUS") or args.allow_usb2:
            return device

        device.close()
        if attempt < attempts:
            print(
                f"DepthAI connected at {last_speed} (USB 2.0-class), "
                f"retrying USB 3.x negotiation {attempt}/{attempts - 1}..."
            )
            time.sleep(1.0)

    raise RuntimeError(
        f"DepthAI negotiated {last_speed}, not USB 3.x. 1280x720 RGB/depth/confidence at "
        f"{args.fps} FPS cannot be recorded reliably on this link. Connect the OAK camera "
        "directly to a USB 3.x port with a USB 3.x cable, then rerun. Use --allow-usb2 "
        "only for best-effort low-bandwidth testing."
    )


def resolve_transport_options(args, device):
    speed_name = usb_speed_name(device)

    constrained_usb = speed_name in ("LOW", "FULL", "HIGH", "UNKNOWN")
    args.usb_speed = speed_name
    args.rgb_transport_effective = (
        "mjpeg" if args.rgb_transport == "auto" and constrained_usb else args.rgb_transport
    )
    args.confidence_transport_effective = (
        "mjpeg"
        if args.confidence_transport == "auto" and constrained_usb
        else args.confidence_transport
    )
    if args.rgb_transport_effective == "auto":
        args.rgb_transport_effective = "raw"
    if args.confidence_transport_effective == "auto":
        args.confidence_transport_effective = "raw"

    print(
        f"DepthAI USB speed: {speed_name}; transports "
        f"rgb={args.rgb_transport_effective}, "
        f"confidence={args.confidence_transport_effective if args.save_confidence_map else 'disabled'}"
    )
    if constrained_usb:
        print("USB 2.0-class link detected; using compressed image transport to prevent RGB starvation.")


def create_mjpeg_output(pipeline, input_output, fps, quality):
    encoder = pipeline.create(dai.node.VideoEncoder)
    encoder.setDefaultProfilePreset(fps, dai.VideoEncoderProperties.Profile.MJPEG)
    encoder.setQuality(quality)
    input_output.link(encoder.input)
    return encoder.bitstream


def configure_pipeline(pipeline, args):
    require_depthai_v3()

    color_socket = dai.CameraBoardSocket.CAM_A
    left_socket = dai.CameraBoardSocket.CAM_B
    right_socket = dai.CameraBoardSocket.CAM_C

    cam_rgb = pipeline.create(dai.node.Camera).build(color_socket)
    mono_left = pipeline.create(dai.node.Camera).build(left_socket)
    mono_right = pipeline.create(dai.node.Camera).build(right_socket)
    stereo = pipeline.create(dai.node.StereoDepth)
    image_align = None
    confidence_align = None
    if args.depth_alignment_mode == "image-align":
        image_align = pipeline.create(dai.node.ImageAlign)
        if args.save_confidence_map:
            confidence_align = pipeline.create(dai.node.ImageAlign)
    imu = pipeline.create(dai.node.IMU)
    sync = pipeline.create(dai.node.Sync) if args.sync_mode == "device" else None

    rgb_output = request_camera_output(cam_rgb, args.fps, frame_type=dai.ImgFrame.Type.NV12)
    confidence_align_to_output = (
        request_camera_output(cam_rgb, args.fps, frame_type=dai.ImgFrame.Type.GRAY8)
        if confidence_align is not None
        else None
    )
    left_output = request_camera_output(mono_left, args.fps, frame_type=dai.ImgFrame.Type.GRAY8)
    right_output = request_camera_output(mono_right, args.fps, frame_type=dai.ImgFrame.Type.GRAY8)

    stereo.setDefaultProfilePreset(enum_by_name(dai.node.StereoDepth.PresetMode, args.depth_preset))
    stereo.setLeftRightCheck(args.lr_check)
    stereo.setSubpixel(args.subpixel)
    if args.subpixel:
        stereo.setSubpixelFractionalBits(args.subpixel_fractional_bits)
    median_filter_name = args.stereo_median_filter
    if args.subpixel and args.subpixel_fractional_bits > 3 and median_filter_name != "off":
        print("Stereo median filter disabled because subpixel fractional bits 4/5 do not support it.")
        median_filter_name = "off"
    median_filter = {
        "off": dai.MedianFilter.MEDIAN_OFF,
        "3x3": dai.MedianFilter.KERNEL_3x3,
        "5x5": dai.MedianFilter.KERNEL_5x5,
        "7x7": dai.MedianFilter.KERNEL_7x7,
    }[median_filter_name]
    stereo.initialConfig.setMedianFilter(median_filter)
    args.stereo_median_filter_effective = median_filter_name
    stereo.setExtendedDisparity(False)
    stereo.setOutputSize(WIDTH, HEIGHT)
    if args.depth_alignment_mode == "stereo":
        stereo.setDepthAlign(color_socket)
    else:
        image_align.setOutputSize(WIDTH, HEIGHT)
        if confidence_align is not None:
            confidence_align.setOutputSize(WIDTH, HEIGHT)

    imu.enableIMUSensor([
        dai.IMUSensor.ACCELEROMETER_RAW,
        dai.IMUSensor.GYROSCOPE_RAW,
    ], args.imu_rate)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(args.imu_batch)

    if sync is not None:
        sync.setSyncThreshold(timedelta(milliseconds=args.sync_threshold_ms))
        sync.setSyncAttempts(args.sync_attempts)

    left_output.link(stereo.left)
    right_output.link(stereo.right)
    depth_output = None
    if image_align is None:
        depth_output = stereo.depth
        rgb_stream_output = rgb_output
        confidence_output = stereo.confidenceMap if args.save_confidence_map else None
    else:
        stereo.depth.link(image_align.input)
        rgb_output.link(image_align.inputAlignTo)
        rgb_stream_output = rgb_output
        depth_output = image_align.outputAligned
        if args.save_confidence_map:
            stereo.confidenceMap.link(confidence_align.input)
            confidence_align_to_output.link(confidence_align.inputAlignTo)
            confidence_output = confidence_align.outputAligned
        else:
            confidence_output = None

    if args.rgb_transport_effective == "mjpeg":
        rgb_stream_output = create_mjpeg_output(
            pipeline, rgb_stream_output, args.fps, args.rgb_transport_quality
        )
    if confidence_output is not None and args.confidence_transport_effective == "mjpeg":
        confidence_output = create_mjpeg_output(
            pipeline, confidence_output, args.fps, args.confidence_transport_quality
        )

    if sync is not None:
        rgb_stream_output.link(sync.inputs["rgb"])
        depth_output.link(sync.inputs["depth"])
        imu.out.link(sync.inputs["imu"])

    if sync is not None:
        return {
            "mode": "device",
            "sync": sync.out,
            "confidence": confidence_output,
        }
    return {
        "mode": "host",
        "rgb": rgb_stream_output,
        "depth": depth_output,
        "imu": imu.out,
        "confidence": confidence_output,
    }


class ImageDatasetWriter:
    def __init__(self, output_dir, args, camera_model=None):
        self.args = args
        self.camera_model = camera_model or {}
        self.started_wall = datetime.now()
        self.started_monotonic = time.monotonic()
        self.output_dir = Path(output_dir) / self.started_wall.strftime("%Y-%m-%d_%H-%M-%S")
        self.rgb_dir = self.output_dir / "rgb"
        self.depth_dir = self.output_dir / "depth_mm"
        self.confidence_dir = self.output_dir / "confidence" if args.save_confidence_map else None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        if self.confidence_dir is not None:
            self.confidence_dir.mkdir(parents=True, exist_ok=True)

        self.timestamps_file = open(self.output_dir / "timestamps.csv", "w", newline="")
        self.imu_file = open(self.output_dir / "imu.csv", "w", newline="")
        self.gps_file = open(self.output_dir / "gps.csv", "w", newline="") if args.enable_gps else None
        self.external_imu_file = (
            open(self.output_dir / "external_imu.csv", "w", newline="") if args.enable_external_imu else None
        )

        self.timestamps_writer = csv.writer(self.timestamps_file)
        self.timestamps_writer.writerow([
            "frame_index",
            "stem",
            "frame_host_wall_time",
            "frame_host_monotonic_ns",
            "frame_dequeue_host_wall_time",
            "frame_dequeue_host_monotonic_ns",
            "frame_queue_lag_ms",
            "rgb_file",
            "depth_file",
            "confidence_file",
            "rgb_sequence",
            "depth_sequence",
            "confidence_sequence",
            "rgb_device_ts_ns",
            "depth_device_ts_ns",
            "confidence_device_ts_ns",
            "imu_message_device_ts_ns",
            "rgb_depth_delta_ms",
            "depth_confidence_delta_ms",
            "rgb_imu_delta_ms",
            "depth_imu_delta_ms",
            "imu_packets",
            "gps_sample_index",
            "gps_host_monotonic_ns",
            "gps_measurement_host_monotonic_ns",
            "gps_receive_latency_ms",
            "gps_frame_delta_ms",
            "gps_nmea_type",
            "gps_latitude_deg",
            "gps_longitude_deg",
            "gps_altitude_m",
            "gps_fix_quality",
            "gps_fix_quality_name",
            "gps_rtk_status",
            "gps_rtk_fixed",
            "gps_rtk_corrected",
            "gps_position_valid",
            "gps_satellites",
            "gps_hdop",
            "gps_differential_age_s",
            "gps_reference_station_id",
            "external_imu_sample_index",
            "external_imu_host_monotonic_ns",
            "external_imu_frame_delta_ms",
        ])

        self.imu_writer = csv.writer(self.imu_file)
        self.imu_writer.writerow([
            "frame_index",
            "stem",
            "packet_index",
            "imu_message_device_ts_ns",
            "accel_device_ts_ns",
            "accel_x_m_s2",
            "accel_y_m_s2",
            "accel_z_m_s2",
            "gyro_device_ts_ns",
            "gyro_x_rad_s",
            "gyro_y_rad_s",
            "gyro_z_rad_s",
        ])

        self.gps_writer = None
        if self.gps_file is not None:
            self.gps_writer = csv.DictWriter(self.gps_file, fieldnames=[
                "sample_index",
                "host_wall_time",
                "host_monotonic_ns",
                "measurement_wall_time_utc",
                "measurement_host_monotonic_ns",
                "receive_latency_ms",
                "device",
                "nmea_type",
                "gps_time_utc",
                "date_utc",
                "latitude_deg",
                "longitude_deg",
                "altitude_m",
                "fix_quality",
                "fix_quality_name",
                "rtk_status",
                "rtk_fixed",
                "rtk_corrected",
                "position_valid",
                "satellites",
                "hdop",
                "geoid_separation_m",
                "differential_age_s",
                "reference_station_id",
                "status",
                "speed_knots",
                "course_deg",
                "raw",
            ], extrasaction="ignore")
            self.gps_writer.writeheader()

        self.external_imu_writer = None
        if self.external_imu_file is not None:
            self.external_imu_writer = csv.DictWriter(self.external_imu_file, fieldnames=[
                "sample_index",
                "host_wall_time",
                "host_monotonic_ns",
                "device",
                "external_imu_format",
                "orientation_format",
                "q_x",
                "q_y",
                "q_z",
                "q_w",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
                "gyro_x",
                "gyro_y",
                "gyro_z",
                "accel_x",
                "accel_y",
                "accel_z",
                "mag_x",
                "mag_y",
                "mag_z",
                "ebimu_timestamp_ms",
                "raw",
            ], extrasaction="ignore")
            self.external_imu_writer.writeheader()

        self.frame_count = 0
        self.imu_packet_count = 0
        self.confidence_frame_count = 0
        self.gps_sample_count = 0
        self.gps_gga_fix_quality_counts = {}
        self.gps_latest_solution = {}
        self.external_imu_sample_count = 0
        print(f"Dataset opened: {self.output_dir}")

    def write_group(self, group, serial_readers=None, confidence_msg=None):
        if isinstance(group, dict) and "_host_monotonic_ns" in group:
            frame_host_monotonic_ns = group["_host_monotonic_ns"]
            frame_host_wall_time = group["_host_wall_time"]
            frame_dequeue_host_monotonic_ns = group.get("_dequeue_host_monotonic_ns", "")
            frame_dequeue_host_wall_time = group.get("_dequeue_host_wall_time", "")
            frame_queue_lag_ms = group.get("_queue_lag_ms", "")
        else:
            frame_host_monotonic_ns = time.monotonic_ns()
            frame_host_wall_time = now_wall_iso()
            frame_dequeue_host_monotonic_ns = frame_host_monotonic_ns
            frame_dequeue_host_wall_time = frame_host_wall_time
            frame_queue_lag_ms = 0.0
        rgb_msg = get_group_item(group, "rgb")
        depth_msg = get_group_item(group, "depth")
        imu_msg = get_group_item(group, "imu")

        rgb_frame = get_color_cv_frame(rgb_msg)
        depth_frame = depth_msg.getFrame()
        confidence_frame = get_confidence_cv_frame(confidence_msg) if confidence_msg is not None else None

        if rgb_frame.ndim != 3 or rgb_frame.shape[2] < 3:
            data_size = len(rgb_msg.getData()) if hasattr(rgb_msg, "getData") else ""
            raise RuntimeError(
                f"RGB stream produced a non-color frame with shape {rgb_frame.shape}, "
                f"type={imgframe_type_name(rgb_msg)}, data_size={data_size}. "
                "The color camera output must be BGR888i/RGB888i/NV12, not GRAY8."
            )
        if rgb_frame.shape[2] == 4:
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGRA2BGR)

        if (
            np.array_equal(rgb_frame[:, :, 0], rgb_frame[:, :, 1])
            and np.array_equal(rgb_frame[:, :, 1], rgb_frame[:, :, 2])
        ):
            raise RuntimeError(
                "RGB stream contains three identical channels instead of color data. "
                "Check CAM_A output negotiation and do not request a separate GRAY8 "
                "alignment target from the color camera."
            )

        if rgb_frame.shape[:2] != (HEIGHT, WIDTH):
            rgb_frame = cv2.resize(rgb_frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        if depth_frame.shape[:2] != (HEIGHT, WIDTH):
            depth_frame = cv2.resize(depth_frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
        if confidence_frame is not None and confidence_frame.shape[:2] != (HEIGHT, WIDTH):
            confidence_frame = cv2.resize(confidence_frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

        if self.args.flip:
            rgb_frame = cv2.flip(rgb_frame, 0)
            depth_frame = cv2.flip(depth_frame, 0)
            if confidence_frame is not None:
                confidence_frame = cv2.flip(confidence_frame, 0)
        if self.args.rotate_180:
            rgb_frame = cv2.rotate(rgb_frame, cv2.ROTATE_180)
            depth_frame = cv2.rotate(depth_frame, cv2.ROTATE_180)
            if confidence_frame is not None:
                confidence_frame = cv2.rotate(confidence_frame, cv2.ROTATE_180)

        if rgb_frame.dtype != np.uint8:
            rgb_frame = rgb_frame.astype(np.uint8)
        if depth_frame.dtype != np.uint16:
            depth_frame = depth_frame.astype(np.uint16)
        if confidence_frame is not None and confidence_frame.dtype != np.uint8:
            confidence_frame = confidence_frame.astype(np.uint8)

        try:
            capture_datetime = datetime.fromisoformat(frame_host_wall_time)
        except (TypeError, ValueError):
            capture_datetime = datetime.now()
        stem = make_file_stem(self.frame_count, capture_datetime)
        rgb_path = self.rgb_dir / f"{stem}_rgb.{self.args.rgb_format}"
        depth_path = self.depth_dir / f"{stem}_depth_mm.png"
        confidence_path = (
            self.confidence_dir / f"{stem}_confidence.png"
            if self.confidence_dir is not None and confidence_frame is not None
            else None
        )

        self.write_rgb(rgb_path, rgb_frame)
        ok = cv2.imwrite(
            str(depth_path),
            depth_frame,
            [cv2.IMWRITE_PNG_COMPRESSION, self.args.depth_png_compression],
        )
        if not ok:
            raise RuntimeError(f"Failed to write depth image: {depth_path}")
        if confidence_path is not None:
            ok = cv2.imwrite(
                str(confidence_path),
                confidence_frame,
                [cv2.IMWRITE_PNG_COMPRESSION, self.args.confidence_png_compression],
            )
            if not ok:
                raise RuntimeError(f"Failed to write confidence image: {confidence_path}")
            self.confidence_frame_count += 1

        rgb_ts_ns = get_device_ts_ns(rgb_msg)
        depth_ts_ns = get_device_ts_ns(depth_msg)
        confidence_ts_ns = get_device_ts_ns(confidence_msg) if confidence_msg is not None else ""
        imu_ts_ns = get_device_ts_ns(imu_msg)
        imu_packets = getattr(imu_msg, "packets", [])

        self.timestamps_writer.writerow([
            self.frame_count,
            stem,
            frame_host_wall_time,
            frame_host_monotonic_ns,
            frame_dequeue_host_wall_time,
            frame_dequeue_host_monotonic_ns,
            frame_queue_lag_ms,
            rgb_path.relative_to(self.output_dir),
            depth_path.relative_to(self.output_dir),
            confidence_path.relative_to(self.output_dir) if confidence_path is not None else "",
            get_sequence_num(rgb_msg),
            get_sequence_num(depth_msg),
            get_sequence_num(confidence_msg) if confidence_msg is not None else "",
            rgb_ts_ns,
            depth_ts_ns,
            confidence_ts_ns,
            imu_ts_ns,
            ns_delta_ms(rgb_ts_ns, depth_ts_ns),
            ns_delta_ms(depth_ts_ns, confidence_ts_ns),
            ns_delta_ms(rgb_ts_ns, imu_ts_ns),
            ns_delta_ms(depth_ts_ns, imu_ts_ns),
            len(imu_packets),
            *self.nearest_serial_columns(serial_readers, "gps", frame_host_monotonic_ns),
            *self.nearest_serial_columns(serial_readers, "external_imu", frame_host_monotonic_ns),
        ])

        for packet_index, packet in enumerate(imu_packets):
            self.write_imu_packet(self.frame_count, stem, packet_index, imu_ts_ns, packet)

        self.frame_count += 1

    def nearest_serial_columns(self, serial_readers, name, frame_host_monotonic_ns):
        reader = (serial_readers or {}).get(name)
        if reader is None:
            if name == "gps":
                return [""] * 19
            return ["", "", ""]

        if name == "gps":
            timestamp_key = "measurement_host_monotonic_ns"
            sample = reader.nearest(
                frame_host_monotonic_ns,
                timestamp_key=timestamp_key,
                predicate=lambda value: (
                    value.get("nmea_type") == "GGA"
                    and value.get("position_valid") is True
                ),
            )
        else:
            timestamp_key = "host_monotonic_ns"
            sample = reader.nearest(frame_host_monotonic_ns, timestamp_key=timestamp_key)
        if sample is None:
            if name == "gps":
                return [""] * 19
            return ["", "", ""]

        delta_ms = (sample[timestamp_key] - frame_host_monotonic_ns) / 1_000_000.0
        if name == "gps":
            return [
                sample.get("sample_index", ""),
                sample.get("host_monotonic_ns", ""),
                sample.get("measurement_host_monotonic_ns", ""),
                sample.get("receive_latency_ms", ""),
                delta_ms,
                sample.get("nmea_type", ""),
                sample.get("latitude_deg", ""),
                sample.get("longitude_deg", ""),
                sample.get("altitude_m", ""),
                sample.get("fix_quality", ""),
                sample.get("fix_quality_name", ""),
                sample.get("rtk_status", ""),
                sample.get("rtk_fixed", ""),
                sample.get("rtk_corrected", ""),
                sample.get("position_valid", ""),
                sample.get("satellites", ""),
                sample.get("hdop", ""),
                sample.get("differential_age_s", ""),
                sample.get("reference_station_id", ""),
            ]
        return [
            sample.get("sample_index", ""),
            sample.get("host_monotonic_ns", ""),
            delta_ms,
        ]

    def write_serial_samples(self, serial_readers):
        gps_reader = serial_readers.get("gps") if serial_readers else None
        if gps_reader is not None and self.gps_writer is not None:
            for sample in gps_reader.drain():
                self.gps_writer.writerow(sample)
                self.gps_sample_count += 1
                if sample.get("latitude_deg") not in ("", None):
                    self.gps_latest_solution = {
                        key: sample.get(key, "")
                        for key in (
                            "sample_index",
                            "host_wall_time",
                            "nmea_type",
                            "latitude_deg",
                            "longitude_deg",
                            "altitude_m",
                            "fix_quality",
                            "fix_quality_name",
                            "rtk_status",
                            "rtk_fixed",
                            "rtk_corrected",
                            "position_valid",
                            "satellites",
                            "hdop",
                            "differential_age_s",
                            "reference_station_id",
                        )
                    }
                if sample.get("nmea_type") == "GGA":
                    quality = str(sample.get("fix_quality", "") or "unknown")
                    self.gps_gga_fix_quality_counts[quality] = (
                        self.gps_gga_fix_quality_counts.get(quality, 0) + 1
                    )

        external_reader = serial_readers.get("external_imu") if serial_readers else None
        if external_reader is not None and self.external_imu_writer is not None:
            for sample in external_reader.drain():
                self.external_imu_writer.writerow(sample)
                self.external_imu_sample_count += 1

    def write_rgb(self, path, rgb_frame):
        if self.args.rgb_format == "jpg":
            params = [cv2.IMWRITE_JPEG_QUALITY, self.args.rgb_jpeg_quality]
        else:
            params = [cv2.IMWRITE_PNG_COMPRESSION, self.args.rgb_png_compression]
        ok = cv2.imwrite(str(path), rgb_frame, params)
        if not ok:
            raise RuntimeError(f"Failed to write RGB image: {path}")

    def write_imu_packet(self, frame_index, stem, packet_index, imu_message_ts_ns, packet):
        accel = getattr(packet, "acceleroMeter", None)
        gyro = getattr(packet, "gyroscope", None)
        self.imu_writer.writerow([
            frame_index,
            stem,
            packet_index,
            imu_message_ts_ns,
            get_device_ts_ns(accel),
            getattr(accel, "x", ""),
            getattr(accel, "y", ""),
            getattr(accel, "z", ""),
            get_device_ts_ns(gyro),
            getattr(gyro, "x", ""),
            getattr(gyro, "y", ""),
            getattr(gyro, "z", ""),
        ])
        self.imu_packet_count += 1

    def average_fps(self, now=None):
        if now is None:
            now = time.monotonic()
        elapsed = max(now - self.started_monotonic, 1e-6)
        return self.frame_count / elapsed

    def close(self, serial_readers=None):
        self.timestamps_file.close()
        self.imu_file.close()
        if self.gps_file is not None:
            self.gps_file.close()
        if self.external_imu_file is not None:
            self.external_imu_file.close()
        gps_max_hz, external_imu_max_hz = serial_max_hz_values(self.args)
        gps_reader = (serial_readers or {}).get("gps")
        rtk_client = gps_reader.rtk_client if gps_reader is not None else None
        metadata = {
            "started_wall_time": self.started_wall.isoformat(timespec="milliseconds"),
            "closed_wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "width": WIDTH,
            "height": HEIGHT,
            "camera_sockets": {
                "rgb": "CAM_A",
                "stereo_left": "CAM_B",
                "stereo_right": "CAM_C",
            },
            "depth_alignment": {
                "enabled": True,
                "aligned_to": "rgb",
                "aligned_to_socket": "CAM_A",
                "mode": self.args.depth_alignment_mode,
                "method": (
                    "ImageAlign(depth, rgb)"
                    if self.args.depth_alignment_mode == "image-align"
                    else "StereoDepth.setDepthAlign(CAM_A)"
                ),
                "output_size": [WIDTH, HEIGHT],
                "rgb_size": [WIDTH, HEIGHT],
                "depth_size": [WIDTH, HEIGHT],
                "depth_pixel_coordinates_match_rgb": True,
                "uses_device_calibration": True,
            },
            "image_transform": {
                "flip_vertical": self.args.flip,
                "rotate_180": self.args.rotate_180,
                "operation_order": ["flip_vertical", "rotate_180"],
                "applied_to": ["rgb", "depth_mm"] + (["confidence"] if self.args.save_confidence_map else []),
            },
            "camera_model": self.camera_model,
            "world_coordinates": {
                "enabled_in_debug_ui": True,
                "position_source": "gps.csv nearest frame sample",
                "orientation_source": "external_imu.csv EBIMU quaternion nearest frame sample",
                "local_frame": "ENU meters; +east, +north, +up",
                "camera_frame": "saved OpenCV image frame; +x right, +y down, +z forward",
                "sensor_heights_above_ground_m": {
                    "gps_antenna": self.args.gps_height_above_ground_m,
                    "camera": self.args.camera_height_above_ground_m,
                    "external_imu": self.args.external_imu_height_above_ground_m,
                },
                "height_reference": "ground directly below the sensor mount",
                "imu_from_camera_rpy_deg": [
                    self.args.imu_from_camera_roll_deg,
                    self.args.imu_from_camera_pitch_deg,
                    self.args.imu_from_camera_yaw_deg,
                ],
                "camera_mount_rpy_deg": [
                    self.args.camera_mount_roll_deg,
                    self.args.camera_mount_pitch_deg,
                    self.args.camera_mount_yaw_deg,
                ],
                "camera_mount_frame": (
                    "vehicle frame to saved camera frame convention; roll +clockwise, "
                    "pitch +down, yaw +right from vehicle forward"
                ),
                "gps_to_camera_enu_m": [
                    self.args.gps_to_camera_east_m,
                    self.args.gps_to_camera_north_m,
                    self.args.gps_to_camera_up_m,
                ],
                "gps_from_camera_camera_m": [
                    self.args.gps_from_camera_right_m,
                    self.args.gps_from_camera_down_m,
                    self.args.gps_from_camera_forward_m,
                ],
                "gps_to_camera_camera_m": [
                    -self.args.gps_from_camera_right_m,
                    -self.args.gps_from_camera_down_m,
                    -self.args.gps_from_camera_forward_m,
                ],
                "external_imu_from_camera_camera_m": [
                    self.args.external_imu_from_camera_right_m,
                    self.args.external_imu_from_camera_down_m,
                    self.args.external_imu_from_camera_forward_m,
                ],
                "camera_from_external_imu_camera_m": [
                    -self.args.external_imu_from_camera_right_m,
                    -self.args.external_imu_from_camera_down_m,
                    -self.args.external_imu_from_camera_forward_m,
                ],
                "lever_arm_frame": "saved camera frame; +x right, +y down, +z forward",
                "magnetic_declination_deg": self.args.magnetic_declination_deg,
                "rtk_max_correction_age_s": self.args.rtk_max_correction_age_s,
                "rtk_max_hdop": self.args.rtk_max_hdop,
                "notes": [
                    "Coordinates are only as accurate as GPS fix, EBIMU magnetic calibration, and imu_from_camera/gps_to_camera extrinsics.",
                    "Set imu_from_camera_*, gps_from_camera_*, external_imu_from_camera_*, and gps_to_camera_* for production absolute coordinates.",
                    "Default mount heights: GPS 1.50m, camera 1.30m, external IMU 1.15m above ground.",
                ],
            },
            "requested_fps": self.args.fps,
            "average_saved_fps": self.average_fps(),
            "frame_count": self.frame_count,
            "imu_packet_count": self.imu_packet_count,
            "confidence_frame_count": self.confidence_frame_count,
            "gps_sample_count": self.gps_sample_count,
            "gps_gga_fix_quality_counts": self.gps_gga_fix_quality_counts,
            "gps_latest_solution": self.gps_latest_solution,
            "external_imu_sample_count": self.external_imu_sample_count,
            "rgb_frame_type": (
                "MJPEG-decoded BGR"
                if self.args.rgb_transport_effective == "mjpeg"
                else "NV12-decoded BGR"
            ),
            "stereo_input_frame_type": "GRAY8",
            "usb_speed": self.args.usb_speed,
            "host_transport": {
                "rgb": self.args.rgb_transport_effective,
                "rgb_mjpeg_quality": (
                    self.args.rgb_transport_quality
                    if self.args.rgb_transport_effective == "mjpeg"
                    else None
                ),
                "depth": "RAW16",
                "confidence": (
                    self.args.confidence_transport_effective
                    if self.args.save_confidence_map
                    else "disabled"
                ),
                "confidence_mjpeg_quality": (
                    self.args.confidence_transport_quality
                    if self.args.save_confidence_map
                    and self.args.confidence_transport_effective == "mjpeg"
                    else None
                ),
            },
            "rgb_format": self.args.rgb_format,
            "depth_format": "uint16_png",
            "depth_units": "millimeters",
            "depth_png_compression": self.args.depth_png_compression,
            "sync": {
                "mode": self.args.sync_mode,
                "threshold_ms": self.args.sync_threshold_ms,
                "attempts": self.args.sync_attempts if self.args.sync_mode == "device" else "",
                "host_matching": (
                    "rgb anchored nearest depth/imu device timestamp"
                    if self.args.sync_mode == "host"
                    else ""
                ),
                "frame_time_source": (
                    "RGB device timestamp mapped to host monotonic/wall clock using minimum observed transport latency"
                ),
                "queue_delay_recorded": True,
                "gps_matching": "nearest valid GGA measurement epoch to reconstructed RGB capture time",
            },
            "confidence_map": {
                "saved": self.args.save_confidence_map,
                "directory": "confidence" if self.args.save_confidence_map else "",
                "format": "uint8_png" if self.args.save_confidence_map else "",
                "transport": self.args.confidence_transport_effective if self.args.save_confidence_map else "",
                "transport_is_lossy": (
                    self.args.save_confidence_map
                    and self.args.confidence_transport_effective == "mjpeg"
                ),
                "png_compression": self.args.confidence_png_compression if self.args.save_confidence_map else "",
                "aligned_to_rgb": True if self.args.save_confidence_map else "",
                "matching": "best-effort nearest device timestamp; confidence never gates rgb/depth/imu sync" if self.args.save_confidence_map else "",
                "match_threshold_ms": self.args.confidence_match_threshold_ms if self.args.save_confidence_map else "",
                "note": "Use this with depth_mm to reject low-confidence stereo pixels. Missing confidence files mean the confidence stream lagged or skipped, not that the frame was dropped." if self.args.save_confidence_map else "",
            },
            "sync_threshold_ms": self.args.sync_threshold_ms,
            "depth_preset": self.args.depth_preset,
            "left_right_check": self.args.lr_check,
            "subpixel": {
                "enabled": self.args.subpixel,
                "fractional_bits": self.args.subpixel_fractional_bits if self.args.subpixel else None,
            },
            "stereo_median_filter": self.args.stereo_median_filter_effective,
            "external_serial": {
                "gps": {
                    "enabled": self.args.enable_gps,
                    "device": self.args.gps_device,
                    "baudrate": self.args.gps_baudrate,
                    "max_hz": gps_max_hz,
                    "file": "gps.csv" if self.args.enable_gps else None,
                    "rtk_ntrip": {
                        "enabled": bool(self.args.rtk_ntrip_host),
                        "host": self.args.rtk_ntrip_host,
                        "port": self.args.rtk_ntrip_port if self.args.rtk_ntrip_host else None,
                        "mountpoint": self.args.rtk_ntrip_mountpoint,
                        "username": self.args.rtk_ntrip_username,
                        "password_set": bool(self.args.rtk_ntrip_password),
                        "gga_interval_s": self.args.rtk_ntrip_gga_interval,
                        "initial_position_set": (
                            self.args.rtk_initial_latitude_deg is not None
                            and self.args.rtk_initial_longitude_deg is not None
                        ),
                        "expected_fix_quality": "4=RTK fixed, 5=RTK float",
                        "rtcm_bytes_received": rtk_client.bytes_received if rtk_client is not None else 0,
                        "connected_at_close": rtk_client.connected if rtk_client is not None else False,
                        "last_error": rtk_client.error if rtk_client is not None else None,
                    },
                },
                "external_imu": {
                    "enabled": self.args.enable_external_imu,
                    "device": self.args.external_imu_device,
                    "baudrate": self.args.external_imu_baudrate,
                    "max_hz": external_imu_max_hz,
                    "format": self.args.external_imu_format,
                    "file": "external_imu.csv" if self.args.enable_external_imu else None,
                },
                "matching": (
                    "GPS: nearest valid GGA measurement_host_monotonic_ns; "
                    "external IMU: nearest receive host_monotonic_ns"
                ),
                "gps_quality_note": (
                    "Coordinates are receiver solutions after RTCM input. A trusted RTK position requires "
                    f"GGA fix_quality=4 and differential_age_s<={self.args.rtk_max_correction_age_s:.1f}."
                ),
                "rtk_trust": {
                    "required_fix_quality": 4,
                    "maximum_differential_age_s": self.args.rtk_max_correction_age_s,
                    "maximum_hdop": self.args.rtk_max_hdop,
                },
            },
            "timestamps_file": "timestamps.csv",
            "imu_file": "imu.csv",
            "gps_file": "gps.csv" if self.args.enable_gps else None,
            "external_imu_file": "external_imu.csv" if self.args.enable_external_imu else None,
        }
        with open(self.output_dir / "metadata.json", "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)
        print(
            f"Dataset saved: {self.output_dir} "
            f"({self.frame_count} frames, avg_fps={metadata['average_saved_fps']:.1f})"
        )


def record_images(args):
    global stop_requested
    stop_requested = False
    writer = None
    write_queue = None
    write_thread = None
    write_errors = []
    serial_readers = create_serial_readers(args)
    serial_stopped = False
    start_serial_readers(serial_readers)

    try:
        print("Starting DepthAI synced image pipeline...")
        device = connect_depthai_device(args)
        with dai.Pipeline(device) as pipeline:
            device = pipeline.getDefaultDevice()
            resolve_transport_options(args, device)
            outputs = configure_pipeline(pipeline, args)
            if outputs["mode"] == "device":
                sync_queue = outputs["sync"].createOutputQueue(maxSize=args.queue_size, blocking=False)
                rgb_queue = None
                depth_queue = None
                imu_queue = None
            else:
                sync_queue = None
                rgb_queue = outputs["rgb"].createOutputQueue(maxSize=args.queue_size * 2, blocking=False)
                depth_queue = outputs["depth"].createOutputQueue(maxSize=args.queue_size * 2, blocking=False)
                imu_queue = outputs["imu"].createOutputQueue(maxSize=args.queue_size * 4, blocking=False)
            confidence_queue = (
                outputs["confidence"].createOutputQueue(maxSize=args.queue_size * 2, blocking=False)
                if outputs["confidence"] is not None
                else None
            )
            rgb_buffer = []
            depth_buffer = []
            imu_buffer = []
            confidence_buffer = []
            pipeline.start()

            if args.camera_warmup_seconds > 0:
                print(
                    f"Warming up camera exposure for {args.camera_warmup_seconds:.1f}s "
                    "before saving frames..."
                )
                warmup_deadline = time.monotonic() + args.camera_warmup_seconds
                while not stop_requested and time.monotonic() < warmup_deadline:
                    discard_message_queue(sync_queue)
                    discard_message_queue(rgb_queue)
                    discard_message_queue(depth_queue)
                    discard_message_queue(imu_queue)
                    discard_message_queue(confidence_queue)
                    time.sleep(0.005)
                for reader in serial_readers.values():
                    reader.drain()

            try:
                print(f"Connected IMU: {device.getConnectedIMU()}, firmware: {device.getIMUFirmwareVersion()}")
            except Exception:
                print("Connected IMU information is unavailable, continuing.")

            camera_model = read_camera_model_metadata(device, args)
            writer = ImageDatasetWriter(args.output_dir, args, camera_model=camera_model)
            device_host_clock = DeviceHostClockMapper()
            if args.async_write:
                write_queue = queue.Queue(maxsize=args.write_queue_size)

                def write_worker():
                    while True:
                        item = write_queue.get()
                        try:
                            if item is None:
                                return
                            group_item, confidence_item = item
                            writer.write_group(group_item, serial_readers, confidence_msg=confidence_item)
                        except Exception as exc:
                            write_errors.append(exc)
                            request_stop(None, None)
                        finally:
                            write_queue.task_done()

                write_thread = threading.Thread(target=write_worker, name="image-writer", daemon=True)
                write_thread.start()
            last_status = time.monotonic()
            last_frame_count = 0
            print(f"Saving RGB/depth/IMU images with {args.sync_mode} sync. Press Ctrl+C to stop.")

            try:
                while not stop_requested:
                    pending_writes = write_queue.qsize() if write_queue is not None else 0
                    if args.max_frames and writer.frame_count + pending_writes >= args.max_frames:
                        break
                    if write_errors:
                        raise write_errors[0]
                    if args.duration and time.monotonic() - writer.started_monotonic >= args.duration:
                        break

                    writer.write_serial_samples(serial_readers)
                    drain_message_queue(
                        confidence_queue,
                        confidence_buffer,
                        max_buffer=max(args.queue_size * 2, 4),
                    )

                    if args.sync_mode == "device":
                        group = sync_queue.tryGet()
                    else:
                        drain_message_queue(
                            rgb_queue,
                            rgb_buffer,
                            max_buffer=max(args.queue_size * 2, 4),
                        )
                        drain_message_queue(
                            depth_queue,
                            depth_buffer,
                            max_buffer=max(args.queue_size * 2, 4),
                        )
                        drain_message_queue(
                            imu_queue,
                            imu_buffer,
                            max_buffer=max(args.queue_size * 8, 32),
                        )
                        group = build_host_synced_group(
                            rgb_buffer,
                            depth_buffer,
                            imu_buffer,
                            args.sync_threshold_ms,
                        )
                    if group is None:
                        now = time.monotonic()
                        if now - last_status >= args.status_interval:
                            current_fps = (writer.frame_count - last_frame_count) / max(now - last_status, 1e-6)
                            buffer_status = (
                                f", buffers rgb/depth/imu={len(rgb_buffer)}/{len(depth_buffer)}/{len(imu_buffer)}"
                                if args.sync_mode == "host"
                                else ""
                            )
                            print(
                                f"Waiting for synced group ({args.sync_mode})... frames={writer.frame_count}, "
                                f"fps={current_fps:.1f}, avg_fps={writer.average_fps(now):.1f}, "
                                f"gps_samples={writer.gps_sample_count}, "
                                f"{gps_status_text(serial_readers)}, "
                                f"external_imu_samples={writer.external_imu_sample_count}"
                                f"{buffer_status}"
                            )
                            last_status = now
                            last_frame_count = writer.frame_count
                        time.sleep(0.001)
                        continue

                    if not isinstance(group, dict):
                        group = {
                            "rgb": get_group_item(group, "rgb"),
                            "depth": get_group_item(group, "depth"),
                            "imu": get_group_item(group, "imu"),
                        }
                    frame_stamp = device_host_clock.stamp(
                        get_device_ts_ns(get_group_item(group, "rgb"))
                    )
                    group["_host_monotonic_ns"] = frame_stamp["capture_monotonic_ns"]
                    group["_host_wall_time"] = frame_stamp["capture_wall_time"]
                    group["_dequeue_host_monotonic_ns"] = frame_stamp["dequeue_monotonic_ns"]
                    group["_dequeue_host_wall_time"] = frame_stamp["dequeue_wall_time"]
                    group["_queue_lag_ms"] = frame_stamp["queue_lag_ms"]
                    drain_message_queue(
                        confidence_queue,
                        confidence_buffer,
                        max_buffer=max(args.queue_size * 2, 4),
                    )
                    depth_msg = get_group_item(group, "depth")
                    confidence_msg = take_nearest_message(
                        confidence_buffer,
                        get_device_ts_ns(depth_msg),
                        args.confidence_match_threshold_ms,
                    )
                    if write_queue is not None:
                        while not stop_requested:
                            try:
                                write_queue.put((group, confidence_msg), timeout=0.2)
                                break
                            except queue.Full:
                                if write_errors:
                                    raise write_errors[0]
                    else:
                        writer.write_group(group, serial_readers, confidence_msg=confidence_msg)

                    now = time.monotonic()
                    if now - last_status >= args.status_interval:
                        pending_writes = write_queue.qsize() if write_queue is not None else 0
                        current_fps = (writer.frame_count - last_frame_count) / max(now - last_status, 1e-6)
                        confidence_status = (
                            f", confidence={writer.confidence_frame_count}"
                            if args.save_confidence_map
                            else ""
                        )
                        print(
                            f"Saving: frames={writer.frame_count}, "
                            f"fps={current_fps:.1f}, avg_fps={writer.average_fps(now):.1f}, "
                            f"imu_packets={writer.imu_packet_count}, "
                            f"gps_samples={writer.gps_sample_count}, "
                            f"{gps_status_text(serial_readers)}, "
                            f"external_imu_samples={writer.external_imu_sample_count}"
                            f"{confidence_status}, pending_writes={pending_writes}"
                        )
                        last_status = now
                        last_frame_count = writer.frame_count
            except KeyboardInterrupt:
                print("Recording stopped.")
            finally:
                if writer is not None:
                    if write_queue is not None:
                        write_queue.put(None)
                        write_queue.join()
                        if write_thread is not None:
                            write_thread.join(timeout=5.0)
                        if write_errors:
                            raise write_errors[0]
                    stop_serial_readers(serial_readers)
                    serial_stopped = True
                    writer.write_serial_samples(serial_readers)
                    writer.close(serial_readers)
    finally:
        if not serial_stopped:
            stop_serial_readers(serial_readers)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save synchronized Luxonis RGB images, uint16 depth-mm images, and IMU CSV."
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output root directory")
    parser.add_argument("--fps", type=int, default=30, help="Requested RGB/depth FPS")
    parser.add_argument("--imu-rate", type=int, default=200, help="IMU report rate in Hz")
    parser.add_argument("--imu-batch", type=int, default=1, help="Maximum IMU packets per batch")
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds; 0 means run until Ctrl+C")
    parser.add_argument(
        "--camera-warmup-seconds",
        type=nonnegative_float,
        default=3.0,
        help="Discard initial camera frames while auto-exposure and white balance settle",
    )
    parser.add_argument("--max-frames", type=positive_int, default=0, help="Stop after N frames; 0 means no limit")
    parser.add_argument("--status-interval", type=float, default=5.0, help="Seconds between FPS prints")
    parser.add_argument(
        "--queue-size",
        type=positive_int,
        default=4,
        help="Small host queue for latest-frame behavior; old device frames are dropped before they become stale",
    )
    parser.add_argument(
        "--usb3-retries",
        type=int,
        default=2,
        help="Additional device reconnect attempts when the camera negotiates USB 2.0 instead of USB 3.x",
    )
    parser.add_argument(
        "--allow-usb2",
        action="store_true",
        help="Allow best-effort compressed recording on USB 2.0 instead of aborting",
    )
    parser.add_argument(
        "--depth-preset",
        type=str,
        default="FAST_ACCURACY",
        choices=[item for item in dir(dai.node.StereoDepth.PresetMode) if item.isupper()],
        help="DepthAI v3 StereoDepth preset",
    )
    parser.add_argument("--sync-mode", choices=["host", "device"], default="host", help="Use host timestamp matching or DepthAI Sync node")
    parser.add_argument("--sync-threshold-ms", type=float, default=50.0)
    parser.add_argument("--sync-attempts", type=int, default=-1)
    parser.add_argument("--async-write", type=str2bool, nargs="?", const=True, default=True, help="Write images on a background thread so capture does not block on disk I/O")
    parser.add_argument("--write-queue-size", type=positive_int, default=180, help="Maximum pending frames for async image writing")
    parser.add_argument(
        "--depth-alignment-mode",
        choices=["image-align", "stereo"],
        default="stereo",
        help="Use ImageAlign node to align depth to the RGB stream, or legacy StereoDepth.setDepthAlign",
    )
    parser.add_argument("--lr-check", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument(
        "--subpixel",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable StereoDepth subpixel disparity for smoother, more precise depth",
    )
    parser.add_argument(
        "--subpixel-fractional-bits",
        type=int,
        choices=[3, 4, 5],
        default=3,
        help="Subpixel precision bits; 3 keeps median filtering available",
    )
    parser.add_argument(
        "--no-subpixel",
        dest="subpixel",
        action="store_false",
        help="Disable StereoDepth subpixel without changing image resolution",
    )
    parser.add_argument(
        "--stereo-median-filter",
        choices=["off", "3x3", "5x5", "7x7"],
        default="7x7",
        help="Stereo disparity/depth median filter; 7x7 follows the robust Geonova setting",
    )
    parser.add_argument("--flip", type=str2bool, nargs="?", const=True, default=False, help="Legacy vertical flip applied before --rotate-180")
    parser.add_argument("--rotate-180", action="store_true", help="Save RGB and depth frames rotated 180 degrees")
    parser.add_argument("--rgb-format", choices=["png", "jpg"], default="png")
    parser.add_argument("--rgb-png-compression", type=png_compression_level, default=1)
    parser.add_argument(
        "--rgb-jpeg-quality",
        "--jpg-quality",
        dest="rgb_jpeg_quality",
        type=jpeg_quality,
        default=100,
        metavar="1-100",
        help="Saved JPG quality; 100 is least compression, lower values produce smaller files",
    )
    parser.add_argument(
        "--rgb-transport",
        choices=["auto", "raw", "mjpeg"],
        default="auto",
        help="Host transport; auto uses MJPEG on USB 2.0-class links and raw NV12 on USB 3.x",
    )
    parser.add_argument(
        "--rgb-transport-quality",
        type=jpeg_quality,
        default=100,
        help="Device MJPEG quality when compressed RGB transport is selected",
    )
    parser.add_argument("--depth-png-compression", type=png_compression_level, default=0)
    parser.add_argument("--save-confidence-map", action="store_true", help="Also save StereoDepth confidence map PNGs aligned with RGB/depth")
    parser.add_argument("--confidence-png-compression", type=png_compression_level, default=0)
    parser.add_argument(
        "--confidence-transport",
        choices=["auto", "raw", "mjpeg"],
        default="auto",
        help="Host transport; auto compresses confidence on USB 2.0-class links",
    )
    parser.add_argument(
        "--confidence-transport-quality",
        type=jpeg_quality,
        default=100,
        help="Device MJPEG quality for confidence transport; decoded frames are still saved as PNG",
    )
    parser.add_argument("--confidence-match-threshold-ms", type=nonnegative_float, default=50.0, help="Nearest confidence frame match window; confidence does not gate RGB/depth/IMU recording")
    parser.add_argument("--gps-device", default="/dev/ttyACM0", help="GPS serial device")
    parser.add_argument("--gps-baudrate", type=int, default=921600, help="GPS serial baudrate")
    parser.add_argument("--rtk-ntrip-host", default="www.gnssdata.or.kr", help="NTRIP caster host; enables RTK RTCM injection when set")
    parser.add_argument("--rtk-ntrip-port", type=int, default=2101, help="NTRIP caster TCP port")
    parser.add_argument("--rtk-ntrip-mountpoint", default="YANJ-RTCM31", help="NTRIP mountpoint name")
    parser.add_argument("--rtk-ntrip-username", default="pjmsm0319@gmail.com", help="NTRIP username, if required")
    parser.add_argument("--rtk-ntrip-password", default="gnss", help="NTRIP password, if required")
    parser.add_argument("--rtk-ntrip-gga", default="", help="Explicit NMEA GGA sentence sent to the NTRIP caster")
    parser.add_argument("--rtk-ntrip-gga-interval", type=nonnegative_float, default=10.0, help="Seconds between GGA updates sent to the NTRIP caster; 0 disables repeats")
    parser.add_argument("--rtk-ntrip-reconnect-delay", type=nonnegative_float, default=5.0, help="Seconds to wait before reconnecting to NTRIP after an error")
    parser.add_argument("--rtk-initial-latitude-deg", type=float, default=None, help="Initial approximate antenna latitude for generated NTRIP GGA")
    parser.add_argument("--rtk-initial-longitude-deg", type=float, default=None, help="Initial approximate antenna longitude for generated NTRIP GGA")
    parser.add_argument("--rtk-initial-altitude-m", type=float, default=0.0, help="Initial approximate antenna altitude for generated NTRIP GGA")
    parser.add_argument("--external-imu-device", default="/dev/ttyUSB0", help="External IMU serial device")
    parser.add_argument("--external-imu-baudrate", type=int, default=921600, help="External IMU serial baudrate")
    parser.add_argument("--external-imu-format", choices=["ebimu", "raw"], default="ebimu", help="Parse EBIMU quaternion output or save raw lines only")
    parser.add_argument("--serial-max-hz", type=nonnegative_float, default=120.0, help="Fallback maximum saved serial samples per second per device")
    parser.add_argument("--gps-max-hz", type=nonnegative_float, default=None, help="Maximum saved GPS serial samples per second; defaults to --serial-max-hz")
    parser.add_argument("--external-imu-max-hz", type=nonnegative_float, default=None, help="Maximum saved external IMU serial samples per second; defaults to --serial-max-hz")
    parser.add_argument("--imu-from-camera-roll-deg", type=float, default=0.0, help="Fixed roll from saved camera frame to external IMU frame")
    parser.add_argument("--imu-from-camera-pitch-deg", type=float, default=0.0, help="Fixed pitch from saved camera frame to external IMU frame")
    parser.add_argument("--imu-from-camera-yaw-deg", type=float, default=0.0, help="Fixed yaw from saved camera frame to external IMU frame")
    parser.add_argument("--camera-mount-roll-deg", type=float, default=0.0, help="Camera mount roll; +clockwise around optical axis")
    parser.add_argument("--camera-mount-pitch-deg", type=float, default=0.0, help="Camera mount pitch from vehicle forward; +down degrees")
    parser.add_argument("--camera-mount-yaw-deg", type=float, default=0.0, help="Camera mount yaw from vehicle forward; +right degrees")
    parser.add_argument("--gps-to-camera-east-m", type=float, default=0.0, help="Camera offset east of GPS antenna in meters")
    parser.add_argument("--gps-to-camera-north-m", type=float, default=0.0, help="Camera offset north of GPS antenna in meters")
    parser.add_argument(
        "--gps-to-camera-up-m",
        type=float,
        default=-0.20,
        help="Camera vertical ENU offset from GPS antenna; default -0.20m because camera is lower",
    )
    parser.add_argument("--gps-from-camera-right-m", type=float, default=0.0, help="GPS antenna offset from saved camera frame: +right meters")
    parser.add_argument("--gps-from-camera-down-m", type=float, default=0.0, help="GPS antenna offset from saved camera frame: +down meters")
    parser.add_argument("--gps-from-camera-forward-m", type=float, default=0.0, help="GPS antenna offset from saved camera frame: +forward meters")
    parser.add_argument("--external-imu-from-camera-right-m", type=float, default=0.0, help="External IMU offset from saved camera frame: +right meters")
    parser.add_argument(
        "--external-imu-from-camera-down-m",
        type=float,
        default=0.15,
        help="External IMU offset from camera: default +0.15m down",
    )
    parser.add_argument("--external-imu-from-camera-forward-m", type=float, default=0.0, help="External IMU offset from saved camera frame: +forward meters")
    parser.add_argument("--gps-height-above-ground-m", type=float, default=1.50, help="GPS antenna height above ground")
    parser.add_argument("--camera-height-above-ground-m", type=float, default=1.30, help="Camera height above ground")
    parser.add_argument("--external-imu-height-above-ground-m", type=float, default=1.15, help="External IMU height above ground")
    parser.add_argument("--magnetic-declination-deg", type=float, default=0.0, help="Local magnetic declination; east-positive degrees")
    parser.add_argument(
        "--rtk-max-correction-age-s",
        type=nonnegative_float,
        default=2.0,
        help="Maximum differential correction age considered trustworthy for position diagnostics",
    )
    parser.add_argument(
        "--rtk-max-hdop",
        type=nonnegative_float,
        default=2.0,
        help="Maximum HDOP considered trustworthy for position diagnostics",
    )
    parser.add_argument("--no-gps", dest="enable_gps", action="store_false", help="Disable GPS serial logging")
    parser.add_argument("--no-external-imu", dest="enable_external_imu", action="store_false", help="Disable external IMU serial logging")
    parser.set_defaults(enable_gps=True, enable_external_imu=True)
    return parser.parse_args()


def main():
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    record_images(parse_args())


if __name__ == "__main__":
    main()
