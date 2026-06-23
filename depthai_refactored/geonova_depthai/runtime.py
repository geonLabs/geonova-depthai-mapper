#!/usr/bin/env python3
"""Shared DepthAI v3 camera, timestamp, GPS/NTRIP, and EBIMU runtime helpers."""

import argparse
import base64
import queue
import socket
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone

import cv2
import depthai as dai
import numpy as np
import serial


WIDTH = 1280
HEIGHT = 720
MIN_DEPTHAI_MAJOR = 3

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


def request_camera_output(camera, fps, size=(WIDTH, HEIGHT), frame_type=None, enable_undistortion=False):
    """Request a DepthAI v3 camera output with explicit undistortion control.

    DepthAI v3 exposes undistortion through the named requestOutput(...,
    enableUndistortion=True) argument.  The older capability overload used in this
    script can look similar but does not make the RGB image undistorted, which can
    make RGB/depth coordinates drift more severely toward the image edges.
    """
    kwargs = {
        "size": size,
        "fps": fps,
        "enableUndistortion": bool(enable_undistortion),
    }
    if frame_type is not None:
        kwargs["type"] = frame_type
    try:
        return camera.requestOutput(**kwargs)
    except TypeError as exc:
        if enable_undistortion:
            raise RuntimeError(
                "This DepthAI build does not support Camera.requestOutput(..., "
                "enableUndistortion=True). Please upgrade depthai v3; production "
                "RGB-D capture requires factory-undistorted RGB geometry."
            ) from exc
        capability = dai.ImgFrameCapability()
        capability.size.fixed(size)
        capability.fps.fixed(fps)
        if frame_type is not None:
            capability.type = frame_type
        return camera.requestOutput(capability, True)


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
        image_stream_undistorted = bool(getattr(args, "rgb_undistort", False))
        metadata = {
            "model": "pinhole",
            "source": "DepthAI device calibration",
            "socket": "CAM_A",
            "width": WIDTH,
            "height": HEIGHT,
            "intrinsics_original": matrix_to_json(rgb_intrinsics),
            "intrinsics": adjust_intrinsics_for_saved_transform(rgb_intrinsics, args),
            "image_stream_undistorted": image_stream_undistorted,
            "coordinate_unprojection": (
                "pinhole projection with saved-image intrinsics; distortion was removed on-device"
                if image_stream_undistorted
                else "cv2.undistortPoints with original intrinsics and distortion coefficients"
            ),
            "notes": (
                "Saved RGB is requested as an on-device undistorted image. Depth is aligned to that "
                "same RGB output, so UI/world-coordinate unprojection must not undistort the pixel again."
                if image_stream_undistorted
                else "Saved RGB remains in the camera's distorted pixel geometry. Intrinsics are adjusted "
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


def configure_pipeline(pipeline, args, sync_imu=True):
    require_depthai_v3()

    # This is a production invariant, not a tuning option. Saved RGB and aligned
    # depth must describe the same factory-undistorted pixel geometry.
    args.rgb_undistort = True
    args.rgb_undistort_effective = True

    platform = pipeline.getDefaultDevice().getPlatform()
    requested_alignment = args.depth_alignment_mode
    if requested_alignment == "auto":
        effective_alignment = "image-align" if platform == dai.Platform.RVC4 else "stereo"
    else:
        effective_alignment = requested_alignment
    if effective_alignment == "image-align" and platform != dai.Platform.RVC4:
        raise RuntimeError(
            f"ImageAlign RGB-D mode is not used by the official DepthAI v3 example on {platform}. "
            "Use --depth-alignment-mode auto (recommended) or stereo."
        )
    args.depth_alignment_effective = effective_alignment
    args.depthai_platform = str(platform).split(".")[-1]
    print(
        "RGB-D geometry: factory-undistorted RGB -> "
        + (
            "ImageAlign.inputAlignTo"
            if effective_alignment == "image-align"
            else "StereoDepth.inputAlignTo"
        )
    )

    color_socket = dai.CameraBoardSocket.CAM_A
    left_socket = dai.CameraBoardSocket.CAM_B
    right_socket = dai.CameraBoardSocket.CAM_C

    cam_rgb = pipeline.create(dai.node.Camera).build(color_socket)
    mono_left = pipeline.create(dai.node.Camera).build(left_socket)
    mono_right = pipeline.create(dai.node.Camera).build(right_socket)
    stereo = pipeline.create(dai.node.StereoDepth)
    image_align = None
    confidence_align = None
    if effective_alignment == "image-align":
        image_align = pipeline.create(dai.node.ImageAlign)
        if args.save_confidence_map:
            confidence_align = pipeline.create(dai.node.ImageAlign)
    imu = pipeline.create(dai.node.IMU)
    sync = pipeline.create(dai.node.Sync) if args.sync_mode == "device" else None

    rgb_output = request_camera_output(
        cam_rgb,
        args.fps,
        frame_type=dai.ImgFrame.Type.NV12,
        enable_undistortion=True,
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
    if effective_alignment == "image-align":
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
    # DepthAI v3 official depth-align example uses the actual RGB output as
    # StereoDepth.inputAlignTo on RVC2/RVC3. setDepthAlign(CAM_A) alone does not
    # describe the requested crop/undistortion geometry of that output.
    if effective_alignment == "stereo":
        rgb_output.link(stereo.inputAlignTo)
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
            rgb_output.link(confidence_align.inputAlignTo)
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
        if sync_imu:
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
