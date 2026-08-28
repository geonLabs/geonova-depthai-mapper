#!/usr/bin/env python3
"""Shared DepthAI v3 camera, timestamp, GPS/NTRIP, and EBIMU runtime helpers."""

import argparse
import base64
import math
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

from geonova_depthai.serial_devices import resolve_serial_device, resolve_serial_devices


WIDTH = 1280
HEIGHT = 720
MIN_DEPTHAI_MAJOR = 3
PREFERRED_RGB_SIZES = (
    (1920, 1200),
    (1920, 1080),
    (WIDTH, HEIGHT),
)
MAX_RVC2_STEREO_WIDTH = 1280
PREFERRED_STEREO_SIZES = (
    (1280, 800),
    (1280, 720),
    (640, 400),
)

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

DEFAULT_RTK_MOUNTPOINT_FORMAT = "RTCM31"
NTRIP_STREAM_POLL_TIMEOUT_S = 0.25
NTRIP_HANDOVER_MIN_VALID_PAYLOADS = 2
FALLBACK_RTK_MOUNTPOINTS = (
    ("GANS-RTCM31", 37.500000, 126.900000, "SMG"),
    ("GUMC-RTCM31", 37.500000, 126.900000, "SMG"),
    ("DBON-RTCM31", 37.600000, 127.000000, "SMG"),
    ("PAJU-RTCM31", 37.750000, 126.740000, "Single Base"),
    ("YONS-RTCM31", 37.500000, 127.000000, "SMG"),
    ("SOUL-RTCM31", 37.620000, 127.100000, "Single Base"),
    ("INCH-RTCM31", 37.420000, 126.690000, "Single Base"),
    ("ICOR-RTCM31", 37.420000, 126.640000, "Single Base"),
    ("SONP-RTCM31", 37.500000, 127.100000, "SMG"),
    ("OJBU-RTCM31", 37.450000, 127.060000, "Single Base"),
    ("PJMS-RTCM31", 37.886041, 126.766214, "Single Base"),
    ("YANJ-RTCM31", 37.410000, 126.550000, "Single Base"),
    ("DOND-RTCM31", 37.900000, 127.060000, "Single Base"),
    ("NAMY-RTCM31", 37.430000, 127.180000, "Single Base"),
    ("GANH-RTCM31", 37.720000, 126.390000, "Single Base"),
    ("SWGS-RTCM31", 37.260000, 126.980000, "Single Base"),
    ("SUWN-RTCM31", 37.280000, 127.050000, "Single Base"),
    ("POCN-RTCM31", 38.018582, 127.190970, "Single Base"),
    ("YANP-RTCM31", 37.450000, 127.510000, "Single Base"),
    ("YEOJ-RTCM31", 37.280000, 127.580000, "Single Base"),
    ("ANSG-RTCM31", 37.010000, 127.270000, "Single Base"),
    ("DANJ-RTCM31", 36.890000, 126.820000, "Single Base"),
    ("CHEN-RTCM31", 36.880000, 127.160000, "Single Base"),
)


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


def enum_name(value):
    return getattr(value, "name", str(value).split(".")[-1])


def socket_matches(value, expected):
    return value == expected or enum_name(value) == enum_name(expected)


def camera_feature_size(feature):
    try:
        width = int(getattr(feature, "width", 0) or 0)
        height = int(getattr(feature, "height", 0) or 0)
    except (TypeError, ValueError):
        return 0, 0
    return width, height


def camera_feature_type_names(feature):
    return [enum_name(sensor_type) for sensor_type in (getattr(feature, "supportedTypes", []) or [])]


def connected_camera_features(device):
    try:
        return list(device.getConnectedCameraFeatures())
    except Exception:
        return []


def find_connected_camera_feature(device, board_socket):
    for feature in connected_camera_features(device):
        if socket_matches(getattr(feature, "socket", None), board_socket):
            return feature
    return None


def camera_feature_to_metadata(feature):
    width, height = camera_feature_size(feature) if feature is not None else (0, 0)
    return {
        "socket": enum_name(getattr(feature, "socket", "")) if feature is not None else "",
        "name": getattr(feature, "sensorName", "") if feature is not None else "",
        "width": width,
        "height": height,
        "types": camera_feature_type_names(feature) if feature is not None else [],
    }


def is_color_camera_feature(feature):
    return "COLOR" in camera_feature_type_names(feature)


def select_color_camera_feature(device, preferred_socket):
    features = connected_camera_features(device)
    preferred = None
    for feature in features:
        if socket_matches(getattr(feature, "socket", None), preferred_socket):
            preferred = feature
            if is_color_camera_feature(feature):
                return feature
    for feature in features:
        if is_color_camera_feature(feature):
            return feature
    return preferred


def rgb_size_from_args(args):
    width = int(getattr(args, "rgb_width", 0) or WIDTH)
    height = int(getattr(args, "rgb_height", 0) or HEIGHT)
    return width, height


def resolve_depth_fps(args):
    depth_fps = float(getattr(args, "depth_fps", 0.0) or 0.0)
    if depth_fps < 0:
        raise ValueError("depth_fps must be >= 0")
    effective = depth_fps if depth_fps > 0 else float(args.fps)
    args.depth_fps_effective = effective
    return effective


def choose_rgb_output_size(sensor_width, sensor_height, candidates=PREFERRED_RGB_SIZES):
    if sensor_width <= 0 or sensor_height <= 0:
        return WIDTH, HEIGHT, "fallback_unknown_sensor"

    fitting = [
        (width, height)
        for width, height in candidates
        if width <= sensor_width and height <= sensor_height
    ]
    if not fitting:
        return WIDTH, HEIGHT, "fallback_sensor_too_small"

    sensor_aspect = sensor_width / sensor_height
    width, height = min(
        fitting,
        key=lambda size: (abs((size[0] / size[1]) - sensor_aspect), -size[0] * size[1]),
    )
    return width, height, "sensor_aspect"


def choose_stereo_input_size(sensor_width, sensor_height, platform, candidates=PREFERRED_STEREO_SIZES):
    if sensor_width <= 0 or sensor_height <= 0:
        return WIDTH, 800, "fallback_unknown_sensor"

    max_width = MAX_RVC2_STEREO_WIDTH if platform == dai.Platform.RVC2 else sensor_width
    fitting = [
        (width, height)
        for width, height in candidates
        if width <= sensor_width and height <= sensor_height and width <= max_width
    ]
    if not fitting:
        width = min(sensor_width, max_width)
        height = max(1, int(round(width * sensor_height / sensor_width)))
        return width, height, "scaled_to_platform_limit"

    sensor_aspect = sensor_width / sensor_height
    width, height = min(
        fitting,
        key=lambda size: (abs((size[0] / size[1]) - sensor_aspect), -size[0] * size[1]),
    )
    return width, height, "sensor_aspect_platform_limit"


def resolve_stereo_sockets(device, preferred_left=None, preferred_right=None):
    left_socket = preferred_left
    right_socket = preferred_right
    try:
        calibration = device.readCalibration()
        left_socket = left_socket or calibration.getStereoLeftCameraId()
        right_socket = right_socket or calibration.getStereoRightCameraId()
    except Exception:
        pass
    left_socket = left_socket or dai.CameraBoardSocket.CAM_B
    right_socket = right_socket or dai.CameraBoardSocket.CAM_C
    return left_socket, right_socket


def resolve_stereo_input_size(device, args, platform, left_socket, right_socket):
    left_feature = find_connected_camera_feature(device, left_socket)
    right_feature = find_connected_camera_feature(device, right_socket)
    left_width, left_height = camera_feature_size(left_feature)
    right_width, right_height = camera_feature_size(right_feature)
    sensor_width = min(value for value in (left_width, right_width) if value > 0) if left_width or right_width else 0
    sensor_height = min(value for value in (left_height, right_height) if value > 0) if left_height or right_height else 0
    width, height, source = choose_stereo_input_size(sensor_width, sensor_height, platform)

    args.left_socket = left_socket
    args.right_socket = right_socket
    args.left_socket_name = enum_name(left_socket)
    args.right_socket_name = enum_name(right_socket)
    args.depth_input_width = width
    args.depth_input_height = height
    args.depth_input_resolution_source = source
    args.left_sensor = camera_feature_to_metadata(left_feature)
    args.right_sensor = camera_feature_to_metadata(right_feature)

    print(
        f"Stereo input size: {width}x{height} "
        f"(source={source}, left={args.left_socket_name} "
        f"{args.left_sensor.get('name') or 'unknown'} "
        f"{args.left_sensor.get('width')}x{args.left_sensor.get('height')}, "
        f"right={args.right_socket_name} "
        f"{args.right_sensor.get('name') or 'unknown'} "
        f"{args.right_sensor.get('width')}x{args.right_sensor.get('height')})"
    )
    return width, height


def set_device_identity_metadata(device, args):
    try:
        args.depthai_device_name = device.getDeviceName()
    except Exception:
        args.depthai_device_name = ""
    try:
        args.depthai_device_id = device.getDeviceId()
    except Exception:
        try:
            args.depthai_device_id = device.getMxId()
        except Exception:
            args.depthai_device_id = ""


def resolve_rgb_output_size(device, args, color_socket=None):
    color_socket = color_socket or dai.CameraBoardSocket.CAM_A
    configured_width = int(getattr(args, "rgb_width", 0) or 0)
    configured_height = int(getattr(args, "rgb_height", 0) or 0)
    if configured_width < 0 or configured_height < 0:
        raise ValueError("rgb_width/rgb_height must be >= 0")
    if bool(configured_width) != bool(configured_height):
        raise ValueError("Set both rgb_width and rgb_height, or set both to 0 for auto")
    feature = select_color_camera_feature(device, color_socket)
    actual_socket = getattr(feature, "socket", color_socket) if feature is not None else color_socket
    sensor_width, sensor_height = camera_feature_size(feature) if feature is not None else (0, 0)
    sensor_name = getattr(feature, "sensorName", "") if feature is not None else ""
    sensor_types = camera_feature_type_names(feature) if feature is not None else []

    if configured_width > 0 and configured_height > 0:
        width, height = configured_width, configured_height
        source = "configured"
    else:
        width, height, source = choose_rgb_output_size(sensor_width, sensor_height)

    args.rgb_width = width
    args.rgb_height = height
    args.rgb_resolution_source = source
    args.rgb_sensor_name = sensor_name
    args.rgb_sensor_width = sensor_width
    args.rgb_sensor_height = sensor_height
    args.rgb_sensor_types = sensor_types
    args.rgb_socket = actual_socket
    args.rgb_socket_name = enum_name(actual_socket)

    sensor_text = (
        f"{sensor_name or 'unknown'} {sensor_width}x{sensor_height}"
        if sensor_width and sensor_height
        else sensor_name or "unknown"
    )
    print(
        f"RGB output size: {width}x{height} "
        f"(source={source}, socket={args.rgb_socket_name}, sensor={sensor_text})"
    )
    if sensor_types and "COLOR" not in sensor_types:
        print(f"Warning: {args.rgb_socket_name} sensor types are {sensor_types}, not COLOR.")
    return width, height


def resolve_rgb_camera_output_size(args, requested_size):
    """Choose the physical camera output used before optional RGB upscaling.

    Some OAK-D Pro W units use an OV9782 color sensor whose maximum output is
    1280x800.  A 1920x1200 request has the same 16:10 geometry but cannot be
    produced directly by that sensor.  Capture the full sensor output in that
    case; configure_pipeline() scales this undistorted RGB frame before using it
    as StereoDepth.inputAlignTo, so RGB and aligned depth retain identical pixel
    coordinates at the requested saved size.
    """
    requested_width, requested_height = (int(value) for value in requested_size)
    sensor_width = int(getattr(args, "rgb_sensor_width", 0) or 0)
    sensor_height = int(getattr(args, "rgb_sensor_height", 0) or 0)
    camera_width, camera_height = requested_width, requested_height
    source = "requested"
    if (
        sensor_width > 0
        and sensor_height > 0
        and (requested_width > sensor_width or requested_height > sensor_height)
    ):
        requested_aspect = requested_width / requested_height
        sensor_aspect = sensor_width / sensor_height
        if abs(requested_aspect - sensor_aspect) > 1e-6:
            raise RuntimeError(
                f"Requested RGB size {requested_width}x{requested_height} exceeds "
                f"the {sensor_width}x{sensor_height} color sensor and has a different "
                "aspect ratio; refusing a geometry-changing resize."
            )
        camera_width, camera_height = sensor_width, sensor_height
        source = "sensor_max_then_uniform_upscale"

    args.rgb_camera_width = camera_width
    args.rgb_camera_height = camera_height
    args.rgb_camera_resolution_source = source
    return camera_width, camera_height


def resize_camera_output(pipeline, camera_output, source_size, target_size):
    """Uniformly resize an undistorted camera frame on-device when required."""
    if tuple(source_size) == tuple(target_size):
        return camera_output
    source_width, source_height = source_size
    target_width, target_height = target_size
    if source_width * target_height != target_width * source_height:
        raise RuntimeError(
            f"RGB resize must preserve geometry: {source_width}x{source_height} -> "
            f"{target_width}x{target_height}"
        )
    resize = pipeline.create(dai.node.ImageManip)
    resize.initialConfig.setOutputSize(
        target_width,
        target_height,
        dai.ImageManipConfig.ResizeMode.STRETCH,
    )
    resize.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
    resize.setMaxOutputFrameSize(target_width * target_height * 3 // 2)
    camera_output.link(resize.inputImage)
    print(
        f"RGB on-device uniform upscale: {source_width}x{source_height} -> "
        f"{target_width}x{target_height}"
    )
    return resize.out


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


def _float_or_none(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_ntrip_mountpoint_list(value):
    if value in ("", None):
        return []
    if isinstance(value, str):
        items = value.replace("\n", ",").replace(";", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [value]
    mountpoints = []
    seen = set()
    for item in items:
        mountpoint = str(item).strip()
        if not mountpoint or mountpoint in seen:
            continue
        seen.add(mountpoint)
        mountpoints.append(mountpoint)
    return mountpoints


def ntrip_mountpoint_matches_format(mountpoint, mountpoint_format):
    if not mountpoint_format:
        return True
    normalized_mountpoint = str(mountpoint).replace(" ", "").upper()
    normalized_format = str(mountpoint_format).replace(" ", "").upper()
    return normalized_mountpoint.endswith(f"-{normalized_format}") or normalized_mountpoint.endswith(normalized_format)


def fallback_rtk_mountpoint_entries(mountpoint_format=DEFAULT_RTK_MOUNTPOINT_FORMAT):
    entries = []
    for mountpoint, latitude, longitude, network in FALLBACK_RTK_MOUNTPOINTS:
        if not ntrip_mountpoint_matches_format(mountpoint, mountpoint_format):
            continue
        entries.append({
            "mountpoint": mountpoint,
            "identifier": mountpoint,
            "format": mountpoint_format,
            "navigation_system": "GPS+GLONASS",
            "network": network,
            "country": "KOR",
            "latitude": latitude,
            "longitude": longitude,
            "source": "fallback",
        })
    return entries


def haversine_distance_m(latitude_a, longitude_a, latitude_b, longitude_b):
    radius_m = 6_371_000.0
    lat_a = math.radians(float(latitude_a))
    lat_b = math.radians(float(latitude_b))
    delta_lat = math.radians(float(latitude_b) - float(latitude_a))
    delta_lon = math.radians(float(longitude_b) - float(longitude_a))
    value = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * radius_m * math.atan2(math.sqrt(value), math.sqrt(1.0 - value))


def parse_ntrip_source_table(text, mountpoint_format=DEFAULT_RTK_MOUNTPOINT_FORMAT):
    if "\r\n\r\n" in text:
        text = text.split("\r\n\r\n", 1)[1]
    elif "\n\n" in text:
        text = text.split("\n\n", 1)[1]

    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("STR;"):
            continue
        parts = line.split(";")
        if len(parts) < 11:
            continue
        mountpoint = parts[1].strip()
        if not ntrip_mountpoint_matches_format(mountpoint, mountpoint_format):
            continue
        latitude = _float_or_none(parts[9])
        longitude = _float_or_none(parts[10])
        if latitude is None or longitude is None:
            continue
        entries.append({
            "mountpoint": mountpoint,
            "identifier": parts[2].strip() if len(parts) > 2 else mountpoint,
            "format": parts[3].strip() if len(parts) > 3 else "",
            "format_details": parts[4].strip() if len(parts) > 4 else "",
            "carrier": parts[5].strip() if len(parts) > 5 else "",
            "navigation_system": parts[6].strip() if len(parts) > 6 else "",
            "network": parts[7].strip() if len(parts) > 7 else "",
            "country": parts[8].strip() if len(parts) > 8 else "",
            "latitude": latitude,
            "longitude": longitude,
            "source": "sourcetable",
        })
    return unique_ntrip_mountpoint_entries(entries)


def unique_ntrip_mountpoint_entries(entries):
    unique = []
    seen = set()
    for entry in entries:
        mountpoint = entry.get("mountpoint")
        if not mountpoint or mountpoint in seen:
            continue
        seen.add(mountpoint)
        unique.append(entry)
    return unique


def ntrip_entry_for_mountpoint(mountpoint, source="configured"):
    for entry in fallback_rtk_mountpoint_entries():
        if entry["mountpoint"] == mountpoint:
            configured = dict(entry)
            configured["source"] = source
            return configured
    return {"mountpoint": mountpoint, "source": source}


def sort_ntrip_mountpoint_entries(entries, latitude_deg=None, longitude_deg=None, max_count=None, preferred_mountpoints=None):
    unique = unique_ntrip_mountpoint_entries(entries)
    preferred = {
        mountpoint: index
        for index, mountpoint in enumerate(parse_ntrip_mountpoint_list(preferred_mountpoints))
    }
    latitude = _float_or_none(latitude_deg)
    longitude = _float_or_none(longitude_deg)
    has_position = latitude is not None and longitude is not None

    ranked = []
    for order, entry in enumerate(unique):
        entry = dict(entry)
        entry_lat = _float_or_none(entry.get("latitude"))
        entry_lon = _float_or_none(entry.get("longitude"))
        distance = None
        if has_position and entry_lat is not None and entry_lon is not None:
            distance = haversine_distance_m(latitude, longitude, entry_lat, entry_lon)
            entry["distance_m"] = distance
        mountpoint = entry.get("mountpoint", "")
        if distance is not None:
            sort_key = (0, distance, preferred.get(mountpoint, len(preferred)), order)
        elif mountpoint in preferred:
            sort_key = (1, preferred[mountpoint], order)
        else:
            sort_key = (2, order)
        ranked.append((sort_key, entry))

    ranked.sort(key=lambda item: item[0])
    result = [entry for _, entry in ranked]
    if max_count and max_count > 0:
        result = result[:int(max_count)]
    return result


def closer_ntrip_mountpoint_entries(
    entries,
    current_entry,
    latitude_deg,
    longitude_deg,
    min_improvement_m=0.0,
):
    """Return only stations meaningfully closer than the active mountpoint."""

    latitude = _float_or_none(latitude_deg)
    longitude = _float_or_none(longitude_deg)
    if latitude is None or longitude is None or not current_entry:
        return []

    current = dict(current_entry)
    current_mountpoint = current.get("mountpoint")
    if _float_or_none(current.get("latitude")) is None or _float_or_none(current.get("longitude")) is None:
        for entry in entries:
            if entry.get("mountpoint") == current_mountpoint:
                current.update(entry)
                break

    current_latitude = _float_or_none(current.get("latitude"))
    current_longitude = _float_or_none(current.get("longitude"))
    if current_latitude is None or current_longitude is None:
        return []

    current_distance = haversine_distance_m(
        latitude,
        longitude,
        current_latitude,
        current_longitude,
    )
    minimum = max(0.0, float(min_improvement_m or 0.0))
    closer = []
    for entry in sort_ntrip_mountpoint_entries(entries, latitude, longitude):
        if entry.get("mountpoint") == current_mountpoint:
            continue
        distance = _float_or_none(entry.get("distance_m"))
        if distance is None:
            continue
        improvement = current_distance - distance
        if improvement <= 0.0 or improvement < minimum:
            continue
        candidate = dict(entry)
        candidate["current_distance_m"] = current_distance
        candidate["distance_improvement_m"] = improvement
        closer.append(candidate)
    return closer


def rtcm3_crc24q(data):
    """Return the CRC-24Q used by RTCM version 3 frames."""

    crc = 0
    for value in data:
        crc ^= int(value) << 16
        for _ in range(8):
            crc <<= 1
            if crc & 0x1000000:
                crc ^= 0x1864CFB
    return crc & 0xFFFFFF


class Rtcm3Framer:
    """Buffer a TCP byte stream and emit complete, CRC-valid RTCM3 frames."""

    def __init__(self):
        self.buffer = bytearray()

    def feed(self, data):
        if data:
            self.buffer.extend(data)
        frames = []
        while self.buffer:
            preamble = self.buffer.find(b"\xD3")
            if preamble < 0:
                self.buffer.clear()
                break
            if preamble:
                del self.buffer[:preamble]
            if len(self.buffer) < 3:
                break
            if self.buffer[1] & 0xFC:
                del self.buffer[0]
                continue
            payload_length = ((self.buffer[1] & 0x03) << 8) | self.buffer[2]
            frame_length = 3 + payload_length + 3
            if len(self.buffer) < frame_length:
                break
            frame = bytes(self.buffer[:frame_length])
            expected_crc = int.from_bytes(frame[-3:], "big")
            if rtcm3_crc24q(frame[:-3]) != expected_crc:
                del self.buffer[0]
                continue
            del self.buffer[:frame_length]
            frames.append(frame)
        return frames


def _fetch_ntrip_source_table_response(config, authenticate=False):
    host = config["host"]
    port = int(config["port"])
    timeout = float(config.get("sourcetable_timeout", 5.0) or 5.0)
    lines = [
        "GET / HTTP/1.0",
        f"Host: {host}",
        "User-Agent: NTRIP synced-image-recorder",
        "Ntrip-Version: Ntrip/2.0",
        "Connection: close",
    ]
    username = config.get("username") if authenticate else None
    password = config.get("password") if authenticate else None
    if authenticate and (username or password):
        token = base64.b64encode(f"{username or ''}:{password or ''}".encode("utf-8")).decode("ascii")
        lines.append(f"Authorization: Basic {token}")
    lines.extend(["", ""])

    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall("\r\n".join(lines).encode("ascii"))
        chunks = []
        total = 0
        while total < 1_000_000:
            try:
                data = sock.recv(65536)
            except socket.timeout:
                if chunks:
                    break
                raise
            if not data:
                break
            chunks.append(data)
            total += len(data)
    return b"".join(chunks).decode("utf-8", errors="replace")


def _ntrip_response_status(text):
    first_line = str(text or "").splitlines()[0].strip() if text else "empty response"
    return first_line[:160]


def fetch_ntrip_source_table(config):
    """Fetch a caster catalogue without breaking public source-table endpoints.

    Some casters, including gnssdata.or.kr, return ``404 Not Found`` for their
    public ``/`` catalogue when an otherwise valid stream Authorization header
    is present. Private casters may require the opposite. Try the public request
    first and only retry with credentials when the unauthenticated response did
    not contain a source table. Mountpoint stream requests remain authenticated.
    """

    text = _fetch_ntrip_source_table_response(config, authenticate=False)
    if "STR;" in text:
        return text

    statuses = [_ntrip_response_status(text)]
    if config.get("username") or config.get("password"):
        authenticated_text = _fetch_ntrip_source_table_response(config, authenticate=True)
        if "STR;" in authenticated_text:
            return authenticated_text
        statuses.append(_ntrip_response_status(authenticated_text))

    raise RuntimeError(
        "NTRIP source table did not contain STR mountpoint entries "
        f"(responses: {'; '.join(statuses)})"
    )


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
        self.current_mountpoint = None
        self.mountpoint_sequence = []
        self.source_table_entries = None
        self.source_table_error = None
        self.source_table_attempted = False

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

    def _live_position_is_fresh(self):
        updated = _float_or_none(self.latest_nmea.get("position_monotonic"))
        max_age = float(self.config.get("position_max_age", 30.0) or 0.0)
        if updated is None or max_age <= 0.0:
            return True
        return time.monotonic() - updated <= max_age

    def _rover_position(self):
        if "position_valid" in self.latest_nmea:
            if not self.latest_nmea.get("position_valid"):
                return None
            latitude = _float_or_none(self.latest_nmea.get("latitude_deg"))
            longitude = _float_or_none(self.latest_nmea.get("longitude_deg"))
            if (
                latitude is not None
                and longitude is not None
                and self._live_position_is_fresh()
            ):
                return latitude, longitude
            return None

        gga = self.latest_nmea.get("gga")
        if gga:
            parsed = parse_nmea_line(gga)
            if str(parsed.get("fix_quality", "")) in ("", "0"):
                return None
            latitude = _float_or_none(parsed.get("latitude_deg"))
            longitude = _float_or_none(parsed.get("longitude_deg"))
            if (
                latitude is not None
                and longitude is not None
                and self._live_position_is_fresh()
            ):
                return latitude, longitude
            return None

        latitude = _float_or_none(self.latest_nmea.get("latitude_deg"))
        longitude = _float_or_none(self.latest_nmea.get("longitude_deg"))
        if latitude is not None and longitude is not None:
            if self._live_position_is_fresh():
                return latitude, longitude
            return None
        if self.latest_nmea.get("position_monotonic") is not None:
            return None

        latitude = _float_or_none(self.config.get("latitude"))
        longitude = _float_or_none(self.config.get("longitude"))
        if latitude is not None and longitude is not None:
            return latitude, longitude
        return None

    def _wait_for_rover_position(self):
        position = self._rover_position()
        if position is not None:
            return position

        wait_s = float(self.config.get("position_wait_s", 0.0) or 0.0)
        deadline = time.monotonic() + wait_s
        while wait_s > 0.0 and not self.stop_event.is_set():
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
            position = self._rover_position()
            if position is not None:
                return position
        return None

    def _load_source_table_entries(self, force=False):
        if self.source_table_attempted and not force:
            return self.source_table_entries or []
        self.source_table_attempted = True
        previous_entries = self.source_table_entries
        try:
            text = fetch_ntrip_source_table(self.config)
            self.source_table_entries = parse_ntrip_source_table(
                text,
                self.config.get("mountpoint_format", DEFAULT_RTK_MOUNTPOINT_FORMAT),
            )
            self.source_table_error = None
            if self.source_table_entries:
                print(f"RTK NTRIP source table loaded: {len(self.source_table_entries)} mountpoints")
        except Exception as exc:
            self.source_table_error = str(exc)
            if previous_entries is None:
                self.source_table_entries = []
            print(f"RTK NTRIP source table unavailable: {self.source_table_error}")
        return self.source_table_entries or []

    def _configured_mountpoints(self, auto_mountpoint):
        primary = parse_ntrip_mountpoint_list(self.config.get("mountpoint"))
        candidates = parse_ntrip_mountpoint_list(self.config.get("mountpoint_candidates"))
        ordered = candidates + primary if auto_mountpoint else primary + candidates
        return parse_ntrip_mountpoint_list(ordered)

    def _mountpoint_candidates(self, wait_for_position=True, refresh_source_table=False):
        auto_mountpoint = bool(self.config.get("auto_mountpoint", True))
        position = (
            self._wait_for_rover_position()
            if auto_mountpoint and wait_for_position
            else self._rover_position()
        )
        configured_mountpoints = self._configured_mountpoints(auto_mountpoint)

        if auto_mountpoint:
            entries = self._load_source_table_entries(force=refresh_source_table)
            if not entries:
                entries = fallback_rtk_mountpoint_entries(
                    self.config.get("mountpoint_format", DEFAULT_RTK_MOUNTPOINT_FORMAT)
                )
            entries = list(entries)
            entries.extend(ntrip_entry_for_mountpoint(mountpoint) for mountpoint in configured_mountpoints)
        else:
            entries = [ntrip_entry_for_mountpoint(mountpoint) for mountpoint in configured_mountpoints]

        if not entries and self.config.get("mountpoint"):
            entries.append(ntrip_entry_for_mountpoint(self.config["mountpoint"]))

        latitude = position[0] if position else None
        longitude = position[1] if position else None
        ranked = sort_ntrip_mountpoint_entries(
            entries,
            latitude,
            longitude,
            max_count=int(self.config.get("max_mountpoints", 0) or 0),
            preferred_mountpoints=configured_mountpoints,
        )
        self.mountpoint_sequence = ranked
        return ranked

    def _periodic_switch_candidates(self, current_entry):
        if not self.config.get("auto_mountpoint", True):
            return []
        position = self._rover_position()
        if position is None:
            return []
        ranked = self._mountpoint_candidates(
            wait_for_position=False,
            refresh_source_table=False,
        )
        return closer_ntrip_mountpoint_entries(
            ranked,
            current_entry,
            position[0],
            position[1],
            self.config.get("switch_min_improvement_m", 0.0),
        )

    def _revalidate_periodic_candidate(self, current_entry, candidate):
        position = self._rover_position()
        if position is None:
            return None
        closer = closer_ntrip_mountpoint_entries(
            [current_entry, candidate],
            current_entry,
            position[0],
            position[1],
            self.config.get("switch_min_improvement_m", 0.0),
        )
        candidate_mountpoint = candidate.get("mountpoint")
        return next(
            (entry for entry in closer if entry.get("mountpoint") == candidate_mountpoint),
            None,
        )

    def _request(self, mountpoint):
        mountpoint = mountpoint.lstrip("/")
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

    def _uses_rtcm3_framing(self, entry):
        descriptions = [entry.get("format"), entry.get("mountpoint")]
        if self.config.get("auto_mountpoint", True):
            descriptions.append(self.config.get("mountpoint_format"))
        return any(
            "RTCM3" in str(value).replace(" ", "").replace(".", "").upper()
            for value in descriptions
            if value
        )

    def _is_cancelled(self, cancel_event=None):
        return self.stop_event.is_set() or bool(cancel_event and cancel_event.is_set())

    @staticmethod
    def _split_response_header(response):
        """Return a complete NTRIP response header and its stream payload.

        NTRIP v2 casters use a regular HTTP header terminated by an empty
        line. Older NTRIP v1 casters commonly reply with only
        ``ICY 200 OK\r\n`` and start the RTCM stream immediately afterwards.
        Waiting for a second CRLF in that response consumes RTCM data as if it
        were a header and eventually reports a false incomplete-header error.
        """

        for separator in (b"\r\n\r\n", b"\n\n"):
            if separator in response:
                return response.split(separator, 1)

        if response.startswith(b"ICY "):
            for separator in (b"\r\n", b"\n"):
                if separator in response:
                    return response.split(separator, 1)
        return None

    @staticmethod
    def _close_socket(sock):
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass

    def _register_handover_socket(self, handover_state, sock):
        if handover_state is None:
            return True
        with handover_state["lock"]:
            if handover_state["cancel"].is_set():
                accepted = False
            else:
                handover_state["socket"] = sock
                accepted = True
        if not accepted:
            self._close_socket(sock)
        return accepted

    @staticmethod
    def _clear_handover_socket(handover_state, sock):
        if handover_state is None:
            return
        with handover_state["lock"]:
            if handover_state.get("socket") is sock:
                handover_state["socket"] = None

    def _open_mountpoint_stream(
        self,
        entry,
        require_frame=False,
        cancel_event=None,
        handover_state=None,
    ):
        """Open a caster stream without mutating the public active-stream state."""

        mountpoint = entry["mountpoint"]
        connect_timeout = float(self.config.get("connect_timeout", 10.0) or 10.0)
        data_timeout = float(self.config.get("data_timeout", 15.0) or 0.0)
        sock = None
        try:
            if self._is_cancelled(cancel_event):
                raise RuntimeError("NTRIP connection cancelled")
            sock = socket.create_connection(
                (self.config["host"], self.config["port"]),
                timeout=connect_timeout,
            )
            if not self._register_handover_socket(handover_state, sock):
                raise RuntimeError("NTRIP connection cancelled")
            sock.settimeout(connect_timeout)
            sock.sendall(self._request(mountpoint))
            response = b""
            header_deadline = time.monotonic() + connect_timeout
            parsed_response = None
            while len(response) < 4096:
                parsed_response = self._split_response_header(response)
                if parsed_response is not None:
                    break
                if self._is_cancelled(cancel_event):
                    raise RuntimeError("NTRIP connection cancelled")
                remaining = header_deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(f"NTRIP response header timed out after {connect_timeout:.1f}s")
                sock.settimeout(min(0.5, remaining))
                try:
                    chunk = sock.recv(256)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                response += chunk
            if parsed_response is None:
                parsed_response = self._split_response_header(response)
            if parsed_response is None:
                if not response:
                    raise RuntimeError(
                        "NTRIP caster closed the connection without a response; "
                        "verify NTRIP username and password"
                    )
                raise RuntimeError("NTRIP caster returned an incomplete response header")
            header, remainder = parsed_response
            first_line = header.splitlines()[0].decode("ascii", errors="replace") if header else ""
            status_parts = first_line.split()
            if len(status_parts) < 2 or status_parts[1] != "200":
                raise RuntimeError(f"NTRIP caster rejected request: {first_line}")

            last_gga_time = 0.0
            gga = self._current_gga()
            if gga:
                sock.sendall(gga.encode("ascii", errors="ignore"))
                last_gga_time = time.monotonic()

            framer = Rtcm3Framer() if self._uses_rtcm3_framing(entry) else None
            frames = framer.feed(remainder) if framer is not None else ([remainder] if remainder else [])
            required_payloads = NTRIP_HANDOVER_MIN_VALID_PAYLOADS if require_frame else 0
            if len(frames) < required_payloads:
                validation_timeout = data_timeout if data_timeout > 0.0 else connect_timeout
                validation_deadline = time.monotonic() + validation_timeout
                while len(frames) < required_payloads:
                    if self._is_cancelled(cancel_event):
                        raise RuntimeError("NTRIP connection cancelled")
                    remaining = validation_deadline - time.monotonic()
                    if remaining <= 0.0:
                        raise TimeoutError(
                            f"fewer than {required_payloads} valid correction payloads "
                            f"received in {validation_timeout:.1f}s"
                        )
                    sock.settimeout(min(0.5, remaining))
                    try:
                        data = sock.recv(4096)
                    except socket.timeout:
                        continue
                    if not data:
                        raise RuntimeError("NTRIP stream closed before valid RTCM data")
                    if framer is not None:
                        frames.extend(framer.feed(data))
                    else:
                        frames.append(data)

            sock.settimeout(NTRIP_STREAM_POLL_TIMEOUT_S)
            return {
                "socket": sock,
                "entry": dict(entry),
                "framer": framer,
                "frames": frames,
                "last_gga_time": last_gga_time,
            }
        except Exception:
            self._clear_handover_socket(handover_state, sock)
            self._close_socket(sock)
            raise

    def _forward_rtcm_frames(self, frames):
        payload = b"".join(frames)
        if not payload:
            return False
        self.serial_write(payload)
        self.bytes_received += len(payload)
        return True

    def _handover_worker(
        self,
        current_entry,
        result_queue,
        cancel_event,
        handover_state=None,
    ):
        errors = []
        cycle_started = time.monotonic()
        cycle_timeout = float(self.config.get("reselect_interval", 300.0) or 300.0)
        try:
            candidates = self._periodic_switch_candidates(current_entry)
        except Exception as exc:
            candidates = []
            errors.append(f"candidate refresh: {exc}")

        if not candidates and not errors:
            result = {"status": "keep", "reason": "no meaningfully closer mountpoint"}
        else:
            result = None
            for candidate in candidates:
                if self._is_cancelled(cancel_event):
                    return
                if time.monotonic() - cycle_started >= cycle_timeout:
                    errors.append(f"handover cycle exceeded {cycle_timeout:.1f}s")
                    break
                candidate = self._revalidate_periodic_candidate(current_entry, candidate)
                if candidate is None:
                    errors.append("candidate is no longer closer at the current rover position")
                    continue
                current_name = current_entry.get("mountpoint", "-")
                improvement = candidate.get("distance_improvement_m")
                improvement_text = (
                    f", {improvement / 1000.0:.1f}km closer"
                    if improvement is not None
                    else ""
                )
                print(
                    f"RTK NTRIP validating handover: {current_name} -> "
                    f"{candidate.get('mountpoint', '-')}{improvement_text}; current stream remains active"
                )
                try:
                    stream = self._open_mountpoint_stream(
                        candidate,
                        require_frame=True,
                        cancel_event=cancel_event,
                        handover_state=handover_state,
                    )
                except Exception as exc:
                    if self._is_cancelled(cancel_event):
                        return
                    errors.append(f"{candidate.get('mountpoint', '-')}: {exc}")
                    continue
                validated_candidate = self._revalidate_periodic_candidate(
                    current_entry,
                    candidate,
                )
                if validated_candidate is None:
                    self._clear_handover_socket(handover_state, stream.get("socket"))
                    self._close_socket(stream.get("socket"))
                    errors.append(
                        f"{candidate.get('mountpoint', '-')}: rover position changed during validation"
                    )
                    continue
                stream["entry"] = validated_candidate
                if self._is_cancelled(cancel_event):
                    self._clear_handover_socket(handover_state, stream.get("socket"))
                    self._close_socket(stream.get("socket"))
                    return
                result = {"status": "ready", "stream": stream}
                break
            if result is None:
                result = {
                    "status": "failed",
                    "reason": "; ".join(errors) or "no closer mountpoint accepted RTCM data",
                }

        if handover_state is not None:
            with handover_state["lock"]:
                if handover_state["cancel"].is_set() or self.stop_event.is_set():
                    cancelled = True
                else:
                    result_queue.put_nowait(result)
                    cancelled = False
            if cancelled:
                stream = result.get("stream") if result else None
                self._clear_handover_socket(handover_state, stream.get("socket") if stream else None)
                self._close_socket(stream.get("socket") if stream else None)
            return
        if not self._is_cancelled(cancel_event):
            result_queue.put(result)

    def _start_handover(self, current_entry):
        result_queue = queue.Queue(maxsize=1)
        cancel_event = threading.Event()
        handover = {
            "queue": result_queue,
            "cancel": cancel_event,
            "thread": None,
            "lock": threading.Lock(),
            "socket": None,
        }
        thread = threading.Thread(
            target=self._handover_worker,
            args=(dict(current_entry), result_queue, cancel_event, handover),
            name="gps-ntrip-handover",
            daemon=True,
        )
        handover["thread"] = thread
        thread.start()
        return handover

    def _cancel_handover(self, handover):
        if handover is None:
            return
        with handover["lock"]:
            handover["cancel"].set()
            sock = handover.get("socket")
            handover["socket"] = None
        self._close_socket(sock)
        handover["thread"].join(timeout=0.2)
        while True:
            try:
                result = handover["queue"].get_nowait()
            except queue.Empty:
                break
            stream = result.get("stream") if result else None
            self._close_socket(stream.get("socket") if stream else None)

    def _stream_mountpoint(self, entry):
        data_timeout = float(self.config.get("data_timeout", 15.0) or 0.0)
        reselect_interval = float(self.config.get("reselect_interval", 300.0) or 0.0)
        periodic_enabled = bool(self.config.get("auto_mountpoint", True) and reselect_interval > 0.0)
        stream = self._open_mountpoint_stream(entry, require_frame=True)
        sock = stream["socket"]
        framer = stream["framer"]
        active_entry = stream["entry"]
        last_gga_time = stream["last_gga_time"]
        last_data_time = time.monotonic()
        handover = None

        self.connected = True
        self.current_mountpoint = active_entry["mountpoint"]
        self.error = None
        distance = active_entry.get("distance_m")
        distance_text = f", distance={distance / 1000.0:.1f}km" if distance is not None else ""
        print(
            f"RTK NTRIP connected: {self.config['host']}:{self.config['port']}/"
            f"{active_entry['mountpoint']}{distance_text}"
        )
        if self._forward_rtcm_frames(stream["frames"]):
            last_data_time = time.monotonic()
        next_reselect_time = (
            time.monotonic() + reselect_interval if periodic_enabled else None
        )

        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                if (
                    next_reselect_time is not None
                    and handover is None
                    and now >= next_reselect_time
                ):
                    handover = self._start_handover(active_entry)
                    next_reselect_time = now + reselect_interval

                if handover is not None:
                    try:
                        handover_result = handover["queue"].get_nowait()
                    except queue.Empty:
                        handover_result = None
                    if handover_result is not None:
                        completed_handover = handover
                        completed_handover["thread"].join(timeout=0.2)
                        handover = None
                        next_reselect_time = time.monotonic() + reselect_interval
                        status = handover_result.get("status")
                        if status == "ready":
                            new_stream = handover_result["stream"]
                            validated_candidate = self._revalidate_periodic_candidate(
                                active_entry,
                                new_stream["entry"],
                            )
                            if validated_candidate is None:
                                self._clear_handover_socket(
                                    completed_handover,
                                    new_stream.get("socket"),
                                )
                                self._close_socket(new_stream.get("socket"))
                                print(
                                    "RTK NTRIP handover skipped; current stream remains active: "
                                    "rover position/fix changed before promotion"
                                )
                                continue
                            new_stream["entry"] = validated_candidate
                            try:
                                forwarded = self._forward_rtcm_frames(new_stream["frames"])
                            except Exception:
                                self._clear_handover_socket(
                                    completed_handover,
                                    new_stream.get("socket"),
                                )
                                self._close_socket(new_stream.get("socket"))
                                raise
                            if not forwarded:
                                self._clear_handover_socket(
                                    completed_handover,
                                    new_stream.get("socket"),
                                )
                                self._close_socket(new_stream.get("socket"))
                                raise RuntimeError("validated NTRIP handover contained no RTCM frame")
                            self._clear_handover_socket(
                                completed_handover,
                                new_stream.get("socket"),
                            )
                            old_sock = sock
                            old_mountpoint = active_entry.get("mountpoint", "-")
                            sock = new_stream["socket"]
                            framer = new_stream["framer"]
                            active_entry = new_stream["entry"]
                            last_gga_time = new_stream["last_gga_time"]
                            last_data_time = time.monotonic()
                            self.current_mountpoint = active_entry["mountpoint"]
                            self.error = None
                            self._close_socket(old_sock)
                            improvement = active_entry.get("distance_improvement_m")
                            improvement_text = (
                                f", {improvement / 1000.0:.1f}km closer"
                                if improvement is not None
                                else ""
                            )
                            print(
                                f"RTK NTRIP seamless handover complete: {old_mountpoint} -> "
                                f"{active_entry['mountpoint']}{improvement_text}"
                            )
                        elif status == "failed":
                            print(
                                "RTK NTRIP handover skipped; current stream remains active: "
                                f"{handover_result.get('reason', 'candidate validation failed')}"
                            )
                        else:
                            print(
                                f"RTK NTRIP reevaluation: keeping {active_entry['mountpoint']} "
                                f"({handover_result.get('reason', 'already nearest')})"
                            )

                if self.config.get("gga_interval", 10.0) > 0:
                    if now - last_gga_time >= self.config.get("gga_interval", 10.0):
                        gga = self._current_gga()
                        if gga:
                            sock.sendall(gga.encode("ascii", errors="ignore"))
                            last_gga_time = now
                if data_timeout > 0.0 and now - last_data_time >= data_timeout:
                    raise TimeoutError(f"no RTCM data for {data_timeout:.1f}s")
                try:
                    data = sock.recv(4096)
                except socket.timeout:
                    continue
                if not data:
                    raise RuntimeError("NTRIP stream closed")
                frames = framer.feed(data) if framer is not None else [data]
                if self._forward_rtcm_frames(frames):
                    last_data_time = time.monotonic()
        finally:
            self._cancel_handover(handover)
            self._close_socket(sock)

    def _run(self):
        reconnect_delay = float(self.config.get("reconnect_delay", 5.0) or 0.0)
        while not self.stop_event.is_set():
            candidates = self._mountpoint_candidates()
            if not candidates:
                self.connected = False
                self.error = "no NTRIP mountpoints available"
                print(f"RTK NTRIP reconnecting after error: {self.error}")
                self.stop_event.wait(reconnect_delay)
                continue

            for entry in candidates:
                if self.stop_event.is_set():
                    break
                self.connected = False
                try:
                    self._stream_mountpoint(entry)
                except Exception as exc:
                    had_active_stream = self.connected
                    failed_mountpoint = (
                        self.current_mountpoint
                        if had_active_stream and self.current_mountpoint
                        else entry.get("mountpoint", "-")
                    )
                    self.connected = False
                    self.error = f"{failed_mountpoint}: {exc}"
                    if not self.stop_event.is_set():
                        print(f"RTK NTRIP switching after error: {self.error}")
                    if had_active_stream:
                        # A long-lived stream may have handed over and moved far
                        # from this stale candidate list. Re-rank immediately on
                        # the next outer reconnect cycle.
                        break
                    continue

            if not self.stop_event.is_set():
                self.stop_event.wait(reconnect_delay)
        self.connected = False


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


def fov_from_intrinsics(intrinsics, width, height):
    fx = float(intrinsics[0][0])
    fy = float(intrinsics[1][1])
    width = float(width)
    height = float(height)
    diagonal = math.hypot(width, height)
    focal_diagonal = math.sqrt(fx * fy)
    return {
        "horizontal_deg": math.degrees(2.0 * math.atan(width / (2.0 * fx))),
        "vertical_deg": math.degrees(2.0 * math.atan(height / (2.0 * fy))),
        "diagonal_deg": math.degrees(2.0 * math.atan(diagonal / (2.0 * focal_diagonal))),
    }


def adjust_intrinsics_for_saved_transform(intrinsics, args, width=None, height=None):
    width = int(width or rgb_size_from_args(args)[0])
    height = int(height or rgb_size_from_args(args)[1])
    adjusted = matrix_to_json(intrinsics)
    if getattr(args, "rotate_180", False):
        adjusted[0][2] = (width - 1) - adjusted[0][2]
        adjusted[1][2] = (height - 1) - adjusted[1][2]
    if getattr(args, "flip", False):
        adjusted[1][2] = (height - 1) - adjusted[1][2]
    return adjusted


def optional_call(obj, method_name, default=None):
    try:
        method = getattr(obj, method_name)
    except AttributeError:
        return default
    try:
        return method()
    except Exception:
        return default


def rotated_rect_to_json(rect):
    if rect is None:
        return None
    center = getattr(rect, "center", None)
    size = getattr(rect, "size", None)
    return {
        "angle_deg": float(getattr(rect, "angle", 0.0) or 0.0),
        "center": {
            "x": float(getattr(center, "x", 0.0) or 0.0),
            "y": float(getattr(center, "y", 0.0) or 0.0),
        },
        "size": {
            "width": float(getattr(size, "width", 0.0) or 0.0),
            "height": float(getattr(size, "height", 0.0) or 0.0),
        },
        "normalized": bool(optional_call(rect, "isNormalized", False)),
    }


def imgframe_camera_model_metadata(message, args):
    """Extract the actual saved-image camera model from a DepthAI ImgFrame.

    Camera.requestOutput(..., enableUndistortion=True) may crop/scale the sensor
    before returning the frame. The ImgFrame transformation carries the resulting
    output intrinsics; those are the values needed for pixel+depth unprojection.
    """
    transformation = optional_call(message, "getTransformation")
    if transformation is None:
        return {}

    intrinsics = optional_call(transformation, "getIntrinsicMatrix")
    if not intrinsics:
        return {}

    width, height = rgb_size_from_args(args)
    source_intrinsics = optional_call(transformation, "getSourceIntrinsicMatrix")
    transform_matrix = optional_call(transformation, "getMatrix")
    source_size = optional_call(transformation, "getSourceSize")
    distortion = optional_call(transformation, "getDistortionCoefficients", [])
    distortion_model = optional_call(transformation, "getDistortionModel")
    source_crops = optional_call(transformation, "getSrcCrops", [])

    frame_transform = {
        "intrinsics_before_saved_transform": matrix_to_json(intrinsics),
        "intrinsics_after_saved_transform": adjust_intrinsics_for_saved_transform(
            intrinsics, args, width, height
        ),
        "distortion_coefficients": [float(value) for value in (distortion or [])],
        "distortion_model": enum_name(distortion_model) if distortion_model is not None else "",
        "source_intrinsics": matrix_to_json(source_intrinsics) if source_intrinsics else None,
        "source_size": {
            "width": int(source_size[0]),
            "height": int(source_size[1]),
        } if source_size else None,
        "matrix": matrix_to_json(transform_matrix) if transform_matrix else None,
        "source_crops": [
            crop for crop in (rotated_rect_to_json(rect) for rect in (source_crops or []))
            if crop is not None
        ],
        "frame_lens_position": int(optional_call(message, "getLensPosition", -1)),
        "frame_lens_position_raw": float(optional_call(message, "getLensPositionRaw", -1.0)),
        "source_width": int(optional_call(message, "getSourceWidth", 0) or 0),
        "source_height": int(optional_call(message, "getSourceHeight", 0) or 0),
        "source_fov_deg": {
            "horizontal": float(optional_call(message, "getSourceHFov", 0.0) or 0.0),
            "vertical": float(optional_call(message, "getSourceVFov", 0.0) or 0.0),
            "diagonal": float(optional_call(message, "getSourceDFov", 0.0) or 0.0),
        },
    }

    return {
        "intrinsics": frame_transform["intrinsics_after_saved_transform"],
        "intrinsics_frame": frame_transform["intrinsics_before_saved_transform"],
        "intrinsics_source": "DepthAI ImgFrame.getTransformation().getIntrinsicMatrix",
        "image_stream_undistorted": True,
        "distortion_coefficients": frame_transform["distortion_coefficients"],
        "distortion_model": frame_transform["distortion_model"] or "DepthAI output",
        "output_fov_deg": fov_from_intrinsics(intrinsics, width, height),
        "lens_position": frame_transform["frame_lens_position"],
        "lens_position_raw": frame_transform["frame_lens_position_raw"],
        "depthai_frame_transformation": frame_transform,
        "coordinate_unprojection": (
            "pinhole projection using DepthAI ImgFrame transformation intrinsics "
            "for the saved, already-undistorted RGB/depth pixels"
        ),
        "notes": (
            "DepthAI reported the actual output intrinsics on the first saved RGB frame. "
            "Factory lens distortion is kept in factory_distortion_coefficients for audit, "
            "but saved pixels are already undistorted and should not be undistorted again."
        ),
    }


def read_camera_model_metadata(device, args):
    width, height = rgb_size_from_args(args)
    rgb_socket = getattr(args, "rgb_socket", dai.CameraBoardSocket.CAM_A)
    try:
        calibration = device.readCalibration()
        rgb_intrinsics = calibration.getCameraIntrinsics(rgb_socket, width, height)
        image_stream_undistorted = bool(getattr(args, "rgb_undistort", False))
        metadata = {
            "model": "pinhole",
            "source": "DepthAI device calibration",
            "socket": getattr(args, "rgb_socket_name", "CAM_A"),
            "sensor_name": getattr(args, "rgb_sensor_name", ""),
            "sensor_size": {
                "width": int(getattr(args, "rgb_sensor_width", 0) or 0),
                "height": int(getattr(args, "rgb_sensor_height", 0) or 0),
            },
            "sensor_types": list(getattr(args, "rgb_sensor_types", []) or []),
            "resolution_source": getattr(args, "rgb_resolution_source", "unknown"),
            "width": width,
            "height": height,
            "intrinsics_original": matrix_to_json(rgb_intrinsics),
            "intrinsics": adjust_intrinsics_for_saved_transform(rgb_intrinsics, args, width, height),
            "intrinsics_source": (
                "DepthAI calibration.getCameraIntrinsics fallback; first RGB frame "
                "will replace this with ImgFrame transformation intrinsics"
                if image_stream_undistorted
                else "DepthAI calibration.getCameraIntrinsics"
            ),
            "factory_intrinsics": matrix_to_json(rgb_intrinsics),
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
            factory_distortion = [
                float(value)
                for value in calibration.getDistortionCoefficients(rgb_socket)
            ]
            metadata["distortion_coefficients"] = factory_distortion
            metadata["factory_distortion_coefficients"] = factory_distortion
            metadata["distortion_model"] = (
                "opencv_rational_thin_prism_tilt_14"
                if len(metadata["distortion_coefficients"]) == 14
                else "opencv"
            )
            metadata["factory_distortion_model"] = metadata["distortion_model"]
        except Exception:
            metadata["distortion_coefficients"] = []
            metadata["factory_distortion_coefficients"] = []
            metadata["distortion_model"] = "unavailable"
            metadata["factory_distortion_model"] = "unavailable"
        try:
            metadata["fov_deg"] = float(calibration.getFov(rgb_socket))
        except Exception:
            metadata["fov_deg"] = None
        try:
            metadata["factory_lens_position"] = int(calibration.getLensPosition(rgb_socket))
        except Exception:
            metadata["factory_lens_position"] = None
        return metadata
    except Exception as exc:
        print(f"Camera calibration metadata unavailable: {exc}")
        return {
            "model": "pinhole",
            "source": "unavailable",
            "socket": getattr(args, "rgb_socket_name", "CAM_A"),
            "width": width,
            "height": height,
            "intrinsics": None,
            "error": str(exc),
        }


def read_stereo_depth_metadata(device, args):
    left_socket = getattr(args, "left_socket", dai.CameraBoardSocket.CAM_B)
    right_socket = getattr(args, "right_socket", dai.CameraBoardSocket.CAM_C)
    width = int(getattr(args, "depth_input_width", WIDTH) or WIDTH)
    height = int(getattr(args, "depth_input_height", HEIGHT) or HEIGHT)
    output_width, output_height = rgb_size_from_args(args)
    metadata = {
        "source": "DepthAI device calibration",
        "left_socket": enum_name(left_socket),
        "right_socket": enum_name(right_socket),
        "stereo_matching_input_size": {"width": width, "height": height},
        "aligned_depth_output_size": {"width": output_width, "height": output_height},
        "input_size": {"width": width, "height": height},
        "input_resolution_source": getattr(args, "depth_input_resolution_source", "unknown"),
        "left_sensor": dict(getattr(args, "left_sensor", {}) or {}),
        "right_sensor": dict(getattr(args, "right_sensor", {}) or {}),
        "units": "millimeters",
        "notes": (
            "DepthAI StereoDepth computes raw depth from the stereo pair calibration. "
            "If physical stereo lenses differ from EEPROM calibration, recalibrate "
            "and flash stereo intrinsics/extrinsics instead of applying a scale patch. "
            "On RVC2 devices such as OAK-D-LR, stereo matching input width is limited "
            "to 1280, but the aligned depth output is saved at the RGB output size."
        ),
    }
    try:
        calibration = device.readCalibration()
        left_intrinsics = calibration.getCameraIntrinsics(left_socket, width, height)
        right_intrinsics = calibration.getCameraIntrinsics(right_socket, width, height)
        metadata.update({
            "left_intrinsics": matrix_to_json(left_intrinsics),
            "right_intrinsics": matrix_to_json(right_intrinsics),
            "left_fov_deg": float(calibration.getFov(left_socket)),
            "right_fov_deg": float(calibration.getFov(right_socket)),
            "baseline_cm": float(calibration.getBaselineDistance(left_socket, right_socket)),
            "left_to_right_extrinsics": matrix_to_json(
                calibration.getCameraExtrinsics(left_socket, right_socket)
            ),
            "left_distortion_coefficients": [
                float(value) for value in calibration.getDistortionCoefficients(left_socket)
            ],
            "right_distortion_coefficients": [
                float(value) for value in calibration.getDistortionCoefficients(right_socket)
            ],
        })
    except Exception as exc:
        metadata["source"] = "unavailable"
        metadata["error"] = str(exc)
    return metadata


class SerialRateLimitedReader:
    def __init__(
        self,
        name,
        device,
        baudrate,
        max_hz,
        parser=None,
        timeout=0.2,
        rtk_config=None,
        device_resolver=None,
        reconnect_delay_s=1.0,
    ):
        self.name = name
        self.device = device
        self.device_resolver = device_resolver
        self.baudrate = baudrate
        self.max_hz = max_hz
        self.parser = parser
        self.timeout = timeout
        self.rtk_config = rtk_config
        self.reconnect_delay_s = max(0.1, float(reconnect_delay_s))
        self.rtk_client = None
        self.connection_stop_event = None
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
        connection_stop_event = self.connection_stop_event
        if connection_stop_event is not None:
            connection_stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def _resolve_device(self):
        if self.device_resolver is None:
            return self.device
        return self.device_resolver()

    def _open_serial(self, device):
        try:
            return serial.Serial(
                device,
                self.baudrate,
                timeout=self.timeout,
                exclusive=True,
            )
        except TypeError:
            # Older pyserial releases and some non-POSIX backends do not expose
            # the exclusive keyword. Keep those platforms compatible.
            return serial.Serial(device, self.baudrate, timeout=self.timeout)

    def _read_connection(self, port, connection_stop_event):
        min_interval = 1.0 / self.max_hz if self.max_hz > 0 else 0.0
        last_saved_monotonic = 0.0
        if self.rtk_config is not None:
            def write_corrections(data):
                with self.write_lock:
                    port.write(data)

            self.rtk_client = NtripCorrectionClient(
                self.rtk_config,
                write_corrections,
                connection_stop_event,
                latest_nmea=self.latest_nmea,
            )
            self.rtk_client.start()
            mountpoint_text = (
                self.rtk_config.get("mountpoint")
                or self.rtk_config.get("mountpoint_candidates")
                or "auto"
            )
            print(
                f"RTK NTRIP enabled: {self.rtk_config['host']}:{self.rtk_config['port']}/"
                f"{mountpoint_text}"
            )
        while not self.stop_event.is_set() and not connection_stop_event.is_set():
            raw = port.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            raw_nmea = parse_nmea_line(line) if self.rtk_config is not None else {}
            if self.rtk_config is not None:
                nmea_type = raw_nmea.get("nmea_type")
                if nmea_type == "GGA":
                    self.latest_nmea["gga"] = line
                    valid_fix = (
                        str(raw_nmea.get("fix_quality", "")) not in ("", "0")
                        and raw_nmea.get("latitude_deg") not in ("", None)
                        and raw_nmea.get("longitude_deg") not in ("", None)
                    )
                    if valid_fix:
                        for key in ("latitude_deg", "longitude_deg", "altitude_m"):
                            value = raw_nmea.get(key)
                            if value not in ("", None):
                                self.latest_nmea[key] = value
                        self.latest_nmea["position_monotonic"] = time.monotonic()
                        self.latest_nmea["position_valid"] = True
                    else:
                        for key in ("latitude_deg", "longitude_deg", "altitude_m"):
                            self.latest_nmea.pop(key, None)
                        self.latest_nmea["position_monotonic"] = time.monotonic()
                        self.latest_nmea["position_valid"] = False
                elif nmea_type == "RMC" and raw_nmea.get("status") == "A":
                    for key in ("latitude_deg", "longitude_deg"):
                        value = raw_nmea.get(key)
                        if value not in ("", None):
                            self.latest_nmea[key] = value
                    self.latest_nmea["position_monotonic"] = time.monotonic()
                    self.latest_nmea["position_valid"] = True
                elif nmea_type == "RMC" and raw_nmea.get("status") == "V":
                    for key in ("latitude_deg", "longitude_deg"):
                        self.latest_nmea.pop(key, None)
                    self.latest_nmea["position_monotonic"] = time.monotonic()
                    self.latest_nmea["position_valid"] = False
            parsed = self.parser(line) if self.parser is not None else {}
            received_monotonic = time.monotonic()
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

    def _run(self):
        previous_error = None
        while not self.stop_event.is_set():
            connection_stop_event = threading.Event()
            self.connection_stop_event = connection_stop_event
            try:
                device = self._resolve_device()
                if not device:
                    raise RuntimeError(f"No safe {self.name} serial device was found")
                self.device = device
                with self._open_serial(device) as port:
                    self.started = True
                    self.error = None
                    previous_error = None
                    print(f"{self.name} serial connected: {device}")
                    self._read_connection(port, connection_stop_event)
            except Exception as exc:
                self.error = str(exc)
                if not self.stop_event.is_set() and self.error != previous_error:
                    print(f"{self.name} serial reconnecting after error: {self.error}")
                previous_error = self.error
            finally:
                self.started = False
                connection_stop_event.set()
                if self.rtk_client is not None:
                    self.rtk_client.stop()
                    self.rtk_client.connected = False
                self.connection_stop_event = None
            if not self.stop_event.is_set():
                self.stop_event.wait(self.reconnect_delay_s)

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
            mountpoint = self.rtk_client.current_mountpoint or "-"
            text += f", ntrip={state}({mountpoint}), rtcm_bytes={self.rtk_client.bytes_received}"
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
    host = getattr(args, "rtk_ntrip_host", "")
    mountpoint = getattr(args, "rtk_ntrip_mountpoint", "")
    if not host:
        return None
    auto_mountpoint = bool(getattr(args, "rtk_ntrip_auto_mountpoint", True))
    if not mountpoint and not auto_mountpoint:
        raise ValueError("--rtk-ntrip-mountpoint is required when --rtk-ntrip-host is set")
    return {
        "host": host,
        "port": getattr(args, "rtk_ntrip_port", 2101),
        "mountpoint": mountpoint,
        "auto_mountpoint": auto_mountpoint,
        "mountpoint_format": getattr(args, "rtk_ntrip_mountpoint_format", DEFAULT_RTK_MOUNTPOINT_FORMAT),
        "mountpoint_candidates": getattr(args, "rtk_ntrip_mountpoint_candidates", ""),
        "username": getattr(args, "rtk_ntrip_username", ""),
        "password": getattr(args, "rtk_ntrip_password", ""),
        "latitude": getattr(args, "rtk_initial_latitude_deg", None),
        "longitude": getattr(args, "rtk_initial_longitude_deg", None),
        "altitude": getattr(args, "rtk_initial_altitude_m", 0.0),
        "gga": getattr(args, "rtk_ntrip_gga", ""),
        "gga_interval": getattr(args, "rtk_ntrip_gga_interval", 10.0),
        "reconnect_delay": getattr(args, "rtk_ntrip_reconnect_delay", 5.0),
        "position_wait_s": getattr(args, "rtk_ntrip_position_wait_s", 10.0),
        "connect_timeout": getattr(args, "rtk_ntrip_connect_timeout_s", 10.0),
        "data_timeout": getattr(args, "rtk_ntrip_data_timeout_s", 15.0),
        "sourcetable_timeout": getattr(args, "rtk_ntrip_sourcetable_timeout_s", 5.0),
        "max_mountpoints": getattr(args, "rtk_ntrip_max_mountpoints", 12),
        "reselect_interval": getattr(args, "rtk_ntrip_reselect_interval_s", 300.0),
        "switch_min_improvement_m": getattr(args, "rtk_ntrip_switch_min_improvement_m", 1000.0),
        "position_max_age": getattr(args, "rtk_ntrip_position_max_age_s", 30.0),
    }


def create_serial_readers(args):
    readers = {}
    gps_selector = getattr(args, "gps_device", "auto")
    external_imu_selector = getattr(args, "external_imu_device", "auto")
    resolve_serial_devices(args)
    gps_max_hz, external_imu_max_hz = serial_max_hz_values(args)
    if args.enable_gps:
        readers["gps"] = SerialRateLimitedReader(
            "gps",
            args.gps_device,
            args.gps_baudrate,
            gps_max_hz,
            parser=NmeaParserState(),
            rtk_config=build_rtk_config(args),
            device_resolver=lambda: resolve_serial_device(gps_selector, "gps"),
        )
    if args.enable_external_imu:
        parser = parse_ebimu_line if args.external_imu_format == "ebimu" else None

        def resolve_external_imu():
            gps_reader = readers.get("gps")
            excluded = [gps_reader.device] if gps_reader is not None else []
            return resolve_serial_device(
                external_imu_selector,
                "external_imu",
                exclude_devices=excluded,
            )

        readers["external_imu"] = SerialRateLimitedReader(
            "external_imu",
            args.external_imu_device,
            args.external_imu_baudrate,
            external_imu_max_hz,
            parser=parser,
            device_resolver=resolve_external_imu,
        )
    return readers


def start_serial_readers(readers):
    for reader in readers.values():
        device = reader.device or "<unresolved>"
        print(f"Starting {reader.name} serial: {device} @ {reader.baudrate}, max {reader.max_hz:g} Hz")
        reader.start()


def stop_serial_readers(readers):
    first_error = None
    for reader in readers.values():
        try:
            reader.stop()
        except Exception as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


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
        f"DepthAI negotiated {last_speed}, not USB 3.x. RGB/depth/confidence at "
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


def configure_monitor_pipeline(pipeline, args):
    """Build the low-bandwidth RGB + OAK IMU pipeline used by --monitor-only."""
    require_depthai_v3()
    args.rgb_undistort = True
    args.rgb_undistort_effective = True

    device = pipeline.getDefaultDevice()
    set_device_identity_metadata(device, args)
    try:
        args.depthai_platform = enum_name(device.getPlatform())
    except Exception:
        args.depthai_platform = "unknown"
    rgb_size = resolve_rgb_output_size(device, args)
    rgb_camera_size = resolve_rgb_camera_output_size(args, rgb_size)
    color_socket = getattr(args, "rgb_socket", dai.CameraBoardSocket.CAM_A)

    camera = pipeline.create(dai.node.Camera).build(color_socket)
    rgb_output = request_camera_output(
        camera,
        args.fps,
        size=rgb_camera_size,
        frame_type=dai.ImgFrame.Type.NV12,
        enable_undistortion=True,
    )
    rgb_output = resize_camera_output(
        pipeline,
        rgb_output,
        rgb_camera_size,
        rgb_size,
    )
    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(
        [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW],
        args.imu_rate,
    )
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(args.imu_batch)

    if args.rgb_transport_effective == "mjpeg":
        rgb_output = create_mjpeg_output(
            pipeline, rgb_output, args.fps, args.rgb_transport_quality
        )
    print(
        f"Monitor-only camera pipeline: rgb={rgb_size[0]}x{rgb_size[1]} "
        f"@ {float(args.fps):g} FPS, depth=disabled, imu={int(args.imu_rate)} Hz"
    )
    return {"mode": "monitor", "rgb": rgb_output, "imu": imu.out}


def configure_pipeline(pipeline, args, sync_imu=True):
    require_depthai_v3()

    # This is a production invariant, not a tuning option. Saved RGB and aligned
    # depth must describe the same factory-undistorted pixel geometry.
    args.rgb_undistort = True
    args.rgb_undistort_effective = True

    device = pipeline.getDefaultDevice()
    set_device_identity_metadata(device, args)
    platform = device.getPlatform()
    rgb_size = resolve_rgb_output_size(device, args)
    rgb_camera_size = resolve_rgb_camera_output_size(args, rgb_size)
    depth_fps = resolve_depth_fps(args)
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
    print(f"Camera FPS: rgb={float(args.fps):g}, depth={depth_fps:g}")
    if depth_fps > float(args.fps):
        print(
            "Depth FPS is higher than RGB FPS; RGB-aligned depth output may still be "
            "limited by the RGB alignment stream cadence."
        )

    color_socket = getattr(args, "rgb_socket", dai.CameraBoardSocket.CAM_A)
    left_socket, right_socket = resolve_stereo_sockets(
        device,
        getattr(args, "left_socket", None),
        getattr(args, "right_socket", None),
    )
    mono_size = resolve_stereo_input_size(device, args, platform, left_socket, right_socket)

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

    rgb_camera_output = request_camera_output(
        cam_rgb,
        args.fps,
        size=rgb_camera_size,
        frame_type=dai.ImgFrame.Type.NV12,
        enable_undistortion=True,
    )
    rgb_output = resize_camera_output(
        pipeline,
        rgb_camera_output,
        rgb_camera_size,
        rgb_size,
    )
    left_output = request_camera_output(mono_left, depth_fps, size=mono_size, frame_type=dai.ImgFrame.Type.GRAY8)
    right_output = request_camera_output(mono_right, depth_fps, size=mono_size, frame_type=dai.ImgFrame.Type.GRAY8)

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
        image_align.setOutputSize(*rgb_size)
        if confidence_align is not None:
            confidence_align.setOutputSize(*rgb_size)

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
            pipeline, confidence_output, depth_fps, args.confidence_transport_quality
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
