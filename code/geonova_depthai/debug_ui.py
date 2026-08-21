#!/usr/bin/env python3

import argparse
import bisect
import csv
import json
import mimetypes
import os
import socket
import subprocess
import threading
import time
import urllib.parse
from collections import OrderedDict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config_cli import parse_args_with_yaml

import cv2
import numpy as np


HOST = "0.0.0.0"
DEFAULT_PORT = 8088
WIDTH = 1280
HEIGHT = 720


DATASET_CACHE = OrderedDict()
DEPTH_CACHE = OrderedDict()
CACHE_LOCK = threading.Lock()
MAX_DATASET_CACHE = 6
MAX_DEPTH_CACHE = 12


def resolve_path(path_text):
    if not path_text:
        raise ValueError("Dataset path is empty.")
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def json_bytes(payload):
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def read_csv(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value, default=None):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "y"):
        return True
    if normalized in ("0", "false", "no", "n"):
        return False
    return default


def clamp(value, low, high):
    return max(low, min(high, value))


def metadata_image_size(metadata):
    size = (metadata or {}).get("image_size") or {}
    if isinstance(size, dict):
        width = safe_int(size.get("width"), WIDTH)
        height = safe_int(size.get("height"), HEIGHT)
    elif isinstance(size, (list, tuple)) and len(size) >= 2:
        width = safe_int(size[0], WIDTH)
        height = safe_int(size[1], HEIGHT)
    else:
        width, height = WIDTH, HEIGHT
    return max(1, width), max(1, height)


class Dataset:
    def __init__(self, root):
        self.root = root
        timestamps_path = root / "timestamps.csv"
        imu_path = root / "imu.csv"
        metadata_path = root / "metadata.json"

        if not timestamps_path.exists():
            raise ValueError(f"timestamps.csv not found in {root}")
        if not imu_path.exists():
            raise ValueError(f"imu.csv not found in {root}")

        self.timestamps = read_csv(timestamps_path)
        self.imu_by_frame = {}
        for row in read_csv(imu_path):
            frame_index = safe_int(row.get("frame_index"), -1)
            self.imu_by_frame.setdefault(frame_index, []).append(row)

        self.gps_rows = read_csv(root / "gps.csv") if (root / "gps.csv").exists() else []
        self.external_imu_rows = read_csv(root / "external_imu.csv") if (root / "external_imu.csv").exists() else []
        self.gps_by_sample_index = {
            safe_int(row.get("sample_index"), -1): row
            for row in self.gps_rows
        }
        self.valid_course_rows = []
        for row in self.gps_rows:
            host_ns = safe_int(row.get("host_monotonic_ns"), None)
            course_deg = safe_float(row.get("course_deg"))
            speed_knots = safe_float(row.get("speed_knots"))
            hdop = safe_float(row.get("hdop"))
            if host_ns is None or course_deg is None or speed_knots is None:
                continue
            if speed_knots * 0.514444 < 2.0:
                continue
            if hdop is not None and hdop > 2.5:
                continue
            self.valid_course_rows.append((host_ns, row))
        self.valid_course_times = [item[0] for item in self.valid_course_rows]
        self.external_imu_by_sample_index = {
            safe_int(row.get("sample_index"), -1): row
            for row in self.external_imu_rows
        }

        self.metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as file:
                self.metadata = json.load(file)
        self.image_width, self.image_height = metadata_image_size(self.metadata)

        if not self.timestamps:
            raise ValueError(f"No frames listed in {timestamps_path}")

        self.yolo_by_frame = {}
        self.yolo_results_mtime_ns = None
        self.reload_yolo_results()

    def reload_yolo_results(self):
        results_path = self.root / "yolo_seg" / "detections.jsonl"
        mtime_ns = results_path.stat().st_mtime_ns if results_path.exists() else None
        if mtime_ns == self.yolo_results_mtime_ns:
            return
        results = {}
        if results_path.exists():
            with results_path.open(encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    results[safe_int(row.get("frame_index"), -1)] = row
        self.yolo_by_frame = results
        self.yolo_results_mtime_ns = mtime_ns

    def yolo_result(self, index):
        self.reload_yolo_results()
        return self.yolo_by_frame.get(index)

    @property
    def frame_count(self):
        return len(self.timestamps)

    def frame(self, index):
        index = clamp(index, 0, self.frame_count - 1)
        row = self.timestamps[index]
        rgb_file = row.get("rgb_file")
        depth_file = row.get("depth_file")
        if not rgb_file or not depth_file:
            raise ValueError("timestamps.csv must include rgb_file and depth_file columns.")
        return {
            "index": index,
            "row": row,
            "rgb_path": self.root / rgb_file,
            "depth_path": self.root / depth_file,
            "imu": self.imu_by_frame.get(index, []),
            "gps": self.gps_by_sample_index.get(safe_int(row.get("gps_sample_index"), -1)),
            "external_imu": self.external_imu_by_sample_index.get(safe_int(row.get("external_imu_sample_index"), -1)),
        }

    def nearest_valid_course(self, host_monotonic_ns, max_delta_ms=5000.0):
        if host_monotonic_ns is None or not self.valid_course_rows:
            return None, None
        pos = bisect.bisect_left(self.valid_course_times, host_monotonic_ns)
        candidates = []
        if pos < len(self.valid_course_rows):
            candidates.append(self.valid_course_rows[pos])
        if pos > 0:
            candidates.append(self.valid_course_rows[pos - 1])
        if not candidates:
            return None, None
        best_host_ns, best_row = min(candidates, key=lambda item: abs(item[0] - host_monotonic_ns))
        delta_ms = (best_host_ns - host_monotonic_ns) / 1_000_000.0
        if abs(delta_ms) > max_delta_ms:
            return None, None
        return best_row, delta_ms

    def nearest_gps_altitude(self, host_monotonic_ns):
        if not self.gps_rows or host_monotonic_ns is None:
            return None
        candidates = [
            row for row in self.gps_rows
            if row.get("altitude_m") not in ("", None)
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda row: abs(safe_int(row.get("host_monotonic_ns"), 0) - host_monotonic_ns),
        )


def get_dataset(path_text):
    root = resolve_path(path_text)
    key = str(root)
    with CACHE_LOCK:
        cached = DATASET_CACHE.get(key)
        if cached is not None:
            DATASET_CACHE.move_to_end(key)
            return cached

    dataset = Dataset(root)
    with CACHE_LOCK:
        DATASET_CACHE[key] = dataset
        DATASET_CACHE.move_to_end(key)
        while len(DATASET_CACHE) > MAX_DATASET_CACHE:
            DATASET_CACHE.popitem(last=False)
    return dataset


def get_depth_frame(dataset, index):
    frame = dataset.frame(index)
    depth_path = frame["depth_path"]
    key = str(depth_path)
    with CACHE_LOCK:
        cached = DEPTH_CACHE.get(key)
        if cached is not None:
            DEPTH_CACHE.move_to_end(key)
            return cached

    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Failed to read depth image: {depth_path}")
    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16)

    with CACHE_LOCK:
        DEPTH_CACHE[key] = depth
        DEPTH_CACHE.move_to_end(key)
        while len(DEPTH_CACHE) > MAX_DEPTH_CACHE:
            DEPTH_CACHE.popitem(last=False)
    return depth


def latest_dataset_under(root_text):
    root = resolve_path(root_text)
    if (root / "timestamps.csv").exists():
        return root
    if not root.exists():
        raise ValueError(f"Path does not exist: {root}")

    candidates = [
        path for path in root.iterdir()
        if path.is_dir() and (path / "timestamps.csv").exists()
    ]
    if not candidates:
        raise ValueError(f"No dataset folders found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def summarize_imu(rows):
    if not rows:
        return None
    row = rows[-1]
    return {
        "packet_count": len(rows),
        "accel": {
            "x": safe_float(row.get("accel_x_m_s2")),
            "y": safe_float(row.get("accel_y_m_s2")),
            "z": safe_float(row.get("accel_z_m_s2")),
        },
        "gyro": {
            "x": safe_float(row.get("gyro_x_rad_s")),
            "y": safe_float(row.get("gyro_y_rad_s")),
            "z": safe_float(row.get("gyro_z_rad_s")),
        },
    }


def summarize_gps(row):
    if not row:
        return None
    return {
        "sample_index": row.get("sample_index"),
        "nmea_type": row.get("nmea_type"),
        "latitude_deg": safe_float(row.get("latitude_deg")),
        "longitude_deg": safe_float(row.get("longitude_deg")),
        "altitude_m": safe_float(row.get("altitude_m")),
        "fix_quality": row.get("fix_quality"),
        "fix_quality_name": row.get("fix_quality_name"),
        "rtk_status": row.get("rtk_status"),
        "rtk_fixed": row.get("rtk_fixed"),
        "rtk_corrected": row.get("rtk_corrected"),
        "position_valid": row.get("position_valid"),
        "status": row.get("status"),
        "satellites": row.get("satellites"),
        "hdop": safe_float(row.get("hdop")),
        "differential_age_s": safe_float(row.get("differential_age_s")),
        "reference_station_id": row.get("reference_station_id"),
        "speed_knots": safe_float(row.get("speed_knots")),
        "course_deg": safe_float(row.get("course_deg")),
    }


def parse_ebimu_row(row):
    if not row:
        return None

    qx = safe_float(row.get("q_x"))
    qy = safe_float(row.get("q_y"))
    qz = safe_float(row.get("q_z"))
    qw = safe_float(row.get("q_w"))
    if all(value is not None for value in (qx, qy, qz, qw)):
        return {
            "orientation_format": "quaternion",
            "q_x": qx,
            "q_y": qy,
            "q_z": qz,
            "q_w": qw,
            "gyro": {
                "x": safe_float(row.get("gyro_x")),
                "y": safe_float(row.get("gyro_y")),
                "z": safe_float(row.get("gyro_z")),
            },
            "accel": {
                "x": safe_float(row.get("accel_x")),
                "y": safe_float(row.get("accel_y")),
                "z": safe_float(row.get("accel_z")),
            },
            "mag": {
                "x": safe_float(row.get("mag_x")),
                "y": safe_float(row.get("mag_y")),
                "z": safe_float(row.get("mag_z")),
            },
            "timestamp_ms": safe_float(row.get("ebimu_timestamp_ms")),
        }

    raw = row.get("raw", "")
    if raw.startswith("*"):
        parts = raw[1:].split(",")
        try:
            values = [float(part) for part in parts]
        except ValueError:
            values = []
        if len(values) >= 14:
            return {
                "orientation_format": "quaternion",
                "q_z": values[0],
                "q_y": values[1],
                "q_x": values[2],
                "q_w": values[3],
                "gyro": {"x": values[4], "y": values[5], "z": values[6]},
                "accel": {"x": values[7], "y": values[8], "z": values[9]},
                "mag": {"x": values[10], "y": values[11], "z": values[12]},
                "timestamp_ms": values[13],
            }
        if len(values) >= 13:
            return {
                "orientation_format": "euler",
                "roll_deg": values[0],
                "pitch_deg": values[1],
                "yaw_deg": values[2],
                "gyro": {"x": values[3], "y": values[4], "z": values[5]},
                "accel": {"x": values[6], "y": values[7], "z": values[8]},
                "mag": {"x": values[9], "y": values[10], "z": values[11]},
                "timestamp_ms": values[12],
            }

    return None


def summarize_external_imu(row):
    parsed = parse_ebimu_row(row)
    if not row and not parsed:
        return None
    summary = {
        "sample_index": row.get("sample_index") if row else None,
        "host_monotonic_ns": safe_int(row.get("host_monotonic_ns"), None) if row else None,
    }
    if parsed:
        summary.update(parsed)
    return summary


def rotation_x(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def rotation_y(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def rotation_z(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def rpy_matrix_deg(roll_deg, pitch_deg, yaw_deg):
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)
    return rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)


def quaternion_to_matrix(qx, qy, qz, qw):
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0:
        return None
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=float)


def orientation_matrix_from_ebimu(parsed):
    if not parsed:
        return None
    if parsed.get("orientation_format") == "quaternion":
        return quaternion_to_matrix(
            parsed.get("q_x"),
            parsed.get("q_y"),
            parsed.get("q_z"),
            parsed.get("q_w"),
        )
    if parsed.get("orientation_format") == "euler":
        return rpy_matrix_deg(
            parsed.get("roll_deg") or 0.0,
            parsed.get("pitch_deg") or 0.0,
            parsed.get("yaw_deg") or 0.0,
        )
    return None


def orientation_matrix_from_gps_course(course_deg, reference_camera_matrix=None):
    yaw = np.deg2rad(course_deg)
    if reference_camera_matrix is not None:
        old_forward = reference_camera_matrix[:, 2]
        horizontal = float(np.hypot(old_forward[0], old_forward[1]))
        forward = np.array([
            np.sin(yaw) * horizontal,
            np.cos(yaw) * horizontal,
            old_forward[2],
        ], dtype=float)
        forward_norm = np.linalg.norm(forward)
        if forward_norm > 1e-9:
            forward /= forward_norm

        old_down = reference_camera_matrix[:, 1]
        down = old_down - forward * float(old_down @ forward)
        down_norm = np.linalg.norm(down)
        if down_norm < 1e-9:
            down = np.array([0.0, 0.0, -1.0], dtype=float)
            down = down - forward * float(down @ forward)
            down_norm = np.linalg.norm(down)
        if down_norm > 1e-9:
            down /= down_norm

        right = np.cross(down, forward)
        right_norm = np.linalg.norm(right)
        if right_norm > 1e-9:
            right /= right_norm
        down = np.cross(forward, right)
        down_norm = np.linalg.norm(down)
        if down_norm > 1e-9:
            down /= down_norm
        return np.column_stack([right, down, forward])

    right = np.array([np.cos(yaw), -np.sin(yaw), 0.0], dtype=float)
    down = np.array([0.0, 0.0, -1.0], dtype=float)
    forward = np.array([np.sin(yaw), np.cos(yaw), 0.0], dtype=float)
    return np.column_stack([right, down, forward])


def camera_mount_matrix_deg(roll_deg, pitch_deg, yaw_deg):
    """Camera frame to vehicle frame. yaw +right, pitch +down, roll +clockwise."""
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)
    return rotation_y(yaw) @ rotation_x(-pitch) @ rotation_z(roll)


def enu_to_llh(lat_deg, lon_deg, alt_m, east_m, north_m, up_m):
    lat = np.deg2rad(lat_deg)
    a = 6378137.0
    e2 = 6.69437999014e-3
    sin_lat = np.sin(lat)
    denom = np.sqrt(1.0 - e2 * sin_lat * sin_lat)
    n_radius = a / denom
    m_radius = a * (1.0 - e2) / (denom ** 3)
    d_lat = north_m / (m_radius + alt_m)
    d_lon = east_m / ((n_radius + alt_m) * max(np.cos(lat), 1e-9))
    return {
        "latitude_deg": lat_deg + np.rad2deg(d_lat),
        "longitude_deg": lon_deg + np.rad2deg(d_lon),
        "altitude_m": alt_m + up_m,
    }


def llh_to_enu(origin_lat_deg, origin_lon_deg, origin_alt_m, lat_deg, lon_deg, alt_m):
    lat = np.deg2rad(origin_lat_deg)
    a = 6378137.0
    e2 = 6.69437999014e-3
    sin_lat = np.sin(lat)
    denom = np.sqrt(1.0 - e2 * sin_lat * sin_lat)
    n_radius = a / denom
    m_radius = a * (1.0 - e2) / (denom ** 3)
    d_lat = np.deg2rad(lat_deg - origin_lat_deg)
    d_lon = np.deg2rad(lon_deg - origin_lon_deg)
    return np.array([
        d_lon * (n_radius + origin_alt_m) * max(np.cos(lat), 1e-9),
        d_lat * (m_radius + origin_alt_m),
        alt_m - origin_alt_m,
    ], dtype=float)


def vector_dict(vector):
    return {
        "x": float(vector[0]),
        "y": float(vector[1]),
        "z": float(vector[2]),
    }


def enu_dict(vector):
    return {
        "east": float(vector[0]),
        "north": float(vector[1]),
        "up": float(vector[2]),
    }


def transformed_pixel_to_original_pixel(x, y, metadata):
    transform = metadata.get("image_transform") or {}
    width, height = metadata_image_size(metadata)
    original_x = float(x)
    original_y = float(y)

    # Inverse of raw -> flip_vertical -> rotate_180.
    if transform.get("rotate_180"):
        original_x = (width - 1) - original_x
        original_y = (height - 1) - original_y
    if transform.get("flip_vertical"):
        original_y = (height - 1) - original_y

    return original_x, original_y


def original_camera_vector_to_saved_frame(vector, metadata):
    transformed = np.array(vector, dtype=float)
    transform = metadata.get("image_transform") or {}

    # Forward transform of camera axes matching raw -> saved image operations.
    if transform.get("flip_vertical"):
        transformed[1] *= -1.0
    if transform.get("rotate_180"):
        transformed[0] *= -1.0
        transformed[1] *= -1.0

    return transformed


def unproject_saved_pixel(dataset, x, y, depth_m):
    camera_model = dataset.metadata.get("camera_model") or {}
    original_intrinsics = camera_model.get("intrinsics_original")
    saved_intrinsics = camera_model.get("intrinsics")
    distortion = camera_model.get("distortion_coefficients") or []
    stream_is_undistorted = safe_bool(camera_model.get("image_stream_undistorted"), False)

    # New refactored captures already save factory-undistorted RGB. Applying
    # undistortPoints again would bend the ray twice and shift WGS84 output.
    if original_intrinsics and not stream_is_undistorted:
        original_x, original_y = transformed_pixel_to_original_pixel(x, y, dataset.metadata)
        camera_matrix = np.array(original_intrinsics, dtype=np.float64)
        dist_coeffs = np.array(distortion, dtype=np.float64) if distortion else None
        point = np.array([[[original_x, original_y]]], dtype=np.float64)
        try:
            normalized = cv2.undistortPoints(point, camera_matrix, dist_coeffs)
            raw_vector = np.array([
                float(normalized[0, 0, 0]),
                float(normalized[0, 0, 1]),
                1.0,
            ])
            saved_vector = original_camera_vector_to_saved_frame(raw_vector, dataset.metadata)
            return saved_vector * depth_m, "distortion_corrected"
        except Exception:
            pass

    if not saved_intrinsics:
        return None, "missing_intrinsics"

    fx = float(saved_intrinsics[0][0])
    fy = float(saved_intrinsics[1][1])
    cx = float(saved_intrinsics[0][2])
    cy = float(saved_intrinsics[1][2])
    return np.array([
        (x - cx) * depth_m / fx,
        (y - cy) * depth_m / fy,
        depth_m,
    ], dtype=float), "pinhole_undistorted" if stream_is_undistorted else "pinhole"


def compute_world_coordinate(dataset, frame, x, y, depth_mm):
    return compute_world_coordinate_for_source(dataset, frame, x, y, depth_mm, "ebimu")


def compute_world_coordinate_for_source(dataset, frame, x, y, depth_mm, orientation_source):
    if depth_mm <= 0:
        return {"status": "unavailable", "reason": "invalid depth"}

    camera_model = dataset.metadata.get("camera_model") or {}
    intrinsics = camera_model.get("intrinsics")
    if not intrinsics:
        return {"status": "unavailable", "reason": "camera intrinsics missing in metadata"}

    row = frame["row"]
    frame_host_ns = safe_int(row.get("frame_host_monotonic_ns"), None)
    gps_row = frame.get("gps")
    gps_lat = safe_float(row.get("gps_latitude_deg"))
    gps_lon = safe_float(row.get("gps_longitude_deg"))
    gps_alt = safe_float(row.get("gps_altitude_m"))

    if gps_lat is None or gps_lon is None:
        gps_summary = summarize_gps(gps_row)
        gps_lat = gps_summary.get("latitude_deg") if gps_summary else None
        gps_lon = gps_summary.get("longitude_deg") if gps_summary else None
        gps_alt = gps_summary.get("altitude_m") if gps_summary else gps_alt

    if gps_alt is None:
        alt_row = dataset.nearest_gps_altitude(frame_host_ns)
        gps_alt = safe_float(alt_row.get("altitude_m")) if alt_row else 0.0

    if gps_lat is None or gps_lon is None:
        return {"status": "unavailable", "reason": "GPS latitude/longitude missing"}

    ebimu = parse_ebimu_row(frame.get("external_imu"))
    world_cfg = dataset.metadata.get("world_coordinates") or {}
    imu_from_camera = world_cfg.get("imu_from_camera_rpy_deg") or [0.0, 0.0, 0.0]
    camera_mount_rpy = normalized_rpy(world_cfg.get("camera_mount_rpy_deg") or world_cfg.get("vehicle_to_camera_rpy_deg"))
    gps_to_camera = world_cfg.get("gps_to_camera_enu_m") or [0.0, 0.0, 0.0]
    gps_to_camera_camera = world_cfg.get("gps_to_camera_camera_m")
    gps_from_camera_camera = world_cfg.get("gps_from_camera_camera_m")
    if gps_to_camera_camera is None and gps_from_camera_camera is not None:
        gps_to_camera_camera = [-float(value) for value in gps_from_camera_camera]
    if gps_to_camera_camera is None:
        gps_to_camera_camera = [0.0, 0.0, 0.0]
    external_imu_from_camera_camera = world_cfg.get("external_imu_from_camera_camera_m")
    camera_from_external_imu_camera = world_cfg.get("camera_from_external_imu_camera_m")
    if camera_from_external_imu_camera is None and external_imu_from_camera_camera is not None:
        camera_from_external_imu_camera = [-float(value) for value in external_imu_from_camera_camera]
    if external_imu_from_camera_camera is None:
        external_imu_from_camera_camera = [0.0, 0.0, 0.0]
    if camera_from_external_imu_camera is None:
        camera_from_external_imu_camera = [0.0, 0.0, 0.0]
    declination_deg = float(world_cfg.get("magnetic_declination_deg") or 0.0)

    r_imu_from_camera = rpy_matrix_deg(*[float(value) for value in imu_from_camera])
    r_declination = rotation_z(np.deg2rad(declination_deg))

    z_m = depth_mm / 1000.0
    try:
        point_camera, unprojection = unproject_saved_pixel(dataset, x, y, z_m)
    except (TypeError, ValueError, IndexError):
        point_camera, unprojection = None, "invalid_camera_model"
    if point_camera is None:
        return {"status": "unavailable", "reason": "invalid camera intrinsics"}

    assumptions = []
    gps_fix_quality = str(row.get("gps_fix_quality") or (gps_row or {}).get("fix_quality") or "")
    gps_position_valid = safe_bool(
        row.get("gps_position_valid", (gps_row or {}).get("position_valid")),
        default=True,
    )
    gps_differential_age_s = safe_float(
        row.get("gps_differential_age_s"),
        safe_float((gps_row or {}).get("differential_age_s")),
    )
    gps_frame_delta_ms = safe_float(row.get("gps_frame_delta_ms"))
    rtk_max_correction_age_s = float(world_cfg.get("rtk_max_correction_age_s") or 2.0)
    rtk_max_hdop = float(world_cfg.get("rtk_max_hdop") or 2.0)
    gps_hdop = safe_float(row.get("gps_hdop"), safe_float((gps_row or {}).get("hdop")))
    position_quality_reasons = []
    if gps_fix_quality != "4":
        label = (gps_row or {}).get("fix_quality_name") or "not RTK fixed"
        assumptions.append(f"GPS solution is {gps_fix_quality or 'unknown'} ({label}); centimeter accuracy is not guaranteed")
        position_quality_reasons.append("fix_quality_is_not_rtk_fixed")
    if gps_position_valid is False:
        assumptions.append("GPS receiver marked the position invalid")
        position_quality_reasons.append("position_invalid")
    if gps_differential_age_s is None:
        assumptions.append("GPS differential correction age is missing")
        position_quality_reasons.append("differential_age_missing")
    elif gps_differential_age_s > rtk_max_correction_age_s:
        assumptions.append(
            f"GPS differential correction age {gps_differential_age_s:.1f}s exceeds "
            f"the trusted {rtk_max_correction_age_s:.1f}s limit"
        )
        position_quality_reasons.append("differential_correction_too_old")
    if gps_frame_delta_ms is not None and abs(gps_frame_delta_ms) > 50.0:
        assumptions.append(f"GPS measurement is {gps_frame_delta_ms:.1f}ms from the RGB capture time")
        position_quality_reasons.append("gps_frame_delta_exceeds_50ms")
    if gps_hdop is None:
        assumptions.append("GPS HDOP is missing")
        position_quality_reasons.append("hdop_missing")
    elif gps_hdop > rtk_max_hdop:
        assumptions.append(f"GPS HDOP {gps_hdop:.2f} exceeds the trusted {rtk_max_hdop:.2f} limit")
        position_quality_reasons.append("hdop_too_high")
    position_quality = {
        "trusted": not position_quality_reasons,
        "reasons": position_quality_reasons,
        "fix_quality": gps_fix_quality,
        "position_valid": gps_position_valid,
        "differential_age_s": gps_differential_age_s,
        "maximum_differential_age_s": rtk_max_correction_age_s,
        "frame_delta_ms": gps_frame_delta_ms,
        "hdop": gps_hdop,
        "maximum_hdop": rtk_max_hdop,
    }
    orientation_details = {"source": orientation_source}
    if orientation_source in ("gps-course", "gps-course-level"):
        course = resolve_course_for_frame(dataset, frame)
        course_row = course["row"]
        course_source = course["source"]
        course_delta_ms = course["delta_ms"]
        gps_course_deg = course["course_deg"]
        gps_speed_m_s = course["speed_m_s"]
        gps_hdop = course["hdop"]
        if gps_course_deg is None:
            return {"status": "unavailable", "reason": "GPS course_deg missing"}
        reference_camera_matrix = None
        tilt_source = "level_assumption"
        if orientation_source == "gps-course":
            r_enu_from_imu = orientation_matrix_from_ebimu(ebimu)
            if r_enu_from_imu is not None:
                reference_camera_matrix = r_declination @ r_enu_from_imu @ r_imu_from_camera
                tilt_source = "ebimu"
        if orientation_source == "gps-course-level":
            r_enu_from_vehicle = orientation_matrix_from_gps_course(gps_course_deg, None)
            r_vehicle_from_camera = camera_mount_matrix_deg(*[float(value) for value in camera_mount_rpy])
            r_enu_from_camera = r_enu_from_vehicle @ r_vehicle_from_camera
        else:
            effective_course_deg = gps_course_deg + float(camera_mount_rpy[2])
            r_enu_from_camera = orientation_matrix_from_gps_course(effective_course_deg, reference_camera_matrix)
        orientation_details.update({
            "course_deg": gps_course_deg,
            "effective_course_deg": (
                gps_course_deg + float(camera_mount_rpy[2])
                if orientation_source == "gps-course"
                else None
            ),
            "speed_m_s": gps_speed_m_s,
            "hdop": gps_hdop,
            "course_source": course_source,
            "course_sample_index": course_row.get("sample_index") if course_row else None,
            "course_frame_delta_ms": course_delta_ms,
            "tilt_source": tilt_source,
            "camera_mount_rpy_deg": camera_mount_rpy,
        })
        if orientation_source == "gps-course-level":
            assumptions.append("GPS course supplies yaw; camera is assumed level")
        else:
            assumptions.append("GPS course supplies yaw; pitch/roll are kept from EBIMU when available")
        assumptions.append("GPS course is movement direction, not optical heading while stopped or reversing")
        if course_source == "nearest_moving_gps":
            assumptions.append("GPS course was taken from nearest moving sample because the frame sample was weak")
        if gps_speed_m_s is not None and gps_speed_m_s < 2.0:
            assumptions.append("GPS speed is below 2 m/s; course may be noisy")
        if gps_hdop is not None and gps_hdop >= 2.0:
            assumptions.append("GPS HDOP is >= 2; position/course confidence is lower")
    else:
        r_enu_from_imu = orientation_matrix_from_ebimu(ebimu)
        if r_enu_from_imu is None:
            return {"status": "unavailable", "reason": "EBIMU quaternion/euler sample missing"}
        r_enu_from_camera = r_declination @ r_enu_from_imu @ r_imu_from_camera
        orientation_details.update({
            "frame_delta_ms": safe_float(row.get("external_imu_frame_delta_ms")),
            "sample_index": row.get("external_imu_sample_index"),
            "orientation_format": ebimu.get("orientation_format") if ebimu else None,
            "timestamp_ms": ebimu.get("timestamp_ms") if ebimu else None,
        })
        assumptions.append("EBIMU quaternion is treated as IMU/body to local ENU rotation")

    forward_axis = r_enu_from_camera[:, 2]
    optical_heading_deg = float(np.rad2deg(np.arctan2(forward_axis[0], forward_axis[1])) % 360.0)
    optical_elevation_deg = float(np.rad2deg(np.arctan2(forward_axis[2], np.hypot(forward_axis[0], forward_axis[1]))))
    orientation_details.update({
        "optical_heading_deg": optical_heading_deg,
        "optical_elevation_deg": optical_elevation_deg,
        "camera_down_up_component": float(r_enu_from_camera[2, 1]),
    })
    if abs(optical_elevation_deg) > 45.0:
        assumptions.append("camera optical axis elevation exceeds 45 deg; check imu_from_camera_rpy_deg/extrinsics")

    gps_to_camera_enu_vec = np.array([float(value) for value in gps_to_camera], dtype=float)
    gps_to_camera_camera_vec = np.array([float(value) for value in gps_to_camera_camera], dtype=float)
    camera_position_enu = gps_to_camera_enu_vec + (r_enu_from_camera @ gps_to_camera_camera_vec)
    point_enu = camera_position_enu + (r_enu_from_camera @ point_camera)
    llh = enu_to_llh(gps_lat, gps_lon, gps_alt, point_enu[0], point_enu[1], point_enu[2])

    if imu_from_camera == [0.0, 0.0, 0.0]:
        assumptions.append("imu_from_camera_rpy_deg is default [0,0,0]")
    if gps_to_camera == [0.0, 0.0, 0.0]:
        assumptions.append("gps_to_camera_enu_m is default [0,0,0]")
    if gps_to_camera_camera == [0.0, 0.0, 0.0]:
        assumptions.append("gps_to_camera_camera_m is default [0,0,0]")

    return {
        "status": "ok",
        "orientation_source": orientation_source,
        "camera_point_m": vector_dict(point_camera),
        "camera_position_enu_m": enu_dict(camera_position_enu),
        "enu_offset_m": enu_dict(point_enu),
        "latitude_deg": float(llh["latitude_deg"]),
        "longitude_deg": float(llh["longitude_deg"]),
        "altitude_m": float(llh["altitude_m"]),
        "gps": {
            "latitude_deg": gps_lat,
            "longitude_deg": gps_lon,
            "altitude_m": gps_alt,
            "frame_delta_ms": gps_frame_delta_ms,
            "sample_index": row.get("gps_sample_index"),
            "position_quality": position_quality,
        },
        "orientation": orientation_details,
        "external_imu": orientation_details if not orientation_source.startswith("gps-course") else {
            "frame_delta_ms": safe_float(row.get("external_imu_frame_delta_ms")),
            "sample_index": row.get("external_imu_sample_index"),
            "orientation_format": ebimu.get("orientation_format") if ebimu else None,
            "timestamp_ms": ebimu.get("timestamp_ms") if ebimu else None,
        },
        "config": {
            "imu_from_camera_rpy_deg": imu_from_camera,
            "camera_mount_rpy_deg": camera_mount_rpy,
            "gps_to_camera_enu_m": gps_to_camera,
            "gps_to_camera_camera_m": gps_to_camera_camera,
            "gps_from_camera_camera_m": gps_from_camera_camera,
            "external_imu_from_camera_camera_m": external_imu_from_camera_camera,
            "camera_from_external_imu_camera_m": camera_from_external_imu_camera,
            "magnetic_declination_deg": declination_deg,
            "rtk_max_correction_age_s": rtk_max_correction_age_s,
            "rtk_max_hdop": rtk_max_hdop,
            "unprojection": unprojection,
        },
        "assumptions": assumptions,
    }


def robust_depth_value(depth, x, y, radius):
    exact = int(depth[y, x])
    if exact > 0 or radius <= 0:
        return {
            "depth_mm": exact,
            "exact_depth_mm": exact,
            "median_depth_mm": exact if exact > 0 else None,
            "sample_count": 1 if exact > 0 else 0,
            "radius": radius,
            "source": "exact" if exact > 0 else "invalid",
        }

    y0 = clamp(y - radius, 0, depth.shape[0] - 1)
    y1 = clamp(y + radius + 1, 1, depth.shape[0])
    x0 = clamp(x - radius, 0, depth.shape[1] - 1)
    x1 = clamp(x + radius + 1, 1, depth.shape[1])
    patch = depth[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return {
            "depth_mm": 0,
            "exact_depth_mm": exact,
            "median_depth_mm": None,
            "sample_count": 0,
            "radius": radius,
            "source": "invalid",
        }

    median = int(np.median(valid))
    return {
        "depth_mm": median,
        "exact_depth_mm": exact,
        "median_depth_mm": median,
        "sample_count": int(valid.size),
        "radius": radius,
        "source": "median",
    }


def make_depth_preview(depth, max_mm):
    max_mm = max(1, int(max_mm))
    clipped = np.clip(depth, 0, max_mm)
    scaled = (clipped * (255.0 / max_mm)).astype(np.uint8)
    color = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    ok, encoded = cv2.imencode(".png", color)
    if not ok:
        raise ValueError("Failed to encode depth preview PNG.")
    return encoded.tobytes()


def estimate_sequence_fps(dataset, index, window=30):
    rows = dataset.timestamps
    if len(rows) < 2:
        return None
    start = clamp(index - window, 0, len(rows) - 2)
    end = clamp(index + window, 1, len(rows) - 1)
    first = safe_int(rows[start].get("rgb_device_ts_ns"), None)
    last = safe_int(rows[end].get("rgb_device_ts_ns"), None)
    if first is None or last is None or last <= first:
        return None
    return (end - start) / ((last - first) / 1_000_000_000.0)


def normalized_rpy(values):
    values = list(values or [])
    while len(values) < 3:
        values.append(0.0)
    return [float(values[0]), float(values[1]), float(values[2])]


def resolve_course_for_frame(dataset, frame):
    row = frame["row"]
    frame_host_ns = safe_int(row.get("frame_host_monotonic_ns"), None)
    gps_row = frame.get("gps")
    course_row = gps_row
    course_source = "frame_gps"
    course_delta_ms = None
    course_deg = safe_float(course_row.get("course_deg")) if course_row else None
    speed_knots = safe_float(course_row.get("speed_knots")) if course_row else None
    speed_m_s = speed_knots * 0.514444 if speed_knots is not None else None
    hdop = safe_float(course_row.get("hdop")) if course_row else None
    weak = (
        course_deg is None
        or speed_m_s is None
        or speed_m_s < 2.0
        or (hdop is not None and hdop > 2.5)
    )
    if weak:
        replacement_row, replacement_delta_ms = dataset.nearest_valid_course(frame_host_ns)
        if replacement_row is not None:
            course_row = replacement_row
            course_source = "nearest_moving_gps"
            course_delta_ms = replacement_delta_ms
            course_deg = safe_float(course_row.get("course_deg"))
            speed_knots = safe_float(course_row.get("speed_knots"))
            speed_m_s = speed_knots * 0.514444 if speed_knots is not None else None
            hdop = safe_float(course_row.get("hdop"))
    return {
        "row": course_row,
        "course_deg": course_deg,
        "speed_m_s": speed_m_s,
        "hdop": hdop,
        "source": course_source,
        "delta_ms": course_delta_ms,
    }


def save_dataset_metadata(dataset):
    metadata_path = dataset.root / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(dataset.metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")


def set_camera_mount_rpy(dataset, roll, pitch, yaw, save=False, calibration=None):
    world_cfg = dataset.metadata.setdefault("world_coordinates", {})
    rpy = [float(roll), float(pitch), float(yaw)]
    world_cfg["camera_mount_rpy_deg"] = rpy
    world_cfg["camera_mount_frame"] = (
        "vehicle frame to saved camera frame convention; roll +clockwise, "
        "pitch +down, yaw +right from vehicle forward"
    )
    if calibration is not None:
        world_cfg["camera_mount_calibration"] = calibration
    world_cfg["updated_wall_time"] = datetime.now().isoformat(timespec="seconds")
    if save:
        save_dataset_metadata(dataset)
    return rpy


def solve_camera_mount_from_point(dataset, frame, x, y, depth_mm, actual_lat, actual_lon, actual_alt):
    if depth_mm <= 0:
        raise ValueError("Clicked point has invalid depth.")

    row = frame["row"]
    gps_lat = safe_float(row.get("gps_latitude_deg"))
    gps_lon = safe_float(row.get("gps_longitude_deg"))
    gps_alt = safe_float(row.get("gps_altitude_m"))
    if gps_lat is None or gps_lon is None:
        gps_summary = summarize_gps(frame.get("gps"))
        gps_lat = gps_summary.get("latitude_deg") if gps_summary else None
        gps_lon = gps_summary.get("longitude_deg") if gps_summary else None
        gps_alt = gps_summary.get("altitude_m") if gps_summary else gps_alt
    if gps_alt is None:
        gps_alt = 0.0
    if gps_lat is None or gps_lon is None:
        raise ValueError("Frame GPS latitude/longitude is missing.")

    course = resolve_course_for_frame(dataset, frame)
    if course["course_deg"] is None:
        raise ValueError("No reliable GPS course is available near this frame.")

    z_m = depth_mm / 1000.0
    point_camera, unprojection = unproject_saved_pixel(dataset, x, y, z_m)
    if point_camera is None:
        raise ValueError("Could not unproject clicked pixel.")

    world_cfg = dataset.metadata.get("world_coordinates") or {}
    current_rpy = normalized_rpy(world_cfg.get("camera_mount_rpy_deg") or world_cfg.get("vehicle_to_camera_rpy_deg"))
    gps_to_camera_enu = np.array(
        [float(value) for value in (world_cfg.get("gps_to_camera_enu_m") or [0.0, 0.0, 0.0])],
        dtype=float,
    )
    gps_to_camera_camera = world_cfg.get("gps_to_camera_camera_m")
    gps_from_camera_camera = world_cfg.get("gps_from_camera_camera_m")
    if gps_to_camera_camera is None and gps_from_camera_camera is not None:
        gps_to_camera_camera = [-float(value) for value in gps_from_camera_camera]
    if gps_to_camera_camera is None:
        gps_to_camera_camera = [0.0, 0.0, 0.0]
    gps_to_camera_camera = np.array([float(value) for value in gps_to_camera_camera], dtype=float)

    target_enu = llh_to_enu(gps_lat, gps_lon, gps_alt, actual_lat, actual_lon, actual_alt)
    target_from_origin = target_enu - gps_to_camera_enu
    camera_vector = point_camera + gps_to_camera_camera
    r_enu_from_vehicle = orientation_matrix_from_gps_course(course["course_deg"], None)

    roll = current_rpy[0]

    def predict(rpy):
        r_vehicle_from_camera = camera_mount_matrix_deg(*rpy)
        return gps_to_camera_enu + (r_enu_from_vehicle @ r_vehicle_from_camera @ camera_vector)

    def error_for(pitch, yaw):
        pred = predict([roll, pitch, yaw]) - gps_to_camera_enu
        return float(np.linalg.norm(pred - target_from_origin))

    best_pitch = current_rpy[1]
    best_yaw = current_rpy[2]
    best_error = error_for(best_pitch, best_yaw)
    for yaw_span, pitch_span, steps in [
        (90.0, 60.0, 41),
        (30.0, 20.0, 41),
        (10.0, 8.0, 41),
        (3.0, 3.0, 31),
        (1.0, 1.0, 31),
        (0.25, 0.25, 21),
    ]:
        yaw_values = np.linspace(best_yaw - yaw_span / 2.0, best_yaw + yaw_span / 2.0, steps)
        pitch_values = np.linspace(best_pitch - pitch_span / 2.0, best_pitch + pitch_span / 2.0, steps)
        for pitch in pitch_values:
            for yaw in yaw_values:
                err = error_for(float(pitch), float(yaw))
                if err < best_error:
                    best_error = err
                    best_pitch = float(pitch)
                    best_yaw = float(yaw)

    before = predict(current_rpy)
    after_rpy = [roll, best_pitch, best_yaw]
    after = predict(after_rpy)
    return {
        "current_rpy_deg": current_rpy,
        "proposed_rpy_deg": after_rpy,
        "error_before_m": float(np.linalg.norm(before - target_enu)),
        "error_after_m": float(np.linalg.norm(after - target_enu)),
        "target_enu_m": enu_dict(target_enu),
        "predicted_before_enu_m": enu_dict(before),
        "predicted_after_enu_m": enu_dict(after),
        "course": course,
        "depth_mm": int(depth_mm),
        "unprojection": unprojection,
    }


INDEX_HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DepthAI Dataset Debugger</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101414;
      --panel: #171d1c;
      --panel-2: #202827;
      --line: #31403d;
      --text: #edf4ef;
      --muted: #aab8b2;
      --accent: #58d68d;
      --accent-2: #67b7ff;
      --warn: #ffd166;
      --bad: #ff6b6b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, input, select {
      font: inherit;
      color: inherit;
    }
    .app {
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 100vh;
    }
    .topbar {
      display: grid;
      grid-template-columns: minmax(280px, 1fr) auto auto auto auto auto auto;
      gap: 8px;
      align-items: center;
      padding: 10px 12px;
      background: #111817;
      border-bottom: 1px solid var(--line);
    }
    .pathInput, .numberInput, select {
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      padding: 0 10px;
      outline: none;
    }
    .pathInput:focus, .numberInput:focus, select:focus {
      border-color: var(--accent);
    }
    .numberInput {
      width: 92px;
    }
    button {
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
      padding: 0 12px;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    button:disabled {
      opacity: 0.45;
      cursor: default;
    }
    .main {
      display: grid;
      grid-template-columns: 1fr 360px;
      min-height: 0;
    }
    .viewer {
      display: grid;
      grid-template-rows: 1fr auto;
      min-width: 0;
      min-height: 0;
    }
    .panes {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1px;
      min-height: 0;
      background: var(--line);
    }
    .pane {
      position: relative;
      min-width: 0;
      min-height: 0;
      background: #070909;
      overflow: hidden;
    }
    .paneTitle {
      position: absolute;
      top: 10px;
      left: 10px;
      z-index: 2;
      height: 28px;
      display: inline-flex;
      align-items: center;
      padding: 0 9px;
      border-radius: 5px;
      background: rgba(10, 14, 13, 0.8);
      border: 1px solid rgba(255,255,255,0.12);
      font-size: 13px;
      color: var(--muted);
    }
    .imageWrap {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
    }
    .debugImage {
      max-width: 100%;
      max-height: 100%;
      width: auto;
      height: auto;
      object-fit: contain;
      image-rendering: auto;
      user-select: none;
      -webkit-user-drag: none;
    }
    .crosshair {
      position: absolute;
      width: 13px;
      height: 13px;
      border: 2px solid var(--accent);
      border-radius: 50%;
      transform: translate(-50%, -50%);
      pointer-events: none;
      display: none;
      box-shadow: 0 0 0 2px rgba(0,0,0,0.65);
    }
    .crosshair::before,
    .crosshair::after {
      content: "";
      position: absolute;
      background: var(--accent);
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
    }
    .crosshair::before { width: 22px; height: 2px; }
    .crosshair::after { width: 2px; height: 22px; }
    .strip {
      display: grid;
      grid-template-columns: auto auto auto 1fr auto auto auto;
      align-items: center;
      gap: 8px;
      padding: 10px 12px;
      border-top: 1px solid var(--line);
      background: #111817;
    }
    .range {
      width: 100%;
      accent-color: var(--accent);
    }
    .side {
      border-left: 1px solid var(--line);
      background: var(--panel);
      padding: 12px;
      overflow: auto;
    }
    .section {
      padding: 12px 0;
      border-bottom: 1px solid var(--line);
    }
    .section:first-child { padding-top: 0; }
    .sectionTitle {
      margin: 0 0 9px 0;
      font-size: 13px;
      color: var(--muted);
      font-weight: 650;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    .sectionTitleRow {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 9px;
    }
    .sectionTitleRow .sectionTitle { margin: 0; }
    .statusBadge {
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 2px 6px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
    }
    .statusBadge.good { border-color: var(--accent); color: var(--accent); }
    .statusBadge.warn { border-color: var(--warn); color: var(--warn); }
    .statusBadge.bad { border-color: var(--bad); color: var(--bad); }
    .metricGrid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .metric {
      min-height: 58px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      background: #141b1a;
    }
    .metric.wide {
      grid-column: span 2;
      min-height: 96px;
    }
    .label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 5px;
    }
    .value {
      font-variant-numeric: tabular-nums;
      font-size: 18px;
      line-height: 1.2;
    }
    .coordValue {
      display: block;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-variant-numeric: tabular-nums;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.55;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      color: #d7e2dc;
    }
    .muted { color: var(--muted); }
    .accent { color: var(--accent); }
    .bad { color: var(--bad); }
    .warn { color: var(--warn); }
    .pair {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 3px 0;
      font-variant-numeric: tabular-nums;
    }
    .kbd {
      display: inline-grid;
      place-items: center;
      min-width: 22px;
      height: 22px;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #101514;
      color: var(--muted);
      font-size: 12px;
      padding: 0 6px;
    }
    .calibGrid {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
    }
    .calibField {
      display: grid;
      gap: 4px;
    }
    .calibField label {
      color: var(--muted);
      font-size: 12px;
    }
    .smallInput {
      width: 100%;
      height: 32px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #101514;
      padding: 0 8px;
      outline: none;
    }
    .smallInput:focus { border-color: var(--accent); }
    .buttonRow {
      display: flex;
      gap: 8px;
      margin-top: 8px;
      flex-wrap: wrap;
    }
    .buttonRow button {
      height: 32px;
      padding: 0 10px;
    }
    .error {
      color: var(--bad);
      min-height: 20px;
      font-size: 13px;
    }
    @media (max-width: 1100px) {
      .main { grid-template-columns: 1fr; }
      .side {
        border-left: 0;
        border-top: 1px solid var(--line);
      }
    }
    @media (max-width: 760px) {
      .topbar { grid-template-columns: 1fr auto; }
      .topbar > select, .topbar > .numberInput { display: none; }
      .panes { grid-template-columns: 1fr; }
      .strip { grid-template-columns: auto auto 1fr auto; }
      #frameText, #saveMode { display: none; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <input id="pathInput" class="pathInput" placeholder="dataset folder path, e.g. ../data/2026-06-17_11-27-13_raw" />
      <button id="latestBtn" title="Open latest dataset under this path">Latest</button>
      <button id="openBtn" title="Open dataset">Open</button>
      <select id="depthMaxSelect" title="Depth color range">
        <option value="3000">3m</option>
        <option value="5000">5m</option>
        <option value="8000" selected>8m</option>
        <option value="12000">12m</option>
      </select>
      <select id="sampleRadiusSelect" title="Depth sample radius">
        <option value="0">1 px</option>
        <option value="2">5 px</option>
        <option value="4" selected>9 px</option>
        <option value="7">15 px</option>
      </select>
      <select id="orientationSourceSelect" title="World coordinate orientation source">
        <option value="compare" selected>Compare</option>
        <option value="ebimu">EBIMU</option>
        <option value="gps-course">GPS Course + Tilt</option>
        <option value="gps-course-level">GPS Course Level</option>
      </select>
      <div class="error" id="errorText"></div>
    </header>
    <main class="main">
      <section class="viewer">
        <div class="panes">
          <div class="pane" id="rgbPane">
            <div class="paneTitle" id="rgbPaneTitle">RGB</div>
            <div class="imageWrap"><img id="rgbImage" class="debugImage" alt="RGB frame" /></div>
            <div id="rgbCrosshair" class="crosshair"></div>
          </div>
          <div class="pane" id="depthPane">
            <div class="paneTitle">Depth mm</div>
            <div class="imageWrap"><img id="depthImage" class="debugImage" alt="Depth frame" /></div>
            <div id="depthCrosshair" class="crosshair"></div>
          </div>
        </div>
        <div class="strip">
          <button id="prevBtn" title="Previous frame (D)">Back</button>
          <button id="nextBtn" title="Next frame (F)">Next</button>
          <button id="firstValidBtn" title="Jump to first frame with valid depth">First Valid</button>
          <input id="frameRange" class="range" type="range" min="0" max="0" value="0" />
          <input id="frameInput" class="numberInput" type="number" min="0" value="0" />
          <span id="frameText" class="mono muted">0 / 0</span>
          <span id="saveMode" class="mono muted"><span class="kbd">D</span> back <span class="kbd">F</span> next</span>
        </div>
      </section>
      <aside class="side">
        <div class="section">
          <h2 class="sectionTitle">Point</h2>
          <div class="metricGrid">
            <div class="metric"><span class="label">Hover XY</span><span id="hoverXY" class="value">-</span></div>
            <div class="metric"><span class="label">Distance</span><span id="hoverDepth" class="value accent">-</span></div>
            <div class="metric"><span class="label">Clicked XY</span><span id="clickXY" class="value">-</span></div>
            <div class="metric"><span class="label">Clicked Distance</span><span id="clickDepth" class="value accent">-</span></div>
            <div class="metric wide"><span class="label">Hover Absolute Coord</span><span id="hoverCoord" class="coordValue">-</span></div>
            <div class="metric wide"><span class="label">Clicked Absolute Coord</span><span id="clickCoord" class="coordValue">-</span></div>
          </div>
        </div>
        <div class="section">
          <h2 class="sectionTitle">Frame</h2>
          <div id="frameInfo" class="mono">No dataset loaded.</div>
        </div>
        <div class="section">
          <h2 class="sectionTitle">YOLO Segmentation</h2>
          <div id="yoloInfo" class="mono">Run tests/test_yolo_seg_shp.py to create results.</div>
        </div>
        <div class="section">
          <h2 class="sectionTitle">Dataset</h2>
          <div id="datasetInfo" class="mono">-</div>
        </div>
        <div class="section">
          <h2 class="sectionTitle">Sync</h2>
          <div id="syncInfo" class="mono">-</div>
        </div>
        <div class="section">
          <h2 class="sectionTitle">Alignment</h2>
          <div id="alignInfo" class="mono">-</div>
        </div>
        <div class="section">
          <h2 class="sectionTitle">IMU</h2>
          <div id="imuInfo" class="mono">-</div>
        </div>
        <div class="section">
          <div class="sectionTitleRow">
            <h2 class="sectionTitle">GPS / EBIMU</h2>
            <span id="rtkBadge" class="statusBadge">UNKNOWN</span>
          </div>
          <div id="externalInfo" class="mono">-</div>
        </div>
        <div class="section">
          <h2 class="sectionTitle">World Config</h2>
          <div id="worldInfo" class="mono">-</div>
        </div>
        <div class="section">
          <h2 class="sectionTitle">Mount Calibration</h2>
          <div class="calibGrid">
            <div class="calibField">
              <label for="mountRoll">Roll</label>
              <input id="mountRoll" class="smallInput" type="number" step="0.1" value="0" />
            </div>
            <div class="calibField">
              <label for="mountPitch">Pitch</label>
              <input id="mountPitch" class="smallInput" type="number" step="0.1" value="0" />
            </div>
            <div class="calibField">
              <label for="mountYaw">Yaw</label>
              <input id="mountYaw" class="smallInput" type="number" step="0.1" value="0" />
            </div>
          </div>
          <div class="buttonRow">
            <button id="applyMountBtn" title="Apply mount values for this UI session">Apply</button>
            <button id="saveMountBtn" title="Save mount values to metadata.json">Save</button>
          </div>
          <div class="calibGrid" style="margin-top:10px;">
            <div class="calibField">
              <label for="actualLat">Actual Lat</label>
              <input id="actualLat" class="smallInput" type="number" step="0.00000001" placeholder="37.x" />
            </div>
            <div class="calibField">
              <label for="actualLon">Actual Lon</label>
              <input id="actualLon" class="smallInput" type="number" step="0.00000001" placeholder="126.x" />
            </div>
            <div class="calibField">
              <label for="actualAlt">Actual Alt</label>
              <input id="actualAlt" class="smallInput" type="number" step="0.01" placeholder="m" />
            </div>
          </div>
          <div class="buttonRow">
            <button id="solveMountBtn" title="Use clicked point and actual coordinate to solve yaw/pitch">Calibrate & Save</button>
          </div>
          <div id="calibrationInfo" class="mono muted" style="margin-top:8px;">Click a point, enter its actual coordinate, then calibrate.</div>
        </div>
        <div class="section">
          <h2 class="sectionTitle">Files</h2>
          <div id="fileInfo" class="mono">-</div>
        </div>
      </aside>
    </main>
  </div>
  <script>
    const state = {
      datasetPath: "",
      frameCount: 0,
      index: 0,
      imageWidth: 1280,
      imageHeight: 720,
      depthMaxMm: 8000,
      sampleRadius: 4,
      orientationSource: "compare",
      hoverRequest: null,
      lastHover: null,
      lastClick: null,
      frame: null
    };

    const el = id => document.getElementById(id);
    const pathInput = el("pathInput");
    const errorText = el("errorText");
    const rgbImage = el("rgbImage");
    const depthImage = el("depthImage");
    const frameRange = el("frameRange");
    const frameInput = el("frameInput");
    const frameText = el("frameText");
    const rgbCrosshair = el("rgbCrosshair");
    const depthCrosshair = el("depthCrosshair");

    function qs(params) {
      return new URLSearchParams(params).toString();
    }

    async function api(path, params) {
      const res = await fetch(`${path}?${qs(params)}`);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || res.statusText);
      }
      return res.json();
    }

    function setError(message) {
      errorText.textContent = message || "";
    }

    function mediaUrl(kind, index = state.index) {
      let path = "/media/depth_preview";
      if (kind === "rgb") path = "/media/rgb";
      return `${path}?${qs({ path: state.datasetPath, index, max_mm: state.depthMaxMm, t: Date.now() })}`;
    }

    async function openDataset(useLatest=false) {
      try {
        setError("");
        let path = pathInput.value.trim();
        if (useLatest) {
          const latest = await api("/api/latest", { path });
          path = latest.path;
          pathInput.value = path;
        }
        const data = await api("/api/dataset", { path });
        state.datasetPath = data.path;
        state.frameCount = data.frame_count;
        setImageSize(data.image_size);
        state.index = 0;
        frameRange.max = Math.max(0, state.frameCount - 1);
        frameRange.value = 0;
        frameInput.max = Math.max(0, state.frameCount - 1);
        frameInput.value = 0;
        await loadFrame(0);
        if (state.frame && state.frame.valid_depth_pixels === 0) {
          try {
            const firstValid = await api("/api/first_valid_depth", { path: state.datasetPath });
            if (firstValid.index > 0) await loadFrame(firstValid.index);
          } catch (err) {
            setError(`Loaded, but no valid depth frame was found: ${err.message}`);
          }
        }
      } catch (err) {
        setError(err.message);
      }
    }

    async function loadFrame(index) {
      if (!state.datasetPath) return;
      index = Math.max(0, Math.min(state.frameCount - 1, Number(index) || 0));
      state.index = index;
      frameRange.value = index;
      frameInput.value = index;
      frameText.textContent = `${index + 1} / ${state.frameCount}`;
      try {
        const frame = await api("/api/frame", { path: state.datasetPath, index });
        state.frame = frame;
        setImageSize(frame.metadata?.image_size);
        rgbImage.src = mediaUrl("rgb", index);
        depthImage.src = mediaUrl("depth", index);
        renderFrame(frame);
        clearPoint(false);
      } catch (err) {
        setError(err.message);
      }
    }

    function mmText(value, source, sampleCount) {
      if (value === null || value === undefined || Number.isNaN(value)) return "-";
      if (value === 0) return source === "invalid" ? "invalid depth" : "0 mm";
      const suffix = source === "median" ? ` median/${sampleCount}px` : "";
      return `${value} mm (${(value / 1000).toFixed(3)} m)${suffix}`;
    }

    function formatNumber(value, digits=4) {
      if (value === null || value === undefined || Number.isNaN(value)) return "-";
      return Number(value).toFixed(digits);
    }

    function setImageSize(size) {
      if (!size || typeof size !== "object") return;
      const width = Number(size.width || size[0] || 0);
      const height = Number(size.height || size[1] || 0);
      if (width > 0 && height > 0) {
        state.imageWidth = width;
        state.imageHeight = height;
      }
    }

    function vectorText(values, digits=2) {
      if (!Array.isArray(values) || values.length === 0) return "-";
      return values.map(value => formatNumber(value, digits)).join(", ");
    }

    function gpsFixCountsText(counts) {
      if (!counts || typeof counts !== "object") return "-";
      const labels = {"0": "invalid", "1": "standalone", "2": "DGPS", "4": "fixed", "5": "float"};
      const entries = Object.entries(counts);
      if (!entries.length) return "-";
      return entries
        .sort(([a], [b]) => Number(a) - Number(b))
        .map(([quality, count]) => `${quality}:${labels[quality] || "other"}=${count}`)
        .join(", ");
    }

    function setMountInputs(values) {
      const rpy = Array.isArray(values) ? values : [0, 0, 0];
      el("mountRoll").value = Number(rpy[0] || 0).toFixed(2);
      el("mountPitch").value = Number(rpy[1] || 0).toFixed(2);
      el("mountYaw").value = Number(rpy[2] || 0).toFixed(2);
    }

    function getMountInputs() {
      return {
        roll: Number(el("mountRoll").value || 0),
        pitch: Number(el("mountPitch").value || 0),
        yaw: Number(el("mountYaw").value || 0)
      };
    }

    function singleCoordText(label, world) {
      if (!world || world.status !== "ok") {
        return `${label}\nunavailable: ${world?.reason || "-"}`;
      }
      const enu = world.enu_offset_m || {};
      const cam = world.camera_point_m || {};
      const cameraOrigin = world.camera_position_enu_m || {};
      const orientation = world.orientation || {};
      const assumptions = (world.assumptions || []).length ? `\n${world.assumptions.join("\n")}` : "";
      const courseLine = world.orientation_source?.startsWith("gps-course")
        ? `\ncourse ${formatNumber(orientation.course_deg, 2)} deg, speed ${formatNumber(orientation.speed_m_s, 2)} m/s (${orientation.course_source || "-"})`
        : "";
      const courseDeltaLine = orientation.course_frame_delta_ms !== null && orientation.course_frame_delta_ms !== undefined
        ? `\ncourse sample delta ${formatNumber(orientation.course_frame_delta_ms, 1)} ms`
        : "";
      const opticalLine =
        `\noptical heading/elev ${formatNumber(orientation.optical_heading_deg, 2)} deg, ${formatNumber(orientation.optical_elevation_deg, 2)} deg`;
      return (
        `${label}\n` +
        `lat ${formatNumber(world.latitude_deg, 8)}\n` +
        `lon ${formatNumber(world.longitude_deg, 8)}\n` +
        `alt ${formatNumber(world.altitude_m, 3)} m\n` +
        `ENU e/n/u ${formatNumber(enu.east, 3)}, ${formatNumber(enu.north, 3)}, ${formatNumber(enu.up, 3)} m\n` +
        `camera origin e/n/u ${formatNumber(cameraOrigin.east, 3)}, ${formatNumber(cameraOrigin.north, 3)}, ${formatNumber(cameraOrigin.up, 3)} m\n` +
        `cam x/y/z ${formatNumber(cam.x, 3)}, ${formatNumber(cam.y, 3)}, ${formatNumber(cam.z, 3)} m` +
        courseLine +
        courseDeltaLine +
        opticalLine +
        assumptions
      );
    }

    function coordDeltaText(a, b, label="Delta GPS-EBIMU") {
      if (!a || !b || a.status !== "ok" || b.status !== "ok") return "";
      const ea = a.enu_offset_m || {};
      const eb = b.enu_offset_m || {};
      const de = (eb.east ?? 0) - (ea.east ?? 0);
      const dn = (eb.north ?? 0) - (ea.north ?? 0);
      const du = (eb.up ?? 0) - (ea.up ?? 0);
      const norm = Math.sqrt(de * de + dn * dn + du * du);
      return `\n${label} e/n/u ${formatNumber(de, 3)}, ${formatNumber(dn, 3)}, ${formatNumber(du, 3)} m | ${formatNumber(norm, 3)} m`;
    }

    function coordText(world, worlds, source) {
      if (source === "compare" && worlds) {
        const ebimu = worlds.ebimu;
        const gpsCourse = worlds.gps_course;
        const gpsCourseLevel = worlds.gps_course_level;
        return (
          singleCoordText("EBIMU", ebimu) +
          "\n\n" +
          singleCoordText("GPS Course + Tilt", gpsCourse) +
          "\n\n" +
          singleCoordText("GPS Course Level", gpsCourseLevel) +
          coordDeltaText(ebimu, gpsCourse, "Delta Tilt-EBIMU") +
          coordDeltaText(gpsCourse, gpsCourseLevel, "Delta Level-Tilt")
        );
      }
      let label = "EBIMU";
      if (source === "gps-course") label = "GPS Course + Tilt";
      if (source === "gps-course-level") label = "GPS Course Level";
      return singleCoordText(label, world);
    }

    function renderFrame(frame) {
      const row = frame.row || {};
      const metadata = frame.metadata || {};
      el("frameInfo").textContent =
        `frame: ${frame.index}\n` +
        `stem: ${row.stem || "-"}\n` +
        `sequence rgb/depth: ${row.rgb_sequence || "-"} / ${row.depth_sequence || "-"}\n` +
        `valid depth pixels: ${frame.valid_depth_pixels ?? "-"}\n` +
        `estimated fps: ${formatNumber(frame.estimated_fps, 2)}`;

      const yolo = frame.yolo;
      const yoloDetections = yolo?.detections || [];
      const yoloPoints = yoloDetections.flatMap(item => item.points || []);
      const worldPoints = yoloPoints.filter(point => point.world_status === "ok").length;
      el("yoloInfo").textContent = yolo ? (
        `detections: ${yolo.detection_count ?? yoloDetections.length}\n` +
        `points: ${yoloPoints.length} (world: ${worldPoints})\n` +
        yoloDetections.map(item =>
          `#${item.detection_id} ${item.class_name} conf=${formatNumber(item.confidence, 3)}\n` +
          (item.points || []).map(point =>
            `  ${point.role}: (${point.pixel_x}, ${point.pixel_y}) ${point.depth_mm}mm ${point.coordinate_quality}`
          ).join("\n")
        ).join("\n")
      ) : "No YOLO result for this frame.";

      const transport = metadata.host_transport || {};
      const confidenceMeta = metadata.confidence_map || {};
      const usbSpeed = metadata.usb_speed || "legacy/unknown";
      const datasetInfo = el("datasetInfo");
      datasetInfo.textContent =
        `USB: ${usbSpeed}\n` +
        `transport rgb/depth/conf: ${transport.rgb || "legacy"} / ${transport.depth || "RAW16"} / ${transport.confidence || (confidenceMeta.saved ? "legacy" : "disabled")}\n` +
        `confidence lossy transport: ${confidenceMeta.transport_is_lossy ?? "-"}\n` +
        `requested / saved fps: ${metadata.requested_fps ?? "-"} / ${formatNumber(metadata.average_saved_fps, 2)}\n` +
        `frames rgb/conf: ${metadata.frame_count ?? state.frameCount} / ${metadata.confidence_frame_count ?? "-"}\n` +
        `GGA fixes: ${gpsFixCountsText(metadata.gps_gga_fix_quality_counts)}\n` +
        `RGB save: ${metadata.rgb_format || "-"}, depth: ${metadata.depth_format || "-"} (${metadata.depth_units || "-"})`;
      datasetInfo.classList.toggle("warn", ["HIGH", "FULL", "LOW"].includes(usbSpeed));

      el("syncInfo").textContent =
        `rgb-depth: ${row.rgb_depth_delta_ms || "-"} ms\n` +
        `rgb-imu: ${row.rgb_imu_delta_ms || "-"} ms\n` +
        `depth-imu: ${row.depth_imu_delta_ms || "-"} ms\n` +
        `camera queue lag: ${row.frame_queue_lag_ms || "-"} ms\n` +
        `gps measurement-frame: ${row.gps_frame_delta_ms || "-"} ms\n` +
        `gps receive latency: ${row.gps_receive_latency_ms || "-"} ms\n` +
        `external IMU-frame: ${row.external_imu_frame_delta_ms || "-"} ms\n` +
        `imu packets in group: ${row.imu_packets || "-"}`;

      const alignment = frame.metadata?.depth_alignment;
      const sockets = frame.metadata?.camera_sockets;
      const transform = frame.metadata?.image_transform;
      el("alignInfo").textContent = alignment ? (
        `enabled: ${alignment.enabled}\n` +
        `aligned to: ${alignment.aligned_to} (${alignment.aligned_to_socket})\n` +
        `rgb socket: ${sockets?.rgb || "-"}\n` +
        `stereo: ${sockets?.stereo_left || "-"} / ${sockets?.stereo_right || "-"}\n` +
        `method: ${alignment.method}\n` +
        `same pixel coords: ${alignment.depth_pixel_coordinates_match_rgb}\n` +
        `rotate 180: ${transform?.rotate_180 ?? "-"}\n` +
        `flip vertical: ${transform?.flip_vertical ?? "-"}`
      ) : "No alignment metadata in this dataset.";

      const imu = frame.imu_summary;
      el("imuInfo").textContent = imu ? (
        `packets: ${imu.packet_count}\n` +
        `accel m/s^2\n` +
        `  x ${formatNumber(imu.accel.x, 6)}\n` +
        `  y ${formatNumber(imu.accel.y, 6)}\n` +
        `  z ${formatNumber(imu.accel.z, 6)}\n` +
        `gyro rad/s\n` +
        `  x ${formatNumber(imu.gyro.x, 6)}\n` +
        `  y ${formatNumber(imu.gyro.y, 6)}\n` +
        `  z ${formatNumber(imu.gyro.z, 6)}`
      ) : "-";

      const gps = frame.gps_summary;
      const ext = frame.external_imu_summary;
      const mag = ext?.mag || {};
      const fixQuality = String(gps?.fix_quality ?? row.gps_fix_quality ?? "");
      const rtkStatus = gps?.rtk_status ?? row.gps_rtk_status ?? "unknown";
      const correctionAge = Number(gps?.differential_age_s ?? row.gps_differential_age_s);
      const maxCorrectionAge = Number(frame.metadata?.world_coordinates?.rtk_max_correction_age_s ?? 2.0);
      const hdop = Number(gps?.hdop ?? row.gps_hdop);
      const maxHdop = Number(frame.metadata?.world_coordinates?.rtk_max_hdop ?? 2.0);
      const rtkTrusted = fixQuality === "4"
        && Number.isFinite(correctionAge) && correctionAge <= maxCorrectionAge
        && Number.isFinite(hdop) && hdop <= maxHdop;
      const rtkBadge = el("rtkBadge");
      rtkBadge.textContent = fixQuality === "4"
        ? (rtkTrusted ? "RTK FIXED" : "RTK FIXED / STALE")
        : (fixQuality === "5" ? "RTK FLOAT" : String(rtkStatus).toUpperCase());
      rtkBadge.classList.toggle("good", rtkTrusted);
      rtkBadge.classList.toggle(
        "warn",
        (fixQuality === "4" && !rtkTrusted) || fixQuality === "5" || fixQuality === "2"
      );
      rtkBadge.classList.toggle("bad", !["2", "4", "5"].includes(fixQuality));
      el("externalInfo").textContent =
        `GPS sample: ${gps?.sample_index ?? row.gps_sample_index ?? "-"}\n` +
        `lat/lon: ${formatNumber(gps?.latitude_deg ?? row.gps_latitude_deg, 8)}, ${formatNumber(gps?.longitude_deg ?? row.gps_longitude_deg, 8)}\n` +
        `alt: ${formatNumber(gps?.altitude_m ?? row.gps_altitude_m, 3)} m\n` +
        `fix: ${gps?.fix_quality ?? row.gps_fix_quality ?? "-"} (${gps?.fix_quality_name ?? row.gps_fix_quality_name ?? "unknown"})\n` +
        `RTK: ${gps?.rtk_status ?? row.gps_rtk_status ?? "unknown"}, corrected=${gps?.rtk_corrected ?? row.gps_rtk_corrected ?? "-"}\n` +
        `position valid: ${gps?.position_valid ?? row.gps_position_valid ?? "-"}\n` +
        `sats/base/age: ${gps?.satellites ?? row.gps_satellites ?? "-"} / ${gps?.reference_station_id ?? row.gps_reference_station_id ?? "-"} / ${formatNumber(gps?.differential_age_s ?? row.gps_differential_age_s, 1)} s\n` +
        `course/speed: ${formatNumber(gps?.course_deg, 2)} deg / ${formatNumber((gps?.speed_knots ?? 0) * 0.514444, 2)} m/s\n` +
        `hdop: ${formatNumber(gps?.hdop, 2)}\n` +
        `EBIMU sample: ${ext?.sample_index ?? row.external_imu_sample_index ?? "-"}\n` +
        `orientation: ${ext?.orientation_format ?? "-"}\n` +
        `q x/y/z/w: ${formatNumber(ext?.q_x, 4)}, ${formatNumber(ext?.q_y, 4)}, ${formatNumber(ext?.q_z, 4)}, ${formatNumber(ext?.q_w, 4)}\n` +
        `mag x/y/z: ${formatNumber(mag.x, 2)}, ${formatNumber(mag.y, 2)}, ${formatNumber(mag.z, 2)}`;

      const world = frame.metadata?.world_coordinates;
      const cameraModel = frame.metadata?.camera_model;
      const sensorHeights = world?.sensor_heights_above_ground_m;
      el("worldInfo").textContent = world ? (
        `camera intrinsics: ${cameraModel?.intrinsics ? "ok" : "missing"}\n` +
        `height GPS/camera/IMU: ${formatNumber(sensorHeights?.gps_antenna, 2)} / ${formatNumber(sensorHeights?.camera, 2)} / ${formatNumber(sensorHeights?.external_imu, 2)} m\n` +
        `imu_from_camera r/p/y: ${vectorText(world.imu_from_camera_rpy_deg, 2)} deg\n` +
        `camera_mount r/p/y: ${vectorText(world.camera_mount_rpy_deg, 2)} deg\n` +
        `gps_to_camera e/n/u: ${vectorText(world.gps_to_camera_enu_m, 2)} m\n` +
        `gps_from_camera x/y/z: ${vectorText(world.gps_from_camera_camera_m, 2)} m\n` +
        `gps_to_camera x/y/z: ${vectorText(world.gps_to_camera_camera_m, 2)} m\n` +
        `external_imu_from_camera x/y/z: ${vectorText(world.external_imu_from_camera_camera_m, 2)} m\n` +
        `camera_from_external_imu x/y/z: ${vectorText(world.camera_from_external_imu_camera_m, 2)} m\n` +
        `mag declination: ${world.magnetic_declination_deg ?? 0} deg\n` +
        `orientation source: ${state.orientationSource}\n` +
        `world frame: ${world.local_frame || "-"}\n` +
        `lever frame: ${world.lever_arm_frame || world.camera_frame || "-"}`
      ) : "No world coordinate metadata in this dataset.";
      if (world) setMountInputs(world.camera_mount_rpy_deg || [0, 0, 0]);

      el("fileInfo").textContent =
        `RGB: ${row.rgb_file || "-"}\n` +
        `depth: ${row.depth_file || "-"}\n` +
        `confidence: ${row.confidence_file || "-"}`;
    }

    function eventToPixel(event, image, surface) {
      let rect = image.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) {
        rect = surface.getBoundingClientRect();
      }
      if (rect.width <= 0 || rect.height <= 0) return null;
      const width = state.imageWidth;
      const height = state.imageHeight;
      const rawX = (event.clientX - rect.left) * width / rect.width;
      const rawY = (event.clientY - rect.top) * height / rect.height;
      const x = Math.max(0, Math.min(width - 1, Math.floor(rawX)));
      const y = Math.max(0, Math.min(height - 1, Math.floor(rawY)));
      return { x, y, inside: rawX >= 0 && rawX < width && rawY >= 0 && rawY < height };
    }

    function setCrosshair(point) {
      for (const [image, crosshair] of [[rgbImage, rgbCrosshair], [depthImage, depthCrosshair]]) {
        const rect = image.getBoundingClientRect();
        const x = rect.left + point.x * rect.width / state.imageWidth;
        const y = rect.top + point.y * rect.height / state.imageHeight;
        const parentRect = crosshair.parentElement.getBoundingClientRect();
        crosshair.style.left = `${x - parentRect.left}px`;
        crosshair.style.top = `${y - parentRect.top}px`;
        crosshair.style.display = "block";
      }
    }

    async function updatePoint(point, mode) {
      if (!point || !state.datasetPath) return;
      setCrosshair(point);
      if (mode === "click") state.lastClick = point;
      const xyEl = mode === "click" ? el("clickXY") : el("hoverXY");
      const depthEl = mode === "click" ? el("clickDepth") : el("hoverDepth");
      const coordEl = mode === "click" ? el("clickCoord") : el("hoverCoord");
      xyEl.textContent = `${point.x}, ${point.y}`;
      try {
        const value = await api("/api/depth_value", {
          path: state.datasetPath,
          index: state.index,
          x: point.x,
          y: point.y,
          radius: state.sampleRadius,
          orientation_source: state.orientationSource
        });
        depthEl.textContent = mmText(value.depth_mm, value.source, value.sample_count);
        depthEl.className = value.depth_mm === 0 ? "value warn" : "value accent";
        coordEl.textContent = coordText(value.world, value.worlds, value.orientation_source);
        coordEl.className = value.world?.status === "ok" ? "coordValue accent" : "coordValue warn";
      } catch (err) {
        depthEl.textContent = "API error";
        coordEl.textContent = "API error";
        setError(err.message);
      }
    }

    async function refreshActivePoints() {
      if (state.lastHover) await updatePoint(state.lastHover, "hover");
      if (state.lastClick) await updatePoint(state.lastClick, "click");
    }

    async function updateMount(save=false) {
      if (!state.datasetPath) return;
      const mount = getMountInputs();
      const data = await api("/api/update_mount_rpy", {
        path: state.datasetPath,
        roll_deg: mount.roll,
        pitch_deg: mount.pitch,
        yaw_deg: mount.yaw,
        save: save ? "true" : "false"
      });
      if (state.frame) {
        state.frame.metadata = data.metadata;
        renderFrame(state.frame);
      }
      el("calibrationInfo").textContent =
        `${save ? "Saved" : "Applied"} camera_mount_rpy_deg: ` +
        data.camera_mount_rpy_deg.map(value => formatNumber(value, 3)).join(", ");
      await refreshActivePoints();
    }

    async function solveMountFromClick() {
      if (!state.datasetPath || !state.lastClick) {
        setError("Click a point before calibration.");
        return;
      }
      const lat = Number(el("actualLat").value);
      const lon = Number(el("actualLon").value);
      const alt = Number(el("actualAlt").value || 0);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
        setError("Actual latitude and longitude are required.");
        return;
      }
      const data = await api("/api/solve_mount_calibration", {
        path: state.datasetPath,
        index: state.index,
        x: state.lastClick.x,
        y: state.lastClick.y,
        radius: state.sampleRadius,
        latitude_deg: lat,
        longitude_deg: lon,
        altitude_m: alt,
        apply: "true",
        save: "true"
      });
      const proposed = data.solution.proposed_rpy_deg;
      setMountInputs(proposed);
      if (state.frame) {
        state.frame.metadata = data.metadata;
        renderFrame(state.frame);
      }
      el("calibrationInfo").textContent =
        `Saved r/p/y: ${proposed.map(value => formatNumber(value, 3)).join(", ")} deg\n` +
        `error: ${formatNumber(data.solution.error_before_m, 3)} m -> ${formatNumber(data.solution.error_after_m, 3)} m\n` +
        `course: ${formatNumber(data.solution.course.course_deg, 2)} deg (${data.solution.course.source})`;
      await refreshActivePoints();
    }

    function scheduleHover(point) {
      state.lastHover = point;
      if (state.hoverRequest) return;
      state.hoverRequest = setTimeout(() => {
        state.hoverRequest = null;
        updatePoint(state.lastHover, "hover");
      }, 35);
    }

    function clearPoint(clearClick=true) {
      el("hoverXY").textContent = "-";
      el("hoverDepth").textContent = "-";
      el("hoverCoord").textContent = "-";
      rgbCrosshair.style.display = "none";
      depthCrosshair.style.display = "none";
      if (clearClick) {
        el("clickXY").textContent = "-";
        el("clickDepth").textContent = "-";
        el("clickCoord").textContent = "-";
        state.lastClick = null;
      }
    }

    function bindPointerSurface(surface, image) {
      surface.addEventListener("pointermove", event => {
        const point = eventToPixel(event, image, surface);
        if (point) scheduleHover(point);
      });
      surface.addEventListener("pointerleave", () => clearPoint(false));
      surface.addEventListener("pointerdown", event => {
        const point = eventToPixel(event, image, surface);
        if (point) updatePoint(point, "click");
      });
    }

    el("openBtn").addEventListener("click", () => openDataset(false));
    el("latestBtn").addEventListener("click", () => openDataset(true));
    pathInput.addEventListener("keydown", event => {
      if (event.key === "Enter") openDataset(false);
    });
    el("prevBtn").addEventListener("click", () => loadFrame(state.index - 1));
    el("nextBtn").addEventListener("click", () => loadFrame(state.index + 1));
    el("firstValidBtn").addEventListener("click", async () => {
      if (!state.datasetPath) return;
      try {
        const data = await api("/api/first_valid_depth", { path: state.datasetPath });
        await loadFrame(data.index);
      } catch (err) {
        setError(err.message);
      }
    });
    frameRange.addEventListener("input", () => loadFrame(frameRange.value));
    frameInput.addEventListener("change", () => loadFrame(frameInput.value));
    el("depthMaxSelect").addEventListener("change", event => {
      state.depthMaxMm = Number(event.target.value);
      if (state.datasetPath) depthImage.src = mediaUrl("depth");
    });
    el("sampleRadiusSelect").addEventListener("change", event => {
      state.sampleRadius = Number(event.target.value);
    });
    el("orientationSourceSelect").addEventListener("change", event => {
      state.orientationSource = event.target.value;
      if (state.frame) renderFrame(state.frame);
      if (state.lastHover) updatePoint(state.lastHover, "hover");
    });
    el("applyMountBtn").addEventListener("click", async () => {
      try {
        setError("");
        await updateMount(false);
      } catch (err) {
        setError(err.message);
      }
    });
    el("saveMountBtn").addEventListener("click", async () => {
      try {
        setError("");
        await updateMount(true);
      } catch (err) {
        setError(err.message);
      }
    });
    el("solveMountBtn").addEventListener("click", async () => {
      try {
        setError("");
        await solveMountFromClick();
      } catch (err) {
        setError(err.message);
      }
    });
    document.addEventListener("keydown", event => {
      const tag = event.target.tagName.toLowerCase();
      if (tag === "input" || tag === "select") return;
      if (event.key === "d" || event.key === "D" || event.key === "ArrowLeft") {
        loadFrame(state.index - 1);
      } else if (event.key === "f" || event.key === "F" || event.key === "ArrowRight") {
        loadFrame(state.index + 1);
      } else if (event.key === "Home") {
        loadFrame(0);
      } else if (event.key === "End") {
        loadFrame(state.frameCount - 1);
      }
    });
    bindPointerSurface(el("rgbPane"), rgbImage);
    bindPointerSurface(el("depthPane"), depthImage);

    const initialParams = new URLSearchParams(location.search);
    pathInput.value = initialParams.get("path") || "../data";
    window.addEventListener("load", () => {
      if (initialParams.get("path")) {
        openDataset(false);
      } else {
        openDataset(true);
      }
    });
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, data, content_type, status=200):
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        except BrokenPipeError:
            pass

    def send_json(self, payload, status=200):
        self.send_bytes(json_bytes(payload), "application/json; charset=utf-8", status)

    def send_error_text(self, message, status=400):
        self.send_bytes(str(message).encode("utf-8"), "text/plain; charset=utf-8", status)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = {key: values[-1] for key, values in urllib.parse.parse_qs(parsed.query).items()}
        try:
            if parsed.path == "/":
                self.send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/latest":
                root = latest_dataset_under(params.get("path", "."))
                self.send_json({"path": str(root)})
            elif parsed.path == "/api/dataset":
                dataset = get_dataset(params.get("path"))
                self.send_json({
                    "path": str(dataset.root),
                    "frame_count": dataset.frame_count,
                    "image_size": {"width": dataset.image_width, "height": dataset.image_height},
                    "metadata": dataset.metadata,
                })
            elif parsed.path == "/api/frame":
                dataset = get_dataset(params.get("path"))
                index = safe_int(params.get("index"), 0)
                frame = dataset.frame(index)
                depth = get_depth_frame(dataset, frame["index"])
                self.send_json({
                    "index": frame["index"],
                    "row": frame["row"],
                    "imu": frame["imu"],
                    "imu_summary": summarize_imu(frame["imu"]),
                    "gps_summary": summarize_gps(frame.get("gps")),
                    "external_imu_summary": summarize_external_imu(frame.get("external_imu")),
                    "estimated_fps": estimate_sequence_fps(dataset, frame["index"]),
                    "valid_depth_pixels": int((depth > 0).sum()),
                    "yolo": dataset.yolo_result(frame["index"]),
                    "metadata": dataset.metadata,
                })
            elif parsed.path == "/api/first_valid_depth":
                dataset = get_dataset(params.get("path"))
                min_valid_pixels = clamp(
                    safe_int(params.get("min_valid_pixels"), 1000),
                    1,
                    dataset.image_width * dataset.image_height,
                )
                found = None
                for index in range(dataset.frame_count):
                    depth = get_depth_frame(dataset, index)
                    valid_pixels = int((depth > 0).sum())
                    if valid_pixels >= min_valid_pixels:
                        found = {"index": index, "valid_depth_pixels": valid_pixels}
                        break
                if found is None:
                    raise ValueError("No frame with enough valid depth pixels was found.")
                self.send_json(found)
            elif parsed.path == "/api/depth_value":
                dataset = get_dataset(params.get("path"))
                index = safe_int(params.get("index"), 0)
                x = clamp(safe_int(params.get("x"), 0), 0, dataset.image_width - 1)
                y = clamp(safe_int(params.get("y"), 0), 0, dataset.image_height - 1)
                radius = clamp(safe_int(params.get("radius"), 4), 0, 20)
                orientation_source = params.get("orientation_source", "compare")
                if orientation_source not in ("compare", "ebimu", "gps-course", "gps-course-level"):
                    orientation_source = "compare"
                frame = dataset.frame(index)
                depth = get_depth_frame(dataset, index)
                depth_value = robust_depth_value(depth, x, y, radius)
                worlds = {}
                if orientation_source in ("compare", "ebimu"):
                    worlds["ebimu"] = compute_world_coordinate_for_source(
                        dataset, frame, x, y, depth_value["depth_mm"], "ebimu"
                    )
                if orientation_source in ("compare", "gps-course"):
                    worlds["gps_course"] = compute_world_coordinate_for_source(
                        dataset, frame, x, y, depth_value["depth_mm"], "gps-course"
                    )
                if orientation_source in ("compare", "gps-course-level"):
                    worlds["gps_course_level"] = compute_world_coordinate_for_source(
                        dataset, frame, x, y, depth_value["depth_mm"], "gps-course-level"
                    )

                if orientation_source == "gps-course":
                    world = worlds.get("gps_course")
                elif orientation_source == "gps-course-level":
                    world = worlds.get("gps_course_level")
                elif orientation_source == "ebimu":
                    world = worlds.get("ebimu")
                else:
                    gps_world = worlds.get("gps_course_level") or worlds.get("gps_course")
                    ebimu_world = worlds.get("ebimu")
                    world = gps_world if gps_world and gps_world.get("status") == "ok" else ebimu_world

                self.send_json({
                    "x": x,
                    "y": y,
                    **depth_value,
                    "orientation_source": orientation_source,
                    "world": world,
                    "worlds": worlds,
                })
            elif parsed.path == "/api/update_mount_rpy":
                dataset = get_dataset(params.get("path"))
                roll = safe_float(params.get("roll_deg"))
                pitch = safe_float(params.get("pitch_deg"))
                yaw = safe_float(params.get("yaw_deg"))
                if roll is None or pitch is None or yaw is None:
                    raise ValueError("roll_deg, pitch_deg, and yaw_deg are required.")
                save = params.get("save", "false").lower() in ("1", "true", "yes")
                rpy = set_camera_mount_rpy(dataset, roll, pitch, yaw, save=save)
                self.send_json({
                    "status": "ok",
                    "saved": save,
                    "camera_mount_rpy_deg": rpy,
                    "metadata": dataset.metadata,
                })
            elif parsed.path == "/api/solve_mount_calibration":
                dataset = get_dataset(params.get("path"))
                index = safe_int(params.get("index"), 0)
                x = clamp(safe_int(params.get("x"), 0), 0, dataset.image_width - 1)
                y = clamp(safe_int(params.get("y"), 0), 0, dataset.image_height - 1)
                radius = clamp(safe_int(params.get("radius"), 4), 0, 20)
                actual_lat = safe_float(params.get("latitude_deg"))
                actual_lon = safe_float(params.get("longitude_deg"))
                actual_alt = safe_float(params.get("altitude_m"), 0.0)
                if actual_lat is None or actual_lon is None:
                    raise ValueError("latitude_deg and longitude_deg are required.")
                frame = dataset.frame(index)
                depth = get_depth_frame(dataset, index)
                depth_value = robust_depth_value(depth, x, y, radius)
                solution = solve_camera_mount_from_point(
                    dataset,
                    frame,
                    x,
                    y,
                    depth_value["depth_mm"],
                    actual_lat,
                    actual_lon,
                    actual_alt,
                )
                save = params.get("save", "false").lower() in ("1", "true", "yes")
                apply_solution = params.get("apply", "true").lower() in ("1", "true", "yes")
                if apply_solution:
                    calibration = {
                        "source": "single_clicked_point",
                        "updated_wall_time": datetime.now().isoformat(timespec="seconds"),
                        "frame_index": index,
                        "x": x,
                        "y": y,
                        "actual_latitude_deg": actual_lat,
                        "actual_longitude_deg": actual_lon,
                        "actual_altitude_m": actual_alt,
                        "error_before_m": solution["error_before_m"],
                        "error_after_m": solution["error_after_m"],
                        "note": "Single-point calibration updates yaw/pitch and keeps roll unchanged.",
                    }
                    set_camera_mount_rpy(dataset, *solution["proposed_rpy_deg"], save=save, calibration=calibration)
                self.send_json({
                    "status": "ok",
                    "saved": save and apply_solution,
                    "applied": apply_solution,
                    **depth_value,
                    "solution": solution,
                    "metadata": dataset.metadata,
                })
            elif parsed.path == "/media/rgb":
                dataset = get_dataset(params.get("path"))
                frame = dataset.frame(safe_int(params.get("index"), 0))
                data = frame["rgb_path"].read_bytes()
                content_type = mimetypes.guess_type(frame["rgb_path"].name)[0] or "image/png"
                self.send_bytes(data, content_type)
            elif parsed.path == "/media/depth_preview":
                dataset = get_dataset(params.get("path"))
                index = safe_int(params.get("index"), 0)
                max_mm = safe_int(params.get("max_mm"), 8000)
                depth = get_depth_frame(dataset, index)
                self.send_bytes(make_depth_preview(depth, max_mm), "image/png")
            else:
                self.send_error_text("Not found", 404)
        except BrokenPipeError:
            pass
        except Exception as exc:
            self.send_error_text(exc, 400)

    def log_message(self, fmt, *args):
        return


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch the DepthAI dataset debug UI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=HOST, help="Bind address. Use 127.0.0.1 for local-only access.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="HTTP server TCP port")
    return parse_args_with_yaml(parser)


def discover_ipv4_addresses():
    addresses = []
    ignored_prefixes = ("docker", "br-", "veth")

    try:
        output = subprocess.check_output(
            ["ip", "-4", "-o", "addr", "show", "scope", "global"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in output.splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            interface = parts[1]
            if interface.startswith(ignored_prefixes):
                continue
            if "inet" not in parts:
                continue
            address = parts[parts.index("inet") + 1].split("/", 1)[0]
            if address and address not in addresses:
                addresses.append(address)
    except Exception:
        pass

    if not addresses:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                address = sock.getsockname()[0]
                if address and not address.startswith("127.") and address not in addresses:
                    addresses.append(address)
        except Exception:
            pass

    return addresses


def print_server_urls(host, port):
    print(f"DepthAI dataset debug UI bind: http://{host}:{port}")
    if host in ("0.0.0.0", ""):
        print(f"Local URL: http://127.0.0.1:{port}")
        addresses = discover_ipv4_addresses()
        if addresses:
            print("External/LAN URL candidates:")
            for address in addresses:
                print(f"  http://{address}:{port}")
        else:
            print("External/LAN URL candidates: unavailable; check `ip -4 addr`.")
        print("External access requires the client to be on a reachable network and the firewall to allow this port.")
    else:
        print(f"URL: http://{host}:{port}")


def main():
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print_server_urls(args.host, args.port)
    print("Open a dataset folder containing rgb/, depth_mm/, timestamps.csv, and imu.csv.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
