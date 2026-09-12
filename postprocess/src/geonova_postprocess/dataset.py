#!/usr/bin/env python3

import bisect
import csv
import json
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path


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
