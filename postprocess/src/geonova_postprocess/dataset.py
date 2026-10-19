#!/usr/bin/env python3

import bisect
import copy
import csv
import json
import os
import tempfile
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path


import cv2
import numpy as np

from geonova_common.config_cli import load_yaml


ORIENTATION_DEFAULTS = load_yaml(Path(__file__).resolve().parents[2] / "configs" / "yolo.yaml")
IMU_SETTINGS = (
    "imu_max_delta_ms", "imu_to_camera_rpy_deg", "imu_filter_time_constant_s",
    "imu_accel_rejection_m_s2", "imu_bias_window_s",
    "imu_bias_accel_tolerance_m_s2", "imu_bias_gyro_threshold_rad_s",
    "imu_to_camera_validated", "imu_max_gap_ms", "imu_accel_innovation_deg",
    "imu_still_duration_s", "imu_still_accel_tolerance_m_s2",
)
SINGLE_POINT_MOUNT_NOTE = "Single-point calibration solves yaw only; current roll and pitch are unchanged."


def orientation_settings(dataset):
    settings = dict(ORIENTATION_DEFAULTS)
    settings.update(dataset.metadata.get("world_coordinates") or {})
    settings.update(getattr(dataset, "orientation_config", {}))
    return settings


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
    except (TypeError, ValueError, OverflowError):
        return default


def safe_float(value, default=None):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
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


def _merge_metadata(base, overrides):
    """Merge JSON objects recursively without mutating capture metadata."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_metadata(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _metadata_overrides(base, current):
    overrides = {}
    for key, value in current.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            nested = _metadata_overrides(base[key], value)
            if nested:
                overrides[key] = nested
        elif key not in base or value != base[key]:
            overrides[key] = copy.deepcopy(value)
    return overrides


def _require_complete_sync(root):
    if (root / "sync.incomplete").exists():
        raise ValueError("Dataset sync is incomplete (sync.incomplete); re-run sync.")


class Dataset:
    def __init__(self, root):
        self.root = root
        _require_complete_sync(root)
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
        self.gps_by_sample_index = {
            safe_int(row.get("sample_index"), -1): row
            for row in self.gps_rows
        }
        self._capture_metadata = {}
        if metadata_path.exists():
            with metadata_path.open(encoding="utf-8") as file:
                self._capture_metadata = json.load(file)
        overrides = {}
        overrides_path = root / "metadata.overrides.json"
        if overrides_path.exists():
            with overrides_path.open(encoding="utf-8") as file:
                overrides = json.load(file)
        if not isinstance(self._capture_metadata, dict) or not isinstance(overrides, dict):
            raise ValueError("Metadata must be a JSON object.")
        self.metadata = _merge_metadata(self._capture_metadata, overrides)
        self.orientation_config = {}
        self._camera_imu_cache = {}
        self._camera_imu_lock = threading.Lock()
        self._index_gps()
        self.image_width, self.image_height = metadata_image_size(self.metadata)

        if not self.timestamps:
            raise ValueError(f"No frames listed in {timestamps_path}")

        self.yolo_by_frame = {}
        self.yolo_results_mtime_ns = None
        self.reload_yolo_results()
        _require_complete_sync(root)

    def _index_gps(self):
        self.valid_course_rows = []
        max_hdop = orientation_settings(self)["trajectory_max_hdop"]
        self._course_max_hdop = max_hdop
        for row in self.gps_rows:
            host_ns = safe_int(row.get("host_monotonic_ns"), None)
            course_deg = safe_float(row.get("course_deg"))
            speed_knots = safe_float(row.get("speed_knots"))
            hdop = safe_float(row.get("hdop"))
            if host_ns is None or course_deg is None or speed_knots is None:
                continue
            if speed_knots * 0.514444 < 2.0:
                continue
            if hdop is not None and hdop > max_hdop:
                continue
            self.valid_course_rows.append((host_ns, row))
        self.valid_course_rows.sort(key=lambda item: item[0])
        self.valid_course_times = [item[0] for item in self.valid_course_rows]
        # Keep the first CSV row at each timestamp and the original-order tie break.
        altitude_by_time = {}
        for index, row in enumerate(self.gps_rows):
            if row.get("altitude_m") not in ("", None):
                timestamp = safe_int(row.get("host_monotonic_ns"), 0)
                altitude_by_time.setdefault(timestamp, (timestamp, index, row))
        self._altitude_rows = sorted(altitude_by_time.values(), key=lambda item: item[0])
        self._altitude_times = [item[0] for item in self._altitude_rows]

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
        }

    def nearest_valid_course(self, host_monotonic_ns, max_delta_ms=5000.0):
        if self._course_max_hdop != orientation_settings(self)["trajectory_max_hdop"]:
            self._index_gps()
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
        if not self._altitude_rows or host_monotonic_ns is None:
            return None
        pos = bisect.bisect_left(self._altitude_times, host_monotonic_ns)
        candidates = self._altitude_rows[max(0, pos - 1):pos + 1]
        return min(candidates, key=lambda item: (abs(item[0] - host_monotonic_ns), item[1]))[2]


def get_dataset(path_text):
    root = resolve_path(path_text)
    _require_complete_sync(root)
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
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=float)


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


def validate_imu_settings(settings):
    for key in IMU_SETTINGS:
        value = settings[key]
        if key == "imu_to_camera_rpy_deg":
            if np.asarray(value).shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError("imu_to_camera_rpy_deg must contain three finite angles.")
        elif key == "imu_to_camera_validated":
            # The shared CLI exposes numeric IMU options; accept only 0/1 there.
            if value not in (False, True):
                raise ValueError("imu_to_camera_validated must be false or true (0 or 1).")
        elif not np.isfinite(value) or value < 0 or (
            key in ("imu_filter_time_constant_s", "imu_accel_rejection_m_s2",
                    "imu_max_gap_ms", "imu_accel_innovation_deg", "imu_still_duration_s",
                    "imu_still_accel_tolerance_m_s2") and value == 0
        ):
            raise ValueError(f"Invalid {key}.")
    if settings["imu_accel_innovation_deg"] >= 90:
        raise ValueError("imu_accel_innovation_deg must be below 90.")
    for key in ("trusted_max_hdop", "trajectory_max_hdop"):
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"{key} must be positive.")


def camera_imu_extrinsics(dataset):
    """Return only a proper IMU-to-original-camera rotation; translations are cm."""
    value = ((dataset.metadata.get("imu") or {}).get("imu_to_camera") or {}).get("matrix")
    try:
        matrix = np.asarray(value, dtype=float)
        rotation = matrix[:3, :3]
        if (
            matrix.shape in ((3, 3), (4, 4))
            and np.all(np.isfinite(rotation))
            and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3, rtol=0)
            and abs(np.linalg.det(rotation) - 1.0) <= 1e-3
        ):
            return rotation, "device_calibration"
    except (TypeError, ValueError, IndexError):
        pass
    settings = orientation_settings(dataset)
    if settings["imu_to_camera_validated"]:
        return rpy_matrix_deg(*settings["imu_to_camera_rpy_deg"]), "config_validated"
    return None, "unavailable"


def _imu_vector(row, prefix, suffix):
    values = [safe_float(row.get(f"{prefix}_{axis}_{suffix}")) for axis in "xyz"]
    if any(value is None for value in values) or not np.all(np.isfinite(values)):
        return None
    return np.array(values, dtype=float)


def _initial_gyro_bias(accels, gyros, settings):
    duration = settings["imu_bias_window_s"] * 1e9
    zero = np.zeros(3)
    if not accels or not gyros or duration <= 0:
        return zero
    start = max(accels[0][0], gyros[0][0])
    end = start + duration
    if min(accels[-1][0], gyros[-1][0]) < end:
        return zero
    a = np.array([v for t, v in accels if start <= t <= end])
    w = np.array([v for t, v in gyros if start <= t <= end])
    if len(a) < 2 or len(w) < 2:
        return zero
    max_gap = settings["imu_max_gap_ms"] * 1e6
    for samples in (accels, gyros):
        times = [t for t, _ in samples if start <= t <= end]
        if times[0] - start > max_gap or end - times[-1] > max_gap or max(np.diff(times)) > max_gap:
            return zero
    tolerance = settings["imu_bias_accel_tolerance_m_s2"]
    if (
        np.max(np.abs(np.linalg.norm(a, axis=1) - 9.80665)) > tolerance
        or np.max(np.linalg.norm(a - np.mean(a, axis=0), axis=1)) > tolerance
        or np.max(np.linalg.norm(w, axis=1)) > settings["imu_bias_gyro_threshold_rad_s"]
    ):
        return zero
    return np.mean(w, axis=0)


def _rotate_gravity(gravity, omega, dt):
    angle = float(np.linalg.norm(omega)) * dt
    if angle <= 1e-12:
        return gravity
    axis = -omega / np.linalg.norm(omega)
    return (gravity * np.cos(angle) + np.cross(axis, gravity) * np.sin(angle)
            + axis * float(axis @ gravity) * (1 - np.cos(angle)))


def _build_camera_imu_series(dataset, settings):
    path = dataset.root / "imu_events.csv"
    if not path.exists():
        return [], [], "unavailable", []
    accels, gyros, vectors = {}, {}, {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            for prefix, suffix, target in (("accel", "m_s2", accels), ("gyro", "rad_s", gyros)):
                timestamp = safe_int(row.get(prefix + "_device_ts_ns"), None)
                if timestamp is None:
                    timestamp = safe_int(row.get("message_device_ts_ns"), None)
                value = _imu_vector(row, prefix, suffix)
                if timestamp is not None and value is not None:
                    target[timestamp] = value
            timestamp = safe_int(row.get("rv_device_ts_ns"), None)
            q = [safe_float(row.get("rv_" + key)) for key in ("i", "j", "k", "real")]
            if timestamp is not None and all(value is not None for value in q):
                rotation = quaternion_to_matrix(*q)
                if rotation is not None:
                    # BNO game quaternion: IMU -> world, with world +Z up.
                    # R.T * down removes its arbitrary/drifting yaw completely.
                    vectors[timestamp] = -rotation[2, :]
    if vectors:
        times = sorted(vectors)
        # Rotation vectors are complete device-filter estimates, not integrations
        # of our held gyro rate. Their timestamp is their freshness support.
        support = [(True, t, t, 0) for t in times]
        return times, np.array([vectors[t] for t in times]), "game_rotation_vector", support
    if not accels or not gyros:
        return [], [], "unavailable", []
    bias = _initial_gyro_bias(sorted(accels.items()), sorted(gyros.items()), settings)
    # Combine simultaneous reports before evaluating validity. Each estimate
    # retains both actual sensor timestamps and the filter's outage generation.
    max_gap = settings["imu_max_gap_ms"] * 1e6
    gravity, gyro_gravity, omega, previous_time = None, None, None, None
    last_accel_time, last_gyro_time, still_since = None, None, None
    still_down = None
    converged, correction_ok, generation = False, False, 0
    times, directions, support = [], [], []
    for timestamp in sorted(set(accels) | set(gyros)):
        outage = previous_time is not None and (
            timestamp - previous_time > max_gap
            or (last_accel_time is not None and timestamp - last_accel_time > max_gap)
            or (last_gyro_time is not None and timestamp - last_gyro_time > max_gap)
        )
        if outage:
            gravity, gyro_gravity, still_since, still_down = None, None, None, None
            converged, correction_ok = False, False
            generation += 1
        elif gravity is not None and omega is not None and previous_time is not None:
            dt = (timestamp - previous_time) / 1e9
            gravity = _rotate_gravity(gravity, omega, dt)
            gyro_gravity = _rotate_gravity(gyro_gravity, omega, dt)
            if still_down is not None:
                still_down = _rotate_gravity(still_down, omega, dt)

        if timestamp in gyros:
            omega = gyros[timestamp] - bias
            last_gyro_time = timestamp
        if timestamp in accels:
            value = accels[timestamp]
            magnitude = float(np.linalg.norm(value))
            error = abs(magnitude - 9.80665)
            weight = max(0.0, 1.0 - error / settings["imu_accel_rejection_m_s2"])
            fresh_gyro = last_gyro_time is not None and timestamp - last_gyro_time <= max_gap
            still = (fresh_gyro and error <= settings["imu_still_accel_tolerance_m_s2"]
                     and np.linalg.norm(omega) <= settings["imu_bias_gyro_threshold_rad_s"])
            correction_ok = False
            if magnitude > 1e-9 and weight > 0:
                # Accelerometers measure specific force, opposite physical down.
                measured = -value / magnitude
                if gravity is None or (not converged and still):
                    gravity = measured
                    gyro_gravity = measured.copy()
                # After initial convergence, this reference is gyro-only until
                # an outage. Quiet accelerometer windows cannot re-anchor it:
                # a slow acceleration ramp can look quiet in every window.
                # Innovation therefore includes cumulative disagreement.
                innovation = float(np.rad2deg(np.arccos(np.clip(gyro_gravity @ measured, -1.0, 1.0))))
                correction_ok = innovation <= settings["imu_accel_innovation_deg"]
                if correction_ok and last_accel_time is not None:
                    dt = min(timestamp - last_accel_time, max_gap) / 1e9
                    alpha = 1.0 - np.exp(-weight * dt / settings["imu_filter_time_constant_s"])
                    gravity = (1.0 - alpha) * gravity + alpha * measured
                    norm = np.linalg.norm(gravity)
                    gravity = gravity / norm if norm > 1e-9 else measured
            if still and magnitude > 1e-9:
                measured = -value / magnitude
                if (still_down is not None
                        and np.linalg.norm(measured - still_down) * 9.80665
                        > settings["imu_still_accel_tolerance_m_s2"]):
                    still_since = None
                if still_since is None:
                    still_since = timestamp
                    still_down = measured.copy()
                if not converged and timestamp - still_since >= settings["imu_still_duration_s"] * 1e9:
                    converged = True
                    gravity = measured.copy()
                    gyro_gravity = measured.copy()
                    correction_ok = True
            else:
                still_since, still_down = None, None
            last_accel_time = timestamp
        fresh = (last_accel_time is not None and last_gyro_time is not None
                 and timestamp - last_accel_time <= max_gap and timestamp - last_gyro_time <= max_gap)
        valid = gravity is not None and fresh and converged and correction_ok
        times.append(timestamp)
        directions.append(gravity.copy() if gravity is not None else np.zeros(3))
        support.append((valid, last_accel_time, last_gyro_time, generation))
        previous_time = timestamp
    return times, np.array(directions), "complementary_filter", support


def _camera_imu_series(dataset, settings):
    # Only filter settings affect the continuous stream; mount and extrinsics do not.
    key = tuple(settings[name] for name in IMU_SETTINGS
                if name not in ("imu_to_camera_rpy_deg", "imu_to_camera_validated", "imu_max_delta_ms"))
    with dataset._camera_imu_lock:
        if key not in dataset._camera_imu_cache:
            dataset._camera_imu_cache[key] = _build_camera_imu_series(dataset, settings)
        return dataset._camera_imu_cache[key]


def summarize_camera_imu(dataset, frame):
    """Native RDF camera tilt; yaw is intentionally absent. Preserve UI keys."""
    settings = orientation_settings(dataset)
    validate_imu_settings(settings)
    rotation, extrinsics_source = camera_imu_extrinsics(dataset)
    summary = {
        "source": "oak", "imu_type": (dataset.metadata.get("imu") or {}).get("type") or "unknown",
        "roll_deg": None, "pitch_deg": None, "sample_delta_ms": None,
        "tilt_method": "unavailable", "extrinsics_source": extrinsics_source, "trusted": False,
    }
    row = frame["row"]
    if (rotation is None or str(summary["imu_type"]).upper() == "NONE"
            or ("imu_message_device_ts_ns" in row
                and safe_int(row.get("imu_message_device_ts_ns"), None) is None)
            or ("imu_packets" in row and safe_int(row.get("imu_packets"), 0) <= 0)):
        return summary
    timestamp = safe_int(row.get("rgb_device_ts_ns"), None)
    times, directions, method, support = _camera_imu_series(dataset, settings)
    if timestamp is None or not times:
        return summary
    pos = bisect.bisect_left(times, timestamp)
    nearest = min(range(max(0, pos - 1), min(len(times), pos + 1)), key=lambda i: abs(times[i] - timestamp))
    delta_ms = (times[nearest] - timestamp) / 1e6
    gravity = directions[nearest]
    used = [nearest]
    if 0 < pos < len(times) and times[pos] != timestamp:
        used = [pos - 1, pos]
        if (times[pos] - times[pos - 1] > settings["imu_max_gap_ms"] * 1e6
                or support[pos - 1][3] != support[pos][3]):
            return summary
        fraction = (timestamp - times[pos - 1]) / float(times[pos] - times[pos - 1])
        gravity = (1.0 - fraction) * directions[pos - 1] + fraction * directions[pos]
    norm = np.linalg.norm(gravity)
    if norm <= 1e-9:
        return summary
    gravity = rotation @ (gravity / norm)
    valid = all(support[i][0] for i in used)
    # A fresh packet from one sensor must not hide the other sensor's age.
    sensor_delta_ms = max(abs(timestamp - t) / 1e6 for i in used for t in support[i][1:3]
                          if t is not None)
    # Camera axes: right/down/forward. Positive roll is clockwise, pitch is down.
    summary.update({
        "roll_deg": float(np.rad2deg(np.arctan2(gravity[0], gravity[1]))),
        "pitch_deg": float(np.rad2deg(np.arctan2(gravity[2], np.hypot(gravity[0], gravity[1])))),
        "sample_delta_ms": delta_ms, "tilt_method": method,
        "trusted": bool(valid and sensor_delta_ms <= settings["imu_max_delta_ms"]),
    })
    return summary


def _camera_pose_from_imu(course_deg, mount_rpy, camera_imu):
    mount = camera_mount_matrix_deg(*mount_rpy)
    roll, pitch = np.deg2rad([camera_imu["roll_deg"], camera_imu["pitch_deg"]])
    camera_down = np.array([np.sin(roll) * np.cos(pitch), np.cos(roll) * np.cos(pitch), np.sin(pitch)])
    vehicle_down = mount @ camera_down
    vehicle_roll = np.rad2deg(np.arctan2(vehicle_down[0], vehicle_down[1]))
    vehicle_pitch = np.rad2deg(np.arctan2(vehicle_down[2], np.hypot(vehicle_down[0], vehicle_down[1])))
    return (orientation_matrix_from_gps_course(course_deg)
            @ camera_mount_matrix_deg(vehicle_roll, vehicle_pitch, 0.0) @ mount)


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
            return raw_vector * depth_m, "distortion_corrected"
        except Exception:
            pass

    if not saved_intrinsics:
        return None, "missing_intrinsics"

    fx = float(saved_intrinsics[0][0])
    fy = float(saved_intrinsics[1][1])
    cx = float(saved_intrinsics[0][2])
    cy = float(saved_intrinsics[1][2])
    saved_vector = np.array([
        (x - cx) * depth_m / fx,
        (y - cy) * depth_m / fy,
        depth_m,
    ], dtype=float)
    # Axis sign changes are self-inverse. Only pixel rays use this reflection;
    # attitude and physical lever arms always stay in the native RDF frame.
    native_vector = original_camera_vector_to_saved_frame(saved_vector, dataset.metadata)
    return native_vector, "pinhole_undistorted" if stream_is_undistorted else "pinhole"


def compute_world_coordinate(dataset, frame, x, y, depth_mm):
    return compute_world_coordinate_for_source(dataset, frame, x, y, depth_mm, "gps-course")


def compute_world_coordinate_for_source(dataset, frame, x, y, depth_mm, orientation_source):
    if orientation_source not in ("gps-course", "gps-course-level"):
        raise ValueError(f"Unknown orientation_source: {orientation_source!r}.")
    if not np.isfinite(depth_mm) or depth_mm <= 0:
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
        gps_alt = safe_float(alt_row.get("altitude_m"), 0.0) if alt_row else 0.0

    if gps_lat is None or gps_lon is None:
        return {"status": "unavailable", "reason": "GPS latitude/longitude missing"}

    world_cfg = dataset.metadata.get("world_coordinates") or {}
    camera_mount_rpy = normalized_rpy(world_cfg.get("camera_mount_rpy_deg") or world_cfg.get("vehicle_to_camera_rpy_deg"))
    gps_to_camera = world_cfg.get("gps_to_camera_enu_m") or [0.0, 0.0, 0.0]
    gps_to_camera_camera = world_cfg.get("gps_to_camera_camera_m")
    gps_from_camera_camera = world_cfg.get("gps_from_camera_camera_m")
    if gps_to_camera_camera is None and gps_from_camera_camera is not None:
        gps_to_camera_camera = [-float(value) for value in gps_from_camera_camera]
    if gps_to_camera_camera is None:
        gps_to_camera_camera = [0.0, 0.0, 0.0]
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
    trusted_max_hdop = float(orientation_settings(dataset)["trusted_max_hdop"])
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
    elif gps_hdop > trusted_max_hdop:
        assumptions.append(f"GPS HDOP {gps_hdop:.2f} exceeds the trusted {trusted_max_hdop:.2f} limit")
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
        "maximum_hdop": trusted_max_hdop,
    }
    course = resolve_course_for_frame(dataset, frame)
    if course["course_deg"] is None:
        return {"status": "unavailable", "trusted": False, "reason": "GPS course_deg missing"}
    camera_imu = summarize_camera_imu(dataset, frame)
    if orientation_source == "gps-course":
        if camera_imu["tilt_method"] == "unavailable":
            return {"status": "unavailable", "trusted": False,
                    "reason": "OAK IMU tilt unavailable", "camera_imu": camera_imu}
        r_enu_from_camera = _camera_pose_from_imu(course["course_deg"], camera_mount_rpy, camera_imu)
        if not camera_imu["trusted"]:
            assumptions.append("OAK IMU timing or filter validity is outside trusted limits")
        assumptions.append("GPS course supplies yaw; OAK IMU supplies pitch/roll")
    else:
        assumptions.append("GPS course supplies yaw; vehicle is assumed level")
        r_enu_from_camera = (orientation_matrix_from_gps_course(course["course_deg"])
                             @ camera_mount_matrix_deg(*camera_mount_rpy))
    course_trusted = (course["speed_m_s"] is not None and course["speed_m_s"] >= 2.0
                      and (course["hdop"] is None or course["hdop"] <= orientation_settings(dataset)["trajectory_max_hdop"]))
    trusted = bool(position_quality["trusted"] and course_trusted
                   and orientation_source == "gps-course" and camera_imu["trusted"])
    orientation_details = {
        "source": orientation_source, "course_deg": course["course_deg"],
        "effective_course_deg": course["course_deg"] + float(camera_mount_rpy[2]),
        "speed_m_s": course["speed_m_s"], "hdop": course["hdop"],
        "course_source": course["source"],
        "course_sample_index": course["row"].get("sample_index") if course["row"] else None,
        "course_frame_delta_ms": course["delta_ms"],
        "tilt_source": "oak" if orientation_source == "gps-course" else "level_assumption",
        "camera_mount_rpy_deg": camera_mount_rpy, "trusted": trusted,
        "camera_imu": camera_imu,
    }
    assumptions.append("GPS course is movement direction, not optical heading while stopped or reversing")
    if course["source"] == "nearest_moving_gps":
        assumptions.append("GPS course was taken from nearest moving sample because the frame sample was weak")

    forward_axis = r_enu_from_camera[:, 2]
    optical_heading_deg = float(np.rad2deg(np.arctan2(forward_axis[0], forward_axis[1])) % 360.0)
    optical_elevation_deg = float(np.rad2deg(np.arctan2(forward_axis[2], np.hypot(forward_axis[0], forward_axis[1]))))
    orientation_details.update({
        "optical_heading_deg": optical_heading_deg,
        "optical_elevation_deg": optical_elevation_deg,
        "camera_down_up_component": float(r_enu_from_camera[2, 1]),
    })
    if abs(optical_elevation_deg) > 45.0:
        assumptions.append("camera optical axis elevation exceeds 45 deg; check IMU-to-camera extrinsics")

    gps_to_camera_enu_vec = np.array([float(value) for value in gps_to_camera], dtype=float)
    gps_to_camera_camera_vec = np.array([float(value) for value in gps_to_camera_camera], dtype=float)
    camera_position_enu = gps_to_camera_enu_vec + (r_enu_from_camera @ gps_to_camera_camera_vec)
    point_enu = camera_position_enu + (r_enu_from_camera @ point_camera)
    llh = enu_to_llh(gps_lat, gps_lon, gps_alt, point_enu[0], point_enu[1], point_enu[2])

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
        "camera_imu": camera_imu,
        "trusted": trusted,
        "config": {
            "camera_mount_rpy_deg": camera_mount_rpy,
            "gps_to_camera_enu_m": gps_to_camera,
            "gps_to_camera_camera_m": gps_to_camera_camera,
            "gps_from_camera_camera_m": gps_from_camera_camera,
            "rtk_max_correction_age_s": rtk_max_correction_age_s,
            "trusted_max_hdop": trusted_max_hdop,
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
        or (hdop is not None and hdop > orientation_settings(dataset)["trajectory_max_hdop"])
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
    """Persist processing overrides atomically; capture files are immutable."""
    overrides = _metadata_overrides(dataset._capture_metadata, dataset.metadata)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(dataset.root),
                                         prefix=".metadata.overrides-", suffix=".tmp", delete=False) as file:
            temporary_path = Path(file.name)
            json.dump(overrides, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(str(temporary_path), str(dataset.root / "metadata.overrides.json"))
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    dataset.metadata = _merge_metadata(dataset._capture_metadata, overrides)


def set_camera_mount_rpy(dataset, roll, pitch, yaw, save=False, calibration=None):
    world_cfg = dataset.metadata.setdefault("world_coordinates", {})
    rpy = [float(roll), float(pitch), float(yaw)]
    world_cfg["camera_mount_rpy_deg"] = rpy
    world_cfg["camera_mount_frame"] = (
        "native RDF camera to vehicle frame; roll +clockwise, "
        "pitch +down, yaw +right from vehicle forward"
    )
    if calibration is not None:
        if calibration.get("source") == "single_clicked_point":
            # Keep legacy debug_ui callers' persisted details accurate too.
            calibration = dict(calibration, solved_axes=["yaw"], fixed_axes=["roll", "pitch"],
                               note=SINGLE_POINT_MOUNT_NOTE)
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
    camera_imu = summarize_camera_imu(dataset, frame)
    if camera_imu["tilt_method"] == "unavailable":
        raise ValueError("OAK IMU tilt unavailable.")
    if not camera_imu["trusted"]:
        raise ValueError("OAK IMU timing or filter validity is outside trusted limits.")

    # One point supplies only one observable rotation when gravity determines
    # tilt for each candidate. Preserve the independently configured bank/tilt.
    roll, pitch = current_rpy[:2]

    def predict(rpy):
        # Each candidate changes camera -> vehicle gravity, so vehicle tilt must
        # be re-derived for every candidate exactly as it is during export.
        rotation = _camera_pose_from_imu(course["course_deg"], rpy, camera_imu)
        return gps_to_camera_enu + (rotation @ camera_vector)

    def error_for(yaw):
        pred = predict([roll, pitch, yaw]) - gps_to_camera_enu
        return float(np.linalg.norm(pred - target_from_origin))

    best_yaw = current_rpy[2]
    best_error = error_for(best_yaw)
    for yaw_span, steps in [(360.0, 361), (2.0, 41), (0.1, 41), (0.005, 41), (0.00025, 41)]:
        yaw_values = np.linspace(best_yaw - yaw_span / 2.0, best_yaw + yaw_span / 2.0, steps)
        for yaw in yaw_values:
            err = error_for(float(yaw))
            if err < best_error:
                best_error = err
                best_yaw = float(yaw)

    before = predict(current_rpy)
    after_rpy = [roll, pitch, best_yaw]
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
        "solved_axes": ["yaw"],
        "fixed_axes": ["roll", "pitch"],
        "note": SINGLE_POINT_MOUNT_NOTE,
    }
