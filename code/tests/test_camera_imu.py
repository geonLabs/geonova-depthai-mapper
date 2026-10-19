"""Synthetic OAK tilt and geometry regressions; also runnable with unittest."""

import csv
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for source in ("postprocess/src", "shared/src"):
    sys.path.insert(0, str(ROOT / source))

from geonova_postprocess import dataset as geometry


def write_rows(path, rows, fields=None):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def down_vector(roll, pitch):
    r, p = np.deg2rad([roll, pitch])
    return np.array([math.sin(r) * math.cos(p), math.cos(r) * math.cos(p), math.sin(p)])


def raw_row(timestamp, down, gyro=(0, 0, 0)):
    row = {"message_device_ts_ns": timestamp, "accel_device_ts_ns": timestamp,
           "gyro_device_ts_ns": timestamp}
    row.update({"accel_" + axis + "_m_s2": -9.80665 * float(value) for axis, value in zip("xyz", down)})
    row.update({"gyro_" + axis + "_rad_s": value for axis, value in zip("xyz", gyro)})
    return row


def rv_row(timestamp, roll=0, pitch=0, yaw=0):
    # Independent IMU-to-world quaternion fixture. World up is +Z.
    r, p, y = np.deg2rad([roll, pitch, yaw]) / 2

    def mul(a, b):
        av, aw, bv, bw = np.array(a[:3]), a[3], np.array(b[:3]), b[3]
        return list(aw * bv + bw * av + np.cross(av, bv)) + [aw * bw - av @ bv]

    q = mul([0, 0, math.sin(y), math.cos(y)], [-math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
    q = mul(q, [-math.sin(p), 0, 0, math.cos(p)])
    q = mul(q, [0, 0, math.sin(r), math.cos(r)])
    row = raw_row(timestamp, down_vector(roll, pitch))
    row.update({"rv_device_ts_ns": timestamp, "rv_accuracy_rad": 0.01})
    row.update({"rv_" + name: value for name, value in zip(("i", "j", "k", "real"), q)})
    return row


def make_dataset(root, events, timestamp=1_000_000_000, metadata=None, gps_rows=None):
    root.mkdir(parents=True, exist_ok=True)
    write_rows(root / "timestamps.csv", [{
        "rgb_file": "rgb.png", "depth_file": "depth.png", "rgb_device_ts_ns": timestamp,
        "frame_host_monotonic_ns": 1_000_000_000, "gps_sample_index": 0,
        "gps_latitude_deg": 37, "gps_longitude_deg": 127, "gps_altitude_m": 100,
        "gps_fix_quality": 4, "gps_position_valid": 1, "gps_hdop": 0.5,
        "gps_differential_age_s": 0.3, "gps_frame_delta_ms": 0,
    }])
    write_rows(root / "imu.csv", [], ["frame_index"])
    if events:
        write_rows(root / "imu_events.csv", events)
    if gps_rows is None:
        gps_rows = [{"sample_index": 0, "host_monotonic_ns": 1_000_000_000,
                     "course_deg": 35, "speed_knots": 10, "hdop": 0.5, "altitude_m": 100}]
    write_rows(root / "gps.csv", gps_rows)
    value = {"camera_model": {"intrinsics": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                              "image_stream_undistorted": True}}
    value.update(metadata or {})
    (root / "metadata.json").write_text(json.dumps(value), encoding="utf-8")
    return geometry.Dataset(root)


class CameraImuTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def summary(self, dataset):
        return geometry.summarize_camera_imu(dataset, dataset.frame(0))

    def test_static_raw_tilts_and_old_metadata(self):
        for i, (roll, pitch) in enumerate(((0, 0), (20, -10), (-25, 15), (60, 35))):
            with self.subTest(roll=roll, pitch=pitch):
                events = [raw_row(1_000_000_000 + j * 10_000_000, down_vector(roll, pitch)) for j in range(50)]
                dataset = make_dataset(self.root / str(i), list(reversed(events)), 1_245_000_000)
                result = self.summary(dataset)
                self.assertAlmostEqual(result["roll_deg"], roll, places=6)
                self.assertAlmostEqual(result["pitch_deg"], pitch, places=6)
                self.assertEqual(result["source"], "oak")
                self.assertEqual(result["imu_type"], "unknown")
                self.assertEqual(result["tilt_method"], "complementary_filter")
                self.assertEqual(result["extrinsics_source"], "config_default")
                self.assertTrue(result["trusted"])

    def test_vehicle_acceleration_rejected(self):
        events = [raw_row(1_000_000_000 + j * 10_000_000, [0, 1, 0]) for j in range(300)]
        for row in events[100:]:
            row["accel_x_m_s2"] = -8.0
        dataset = make_dataset(self.root, events, 3_990_000_000)
        self.assertAlmostEqual(self.summary(dataset)["roll_deg"], 0, places=6)

    def test_initial_still_bias_estimate_and_disable(self):
        events = [raw_row(1_000_000_000 + j * 10_000_000, [0, 1, 0], (0, 0, 0.02)) for j in range(401)]
        for row in events[201:]:
            row["accel_x_m_s2"] = -8.0
        dataset = make_dataset(self.root, events, 5_000_000_000)
        self.assertAlmostEqual(self.summary(dataset)["roll_deg"], 0, places=5)
        dataset.orientation_config["imu_bias_window_s"] = 0
        self.assertGreater(self.summary(dataset)["roll_deg"], 2)

    def test_initial_motion_does_not_become_bias(self):
        events = [raw_row(1_000_000_000 + j * 10_000_000, [0, 1, 0], (0, 0, 0.5)) for j in range(301)]
        dataset = make_dataset(self.root, events, 4_000_000_000)
        self.assertGreater(self.summary(dataset)["roll_deg"], 20)

    def test_gyro_integration_during_rejected_acceleration(self):
        events = [raw_row(1_000_000_000 + j * 10_000_000, [0, 1, 0], (0, 0, 0.5)) for j in range(101)]
        for row in events[1:]:
            row["accel_x_m_s2"] = -8
        dataset = make_dataset(self.root, events, 2_000_000_000)
        self.assertAlmostEqual(self.summary(dataset)["roll_deg"], math.degrees(0.5), places=5)

    def test_rotation_vector_ignores_yaw(self):
        for i, yaw in enumerate((0, 87, -140)):
            events = [rv_row(1_000_000_000, 18, -12, yaw)]
            # Deliberately conflicting raw acceleration; rotation vector wins.
            events[0]["accel_x_m_s2"] = 30
            dataset = make_dataset(self.root / str(i), events, metadata={"imu": {"type": "BNO086"}})
            result = self.summary(dataset)
            self.assertEqual(result["tilt_method"], "game_rotation_vector")
            self.assertEqual(result["imu_type"], "BNO086")
            self.assertAlmostEqual(result["roll_deg"], 18, places=6)
            self.assertAlmostEqual(result["pitch_deg"], -12, places=6)

    def test_interpolation_uses_rv_timestamp(self):
        events = [rv_row(1_000_000_000, 0), rv_row(1_010_000_000, 20)]
        for row in events:
            row["message_device_ts_ns"] += 100_000_000
        dataset = make_dataset(self.root, events, 1_005_000_000)
        self.assertAlmostEqual(self.summary(dataset)["roll_deg"], 10, places=6)
        self.assertAlmostEqual(abs(self.summary(dataset)["sample_delta_ms"]), 5)

    def test_extrinsics_rotation_and_translation_ignored(self):
        rotation = geometry.rpy_matrix_deg(15, -30, 5)
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = [99, -70, 150]  # centimeters must not affect tilt or lever arm
        events = [raw_row(1_000_000_000, rotation.T @ down_vector(25, 12))]
        metadata = {"imu": {"type": "BMI270", "imu_to_camera": {"matrix": matrix.tolist()}}}
        dataset = make_dataset(self.root, events, metadata=metadata)
        result = self.summary(dataset)
        self.assertEqual(result["extrinsics_source"], "device_calibration")
        self.assertAlmostEqual(result["roll_deg"], 25, places=6)
        self.assertAlmostEqual(result["pitch_deg"], 12, places=6)

    def test_invalid_extrinsics_use_config_rotation(self):
        invalid = [None, [[1, 2]], np.diag([-1, 1, 1]).tolist(),
                   (np.eye(3) * 2).tolist(), [[float("nan"), 0, 0], [0, 1, 0], [0, 0, 1]]]
        rotation = geometry.rpy_matrix_deg(10, -20, 30)
        for i, value in enumerate(invalid):
            dataset = make_dataset(self.root / str(i), [raw_row(1_000_000_000, rotation.T @ down_vector(5, 8))],
                                   metadata={"imu": {"imu_to_camera": {"matrix": value}},
                                             "world_coordinates": {"imu_to_camera_rpy_deg": [10, -20, 30]}})
            result = self.summary(dataset)
            self.assertEqual(result["extrinsics_source"], "config_default")
            self.assertAlmostEqual(result["roll_deg"], 5, places=6)
            self.assertAlmostEqual(result["pitch_deg"], 8, places=6)

    def test_sample_time_trust_boundary_and_world_quality(self):
        dataset = make_dataset(self.root, [rv_row(1_000_000_000)], 1_020_000_000)
        self.assertTrue(self.summary(dataset)["trusted"])
        world = geometry.compute_world_coordinate(dataset, dataset.frame(0), 320, 240, 4000)
        self.assertTrue(world["trusted"])
        dataset.timestamps[0]["rgb_device_ts_ns"] = 1_020_001_000
        self.assertFalse(self.summary(dataset)["trusted"])
        world = geometry.compute_world_coordinate(dataset, dataset.frame(0), 320, 240, 4000)
        self.assertEqual(world["status"], "ok")
        self.assertFalse(world["trusted"])
        dataset.orientation_config["imu_max_delta_ms"] = 21
        self.assertTrue(self.summary(dataset)["trusted"])

    def test_level_matches_existing_level_mode_with_mount_and_lever_arm(self):
        dataset = make_dataset(self.root, [rv_row(1_000_000_000)], metadata={"world_coordinates": {
            "camera_mount_rpy_deg": [3, -7, 11], "gps_to_camera_enu_m": [0.2, -0.4, 0.1],
            "gps_to_camera_camera_m": [0.3, 0.1, -0.2],
        }})
        frame = dataset.frame(0)
        actual = geometry.compute_world_coordinate(dataset, frame, 410, 330, 4500)
        level = geometry.compute_world_coordinate_for_source(dataset, frame, 410, 330, 4500, "gps-course-level")
        for key in ("latitude_deg", "longitude_deg", "altitude_m", "enu_offset_m", "camera_position_enu_m"):
            if isinstance(actual[key], dict):
                np.testing.assert_allclose(list(actual[key].values()), list(level[key].values()), atol=1e-12)
            else:
                self.assertAlmostEqual(actual[key], level[key], places=12)
        self.assertEqual(actual["orientation_source"], "gps-course")

    def test_nonlevel_pose_composes_all_mount_angles_and_lever_arm(self):
        dataset = make_dataset(self.root, [raw_row(1_000_000_000, down_vector(20, 10))], metadata={"world_coordinates": {
            "camera_mount_rpy_deg": [-5, 8, 12], "gps_to_camera_enu_m": [1, 2, 3],
            "gps_to_camera_camera_m": [0.2, 0.3, 0.4],
        }})
        world = geometry.compute_world_coordinate(dataset, dataset.frame(0), 320, 240, 4000)
        rotation = (geometry.orientation_matrix_from_gps_course(35)
                    @ geometry.camera_mount_matrix_deg(20, 10, 0)
                    @ geometry.camera_mount_matrix_deg(-5, 8, 12))
        expected = np.array([1, 2, 3]) + rotation @ np.array([0.2, 0.3, 4.4])
        np.testing.assert_allclose(list(world["enu_offset_m"].values()), expected)

    def test_missing_stream_or_timestamp_is_unavailable(self):
        dataset = make_dataset(self.root, [])
        self.assertEqual(self.summary(dataset)["tilt_method"], "unavailable")
        self.assertFalse(self.summary(dataset)["trusted"])
        self.assertEqual(geometry.compute_world_coordinate(dataset, dataset.frame(0), 320, 240, 4000)["status"], "unavailable")
        self.assertEqual(geometry.compute_world_coordinate_for_source(dataset, dataset.frame(0), 320, 240, 4000, "gps-course-level")["status"], "ok")
        with_stream = make_dataset(self.root / "stream", [rv_row(1_000_000_000)])
        with_stream.timestamps[0]["rgb_device_ts_ns"] = ""
        self.assertEqual(self.summary(with_stream)["tilt_method"], "unavailable")

    def test_stream_is_cached_across_frames_and_points(self):
        dataset = make_dataset(self.root, [rv_row(1_000_000_000)])
        with patch.object(geometry, "_build_camera_imu_series", wraps=geometry._build_camera_imu_series) as build:
            for _ in range(4):
                self.summary(dataset)
                geometry.compute_world_coordinate(dataset, dataset.frame(0), 320, 240, 4000)
            dataset.orientation_config["imu_max_delta_ms"] = 30
            self.summary(dataset)
            self.assertEqual(build.call_count, 1)

    def test_named_hdop_settings_control_trust_and_course_selection(self):
        gps = [
            {"sample_index": 0, "host_monotonic_ns": 1_000_000_000, "course_deg": 10,
             "speed_knots": 10, "hdop": 2.2, "altitude_m": 100},
            {"sample_index": 1, "host_monotonic_ns": 1_010_000_000, "course_deg": 60,
             "speed_knots": 10, "hdop": 0.7, "altitude_m": 100},
        ]
        dataset = make_dataset(self.root, [rv_row(1_000_000_000)], gps_rows=gps)
        dataset.timestamps[0]["gps_hdop"] = 2.2
        self.assertEqual(geometry.resolve_course_for_frame(dataset, dataset.frame(0))["course_deg"], 10)
        self.assertFalse(geometry.compute_world_coordinate(dataset, dataset.frame(0), 320, 240, 4000)["trusted"])
        dataset.orientation_config.update(trusted_max_hdop=2.3, trajectory_max_hdop=1.0)
        self.assertEqual(geometry.resolve_course_for_frame(dataset, dataset.frame(0))["course_deg"], 60)
        self.assertTrue(geometry.compute_world_coordinate(dataset, dataset.frame(0), 320, 240, 4000)["trusted"])

    def test_separate_sensor_timestamps_and_duplicate_packets(self):
        events = [raw_row(1_000_000_000 + j * 10_000_000, down_vector(-12, 7)) for j in range(30)]
        for row in events:
            row["gyro_device_ts_ns"] += 2_000_000
            row["message_device_ts_ns"] += 80_000_000
        events = events[::2] + events[1::2] + events[:3]
        dataset = make_dataset(self.root, events, 1_155_000_000)
        result = self.summary(dataset)
        self.assertAlmostEqual(result["roll_deg"], -12, places=6)
        self.assertAlmostEqual(result["pitch_deg"], 7, places=6)
        self.assertAlmostEqual(result["sample_delta_ms"], -3)

    def test_unknown_orientation_is_an_error(self):
        dataset = make_dataset(self.root, [])
        with self.assertRaisesRegex(ValueError, "Unknown orientation_source"):
            geometry.compute_world_coordinate_for_source(dataset, dataset.frame(0), 0, 0, 0, "invalid")

    def test_altitude_binary_search_matches_old_scan_including_ties(self):
        rng = random.Random(42)
        rows = [{"host_monotonic_ns": rng.randrange(20), "altitude_m": i if i % 5 else ""} for i in range(100)]
        rows.extend([{"host_monotonic_ns": "invalid", "altitude_m": "9"},
                     {"host_monotonic_ns": "", "altitude_m": "10"}])
        dataset = make_dataset(self.root, [], gps_rows=rows)
        for timestamp in range(-5, 26):
            expected = min([row for row in dataset.gps_rows if row.get("altitude_m") not in ("", None)],
                           key=lambda row: abs(geometry.safe_int(row.get("host_monotonic_ns"), 0) - timestamp))
            self.assertIs(dataset.nearest_gps_altitude(timestamp), expected)
        self.assertIsNone(dataset.nearest_gps_altitude(None))


if __name__ == "__main__":
    unittest.main()
