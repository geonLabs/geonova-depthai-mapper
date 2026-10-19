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


def reference_vehicle(course, roll, pitch):
    """Construct physical ENU axes from forward slope and bank, without Euler matrices."""
    c, r, p = np.deg2rad([course, roll, pitch])
    forward = np.array([math.sin(c) * math.cos(p), math.cos(c) * math.cos(p), -math.sin(p)])
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    banked_right = right * math.cos(r) + down * math.sin(r)
    return np.column_stack([banked_right, np.cross(forward, banked_right), forward])


def reference_mount(angles):
    """Rodrigues action on basis vectors: bank, downward pitch, then rightward yaw."""
    def turn(v, axis, degrees):
        axis = np.array(axis)
        a = math.radians(degrees)
        return v * math.cos(a) + np.cross(axis, v) * math.sin(a) + axis * (axis @ v) * (1 - math.cos(a))

    columns = []
    for vector in np.eye(3):
        vector = turn(vector, [0, 0, 1], angles[0])
        vector = turn(vector, [1, 0, 0], -angles[1])
        columns.append(turn(vector, [0, 1, 0], angles[2]))
    return np.column_stack(columns)


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
                              "image_stream_undistorted": True},
             "imu": {"imu_to_camera": {"matrix": np.eye(3).tolist()}}}
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

    def test_static_raw_tilts(self):
        for i, (roll, pitch) in enumerate(((0, 0), (20, -10), (-25, 15), (60, 35))):
            with self.subTest(roll=roll, pitch=pitch):
                events = [raw_row(1_000_000_000 + j * 10_000_000, down_vector(roll, pitch)) for j in range(100)]
                dataset = make_dataset(self.root / str(i), list(reversed(events)), 1_745_000_000)
                result = self.summary(dataset)
                self.assertAlmostEqual(result["roll_deg"], roll, places=6)
                self.assertAlmostEqual(result["pitch_deg"], pitch, places=6)
                self.assertEqual(result["source"], "oak")
                self.assertEqual(result["imu_type"], "unknown")
                self.assertEqual(result["tilt_method"], "complementary_filter")
                self.assertEqual(result["extrinsics_source"], "device_calibration")
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
            dataset = make_dataset(self.root / str(i), events, metadata={"imu": {
                "type": "BNO086", "imu_to_camera": {"matrix": np.eye(3).tolist()}}})
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
            self.assertEqual(self.summary(dataset)["extrinsics_source"], "unavailable")
            self.assertFalse(self.summary(dataset)["trusted"])
            dataset.orientation_config["imu_to_camera_validated"] = True
            result = self.summary(dataset)
            self.assertEqual(result["extrinsics_source"], "config_validated")
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
        # The measured camera gravity includes its mount even on a level vehicle.
        mount = reference_mount([3, -7, 11])
        events = [raw_row(1_000_000_000 + j * 10_000_000, mount.T @ [0, 1, 0]) for j in range(101)]
        dataset = make_dataset(self.root, events, 2_000_000_000, metadata={"world_coordinates": {
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
        pose = reference_vehicle(35, 20, 10) @ reference_mount([-5, 8, 12])
        measured_down = pose.T @ [0, 0, -1]
        events = [raw_row(1_000_000_000 + j * 10_000_000, measured_down) for j in range(101)]
        dataset = make_dataset(self.root, events, 2_000_000_000, metadata={"world_coordinates": {
            "camera_mount_rpy_deg": [-5, 8, 12], "gps_to_camera_enu_m": [1, 2, 3],
            "gps_to_camera_camera_m": [0.2, 0.3, 0.4],
        }})
        world = geometry.compute_world_coordinate(dataset, dataset.frame(0), 320, 240, 4000)
        expected = np.array([1, 2, 3]) + pose @ np.array([0.2, 0.3, 4.4])
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

    def test_downward_mount_worked_example(self):
        mount = reference_mount([0, 10, 0])
        events = [raw_row(1_000_000_000 + j * 10_000_000, mount.T @ [0, 1, 0]) for j in range(101)]
        dataset = make_dataset(self.root, events, 2_000_000_000,
                               metadata={"world_coordinates": {"camera_mount_rpy_deg": [0, 10, 0]}})
        dataset.gps_rows[0]["course_deg"] = "0"
        world = geometry.compute_world_coordinate(dataset, dataset.frame(0), 320, 240, 10000)
        np.testing.assert_allclose(list(world["enu_offset_m"].values()), [0, 9.8480775301, -1.7364817767], atol=1e-9)
        self.assertTrue(world["trusted"])

    def test_physical_pose_combinations_against_reference_axes(self):
        cases = [(0, 0, 0, [0, 10, 0]), (35, 20, 10, [-5, 8, 12]),
                 (148, -25, 17, [11, -19, -35]), (286, 32, -21, [-17, 24, 28])]
        for i, (course, roll, pitch, mount) in enumerate(cases):
            with self.subTest(case=i):
                pose = reference_vehicle(course, roll, pitch) @ reference_mount(mount)
                events = [raw_row(1_000_000_000 + j * 10_000_000, pose.T @ [0, 0, -1]) for j in range(101)]
                dataset = make_dataset(self.root / str(i), events, 2_000_000_000, metadata={"world_coordinates": {
                    "camera_mount_rpy_deg": mount, "gps_to_camera_enu_m": [1, -2, 0.5],
                    "gps_to_camera_camera_m": [0.2, -0.3, 0.4]}})
                dataset.gps_rows[0]["course_deg"] = str(course)
                world = geometry.compute_world_coordinate(dataset, dataset.frame(0), 370, 215, 10000)
                expected = [1, -2, 0.5] + pose @ np.array([1.2, -0.8, 10.4])
                np.testing.assert_allclose(list(world["enu_offset_m"].values()), expected, atol=1e-10)
                self.assertTrue(world["trusted"])

    def test_saved_image_transforms_preserve_physical_points(self):
        for distorted in (False, True):
            for flip, rotate in ((False, False), (True, False), (False, True), (True, True)):
                for roll, pitch in ((0, 0), (16, -9)):
                    with self.subTest(distorted=distorted, flip=flip, rotate=rotate, roll=roll):
                        original = [[500, 0, 301], [0, 480, 221], [0, 0, 1]]
                        saved = [row[:] for row in original]
                        # Forward pixel operations on the principal point and the clicked pixel.
                        x, y = 351.0, 221.0  # Physical point: 1 m right, 10 m forward.
                        if flip:
                            saved[1][2], y = 479 - saved[1][2], 479 - y
                        if rotate:
                            saved[0][2], x = 639 - saved[0][2], 639 - x
                            saved[1][2], y = 479 - saved[1][2], 479 - y
                        metadata = {"image_size": [640, 480],
                                    "image_transform": {"flip_vertical": flip, "rotate_180": rotate},
                                    "camera_model": {"intrinsics": saved, "intrinsics_original": original,
                                                     "image_stream_undistorted": not distorted}}
                        dataset = make_dataset(self.root, [rv_row(1_000_000_000, roll, pitch)], metadata=metadata)
                        dataset.gps_rows[0]["course_deg"] = "0"
                        # No lens distortion: test pixel/frame handling in both unprojection paths.
                        def undistort(points, camera_matrix, coefficients):
                            return (points - np.array([301, 221])) / np.array([500, 480])
                        with patch.object(geometry.cv2, "undistortPoints", side_effect=undistort, create=True):
                            world = geometry.compute_world_coordinate(dataset, dataset.frame(0), x, y, 10000)
                        expected = reference_vehicle(0, roll, pitch) @ [1, 0, 10]
                        np.testing.assert_allclose(list(world["enu_offset_m"].values()), expected, atol=1e-12)
                        self.assertAlmostEqual(world["camera_imu"]["roll_deg"], roll)
                        self.assertAlmostEqual(world["camera_imu"]["pitch_deg"], pitch)
                        self.assertTrue(world["trusted"])

    def test_missing_extrinsics_never_imply_identity(self):
        for i, imu in enumerate(({}, {"type": "BMI270", "imu_to_camera": {"matrix": None}},
                                 {"imu_to_camera": {"matrix": np.diag([-1, 1, 1]).tolist()}})):
            dataset = make_dataset(self.root / str(i), [raw_row(1_000_000_000, [1, 0, 0])],
                                   metadata={"imu": imu})
            result = self.summary(dataset)
            self.assertEqual(result["extrinsics_source"], "unavailable")
            self.assertEqual(result["tilt_method"], "unavailable")
            self.assertIsNone(result["roll_deg"])
            self.assertFalse(result["trusted"])
            self.assertFalse(geometry.compute_world_coordinate(dataset, dataset.frame(0), 320, 240, 10000)["trusted"])
            level = geometry.compute_world_coordinate_for_source(dataset, dataset.frame(0), 320, 240, 10000, "gps-course-level")
            self.assertEqual(level["status"], "ok")

    def test_raw_startup_requires_still_period(self):
        events = [raw_row(1_000_000_000 + j * 10_000_000, [0, 1, 0]) for j in range(101)]
        dataset = make_dataset(self.root, events, 1_495_000_000)
        self.assertFalse(self.summary(dataset)["trusted"])  # Interpolation cannot bridge unready -> ready.
        dataset.timestamps[0]["rgb_device_ts_ns"] = 1_500_000_000
        self.assertTrue(self.summary(dataset)["trusted"])

    def test_moderate_acceleration_and_braking_do_not_turn_into_tilt(self):
        for acceleration in (-3.0, -2.0, -1.5, 1.5, 2.0, 3.0):
            with self.subTest(acceleration=acceleration):
                events = [raw_row(1_000_000_000 + j * 10_000_000, [0, 1, 0]) for j in range(501)]
                for row in events[100:400]:
                    row["accel_z_m_s2"] = acceleration
                dataset = make_dataset(self.root, events, 4_990_000_000)
                result = self.summary(dataset)
                self.assertAlmostEqual(result["pitch_deg"], 0, places=8)
                self.assertFalse(result["trusted"])
                dataset.timestamps[0]["rgb_device_ts_ns"] = 6_000_000_000
                self.assertTrue(self.summary(dataset)["trusted"])

    def test_accelerating_startup_is_untrusted_and_recovers(self):
        for acceleration in (-3, -1.5, 1.5, 2, 3):
            events = [raw_row(1_000_000_000 + j * 10_000_000, [0, 1, 0]) for j in range(401)]
            for row in events[:300]:
                row["accel_z_m_s2"] = acceleration
            dataset = make_dataset(self.root, events, 3_990_000_000)
            self.assertFalse(self.summary(dataset)["trusted"])
            dataset.timestamps[0]["rgb_device_ts_ns"] = 4_490_000_000
            self.assertFalse(self.summary(dataset)["trusted"])
            dataset.timestamps[0]["rgb_device_ts_ns"] = 5_000_000_000
            self.assertTrue(self.summary(dataset)["trusted"])
            self.assertAlmostEqual(self.summary(dataset)["pitch_deg"], 0, places=8)

    def test_gradual_acceleration_cannot_move_its_own_innovation_reference(self):
        for acceleration in (-2.0, 2.0):
            events = [raw_row(1_000_000_000 + j * 10_000_000, [0, 1, 0]) for j in range(801)]
            for j, row in enumerate(events):
                row["accel_z_m_s2"] = acceleration * min(1, max(0, (j - 100) / 300.0))
            dataset = make_dataset(self.root, events, 8_000_000_000)
            result = self.summary(dataset)
            self.assertFalse(result["trusted"])
            self.assertLess(abs(result["pitch_deg"]), 7.0)

    def test_outage_reinitializes_instead_of_integrating_held_gyro(self):
        events = [raw_row(1_000_000_000 + j * 10_000_000, [0, 1, 0]) for j in range(101)]
        events.append(raw_row(2_010_000_000, [0, 1, 0], (0.2, 0, 0)))
        events.extend(raw_row(12_010_000_000 + j * 10_000_000, [0, 1, 0]) for j in range(101))
        dataset = make_dataset(self.root, events, 12_010_000_000)
        result = self.summary(dataset)
        self.assertAlmostEqual(result["pitch_deg"], 0, places=8)
        self.assertFalse(result["trusted"])
        dataset.timestamps[0]["rgb_device_ts_ns"] = 13_010_000_000
        self.assertTrue(self.summary(dataset)["trusted"])
        dataset.timestamps[0]["rgb_device_ts_ns"] = 12_009_000_000
        self.assertEqual(self.summary(dataset)["tilt_method"], "unavailable")

    def test_fresh_packet_cannot_hide_other_sensor_outage(self):
        for missing in ("accel", "gyro"):
            events = [raw_row(1_000_000_000 + j * 10_000_000, [0, 1, 0]) for j in range(201)]
            for row in events[101:]:
                # Repeated stale sensor data in a batch is deduplicated by actual timestamp.
                row[missing + "_device_ts_ns"] = 2_000_000_000
            dataset = make_dataset(self.root, events, 2_030_000_000)
            self.assertFalse(self.summary(dataset)["trusted"])
            dataset.timestamps[0]["rgb_device_ts_ns"] = 3_000_000_000
            self.assertFalse(self.summary(dataset)["trusted"])

    def test_rotation_vector_gap_is_not_interpolated(self):
        dataset = make_dataset(self.root, [rv_row(1_000_000_000), rv_row(11_000_000_000, 30)], 10_999_000_000)
        self.assertEqual(self.summary(dataset)["tilt_method"], "unavailable")
        self.assertFalse(self.summary(dataset)["trusted"])

    def test_missing_imu_frame_empty_stream_and_none_chip(self):
        for case in ("missing_frame", "empty", "absent", "none"):
            dataset = make_dataset(self.root / case, [rv_row(1_000_000_000)])
            if case == "missing_frame":
                dataset.timestamps[0].update(imu_message_device_ts_ns="", imu_packets="", rgb_imu_delta_ms="")
            elif case == "empty":
                (dataset.root / "imu_events.csv").write_text("message_device_ts_ns\n", encoding="utf-8")
            elif case == "absent":
                (dataset.root / "imu_events.csv").unlink()
            else:
                dataset.metadata["imu"]["type"] = "NONE"
            self.assertEqual(self.summary(dataset)["tilt_method"], "unavailable")
            self.assertFalse(geometry.compute_world_coordinate(dataset, dataset.frame(0), 320, 240, 10000)["trusted"])
            level = geometry.compute_world_coordinate_for_source(dataset, dataset.frame(0), 320, 240, 10000, "gps-course-level")
            self.assertEqual(level["status"], "ok")

    def test_mount_solver_predictions_match_export_for_each_candidate(self):
        dataset = make_dataset(self.root, [rv_row(1_000_000_000, 12, -7)], metadata={"world_coordinates": {
            "camera_mount_rpy_deg": [3, 5, 8], "gps_to_camera_camera_m": [0.2, -0.1, 0.4]}})
        frame = dataset.frame(0)
        geometry.set_camera_mount_rpy(dataset, 3, 11, 17)
        target = geometry.compute_world_coordinate(dataset, frame, 360, 265, 10000)
        geometry.set_camera_mount_rpy(dataset, 3, 5, 8)
        solution = geometry.solve_camera_mount_from_point(dataset, frame, 360, 265, 10000,
                                                        target["latitude_deg"], target["longitude_deg"], target["altitude_m"])
        before = geometry.compute_world_coordinate(dataset, frame, 360, 265, 10000)
        np.testing.assert_allclose(list(solution["predicted_before_enu_m"].values()), list(before["enu_offset_m"].values()), atol=1e-12)
        geometry.set_camera_mount_rpy(dataset, *solution["proposed_rpy_deg"])
        after = geometry.compute_world_coordinate(dataset, frame, 360, 265, 10000)
        np.testing.assert_allclose(list(solution["predicted_after_enu_m"].values()), list(after["enu_offset_m"].values()), atol=1e-12)
        self.assertLess(solution["error_after_m"], 0.002)

    def test_mount_overrides_merge_and_preserve_capture_bytes(self):
        dataset = make_dataset(self.root, [], metadata={"format_version": "raw_events_v1", "world_coordinates": {
            "camera_mount_rpy_deg": [0, 10, 0], "gps_to_camera_enu_m": [1, 2, 3]}})
        original = (self.root / "metadata.json").read_bytes()
        sidecar = self.root / "metadata.overrides.json"
        sidecar.write_text(json.dumps({"world_coordinates": {"camera_mount_rpy_deg": [1, 11, 2]},
                                      "display": {"units": "m"}}), encoding="utf-8")
        dataset = geometry.Dataset(self.root)
        self.assertEqual(dataset.metadata["world_coordinates"]["gps_to_camera_enu_m"], [1, 2, 3])
        self.assertEqual(dataset.metadata["world_coordinates"]["camera_mount_rpy_deg"], [1, 11, 2])
        geometry.set_camera_mount_rpy(dataset, 2, 12, 3, save=True, calibration={"error_after_m": 0.1})
        self.assertEqual((self.root / "metadata.json").read_bytes(), original)
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
        self.assertNotIn("format_version", saved)
        self.assertNotIn("gps_to_camera_enu_m", saved["world_coordinates"])
        self.assertEqual(geometry.Dataset(self.root).metadata, dataset.metadata)
        # The generic save path also writes the sidecar, including unrelated overrides.
        dataset.metadata["display"]["units"] = "cm"
        geometry.save_dataset_metadata(dataset)
        self.assertEqual(geometry.Dataset(self.root).metadata["display"]["units"], "cm")
        self.assertEqual((self.root / "metadata.json").read_bytes(), original)
        self.assertFalse(list(self.root.glob(".metadata.overrides-*.tmp")))

    def test_sidecar_replace_failure_preserves_previous_file(self):
        dataset = make_dataset(self.root, [])
        geometry.set_camera_mount_rpy(dataset, 1, 2, 3, save=True)
        sidecar = self.root / "metadata.overrides.json"
        original = sidecar.read_bytes()
        raw = (self.root / "metadata.json").read_bytes()
        with patch.object(geometry.os, "replace", side_effect=OSError("Replace failed")):
            with self.assertRaisesRegex(OSError, "Replace failed"):
                geometry.set_camera_mount_rpy(dataset, 4, 5, 6, save=True)
        self.assertEqual(sidecar.read_bytes(), original)
        self.assertEqual((self.root / "metadata.json").read_bytes(), raw)
        self.assertFalse(list(self.root.glob(".metadata.overrides-*.tmp")))


if __name__ == "__main__":
    unittest.main()
