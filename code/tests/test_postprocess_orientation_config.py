"""Defaults and output-preserving orchestration regressions (no YOLO runtime)."""

import contextlib
import csv
import io
import json
import sys
import tempfile
import types
import unittest
from dataclasses import asdict, fields, replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from geonova_depthai import fence_linearization as lines
from geonova_depthai import yolo_seg_shp as yolo
from geonova_common.config_cli import load_yaml
from geonova_postprocess import dataset as geometry


class PostprocessConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_cli_defaults_equal_every_yaml_setting(self):
        for name, module in (("yolo", yolo), ("linearization", lines)):
            config_path = ROOT / "postprocess" / "configs" / (name + ".yaml")
            expected = load_yaml(config_path)
            actual = vars(module.parse_args([]))
            for key, value in expected.items():
                if isinstance(actual[key], Path):
                    value = Path(value)
                self.assertEqual(actual[key], value, key)
            self.assertEqual(config_path.read_bytes(), (ROOT / "code" / "configs" / config_path.name).read_bytes())
        self.assertIsNone(yolo.parse_args([]).output_dir)

    def test_automatic_and_standalone_defaults_include_every_setting(self):
        automatic = lines.linearization_config_from_args(yolo.parse_args([]), for_yolo=True)
        standalone = lines.linearization_config_from_args(lines.parse_args([]))
        self.assertEqual(automatic, standalone)
        self.assertEqual(automatic, lines.LinearizationConfig())
        self.assertEqual(automatic.max_gap_connect_m, 10.0)
        self.assertEqual(yolo.parse_args([]).orientation_source, "gps-course")
        self.assertEqual(yolo.parse_args([]).max_frames, 999999)

    def test_only_explicit_orientation_values_override_merged_dataset_metadata(self):
        with (self.root / "timestamps.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["rgb_file", "depth_file"])
            writer.writeheader()
            writer.writerows({"rgb_file": "rgb.png", "depth_file": "depth.png"} for _ in range(201))
        (self.root / "imu.csv").write_text("frame_index\n", encoding="utf-8")
        angles = [10.0, -20.0, 30.0]
        (self.root / "metadata.json").write_text(json.dumps({"world_coordinates": {
            "imu_to_camera_rpy_deg": angles, "imu_to_camera_validated": False,
            "trusted_max_hdop": 1.2, "imu_accel_innovation_deg": 4.0,
            "orientation_source": "gps-course-level"}}), encoding="utf-8")
        (self.root / "metadata.overrides.json").write_text(json.dumps({"world_coordinates": {
            "imu_to_camera_validated": True, "trajectory_max_hdop": 1.7}}), encoding="utf-8")
        cases = [([], {}),
                 (["--imu-to-camera-validated", "1"], {"imu_to_camera_validated": 1.0}),
                 (["--imu-max-delta-ms", "25"], {"imu_max_delta_ms": 25.0}),
                 (["--imu-to-camera-validated", "0"], {"imu_to_camera_validated": 0.0}),
                 (["--imu-to-camera-rpy-deg", "0", "0", "0"], {"imu_to_camera_rpy_deg": [0.0, 0.0, 0.0]}),
                 (["--orientation-source", "gps-course"], {"orientation_source": "gps-course"})]
        for flags, expected in cases:
            with self.subTest(flags=flags):
                dataset = geometry.Dataset(self.root)
                argv = ["--dataset", str(self.root), "--model", str(self.root / "missing.pt")] + flags
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(yolo, "Dataset", return_value=dataset))
                    run = stack.enter_context(patch.object(yolo, "run_dataset", wraps=yolo.run_dataset))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    with self.assertRaisesRegex(ValueError, "Model not found"):
                        yolo.main(argv)
                self.assertEqual(run.call_args.kwargs["orientation_source"], expected.get("orientation_source"))
                overrides = {key: value for key, value in expected.items() if key != "orientation_source"}
                self.assertEqual(dataset.orientation_config, overrides)
                settings = geometry.orientation_settings(dataset)
                self.assertEqual(settings["trusted_max_hdop"], 1.2)
                self.assertEqual(settings["trajectory_max_hdop"], 1.7)
                self.assertEqual(settings["imu_accel_innovation_deg"], 4.0)
                self.assertEqual(settings["imu_to_camera_rpy_deg"], overrides.get("imu_to_camera_rpy_deg", angles))
                rotation, source = geometry.camera_imu_extrinsics(dataset)
                if settings["imu_to_camera_validated"]:
                    self.assertEqual(source, "config_validated")
                    np.testing.assert_allclose(rotation, geometry.rpy_matrix_deg(*settings["imu_to_camera_rpy_deg"]))
                else:
                    self.assertEqual(source, "unavailable")
                    self.assertIsNone(rotation)

    def test_explicit_yaml_orientation_values_and_cli_flags_override_in_order(self):
        config_path = self.root / "orientation.yaml"
        config_path.write_text("imu_to_camera_validated: true\nimu_to_camera_rpy_deg: [10, 20, 30]\n"
                               "imu_accel_innovation_deg: 4\n", encoding="utf-8")
        args = yolo.parse_args(["--config", str(config_path), "--imu-to-camera-validated", "0"])
        self.assertEqual(args.orientation_overrides, {"imu_to_camera_validated": 0.0,
                                                     "imu_to_camera_rpy_deg": [10.0, 20.0, 30.0],
                                                     "imu_accel_innovation_deg": 4.0})
        self.assertEqual(yolo.parse_args([]).orientation_overrides, {})

    def test_config_home_path_expands_for_values_and_provenance(self):
        config_path = self.root / "review_config.yaml"
        config_path.write_text("trajectory_max_hdop: 1.1\nimu_gyro_bias_bound_deg_s: 0.2\n", encoding="utf-8")
        with patch.dict("os.environ", {"HOME": str(self.root), "USERPROFILE": str(self.root)}):
            args = yolo.parse_args(["--config", "~/review_config.yaml"])
        self.assertEqual(args.config, config_path.resolve())
        self.assertEqual(args.orientation_overrides, {"trajectory_max_hdop": 1.1,
                                                     "imu_gyro_bias_bound_deg_s": 0.2})

    def test_orientation_help_explains_precedence(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
            yolo.parse_args(["--help"])
        self.assertEqual(stopped.exception.code, 0)
        help_text = " ".join(output.getvalue().split())
        self.assertIn("packaged YAML defaults, dataset metadata.json merged with metadata.overrides.json", help_text)
        self.assertIn("explicit --config values, explicit CLI flags", help_text)

    def test_yaml_and_cli_override_all_linearization_fields(self):
        values = asdict(lines.LinearizationConfig())
        values.update(side_window_m=31.0, max_frame_gap=81, spline_smoothing=0.22,
                      trajectory_max_hdop=1.8, max_gap_connect_m=14.0)
        config_path = self.root / "yolo.yaml"
        config_path.write_text("\n".join(key + ": " + str(value) for key, value in values.items()
                                         if key != "depth_sample_min") +
                               "\nlinearization_depth_sample_min: 33\n", encoding="utf-8")
        args = yolo.parse_args(["--config", str(config_path), "--max-frame-gap", "91"])
        expected = lines.LinearizationConfig(**dict(values, depth_sample_min=33, max_frame_gap=91))
        self.assertEqual(lines.linearization_config_from_args(args, True), expected)
        with patch.object(yolo, "run_dataset") as run:
            yolo.main(["--config", str(config_path), "--max-frame-gap", "91"])
        self.assertEqual(run.call_args.kwargs["linearization_config"], expected)
        self.assertEqual(run.call_args.kwargs["orientation_config"]["trajectory_max_hdop"], 1.8)

    def _outputs(self):
        output = self.root / "output"
        for name in ("detected_plot", "detected_rgb", "overlays"):
            path = output / name
            path.mkdir(parents=True, exist_ok=True)
            (path / "keep.txt").write_text("original", encoding="utf-8")
        return output

    def _assert_outputs_untouched(self, output):
        for name in ("detected_plot", "detected_rgb", "overlays"):
            self.assertEqual((output / name / "keep.txt").read_text(encoding="utf-8"), "original")

    def test_invalid_arguments_leave_all_existing_outputs_untouched(self):
        output = self._outputs()
        invalid = ({"start_frame": 199}, {"max_frames": 0}, {"stride": 0}, {"batch_size": 0},
                   {"image_size": 0}, {"confidence": float("nan")}, {"max_depth_mm": 0},
                   {"fragment_min_samples": 0}, {"fragment_erosion_px": -1}, {"depth_radius": -1},
                   {"depth_confidence_max": 256}, {"orientation_source": "invalid"}, {"classes": [-1]},
                   {"max_frames": float("inf")}, {"orientation_config": {"imu_max_delta_ms": -1}},
                   {"linearization_config": replace(lines.LinearizationConfig(), side_window_m=-1)},
                   {"linearization_config": replace(lines.LinearizationConfig(), gps_smoothing_window=4)},
                   {"linearization_config": replace(lines.LinearizationConfig(), gps_savgol_polyorder=11)})
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                yolo.run_dataset(self.root / "missing", self.root / "missing.pt", output, **arguments)
            self._assert_outputs_untouched(output)

    def test_placeholder_output_is_rejected_before_existing_outputs_are_cleared(self):
        output = self.root / "REPLACE_RUN_ID" / "yolo"
        for name in ("detected_plot", "detected_rgb", "overlays"):
            path = output / name
            path.mkdir(parents=True, exist_ok=True)
            (path / "keep.txt").write_text("original", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unresolved placeholder"):
            yolo.run_dataset(
                self.root / "missing",
                self.root / "missing.pt",
                self.root / "REPLACE_RUN_ID" / "yolo",
            )
        for name in ("detected_plot", "detected_rgb", "overlays"):
            self.assertEqual((output / name / "keep.txt").read_text(encoding="utf-8"), "original")

    def test_out_of_bounds_and_bad_model_leave_outputs_untouched(self):
        output = self._outputs()
        dataset = Mock(frame_count=201, metadata={})
        model_path = self.root / "model.pt"
        model_path.touch()
        module = types.ModuleType("ultralytics")
        module.YOLO = Mock(return_value=Mock(task="detect"))
        with patch.object(yolo, "Dataset", return_value=dataset), patch.dict(sys.modules, {"ultralytics": module}):
            with self.assertRaisesRegex(ValueError, "exceeds"):
                yolo.run_dataset(self.root, model_path, output, start_frame=201)
            self._assert_outputs_untouched(output)
            with self.assertRaisesRegex(ValueError, "segmentation model"):
                yolo.run_dataset(self.root, model_path, output)
            self._assert_outputs_untouched(output)

    def test_automatic_run_passes_full_config_and_uses_combined_trust(self):
        output = self.root / "output"
        model_path = self.root / "model.pt"
        model_path.touch()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.ones((32, 32), dtype=np.uint8)
        box = types.SimpleNamespace(cls=np.array(0), conf=np.array(0.9))
        result = types.SimpleNamespace(boxes=[box], masks=object(), plot=lambda: image)
        model = types.SimpleNamespace(task="segment", names={0: "fence"}, predict=lambda **kwargs: [result])
        module = types.ModuleType("ultralytics")
        module.YOLO = lambda _: model
        dataset = Mock(frame_count=201, metadata={"world_coordinates": {"orientation_source": "gps-course-level"}})
        dataset.frame.return_value = {"rgb_path": self.root / "rgb.png", "row": {"rgb_file": "rgb.png"}}
        # Change EVERY configurable field so dropping any one is observable.
        changes = {}
        for field in fields(lines.LinearizationConfig):
            value = getattr(lines.LinearizationConfig(), field.name)
            # The smoothing window must stay odd, so it moves by two.
            step = 2 if field.name == "gps_smoothing_window" else 1
            changes[field.name] = ("EPSG:32652" if isinstance(value, str) else value + step)
        config = lines.LinearizationConfig(**changes)
        world = {"status": "ok", "trusted": False, "gps": {"position_quality": {"trusted": True}},
                 "longitude_deg": 127, "latitude_deg": 37, "altitude_m": 100}
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"ultralytics": module}))
            stack.enter_context(patch.object(yolo, "Dataset", return_value=dataset))
            stack.enter_context(patch.object(yolo, "_binary_masks", return_value=[(0, mask)]))
            stack.enter_context(patch.object(yolo, "get_depth_frame", return_value=np.full((32, 32), 4000, dtype=np.uint16)))
            stack.enter_context(patch.object(yolo, "_confidence_frame", return_value=None))
            stack.enter_context(patch.object(yolo, "compute_world_coordinate_for_source", return_value=world))
            stack.enter_context(patch.object(yolo, "_prepare_writer", return_value=Mock()))
            stack.enter_context(patch.object(yolo, "_write_prj"))
            stack.enter_context(patch.object(yolo.cv2, "imread", return_value=image))
            stack.enter_context(patch.object(yolo.cv2, "imwrite", return_value=True))
            run = stack.enter_context(patch.object(yolo, "linearize_fence_points", return_value={}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            summary = yolo.run_dataset(self.root, model_path, output, linearization_config=config)
        self.assertEqual(run.call_args.args[3], config)
        self.assertEqual(summary["orientation_source"], "gps-course-level")
        with (output / "points.csv").open(encoding="utf-8-sig", newline="") as file:
            self.assertTrue(all(row["coordinate_quality"] == "approximate" for row in csv.DictReader(file)))

    def test_main_resolves_shared_hdop_for_orientation_and_automatic_linearization(self):
        with (self.root / "timestamps.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["rgb_file", "depth_file"])
            writer.writeheader()
            writer.writerows({"rgb_file": "rgb.png", "depth_file": "depth.png"} for _ in range(201))
        (self.root / "imu.csv").write_text("frame_index\n", encoding="utf-8")
        (self.root / "metadata.json").write_text(json.dumps({"world_coordinates": {
            "trajectory_max_hdop": 1.5}}), encoding="utf-8")
        (self.root / "metadata.overrides.json").write_text(json.dumps({"world_coordinates": {
            "trajectory_max_hdop": 1.0}}), encoding="utf-8")
        config_path = self.root / "review.yaml"
        config_path.write_text("trajectory_max_hdop: 1.2\n", encoding="utf-8")
        model_path = self.root / "model.pt"
        model_path.touch()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        result = types.SimpleNamespace(boxes=[types.SimpleNamespace(cls=np.array(0), conf=np.array(0.9))],
                                       masks=object(), plot=lambda: image)
        module = types.ModuleType("ultralytics")
        module.YOLO = lambda _: types.SimpleNamespace(task="segment", names={0: "fence"},
                                                     predict=lambda **kwargs: [result])
        world = {"status": "ok", "trusted": False, "longitude_deg": 127, "latitude_deg": 37, "altitude_m": 100}
        cases = [(None, 1.0), ([], 1.0), (["--config", str(config_path)], 1.2),
                 (["--trajectory-max-hdop", "2.5"], 2.5),
                 (["--config", str(config_path), "--trajectory-max-hdop", "1.8"], 1.8)]
        for flags, expected in cases:
            with self.subTest(flags=flags), contextlib.ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules, {"ultralytics": module}))
                stack.enter_context(patch.object(yolo, "_binary_masks", return_value=[(0, np.ones((32, 32), dtype=np.uint8))]))
                stack.enter_context(patch.object(yolo, "get_depth_frame", return_value=np.full((32, 32), 4000, dtype=np.uint16)))
                stack.enter_context(patch.object(yolo, "_confidence_frame", return_value=None))
                coordinates = stack.enter_context(patch.object(yolo, "compute_world_coordinate_for_source", return_value=world))
                stack.enter_context(patch.object(yolo, "_prepare_writer", return_value=Mock()))
                stack.enter_context(patch.object(yolo, "_write_prj"))
                stack.enter_context(patch.object(yolo.cv2, "imread", return_value=image))
                stack.enter_context(patch.object(yolo.cv2, "imwrite", return_value=True))
                linearize = stack.enter_context(patch.object(yolo, "linearize_fence_points", return_value={}))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                if flags is None:
                    yolo.run_dataset(self.root, model_path, None, linearization_config=lines.LinearizationConfig())
                else:
                    yolo.main(["--dataset", str(self.root), "--model", str(model_path)] + flags)
                dataset = coordinates.call_args.args[0]
                self.assertEqual(geometry.orientation_settings(dataset)["trajectory_max_hdop"], expected)
                self.assertEqual(linearize.call_args.args[3].trajectory_max_hdop, expected)


if __name__ == "__main__":
    unittest.main()
