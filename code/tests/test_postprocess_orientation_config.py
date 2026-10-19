"""Defaults and output-preserving orchestration regressions (no YOLO runtime)."""

import contextlib
import csv
import io
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
        dataset = Mock(frame_count=201, metadata={})
        dataset.frame.return_value = {"rgb_path": self.root / "rgb.png", "row": {"rgb_file": "rgb.png"}}
        # Change EVERY configurable field so dropping any one is observable.
        changes = {}
        for field in fields(lines.LinearizationConfig):
            value = getattr(lines.LinearizationConfig(), field.name)
            changes[field.name] = ("EPSG:32652" if isinstance(value, str) else value + 1)
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
            yolo.run_dataset(self.root, model_path, output, linearization_config=config)
        self.assertEqual(run.call_args.args[3], config)
        with (output / "points.csv").open(encoding="utf-8-sig", newline="") as file:
            self.assertTrue(all(row["coordinate_quality"] == "approximate" for row in csv.DictReader(file)))


if __name__ == "__main__":
    unittest.main()
