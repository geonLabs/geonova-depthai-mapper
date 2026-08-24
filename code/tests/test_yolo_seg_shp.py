#!/usr/bin/env python3
"""YOLO-seg/SHP test CLI plus a synthetic mask-axis self-test."""

from __future__ import annotations

import sys
from pathlib import Path
import csv

import cv2
import numpy as np
import pytest
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geonova_depthai.yolo_seg_shp import (  # noqa: E402
    _feature_depth_mm,
    default_model_path,
    main,
    mask_axis_points,
    mask_depth_observation,
    median_depth_mm,
    run_dataset,
)
from geonova_depthai.fence_linearization import (  # noqa: E402
    CRS_WGS84,
    CRS_WORK,
    LinearizationConfig,
    linearize_fence_points,
)


def test_default_model_path_prefers_newest_local_model(tmp_path: Path) -> None:
    nano_model = tmp_path / "model" / "n_model" / "best.pt"
    nano_model.parent.mkdir(parents=True)
    nano_model.touch()
    assert default_model_path(tmp_path) == nano_model

    field_model = tmp_path / "model" / "x_model" / "best.pt"
    field_model.parent.mkdir(parents=True)
    field_model.touch()
    assert default_model_path(tmp_path) == field_model


def test_mask_axis_points() -> None:
    mask = np.zeros((400, 400), dtype=np.uint8)
    cv2.line(mask, (120, 40), (270, 360), 35, 1)
    points = mask_axis_points(mask)
    assert [point.role for point in points] == ["endpoint_a", "midpoint", "endpoint_b"]
    assert points[0].y < points[1].y < points[2].y
    expected_mid = np.array([(points[0].x + points[2].x) / 2, (points[0].y + points[2].y) / 2])
    actual_mid = np.array([points[1].x, points[1].y])
    assert np.linalg.norm(expected_mid - actual_mid) < 20


def test_mask_depth_observation_rejects_far_depth() -> None:
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[20:60, 20:80] = 1
    depth = np.zeros_like(mask, dtype=np.uint16)
    depth[20:60, 20:80] = 4_000
    depth[25:35, 25:35] = 12_000
    observation = mask_depth_observation(mask, depth, 0.9, max_depth_mm=8_000)
    assert observation.status == "ok"
    assert observation.depth_mm == 4_000
    assert observation.valid_count > 1_000
    assert 0.0 < observation.weight <= 1.0


def test_mask_depth_observation_rejects_far_mask_with_sparse_near_tail() -> None:
    mask = np.ones((80, 100), dtype=np.uint8)
    depth = np.full(mask.shape, 12_000, dtype=np.uint16)
    depth[20:25, 20:30] = 7_997

    observation = mask_depth_observation(
        mask,
        depth,
        0.9,
        max_depth_mm=8_000,
        erosion_px=0,
    )

    assert observation.status == "beyond_max_depth"
    assert observation.depth_mm == 0
    assert observation.valid_count == 50
    assert observation.valid_ratio == 50 / mask.size
    assert observation.weight == 0.0
    assert observation.measured_count == mask.size
    assert observation.measured_median_mm == 12_000


def test_depth_cutoff_keeps_exact_boundary_and_rejects_above_boundary() -> None:
    mask = np.ones((20, 20), dtype=np.uint8)

    at_limit = np.full(mask.shape, 8_000, dtype=np.uint16)
    observation = mask_depth_observation(
        mask, at_limit, max_depth_mm=8_000, erosion_px=0
    )
    assert observation.status == "ok"
    assert observation.depth_mm == 8_000

    above_limit = np.full(mask.shape, 8_001, dtype=np.uint16)
    observation = mask_depth_observation(
        mask, above_limit, max_depth_mm=8_000, erosion_px=0
    )
    assert observation.status == "beyond_max_depth"
    assert observation.depth_mm == 0

    split_boundary = np.empty(mask.shape, dtype=np.uint16)
    split_boundary.flat[: mask.size // 2] = 8_000
    split_boundary.flat[mask.size // 2 :] = 8_001
    observation = mask_depth_observation(
        mask, split_boundary, max_depth_mm=8_000, erosion_px=0
    )
    assert observation.status == "beyond_max_depth"
    assert observation.measured_median_mm == 8_000.5


def test_local_feature_depth_rejects_far_patch_with_sparse_near_tail() -> None:
    depth = np.full((11, 11), 12_000, dtype=np.uint16)
    depth[5, 5] = 7_997
    assert median_depth_mm(depth, 5, 5, radius=5, max_depth_mm=8_000) == (0, 0)

    depth[:] = 4_000
    depth[0, 0] = 12_000
    assert median_depth_mm(depth, 5, 5, radius=5, max_depth_mm=8_000) == (4_000, 120)


def test_out_of_range_detection_suppresses_all_feature_coordinates() -> None:
    depth = np.full((11, 11), 4_000, dtype=np.uint16)
    assert _feature_depth_mm(
        depth, 5, 5, 5, 8_000, "beyond_max_depth"
    ) == (0, 0)
    assert _feature_depth_mm(depth, 5, 5, 5, 8_000, "ok") == (4_000, 121)


def test_depth_observation_rejects_nonpositive_sample_limit() -> None:
    mask = np.ones((5, 5), dtype=np.uint8)
    depth = np.zeros_like(mask, dtype=np.uint16)
    for min_samples in (0, -1):
        try:
            mask_depth_observation(mask, depth, min_samples=min_samples)
        except ValueError as exc:
            assert "min_samples must be positive" in str(exc)
        else:
            raise AssertionError("nonpositive min_samples should be rejected")


def test_depth_range_gate_uses_confidence_qualified_measurements() -> None:
    mask = np.ones((20, 20), dtype=np.uint8)
    depth = np.full(mask.shape, 12_000, dtype=np.uint16)
    depth[5:15, 5:15] = 4_000
    confidence = np.full(mask.shape, 255, dtype=np.uint8)
    confidence[5:15, 5:15] = 0

    observation = mask_depth_observation(
        mask,
        depth,
        confidence_map=confidence,
        max_depth_mm=8_000,
        min_samples=20,
        erosion_px=0,
        confidence_max=200,
    )

    assert observation.status == "ok"
    assert observation.depth_mm == 4_000
    assert observation.measured_count == 100
    assert observation.measured_median_mm == 4_000


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"max_depth_mm": 0}, "max_depth_mm must be positive"),
        ({"fragment_min_samples": 0}, "fragment_min_samples must be positive"),
        ({"fragment_erosion_px": -1}, "fragment_erosion_px must be nonnegative"),
        ({"depth_radius": -1}, "depth_radius must be nonnegative"),
    ],
)
def test_run_dataset_rejects_invalid_depth_settings_before_creating_output(
    tmp_path: Path,
    arguments: dict,
    message: str,
) -> None:
    dataset = tmp_path / "missing-dataset"
    output = tmp_path / "output"

    with pytest.raises(ValueError, match=message):
        run_dataset(dataset, tmp_path / "missing-model.pt", output, **arguments)

    assert not output.exists()


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_fence_linearization_debug_outputs(tmp_path: Path) -> None:
    """Recover one missing role and one lateral-side flip into one fence line."""
    to_wgs84 = Transformer.from_crs(CRS_WORK, CRS_WGS84, always_xy=True)
    origin_x, origin_y = 953_000.0, 1_952_000.0

    trajectory_rows = []
    for index in range(12):
        lon, lat = to_wgs84.transform(origin_x + index, origin_y)
        trajectory_rows.append({
            "frame_index": index,
            "gps_longitude_deg": lon,
            "gps_latitude_deg": lat,
            "gps_position_valid": 1,
            "gps_hdop": 0.7,
            "gps_measurement_host_monotonic_ns": index,
        })
    trajectory_csv = tmp_path / "timestamps.csv"
    _write_rows(trajectory_csv, trajectory_rows)

    point_rows = []
    roles = (("endpoint_a", -1.0), ("midpoint", 0.0), ("endpoint_b", 1.0))
    for detection in range(8):
        fence_y = origin_y - 3.0 if detection == 5 else origin_y + 3.0
        for order, (role, role_z) in enumerate(roles):
            missing = detection == 3 and role == "midpoint"
            lon, lat = to_wgs84.transform(origin_x + detection + 2.0, fence_y)
            point_rows.append({
                "frame_index": 200 + detection,
                "detection_id": 0,
                "point_order": order,
                "role": role,
                "class_id": 0,
                "class_name": "guard_fence",
                "confidence": 0.95,
                "pixel_x": 100,
                "pixel_y": 100 + order,
                "depth_mm": 4000,
                "depth_sample_count": 0 if missing else 50,
                "coordinate_quality": "trusted",
                "orientation_source": "gps-course-level",
                "longitude_deg": "" if missing else lon,
                "latitude_deg": "" if missing else lat,
                "altitude_m": "" if missing else 100.0 + role_z,
                "world_status": "unavailable" if missing else "ok",
                "world_reason": "synthetic missing depth" if missing else "",
            })
    points_csv = tmp_path / "points.csv"
    _write_rows(points_csv, point_rows)

    output_dir = tmp_path / "linearized"
    summary = linearize_fence_points(
        points_csv,
        output_dir,
        trajectory_csv,
        LinearizationConfig(
            gps_smoothing_window=1,
            side_vote_window=7,
            residual_window=5,
            max_gap_connect_m=2.0,
            max_heading_deg=45.0,
            min_line_support=3,
        ),
    )

    assert summary["trajectory_source"] == "gps"
    assert summary["raw_points"] == 24
    assert summary["corrected_points"] == 24
    assert summary["accepted_points"] == 21
    assert summary["lines"] == 1
    assert 3.5 < summary["total_length_m_2d"] < 4.5

    with (output_dir / "points_corrected.csv").open(encoding="utf-8-sig") as file:
        corrected = list(csv.DictReader(file))
    restored = [row for row in corrected if row["frame_index"] == "203" and row["role"] == "midpoint"]
    flipped = [row for row in corrected if row["frame_index"] == "205"]
    assert restored[0]["source_status"] == "interpolated"
    assert restored[0]["correction_type"] == "role_linear"
    assert all(row["accepted"] == "0" for row in flipped)
    assert all("gross_lateral_outlier" in row["correction_type"] for row in flipped)

    for stem in (
        "debug_00_raw",
        "debug_01_projected",
        "debug_02_local",
        "debug_03_side_corrected",
        "debug_04_interpolated",
        "debug_05_filtered",
        "points_corrected",
        "fence_lines",
    ):
        assert (output_dir / f"{stem}.csv").exists()
        assert (output_dir / f"{stem}.shp").exists()
        assert (output_dir / f"{stem}.shx").exists()
        assert (output_dir / f"{stem}.dbf").exists()


def test_representative_is_the_only_modern_mapping_point(tmp_path: Path) -> None:
    to_wgs84 = Transformer.from_crs(CRS_WORK, CRS_WGS84, always_xy=True)
    origin_x, origin_y = 953_000.0, 1_952_000.0
    trajectory = []
    points = []
    for index in range(8):
        lon, lat = to_wgs84.transform(origin_x + index, origin_y)
        trajectory.append({
            "frame_index": 200 + index,
            "gps_longitude_deg": lon,
            "gps_latitude_deg": lat,
            "gps_position_valid": 1,
            "gps_hdop": 0.5,
            "gps_measurement_host_monotonic_ns": index,
        })
    for detection in range(5):
        lon, lat = to_wgs84.transform(origin_x + detection + 1.0, origin_y + 3.0)
        for role, usage in (
            ("representative", "mapping"),
            ("endpoint_a", "feature"),
            ("midpoint", "feature"),
            ("endpoint_b", "feature"),
        ):
            points.append({
                "frame_index": 200 + detection,
                "detection_id": 0,
                "role": role,
                "point_usage": usage,
                "confidence": 0.9,
                "depth_mm": 4_000,
                "depth_sample_count": 80,
                "observation_weight": 0.8 if usage == "mapping" else 0.0,
                "coordinate_quality": "trusted",
                "longitude_deg": lon,
                "latitude_deg": lat,
                "altitude_m": 100.0,
                "world_status": "ok",
            })
    points_csv = tmp_path / "points.csv"
    trajectory_csv = tmp_path / "timestamps.csv"
    _write_rows(points_csv, points)
    _write_rows(trajectory_csv, trajectory)
    summary = linearize_fence_points(
        points_csv,
        tmp_path / "out",
        trajectory_csv,
        LinearizationConfig(gps_smoothing_window=1, min_line_support=3),
    )
    assert summary["raw_points"] == 20
    assert summary["corrected_points"] == 5
    assert summary["accepted_points"] == 5
    assert summary["lines"] == 1


if __name__ == "__main__":
    test_mask_axis_points()
    main()
