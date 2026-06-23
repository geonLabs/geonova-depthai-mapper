#!/usr/bin/env python3
"""YOLO-seg/SHP test CLI plus a synthetic mask-axis self-test."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geonova_depthai.yolo_seg_shp import main, mask_axis_points  # noqa: E402


def test_mask_axis_points() -> None:
    mask = np.zeros((400, 400), dtype=np.uint8)
    cv2.line(mask, (120, 40), (270, 360), 35, 1)
    points = mask_axis_points(mask)
    assert [point.role for point in points] == ["endpoint_a", "midpoint", "endpoint_b"]
    assert points[0].y < points[1].y < points[2].y
    expected_mid = np.array([(points[0].x + points[2].x) / 2, (points[0].y + points[2].y) / 2])
    actual_mid = np.array([points[1].x, points[1].y])
    assert np.linalg.norm(expected_mid - actual_mid) < 20


if __name__ == "__main__":
    test_mask_axis_points()
    main()
