from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "convert_lidar_bag_to_pcd.py"
SPEC = importlib.util.spec_from_file_location("geonova_lidar_converter", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = converter
SPEC.loader.exec_module(converter)


def point_fields():
    return [
        SimpleNamespace(name="x", offset=0, datatype=7, count=1),
        SimpleNamespace(name="y", offset=4, datatype=7, count=1),
        SimpleNamespace(name="z", offset=8, datatype=7, count=1),
        SimpleNamespace(name="intensity", offset=12, datatype=7, count=1),
    ]


def test_nearest_timestamp_selects_closest_value() -> None:
    assert converter.nearest_timestamp([100, 200, 400], 260) == 200
    assert converter.nearest_timestamp([100, 200, 400], 350) == 400
    assert converter.nearest_timestamp([], 100) is None


def test_build_field_specs_preserves_requested_order_by_offset() -> None:
    specs = converter.build_field_specs(point_fields(), ["z", "x"])
    assert [spec.msg_name for spec in specs] == ["x", "z"]
    assert [spec.pcd_name for spec in specs] == ["x", "z"]
    assert all(spec.size == 4 and spec.pcd_type == "F" for spec in specs)


def test_pointcloud_data_bytes_removes_row_padding() -> None:
    message = SimpleNamespace(
        width=2,
        height=2,
        point_step=2,
        row_step=6,
        data=b"abcdXXefghYY",
    )
    raw, width, height, point_step = converter.pointcloud_data_bytes(message)
    assert raw == b"abcdefgh"
    assert (width, height, point_step) == (2, 2, 2)


def test_binary_pcd_writer_emits_header_and_payload(tmp_path: Path) -> None:
    specs = converter.build_field_specs(point_fields(), None)
    points = np.zeros(
        2,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("intensity", "<f4"),
        ],
    )
    points[0] = (1.0, 2.0, 3.0, 4.0)
    points[1] = (5.0, 6.0, 7.0, 8.0)

    output = tmp_path / "sample.pcd"
    converter.write_binary_pcd(output, points, specs, width=2, height=1)
    data = output.read_bytes()

    assert b"FIELDS x y z intensity\n" in data
    assert b"WIDTH 2\nHEIGHT 1\n" in data
    marker = b"DATA binary\n"
    payload = data.split(marker, 1)[1]
    assert len(payload) == points.nbytes


def test_drop_invalid_xyz_filters_non_finite_rows() -> None:
    specs = converter.build_field_specs(point_fields(), None)
    points = np.zeros(
        3,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("intensity", "<f4"),
        ],
    )
    points[0] = (1.0, 2.0, 3.0, 10.0)
    points[1] = (np.nan, 2.0, 3.0, 20.0)
    points[2] = (4.0, 5.0, np.inf, 30.0)

    filtered = converter.drop_invalid_xyz(points, specs)
    assert filtered.shape == (1,)
    assert float(filtered[0]["x"]) == 1.0
