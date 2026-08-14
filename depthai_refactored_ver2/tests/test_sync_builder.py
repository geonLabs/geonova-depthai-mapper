from __future__ import annotations

import csv
from pathlib import Path

from geonova_depthai.postprocess.sync_builder import build_synced_dataset


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_dataset(tmp_path: Path, rgb_rows, depth_rows, imu_rows) -> Path:
    dataset = tmp_path / "raw"
    dataset.mkdir()
    write_rows(dataset / "rgb_events.csv", rgb_rows)
    write_rows(dataset / "depth_events.csv", depth_rows)
    write_rows(dataset / "imu_events.csv", imu_rows)
    return dataset


def image_row(index: int, stream: str, ts_ns: int) -> dict[str, object]:
    return {
        "event_index": index,
        "file": f"{stream}/{index:07d}.png",
        "sequence": index,
        "device_ts_ns": ts_ns,
        "capture_monotonic_ns": ts_ns,
        "capture_wall_time": "2026-01-01T00:00:00.000",
    }


def imu_row(index: int, ts_ns: int) -> dict[str, object]:
    return {
        "message_index": index,
        "packet_index": 0,
        "message_device_ts_ns": ts_ns,
        "message_sequence": index,
    }


def read_timestamps(dataset: Path) -> list[dict[str, str]]:
    with (dataset / "timestamps.csv").open(newline="") as file:
        return list(csv.DictReader(file))


def test_rgb_rows_are_sorted_by_device_time_before_sync(tmp_path: Path) -> None:
    dataset = make_dataset(
        tmp_path,
        rgb_rows=[
            image_row(2, "rgb", 2_000_000_000),
            image_row(1, "rgb", 1_000_000_000),
        ],
        depth_rows=[
            image_row(1, "depth", 1_000_100_000),
            image_row(2, "depth", 2_000_100_000),
        ],
        imu_rows=[
            imu_row(1, 1_000_000_000),
            imu_row(2, 2_000_000_000),
        ],
    )

    _, report = build_synced_dataset(dataset, threshold_ms=50.0, rgb_depth_threshold_ms=10.0)
    rows = read_timestamps(dataset)

    assert [row["rgb_sequence"] for row in rows] == ["1", "2"]
    assert report["order_inversions"]["rgb_device_ts"] == 1


def test_rgb_depth_threshold_drops_one_frame_depth_match(tmp_path: Path) -> None:
    dataset = make_dataset(
        tmp_path,
        rgb_rows=[image_row(1, "rgb", 1_000_000_000)],
        depth_rows=[image_row(1, "depth", 1_033_000_000)],
        imu_rows=[imu_row(1, 1_000_000_000)],
    )

    _, report = build_synced_dataset(dataset, threshold_ms=50.0, rgb_depth_threshold_ms=10.0)
    rows = read_timestamps(dataset)

    assert rows == []
    assert report["dropped"]["no_depth"] == 1
