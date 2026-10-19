from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from geonova_depthai.postprocess.sync_builder import (
    DEFAULT_GPS_MAX_DELTA_MS,
    SYNCED_IMU_FIELDS,
    TIMESTAMP_FIELDS,
    build_parser,
    build_synced_dataset,
)
from geonova_common.config_cli import load_yaml
from geonova_common.run_contract import validate_capture, write_capture_manifest


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


class SyncSafetyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def make_raw(self, times=(1_000_000_000,)):
        return make_dataset(
            self.root,
            rgb_rows=[image_row(index, "rgb", ts) for index, ts in enumerate(times)],
            depth_rows=[image_row(index, "depth_mm", ts) for index, ts in enumerate(times)],
            imu_rows=[imu_row(index, ts) for index, ts in enumerate(times)],
        )

    def assert_empty_gps(self, row):
        gps_fields = [name for name in TIMESTAMP_FIELDS if name.startswith("gps_")]
        self.assertTrue(gps_fields)
        self.assertTrue(all(row[name] == "" for name in gps_fields))

    def test_gps_inside_and_at_threshold_uses_measurement_time(self):
        dataset = self.make_raw((1_000_000_000, 2_000_000_000, 3_000_000_000))
        write_rows(dataset / "gps.csv", [
            {"sample_index": 1, "measurement_host_monotonic_ns": 1_199_000_000,
             "host_monotonic_ns": 10_000_000_000, "latitude_deg": 37.1},
            {"sample_index": 2, "measurement_host_monotonic_ns": 2_200_000_000,
             "host_monotonic_ns": 11_000_000_000, "latitude_deg": 37.2},
            {"sample_index": 3, "measurement_host_monotonic_ns": 2_800_000_000,
             "host_monotonic_ns": 12_000_000_000, "latitude_deg": 37.3},
        ])

        _, report = build_synced_dataset(dataset)
        rows = read_timestamps(dataset)

        self.assertEqual([row["gps_sample_index"] for row in rows], ["1", "2", "3"])
        self.assertEqual([float(row["gps_frame_delta_ms"]) for row in rows], [199.0, 200.0, -200.0])
        self.assertEqual(report["no_gps_within_threshold"], 0)
        self.assertEqual(report["deltas_ms"]["gps_frame_ms"]["count"], 3)

    def test_gps_outside_threshold_leaves_all_gps_fields_empty(self):
        dataset = self.make_raw()
        write_rows(dataset / "gps.csv", [
            {"sample_index": 1, "host_monotonic_ns": 1_200_000_001, "latitude_deg": 37.1},
        ])

        _, report = build_synced_dataset(dataset)

        self.assertEqual(report["kept_frame_count"], 1)
        self.assertEqual(report["no_gps_within_threshold"], 1)
        self.assertEqual(report["gps_max_delta_ms"], 200.0)
        self.assertEqual(report["deltas_ms"]["gps_frame_ms"]["count"], 0)
        self.assert_empty_gps(read_timestamps(dataset)[0])
        self.assertEqual(json.loads((dataset / "quality_report.json").read_text()), report)

    def test_gps_outage_gap_does_not_attach_stale_samples(self):
        dataset = self.make_raw((1_000_000_000, 1_200_000_000, 2_000_000_000, 3_000_000_000, 3_200_000_000))
        write_rows(dataset / "gps.csv", [
            {"sample_index": 1, "host_monotonic_ns": 1_000_000_000},
            {"sample_index": 2, "host_monotonic_ns": 3_200_000_000},
        ])

        _, report = build_synced_dataset(dataset)
        rows = read_timestamps(dataset)

        self.assertEqual(report["kept_frame_count"], 5)
        self.assertEqual(report["no_gps_within_threshold"], 1)
        self.assertEqual([row["gps_sample_index"] for row in rows], ["1", "1", "", "2", "2"])
        self.assert_empty_gps(rows[2])

    def test_gps_falls_back_to_host_time_for_missing_measurement_time(self):
        dataset = self.make_raw((1_000_000_000, 2_000_000_000))
        write_rows(dataset / "gps.csv", [
            {"sample_index": 1, "host_monotonic_ns": 1_010_000_000},
            {"sample_index": 2, "measurement_host_monotonic_ns": "invalid",
             "host_monotonic_ns": 1_990_000_000},
        ])

        _, report = build_synced_dataset(dataset, gps_max_delta_ms=10.0)

        self.assertEqual(report["no_gps_within_threshold"], 0)
        self.assertEqual([row["gps_sample_index"] for row in read_timestamps(dataset)], ["1", "2"])

    def test_missing_gps_stream_or_frame_host_time_is_reported(self):
        dataset = self.make_raw()
        rgb = image_row(0, "rgb", 1_000_000_000)
        rgb["capture_monotonic_ns"] = ""
        write_rows(dataset / "rgb_events.csv", [rgb])
        write_rows(dataset / "gps.csv", [{"sample_index": 1, "host_monotonic_ns": 1_000_000_000}])

        _, report = build_synced_dataset(dataset)

        self.assertEqual(report["no_gps_within_threshold"], 1)
        self.assert_empty_gps(read_timestamps(dataset)[0])

    def test_external_imu_file_is_not_read_or_copied(self):
        dataset = self.make_raw()
        external = b"\xff\x00not a CSV\xfe"
        (dataset / "external_imu.csv").write_bytes(external)

        output, report = build_synced_dataset(dataset, output_dir=self.root / "synced")

        self.assertEqual(report["kept_frame_count"], 1)
        self.assertFalse((output / "external_imu.csv").exists())
        self.assertEqual((dataset / "external_imu.csv").read_bytes(), external)
        self.assertFalse(any(name.startswith("external_imu_") for name in read_timestamps(output)[0]))
        self.assertNotIn("external_imu", json.dumps(report))

    def test_missing_device_time_has_its_own_drop_count(self):
        dataset = self.make_raw()
        write_rows(dataset / "rgb_events.csv", [
            image_row(0, "rgb", 1_000_000_000),
            image_row(1, "rgb", ""),
            image_row(2, "rgb", "invalid"),
            image_row(3, "rgb", 2_000_000_000),
            image_row(4, "rgb", 3_000_000_000),
        ])
        write_rows(dataset / "depth_events.csv", [
            image_row(0, "depth_mm", 1_000_000_000),
            image_row(4, "depth_mm", 3_000_000_000),
        ])

        _, report = build_synced_dataset(dataset)

        self.assertEqual(report["dropped"], {"no_device_ts": 2, "no_depth": 1, "no_imu": 1})
        self.assertEqual(report["kept_frame_count"], 1)
        self.assertEqual(report["no_gps_within_threshold"], 1)
        self.assertEqual(json.loads((dataset / "quality_report.json").read_text())["dropped"], report["dropped"])
        sidecar = json.loads((dataset / "metadata.sync.json").read_text())
        self.assertEqual(sidecar["sync"]["dropped"], report["dropped"])

    def test_in_place_keeps_all_capture_files_and_metadata_bytes(self):
        dataset = self.make_raw()
        metadata = {
            "format_version": "raw_events_v1",
            "sync": {"mode": "capture", "threshold_ms": 12.0, "custom": [1, 2]},
            "quality_report": {"capture_only": True},
            "imu": {"source": "oak", "type": "BNO086", "imu_to_camera": {"matrix": [[1, 0, 0, 0]]}},
            "custom_capture_field": {"nested": "preserve me"},
        }
        raw_metadata = (json.dumps(metadata, indent=4) + "\n").encode("utf-8")
        (dataset / "metadata.json").write_bytes(raw_metadata)
        for name in ("rgb/0000000.png", "depth_mm/0000000.png"):
            path = dataset / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"capture image")
        write_capture_manifest(dataset, "complete")
        original = {path: path.read_bytes() for path in dataset.rglob("*") if path.is_file()}

        output, report = build_synced_dataset(dataset)

        self.assertEqual(output, dataset)
        for path, content in original.items():
            self.assertEqual(path.read_bytes(), content, str(path))
        self.assertEqual((dataset / "metadata.json").read_bytes(), raw_metadata)
        sidecar = json.loads((dataset / "metadata.sync.json").read_text())
        self.assertEqual(sidecar["sync"]["gps_max_delta_ms"], 200.0)
        self.assertEqual(sidecar["quality_report"], report)
        self.assertEqual(validate_capture(dataset)["state"], "complete")

    def test_existing_output_is_never_overwritten_in_place(self):
        dataset = self.make_raw()
        filenames = ("timestamps.csv", "imu.csv", "metadata.sync.json", "quality_report.json")
        for filename in filenames:
            with self.subTest(filename=filename):
                path = dataset / filename
                original = b"existing capture content\r\n"
                path.write_bytes(original)
                with self.assertRaisesRegex(FileExistsError, "Sync output already exists"):
                    build_synced_dataset(dataset)
                self.assertEqual(path.read_bytes(), original)
                self.assertFalse(any((dataset / other).exists() for other in filenames if other != filename))
                path.unlink()

    def test_continuous_imu_and_rotation_vectors_pass_through_unchanged(self):
        dataset = self.make_raw()
        packet = imu_row(0, 1_000_000_000)
        packet.update({
            "accel_device_ts_ns": "999000000", "gyro_device_ts_ns": "999100000",
            "accel_x_m_s2": "0.125", "gyro_x_rad_s": "0.250",
            "rv_device_ts_ns": "999200000", "rv_i": "0.0010", "rv_j": "-0.0020",
            "rv_k": "0.0030", "rv_real": "0.9999", "rv_accuracy_rad": "0.0100",
        })
        # This packet is outside every frame match but must remain in the raw stream.
        unmatched = imu_row(1, 9_000_000_000)
        write_rows(dataset / "imu_events.csv", [packet, unmatched])
        (dataset / "metadata.json").write_bytes(b'{"format_version":"raw_events_v1", "imu":{"type":"BNO086"}}\n')
        original_imu = (dataset / "imu_events.csv").read_bytes()
        original_metadata = (dataset / "metadata.json").read_bytes()

        separate, _ = build_synced_dataset(dataset, output_dir=self.root / "synced", copy_images=True)
        in_place, _ = build_synced_dataset(dataset)

        for output in (separate, in_place):
            with self.subTest(output=output):
                self.assertEqual((output / "imu_events.csv").read_bytes(), original_imu)
                self.assertEqual((output / "metadata.json").read_bytes(), original_metadata)
                with (output / "imu.csv").open(newline="") as file:
                    reader = csv.DictReader(file)
                    synced_packets = list(reader)
                    self.assertEqual(reader.fieldnames, SYNCED_IMU_FIELDS)
                self.assertEqual(len(synced_packets), 1)
                self.assertEqual(synced_packets[0]["accel_x_m_s2"], "0.125")
                row = read_timestamps(output)[0]
                self.assertEqual(row["imu_message_device_ts_ns"], "1000000000")
                self.assertEqual(float(row["rgb_imu_delta_ms"]), 0.0)
                self.assertEqual(float(row["depth_imu_delta_ms"]), 0.0)
                self.assertEqual(row["imu_packets"], "1")

    def test_gps_default_matches_both_yaml_configs(self):
        root = Path(__file__).resolve().parents[2]
        args = build_parser().parse_args(["--dataset", "raw", "--output-dir", "synced"])
        self.assertEqual(args.gps_max_delta_ms, DEFAULT_GPS_MAX_DELTA_MS)
        self.assertEqual(args.gps_max_delta_ms, 200.0)
        for config in (root / "postprocess/configs/sync.yaml", root / "code/configs/sync.yaml"):
            self.assertEqual(load_yaml(config)["gps_max_delta_ms"], args.gps_max_delta_ms)

    def test_gps_cli_flag_works_through_both_entrypoints_and_overrides_yaml(self):
        dataset = self.make_raw()
        (dataset / "metadata.json").write_text('{"format_version": "raw_events_v1"}', encoding="utf-8")
        for name in ("rgb/0000000.png", "depth_mm/0000000.png"):
            path = dataset / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"capture image")
        write_rows(dataset / "gps.csv", [{"sample_index": 1, "host_monotonic_ns": 1_150_000_000}])
        config = self.root / "sync.yaml"
        config.write_text("gps_max_delta_ms: 275.0\n", encoding="utf-8")
        root = Path(__file__).resolve().parents[2]
        entrypoints = ([root / "postprocess/main.py", "sync"], [root / "code/build_synced_dataset.py"])
        for index, entrypoint in enumerate(entrypoints):
            with self.subTest(entrypoint=entrypoint):
                output = self.root / ("synced" + str(index))
                command = [sys.executable] + [str(part) for part in entrypoint] + [
                    "--dataset", str(dataset), "--output-dir", str(output), "--allow-legacy",
                    "--config", str(config), "--gps-max-delta-ms", "100",
                ]
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assert_empty_gps(read_timestamps(output)[0])
                report = json.loads((output / "quality_report.json").read_text())
                self.assertEqual(report["no_gps_within_threshold"], 1)
                sidecar = json.loads((output / "metadata.sync.json").read_text())
                self.assertEqual(sidecar["sync"]["gps_max_delta_ms"], 100.0)
                self.assertEqual(json.loads((output / "processing_manifest.json").read_text())["state"], "complete")

    def test_invalid_thresholds_fail_before_writing_outputs(self):
        dataset = self.make_raw()
        for setting in ("threshold_ms", "rgb_depth_threshold_ms", "gps_max_delta_ms"):
            for value in (-1.0, float("nan"), float("inf")):
                with self.subTest(setting=setting, value=value):
                    with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                        build_synced_dataset(dataset, **{setting: value})
                    self.assertFalse((dataset / "timestamps.csv").exists())
