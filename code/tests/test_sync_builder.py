from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from geonova_depthai.postprocess import sync_builder
from geonova_depthai.postprocess.sync_builder import (
    DEFAULT_GPS_MAX_DELTA_MS,
    SYNCED_IMU_FIELDS,
    TIMESTAMP_FIELDS,
    build_parser,
    build_synced_dataset,
    SYNC_OUTPUT_FILES,
    validate_sync_capture,
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
    (dataset / "metadata.json").write_text('{"format_version": "raw_events_v1"}\n', encoding="utf-8")
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
        rgb_rows=[image_row(1, "rgb", 1_000_000_000), image_row(2, "rgb", 2_000_000_000)],
        depth_rows=[image_row(1, "depth", 1_033_000_000), image_row(2, "depth", 2_000_000_000)],
        imu_rows=[imu_row(1, 1_000_000_000)],
    )

    _, report = build_synced_dataset(dataset, threshold_ms=50.0, rgb_depth_threshold_ms=10.0)
    rows = read_timestamps(dataset)

    assert [row["rgb_sequence"] for row in rows] == ["2"]
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

    def add_capture_images(self, dataset):
        for manifest in ("rgb_events.csv", "depth_events.csv"):
            with (dataset / manifest).open(newline="") as file:
                for row in csv.DictReader(file):
                    path = dataset / row["file"]
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(b"capture image")

    def assert_empty_imu(self, row):
        for name in ("imu_message_device_ts_ns", "rgb_imu_delta_ms", "depth_imu_delta_ms", "imu_packets"):
            self.assertEqual(row[name], "", name)

    def test_gps_inside_and_at_threshold_uses_measurement_time(self):
        dataset = self.make_raw((1_000_000_000, 2_000_000_000, 3_000_000_000))
        write_rows(dataset / "gps.csv", [
            {"sample_index": 1, "measurement_host_monotonic_ns": 1_149_000_000,
             "host_monotonic_ns": 10_000_000_000, "latitude_deg": 37.1},
            {"sample_index": 2, "measurement_host_monotonic_ns": 2_150_000_000,
             "host_monotonic_ns": 11_000_000_000, "latitude_deg": 37.2},
            {"sample_index": 3, "measurement_host_monotonic_ns": 2_850_000_000,
             "host_monotonic_ns": 12_000_000_000, "latitude_deg": 37.3},
        ])

        _, report = build_synced_dataset(dataset)
        rows = read_timestamps(dataset)

        self.assertEqual([row["gps_sample_index"] for row in rows], ["1", "2", "3"])
        self.assertEqual([float(row["gps_frame_delta_ms"]) for row in rows], [149.0, 150.0, -150.0])
        self.assertEqual(report["no_gps_within_threshold"], 0)
        self.assertEqual(report["deltas_ms"]["gps_frame_ms"]["count"], 3)

    def test_gps_outside_threshold_leaves_all_gps_fields_empty(self):
        dataset = self.make_raw()
        write_rows(dataset / "gps.csv", [
            {"sample_index": 1, "host_monotonic_ns": 1_150_000_001, "latitude_deg": 37.1},
        ])

        _, report = build_synced_dataset(dataset)

        self.assertEqual(report["kept_frame_count"], 1)
        self.assertEqual(report["no_gps_within_threshold"], 1)
        self.assertEqual(report["gps_max_delta_ms"], 150.0)
        self.assertEqual(report["deltas_ms"]["gps_frame_ms"]["count"], 0)
        self.assert_empty_gps(read_timestamps(dataset)[0])
        self.assertEqual(json.loads((dataset / "quality_report.json").read_text()), report)

    def test_gps_outage_gap_does_not_attach_stale_samples(self):
        dataset = self.make_raw((1_000_000_000, 1_150_000_000, 2_000_000_000, 3_050_000_000, 3_200_000_000))
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

        self.assertEqual(report["dropped"], {"no_device_ts": 2, "no_depth": 1})
        self.assertEqual(report["kept_frame_count"], 2)
        self.assertEqual(report["no_imu_match"], 1)
        self.assertEqual(report["no_gps_within_threshold"], 2)
        self.assert_empty_imu(read_timestamps(dataset)[1])
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
        self.add_capture_images(dataset)
        write_rows(dataset / "gps.csv", [{"sample_index": 1, "host_monotonic_ns": 1_100_000_000}])
        (dataset / "metadata.overrides.json").write_bytes(b'{"world_coordinates": {"mount_height_m": 2.1}}\n')
        (dataset / "external_imu.csv").write_bytes(b"untouched legacy stream\n")
        write_capture_manifest(dataset, "complete")
        original = {path: path.read_bytes() for path in dataset.rglob("*") if path.is_file()}

        output, report = build_synced_dataset(dataset)
        first_outputs = {name: (dataset / name).read_bytes() for name in SYNC_OUTPUT_FILES}
        replacements = []
        replace = sync_builder.os.replace

        def record_replace(source, destination):
            self.assertEqual(source.parent, dataset)
            self.assertTrue(source.name.startswith("." + destination.name + "."))
            self.assertTrue((dataset / "sync.incomplete").is_file())
            for name, content in first_outputs.items():
                self.assertEqual((dataset / (name + ".prev")).read_bytes(), content)
            self.assertEqual(destination.read_bytes(), first_outputs[destination.name])
            staged_bytes = source.read_bytes()
            replace(source, destination)
            self.assertEqual(destination.read_bytes(), staged_bytes)
            replacements.append(destination.name)

        with mock.patch.object(sync_builder.os, "replace", side_effect=record_replace):
            output, report = build_synced_dataset(dataset, gps_max_delta_ms=75.0)

        self.assertEqual(output, dataset)
        self.assertEqual(replacements, list(SYNC_OUTPUT_FILES))
        self.assertNotEqual((dataset / "timestamps.csv").read_bytes(), first_outputs["timestamps.csv"])
        self.assert_empty_gps(read_timestamps(dataset)[0])
        for path, content in original.items():
            self.assertEqual(path.read_bytes(), content, str(path))
        self.assertEqual((dataset / "metadata.json").read_bytes(), raw_metadata)
        sidecar = json.loads((dataset / "metadata.sync.json").read_text())
        self.assertEqual(sidecar["sync"]["gps_max_delta_ms"], 75.0)
        self.assertEqual(sidecar["quality_report"], report)
        self.assertEqual(validate_capture(dataset)["state"], "complete")
        self.assertEqual({path for path in dataset.rglob("*") if path.is_file()},
                         set(original) | {dataset / name for name in SYNC_OUTPUT_FILES})

    def test_existing_outputs_require_recognisable_raw_metadata(self):
        dataset = self.make_raw()
        metadata_path = dataset / "metadata.json"
        for metadata in (None, b"{}", b"[]", b"null", b"broken JSON", b"\xff",
                         b'{"format_version":"synced_capture_v1"}',
                         b'{"format_version":"unknown", "sync":{"mode":"postprocess"}}'):
            for filename in SYNC_OUTPUT_FILES:
                with self.subTest(metadata=metadata, filename=filename):
                    if metadata is None:
                        metadata_path.unlink(missing_ok=True)
                    else:
                        metadata_path.write_bytes(metadata)
                    path = dataset / filename
                    path.write_bytes(b"existing capture content\r\n")
                    original = {p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()}
                    with self.assertRaisesRegex(FileExistsError, "Refusing to replace.*raw_events_v1"):
                        build_synced_dataset(dataset)
                    self.assertEqual({p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()}, original)
                    path.unlink()

    def test_partial_previous_sync_can_be_replaced(self):
        dataset = self.make_raw()
        (dataset / "timestamps.csv").write_bytes(b"interrupted old sync\n")

        _, report = build_synced_dataset(dataset)

        self.assertEqual(len(read_timestamps(dataset)), 1)
        self.assertEqual(report["kept_frame_count"], 1)
        self.assertTrue(all((dataset / name).is_file() for name in SYNC_OUTPUT_FILES))

    def test_interruption_before_first_replace_keeps_all_previous_outputs(self):
        dataset = self.make_raw()
        build_synced_dataset(dataset)
        original = {path: path.read_bytes() for path in dataset.rglob("*") if path.is_file()}

        with mock.patch.object(sync_builder.os, "replace", side_effect=OSError("Interrupted before replace")) as replace:
            with self.assertRaisesRegex(OSError, "Interrupted before replace"):
                build_synced_dataset(dataset, gps_max_delta_ms=25.0)

        replace.assert_called_once()
        self.assertEqual({path: path.read_bytes() for path in dataset.rglob("*") if path.is_file()}, original)

    def test_interrupted_serialization_keeps_all_previous_outputs(self):
        dataset = self.make_raw()
        build_synced_dataset(dataset)
        original = {path: path.read_bytes() for path in dataset.rglob("*") if path.is_file()}
        write_csv = sync_builder.write_csv
        dump = json.dump

        for filename in SYNC_OUTPUT_FILES:
            with self.subTest(filename=filename):
                def interrupt_csv(file, *args):
                    if Path(file.name).name.startswith("." + filename + "."):
                        file.write("partial CSV\n")
                        raise OSError("Interrupted write")
                    return write_csv(file, *args)

                def interrupt_json(payload, file, **kwargs):
                    if Path(file.name).name.startswith("." + filename + "."):
                        file.write('{"partial":')
                        raise OSError("Interrupted write")
                    return dump(payload, file, **kwargs)

                with mock.patch.object(sync_builder, "write_csv", side_effect=interrupt_csv):
                    with mock.patch.object(json, "dump", side_effect=interrupt_json):
                        with mock.patch.object(sync_builder.os, "replace") as replace:
                            with self.assertRaisesRegex(OSError, "Interrupted write"):
                                build_synced_dataset(dataset, gps_max_delta_ms=25.0)
                replace.assert_not_called()
                self.assertEqual({p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()}, original)

    def test_each_failed_swap_restores_the_previous_generation(self):
        dataset = self.make_raw()
        build_synced_dataset(dataset)
        first_outputs = {name: (dataset / name).read_bytes() for name in SYNC_OUTPUT_FILES}
        replace = sync_builder.os.replace

        for previous_count in (0, 1, 4):
            for index, name in enumerate(SYNC_OUTPUT_FILES):
                if index < previous_count:
                    (dataset / name).write_bytes(first_outputs[name])
                else:
                    (dataset / name).unlink(missing_ok=True)
            original = {p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()}
            for failed_name in SYNC_OUTPUT_FILES:
                with self.subTest(previous_count=previous_count, failed_name=failed_name):
                    def fail_swap(source, destination):
                        self.assertTrue((dataset / "sync.incomplete").is_file())
                        for name in SYNC_OUTPUT_FILES[:previous_count]:
                            self.assertEqual((dataset / (name + ".prev")).read_bytes(), first_outputs[name])
                        if destination.name == failed_name:
                            raise OSError("Swap failed")
                        replace(source, destination)

                    with mock.patch.object(sync_builder.os, "replace", side_effect=fail_swap):
                        with self.assertRaisesRegex(OSError, "Swap failed"):
                            build_synced_dataset(dataset, gps_max_delta_ms=25.0)
                    self.assertEqual({p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()}, original)

    def test_failed_backup_keeps_the_previous_generation(self):
        dataset = self.make_raw()
        build_synced_dataset(dataset)
        original = {p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()}
        copy = sync_builder.shutil.copy2

        def fail_backup(source, destination):
            if source.name == "imu.csv":
                destination.write_bytes(b"partial backup")
                raise OSError("Backup failed")
            return copy(source, destination)

        with mock.patch.object(sync_builder.shutil, "copy2", side_effect=fail_backup):
            with mock.patch.object(sync_builder.os, "replace") as replace:
                with self.assertRaisesRegex(OSError, "Backup failed"):
                    build_synced_dataset(dataset)
        replace.assert_not_called()
        self.assertEqual({p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()}, original)

    def test_crash_after_each_swap_recovers_before_the_next_sync(self):
        dataset = self.make_raw()
        write_rows(dataset / "depth_events.csv", [image_row(0, "depth_mm", 1_010_000_000)])
        build_synced_dataset(dataset)
        original = {p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()}
        root = Path(__file__).resolve().parents[2]
        script = """import os
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from geonova_depthai.postprocess import sync_builder
replace = sync_builder.os.replace
swaps = 0
def crash(source, destination):
    global swaps
    replace(source, destination)
    swaps += 1
    if swaps == int(sys.argv[3]):
        os._exit(87)
sync_builder.os.replace = crash
sync_builder.build_synced_dataset(Path(sys.argv[2]), gps_max_delta_ms=25.0)
"""
        for swap in range(1, 5):
            with self.subTest(swap=swap):
                result = subprocess.run([
                    sys.executable, "-c", script, str(root / "code"), str(dataset), str(swap),
                ], capture_output=True, text=True)
                self.assertEqual(result.returncode, 87, result.stdout + result.stderr)
                self.assertTrue((dataset / "sync.incomplete").is_file())
                for name in SYNC_OUTPUT_FILES:
                    self.assertEqual((dataset / (name + ".prev")).read_bytes(), original[dataset / name])

                # A failing next sync still recovers the old generation first.
                with self.assertRaisesRegex(ValueError, "No synchronized frames survived"):
                    build_synced_dataset(dataset, rgb_depth_threshold_ms=0.0)
                self.assertEqual({p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()}, original)

        _, report = build_synced_dataset(dataset, gps_max_delta_ms=25.0)
        self.assertEqual(report["gps_max_delta_ms"], 25.0)
        self.assertFalse((dataset / "sync.incomplete").exists())

    def test_incomplete_crash_backups_clear_all_outputs_before_rebuilding(self):
        dataset = self.make_raw()
        raw = {p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()}
        write_outputs = sync_builder.write_sync_outputs
        for backup_count in range(4):
            with self.subTest(backup_count=backup_count):
                (dataset / "sync.incomplete").write_text("interrupted\n")
                for index, name in enumerate(SYNC_OUTPUT_FILES):
                    (dataset / name).write_bytes(b"mixed generation\n")
                    if index < backup_count:
                        (dataset / (name + ".prev")).write_bytes(b"old output\n")
                (dataset / ".timestamps.csv.crashed.tmp").write_bytes(b"partial backup\n")

                def check_recovered(*args):
                    self.assertEqual({p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()}, raw)
                    return write_outputs(*args)

                with mock.patch.object(sync_builder, "write_sync_outputs", side_effect=check_recovered):
                    _, report = build_synced_dataset(dataset)
                self.assertEqual(report["kept_frame_count"], 1)
                self.assertTrue(all((dataset / name).is_file() for name in SYNC_OUTPUT_FILES))

    def test_zero_frame_sync_leaves_previous_outputs_byte_identical(self):
        dataset = self.make_raw()
        write_rows(dataset / "depth_events.csv", [image_row(0, "depth_mm", 1_010_000_000)])
        build_synced_dataset(dataset)
        original = {p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()}

        with self.assertRaisesRegex(ValueError, "kept=0, rgb=1, depth=1, no_device_ts=0, no_depth=1"):
            build_synced_dataset(dataset, rgb_depth_threshold_ms=0.0)

        self.assertEqual({p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()}, original)
        output = self.root / "empty"
        with self.assertRaisesRegex(ValueError, "No synchronized frames survived"):
            build_synced_dataset(dataset, output_dir=output, rgb_depth_threshold_ms=0.0)
        self.assertEqual(list(output.iterdir()), [])

    def test_zero_frame_counts_include_rgb_rows_without_device_timestamps(self):
        dataset = self.make_raw()
        write_rows(dataset / "rgb_events.csv", [
            image_row(0, "rgb", ""), image_row(1, "rgb", 2_000_000_000),
        ])
        original = {p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()}
        with self.assertRaisesRegex(ValueError, "kept=0, rgb=2, depth=1, no_device_ts=1, no_depth=1"):
            build_synced_dataset(dataset)
        self.assertEqual({p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()}, original)

    def test_empty_imu_stream_keeps_frames_with_empty_imu_fields(self):
        dataset = self.make_raw((1_000_000_000, 2_000_000_000))
        metadata_path = dataset / "metadata.json"
        write_rows(dataset / "gps.csv", [
            {"sample_index": 1, "host_monotonic_ns": 1_000_000_000, "latitude_deg": 37.1},
            {"sample_index": 2, "host_monotonic_ns": 2_000_000_000, "latitude_deg": 37.2},
        ])
        for imu_type, content in (("NONE", b""), ("unknown", b"message_index,message_device_ts_ns\r\n")):
            with self.subTest(imu_type=imu_type):
                metadata_path.write_text(json.dumps({"format_version": "raw_events_v1", "imu": {"type": imu_type}}),
                                         encoding="utf-8")
                (dataset / "imu_events.csv").write_bytes(content)
                metadata_bytes = metadata_path.read_bytes()

                _, report = build_synced_dataset(dataset)

                rows = read_timestamps(dataset)
                self.assertEqual([row["frame_index"] for row in rows], ["0", "1"])
                self.assertEqual([row["gps_sample_index"] for row in rows], ["1", "2"])
                for row in rows:
                    self.assert_empty_imu(row)
                with (dataset / "imu.csv").open(newline="") as file:
                    reader = csv.DictReader(file)
                    self.assertEqual(reader.fieldnames, SYNCED_IMU_FIELDS)
                    self.assertEqual(list(reader), [])
                self.assertEqual(report["kept_frame_count"], 2)
                self.assertEqual(report["no_imu_match"], 2)
                self.assertEqual(report["dropped"], {"no_device_ts": 0, "no_depth": 0})
                self.assertEqual(report["deltas_ms"]["rgb_imu_ms"]["count"], 0)
                self.assertEqual(report["no_gps_within_threshold"], 0)
                self.assertEqual((dataset / "imu_events.csv").read_bytes(), content)
                self.assertEqual(metadata_path.read_bytes(), metadata_bytes)

    def test_imu_outside_threshold_is_kept_without_stale_imu_fields(self):
        dataset = self.make_raw((1_000_000_000, 2_000_000_000))
        write_rows(dataset / "imu_events.csv", [imu_row(1, 1_050_000_000), imu_row(2, 2_050_000_001)])

        _, report = build_synced_dataset(dataset)

        rows = read_timestamps(dataset)
        self.assertEqual(rows[0]["imu_packets"], "1")
        self.assertEqual(float(rows[0]["rgb_imu_delta_ms"]), 50.0)
        self.assert_empty_imu(rows[1])
        self.assertEqual(report["kept_frame_count"], 2)
        self.assertEqual(report["no_imu_match"], 1)
        self.assertEqual(report["deltas_ms"]["rgb_imu_ms"]["count"], 1)

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
        args = build_parser().parse_args(["--dataset", "raw"])
        self.assertEqual(args.gps_max_delta_ms, DEFAULT_GPS_MAX_DELTA_MS)
        self.assertEqual(args.gps_max_delta_ms, 150.0)
        self.assertEqual(args.output_dir, "")
        for config in (root / "postprocess/configs/sync.yaml", root / "code/configs/sync.yaml"):
            self.assertEqual(load_yaml(config)["gps_max_delta_ms"], args.gps_max_delta_ms)
            self.assertEqual(load_yaml(config)["output_dir"], args.output_dir)
        self.assertEqual((root / "postprocess/configs/sync.yaml").read_bytes(),
                         (root / "code/configs/sync.yaml").read_bytes())

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

    def test_both_entrypoints_sync_and_resync_in_place_by_default(self):
        dataset = self.make_raw()
        self.add_capture_images(dataset)
        (dataset / "imu_events.csv").write_bytes(b"message_index,message_device_ts_ns\r\n")
        write_capture_manifest(dataset, "complete")
        original = {path: path.read_bytes() for path in dataset.rglob("*") if path.is_file()}
        config = self.root / "sync.yaml"
        config.write_text('dataset: "raw"\noutput_dir: ""\n', encoding="utf-8")
        root = Path(__file__).resolve().parents[2]
        entrypoints = ([root / "postprocess/main.py", "sync"], [root / "code/build_synced_dataset.py"])
        for entrypoint in entrypoints:
            for arguments in (["--dataset", str(dataset)], ["--config", str(config)]):
                with self.subTest(entrypoint=entrypoint, arguments=arguments):
                    command = [sys.executable] + [str(part) for part in entrypoint] + arguments
                    result = subprocess.run(command, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(len(read_timestamps(dataset)), 1)
                    self.assert_empty_imu(read_timestamps(dataset)[0])
                    report = json.loads((dataset / "quality_report.json").read_text())
                    self.assertEqual(report["no_imu_match"], 1)
                    for path, content in original.items():
                        self.assertEqual(path.read_bytes(), content, str(path))
                    self.assertEqual(validate_capture(dataset)["state"], "complete")
                    self.assertEqual({p for p in dataset.rglob("*") if p.is_file()},
                                     set(original) | {dataset / name for name in SYNC_OUTPUT_FILES})

    def test_legacy_synced_format_resyncs_through_both_entrypoints(self):
        dataset = self.make_raw()
        self.add_capture_images(dataset)
        (dataset / "metadata.json").write_bytes(
            b'{ "format_version": "synced_from_raw_events_v1", "sync": {"old": true} }\n'
        )
        original = {path: path.read_bytes() for path in dataset.rglob("*") if path.is_file()}
        for name in SYNC_OUTPUT_FILES:
            (dataset / name).write_bytes(b"old sync output\n")
        root = Path(__file__).resolve().parents[2]
        entrypoints = ([root / "postprocess/main.py", "sync"], [root / "code/build_synced_dataset.py"])
        for entrypoint in entrypoints:
            with self.subTest(entrypoint=entrypoint):
                command = [sys.executable] + [str(part) for part in entrypoint] + [
                    "--dataset", str(dataset), "--allow-legacy",
                ]
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(len(read_timestamps(dataset)), 1)
                for path, content in original.items():
                    self.assertEqual(path.read_bytes(), content, str(path))
                self.assertEqual(json.loads((dataset / "quality_report.json").read_text())["kept_frame_count"], 1)

    def test_both_entrypoints_recover_then_preserve_outputs_on_zero_frames(self):
        dataset = self.make_raw()
        write_rows(dataset / "depth_events.csv", [image_row(0, "depth_mm", 1_010_000_000)])
        self.add_capture_images(dataset)
        write_capture_manifest(dataset, "complete")
        build_synced_dataset(dataset)
        original = {p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()}
        root = Path(__file__).resolve().parents[2]
        entrypoints = ([root / "postprocess/main.py", "sync"], [root / "code/build_synced_dataset.py"])
        for entrypoint in entrypoints:
            with self.subTest(entrypoint=entrypoint):
                for name in SYNC_OUTPUT_FILES:
                    (dataset / (name + ".prev")).write_bytes(original[dataset / name])
                    (dataset / name).write_bytes(b"interrupted generation\n")
                (dataset / "sync.incomplete").write_text("interrupted\n")
                command = [sys.executable] + [str(part) for part in entrypoint] + [
                    "--dataset", str(dataset), "--rgb-depth-threshold-ms", "0",
                ]
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("No synchronized frames survived", result.stderr)
                self.assertIn("kept=0, rgb=1, depth=1, no_device_ts=0, no_depth=1", result.stderr)
                self.assertEqual({p: p.read_bytes() for p in dataset.rglob("*") if p.is_file()}, original)

    def test_legacy_format_keeps_manifest_and_path_validation(self):
        dataset = self.make_raw()
        self.add_capture_images(dataset)
        (dataset / "metadata.json").write_bytes(b'{"format_version":"synced_from_raw_events_v1"}\n')
        with self.assertRaisesRegex(ValueError, "allow-legacy"):
            validate_sync_capture(dataset)
        self.assertIsNone(validate_sync_capture(dataset, allow_legacy=True))
        for state in ("recording", "failed"):
            write_capture_manifest(dataset, state)
            with self.assertRaisesRegex(ValueError, "not complete"):
                validate_sync_capture(dataset, allow_legacy=True)
        write_capture_manifest(dataset, "complete")
        self.assertEqual(validate_sync_capture(dataset)["state"], "complete")
        (dataset / "rgb/0000000.png").write_bytes(b"truncated")
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            validate_sync_capture(dataset, allow_legacy=True)
        (dataset / "capture_manifest.json").unlink()
        write_rows(dataset / "rgb_events.csv", [{"file": "../outside.png"}])
        with self.assertRaisesRegex(ValueError, "relative path"):
            validate_sync_capture(dataset, allow_legacy=True)

    def test_cli_refuses_unknown_format_before_changing_any_files(self):
        dataset = self.make_raw()
        self.add_capture_images(dataset)
        (dataset / "metadata.json").write_bytes(b'{"format_version":"synced_capture_v1"}\n')
        (dataset / "timestamps.csv").write_bytes(b"timestamps from recorder\n")
        original = {path: path.read_bytes() for path in dataset.rglob("*") if path.is_file()}
        root = Path(__file__).resolve().parents[2]

        result = subprocess.run([
            sys.executable, str(root / "postprocess/main.py"), "sync", "--dataset", str(dataset), "--allow-legacy",
        ], capture_output=True, text=True)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to replace sync outputs", result.stderr)
        self.assertEqual({path: path.read_bytes() for path in dataset.rglob("*") if path.is_file()}, original)

    def test_separate_output_must_still_be_new_and_portable(self):
        dataset = self.make_raw()
        self.add_capture_images(dataset)
        (dataset / "metadata.json").write_bytes(
            b'{ "format_version": "raw_events_v1", "world_coordinates": {"mount_height_m": 1.5} }\r\n'
        )
        overrides = b'{ "world_coordinates": {"mount_height_m": 2.1, "mount_pitch_deg": -4.0} }\r\n'
        (dataset / "metadata.overrides.json").write_bytes(overrides)
        original = {path: path.read_bytes() for path in dataset.rglob("*") if path.is_file()}

        output, _ = build_synced_dataset(dataset, output_dir=self.root / "synced", copy_images=True)

        self.assertFalse(any(path.is_symlink() for path in output.rglob("*")))
        self.assertEqual((output / "rgb/0000000.png").read_bytes(), (dataset / "rgb/0000000.png").read_bytes())
        self.assertEqual((output / "metadata.json").read_bytes(), (dataset / "metadata.json").read_bytes())
        self.assertEqual((output / "metadata.overrides.json").read_bytes(), overrides)
        previous = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
        with self.assertRaisesRegex(ValueError, "new or empty"):
            build_synced_dataset(dataset, output_dir=output)
        for path in (dataset, dataset / "nested"):
            with self.assertRaisesRegex(ValueError, "separate"):
                build_synced_dataset(dataset, output_dir=path)
        self.assertEqual({path: path.read_bytes() for path in output.rglob("*") if path.is_file()}, previous)
        self.assertEqual({path: path.read_bytes() for path in dataset.rglob("*") if path.is_file()}, original)

    def test_invalid_thresholds_fail_before_writing_outputs(self):
        dataset = self.make_raw()
        for setting in ("threshold_ms", "rgb_depth_threshold_ms", "gps_max_delta_ms"):
            for value in (-1.0, float("nan"), float("inf")):
                with self.subTest(setting=setting, value=value):
                    with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                        build_synced_dataset(dataset, **{setting: value})
                    self.assertFalse((dataset / "timestamps.csv").exists())
