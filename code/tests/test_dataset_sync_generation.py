"""Dataset readers must not combine two completed sync publications."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
for source in ("postprocess/src", "shared/src"):
    sys.path.insert(0, str(ROOT / source))

from geonova_postprocess import dataset as geometry
from geonova_postprocess import sync_builder as sync


class DatasetSyncGenerationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.publish("old")

    def publish(self, generation):
        # Model a whole publication occurring between reader checks: neither
        # endpoint sees the marker, but the generation must reveal the change.
        marker = self.root / "sync.incomplete"
        marker.touch()
        (self.root / "timestamps.csv").write_text(
            "rgb_file,depth_file\n" + generation + ".png,depth.png\n", encoding="utf-8")
        (self.root / "imu.csv").write_text(
            "frame_index,message_index\n0," + generation + "\n", encoding="utf-8")
        (self.root / "metadata.sync.json").write_text(json.dumps({"generation": generation}), encoding="utf-8")
        marker.unlink()

    def test_complete_publication_between_reads_retries_mixed_generation(self):
        read_csv = geometry.read_csv
        timestamp_reads = []

        def read_then_publish(path):
            rows = read_csv(path)
            if path.name == "timestamps.csv":
                timestamp_reads.append(rows)
                if len(timestamp_reads) == 1:
                    self.publish("new")
            return rows

        with patch.object(geometry, "read_csv", side_effect=read_then_publish):
            dataset = geometry.Dataset(self.root)
        self.assertEqual(len(timestamp_reads), 2)
        self.assertEqual(dataset.sync_generation, "new")
        self.assertEqual(dataset.timestamps[0]["rgb_file"], "new.png")
        self.assertEqual(dataset.imu_by_frame[0][0]["message_index"], "new")

    def test_repeated_publications_fail_clearly_after_bounded_retries(self):
        read_csv = geometry.read_csv
        generations = []

        def read_then_publish(path):
            rows = read_csv(path)
            if path.name == "timestamps.csv":
                generations.append(str(len(generations)))
                self.publish(generations[-1])
            return rows

        with patch.object(geometry, "read_csv", side_effect=read_then_publish):
            with self.assertRaisesRegex(ValueError, "sync changed while loading"):
                geometry.Dataset(self.root)
        self.assertEqual(len(generations), 3)

    def test_rollback_between_checks_retries_even_when_generation_is_unchanged(self):
        (self.root / "quality_report.json").write_text("{}", encoding="utf-8")
        original = {name: (self.root / name).read_bytes() for name in sync.SYNC_OUTPUT_FILES}
        read_csv, replace = geometry.read_csv, sync.os.replace
        timestamp_reads, mixed_imu = [], []
        failed = False

        def replace_then_fail(source, destination):
            nonlocal failed
            replace(source, destination)
            if destination.name == "imu.csv" and not failed:
                failed = True
                mixed_imu.extend(read_csv(destination))
                raise OSError("Publication failed after IMU replacement")

        def read_across_rollback(path):
            if path.name == "timestamps.csv":
                rows = read_csv(path)
                timestamp_reads.append(rows)
                return rows
            if path.name == "imu.csv" and len(timestamp_reads) == 1:
                with sync.sync_writer_lock(self.root):
                    with patch.object(sync.os, "replace", side_effect=replace_then_fail):
                        with self.assertRaisesRegex(OSError, "Publication failed"):
                            sync.write_sync_outputs(
                                self.root, [{"rgb_file": "new.png", "depth_file": "depth.png"}],
                                [{"frame_index": 0, "message_index": "new"}], {"quality_report": {}},
                            )
                # Reader saw IMU B before the failed writer restored A. Both
                # generation checks see A; only the epoch reveals the ABA.
                return mixed_imu
            return read_csv(path)

        with patch.object(geometry, "read_csv", side_effect=read_across_rollback):
            dataset = geometry.Dataset(self.root)
        self.assertEqual(mixed_imu[0]["message_index"], "new")
        self.assertEqual(len(timestamp_reads), 2)
        self.assertEqual(dataset.sync_generation, "old")
        self.assertEqual(dataset.sync_epoch, 1)
        self.assertEqual(dataset.timestamps[0]["rgb_file"], "old.png")
        self.assertEqual(dataset.imu_by_frame[0][0]["message_index"], "old")
        self.assertEqual({name: (self.root / name).read_bytes() for name in sync.SYNC_OUTPUT_FILES}, original)

    def test_old_synced_dataset_without_epoch_loads_as_epoch_zero(self):
        self.assertFalse((self.root / "sync.epoch").exists())
        dataset = geometry.Dataset(self.root)
        self.assertEqual(dataset.sync_epoch, 0)
        self.assertEqual(dataset.sync_generation, "old")
        self.assertEqual(dataset.imu_by_frame[0][0]["message_index"], "old")

    def test_repeated_epoch_changes_fail_after_bounded_retries(self):
        read_csv = geometry.read_csv
        epochs = []

        def read_then_bump(path):
            rows = read_csv(path)
            if path.name == "timestamps.csv":
                epochs.append(len(epochs) + 1)
                (self.root / "sync.epoch").write_text(str(epochs[-1]), encoding="utf-8")
            return rows

        with patch.object(geometry, "read_csv", side_effect=read_then_bump):
            with self.assertRaisesRegex(ValueError, "sync changed while loading"):
                geometry.Dataset(self.root)
        self.assertEqual(epochs, [1, 2, 3])

    def test_legacy_without_generation_still_checks_marker_after_reading(self):
        (self.root / "metadata.sync.json").write_text("{}", encoding="utf-8")
        self.assertIsNone(geometry.Dataset(self.root).sync_generation)
        read_csv = geometry.read_csv

        def read_then_mark(path):
            rows = read_csv(path)
            (self.root / "sync.incomplete").touch()
            return rows

        with patch.object(geometry, "read_csv", side_effect=read_then_mark):
            with self.assertRaisesRegex(ValueError, r"sync.incomplete.*re-run sync"):
                geometry.Dataset(self.root)

    def test_cached_dataset_reloads_after_new_generation(self):
        with patch.dict(geometry.DATASET_CACHE, {}, clear=True):
            old = geometry.get_dataset(str(self.root))
            self.publish("new")
            new = geometry.get_dataset(str(self.root))
            self.assertIsNot(old, new)
            self.assertEqual(new.sync_generation, "new")
            self.assertEqual(new.timestamps[0]["rgb_file"], "new.png")

    def test_cached_dataset_reloads_after_epoch_changes_with_same_generation(self):
        with patch.dict(geometry.DATASET_CACHE, {}, clear=True):
            old = geometry.get_dataset(str(self.root))
            (self.root / "sync.epoch").write_text("1\n", encoding="utf-8")
            new = geometry.get_dataset(str(self.root))
            self.assertIsNot(old, new)
            self.assertEqual(new.sync_generation, old.sync_generation)
            self.assertEqual(new.sync_epoch, 1)


if __name__ == "__main__":
    unittest.main()
