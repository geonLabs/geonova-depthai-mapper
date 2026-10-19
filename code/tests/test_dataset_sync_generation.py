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


if __name__ == "__main__":
    unittest.main()
