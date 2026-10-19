"""Writer serialization, capture safety and durable sync recovery."""
from __future__ import annotations

import errno
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock
from uuid import UUID

from geonova_depthai.postprocess import sync_builder as sync
from tests import test_sync_builder as fixtures


ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINTS = ([ROOT / "postprocess/main.py", "sync"], [ROOT / "code/build_synced_dataset.py"])

# Exit after each filesystem boundary in a real child process, bypassing all
# Python finally blocks. Count successful mutations only (not missing unlinks).
CRASH_SCRIPT = """
import os
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from geonova_depthai.postprocess import sync_builder as sync
dataset = Path(sys.argv[2])
stop = int(sys.argv[3])
events = 0
def boundary():
    global events
    events += 1
    if events == stop:
        os._exit(87)
fsync = os.fsync
replace = os.replace
rename = Path.rename
unlink = Path.unlink
def synced(fd):
    fsync(fd)
    boundary()
def replaced(source, destination):
    replace(source, destination)
    boundary()
def renamed(source, destination):
    result = rename(source, destination)
    boundary()
    return result
def unlinked(path, *args, **kwargs):
    existed = path.exists()
    result = unlink(path, *args, **kwargs)
    if existed:
        boundary()
    return result
os.fsync = synced
os.replace = replaced
Path.rename = renamed
Path.unlink = unlinked
if sys.argv[4] == 'publish':
    sync.build_synced_dataset(dataset, gps_max_delta_ms=25.0)
else:
    with sync.sync_writer_lock(dataset):
        sync.recover_sync_generation(dataset)
print(events)
"""


class SyncPublicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def make_raw(self, name="case"):
        parent = self.root / name
        parent.mkdir(parents=True)
        return fixtures.make_dataset(
            parent, [fixtures.image_row(0, "rgb", 1_000_000_000)],
            [fixtures.image_row(0, "depth_mm", 1_000_000_000)],
            [fixtures.imu_row(0, 1_000_000_000)],
        )

    def make_complete_capture(self, name="case"):
        dataset = self.make_raw(name)
        for directory in ("rgb", "depth_mm"):
            image_dir = dataset / directory
            image_dir.mkdir()
            (image_dir / "0000000.png").write_bytes(b"capture image")
        fixtures.write_capture_manifest(dataset, "complete")
        return dataset

    def recover(self, dataset):
        with sync.sync_writer_lock(dataset):
            sync.recover_sync_generation(dataset)

    def outputs(self, dataset):
        return {name: (dataset / name).read_bytes() for name in sync.SYNC_OUTPUT_FILES}

    def assert_clean(self, dataset):
        self.assertFalse((dataset / sync.SYNC_INCOMPLETE).exists())
        self.assertFalse(list(dataset.glob("*.prev")))
        self.assertFalse(list(dataset.glob(".*.tmp")))
        self.assertTrue((dataset / sync.SYNC_LOCK).is_file())

    def child(self, dataset, stop, action):
        return subprocess.run([
            sys.executable, "-c", CRASH_SCRIPT, str(ROOT / "code"), str(dataset), str(stop), action,
        ], capture_output=True, text=True, timeout=20)

    def test_active_writer_blocks_library_and_both_clis_without_changes(self):
        dataset = self.make_raw()
        # CLI validation requires real referenced files; use a stopped legacy run.
        for stream in ("rgb", "depth_mm"):
            (dataset / stream).mkdir()
            (dataset / stream / "0000000.png").write_bytes(b"capture image")
        sync.build_synced_dataset(dataset)
        script = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from geonova_depthai.postprocess import sync_builder as sync
replace = sync.os.replace
def paused(source, destination):
    replace(source, destination)
    if destination.name == 'timestamps.csv':
        print('locked', flush=True)
        sys.stdin.readline()
sync.os.replace = paused
sync.build_synced_dataset(Path(sys.argv[2]), gps_max_delta_ms=25.0)
"""
        writer = subprocess.Popen([
            sys.executable, "-c", script, str(ROOT / "code"), str(dataset),
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(writer.stdout.readline().strip(), "locked")
            self.assertTrue((dataset / sync.SYNC_INCOMPLETE).exists())
            active = fixtures.file_bytes(dataset)
            with self.assertRaisesRegex(RuntimeError, "Sync writer already active"):
                sync.build_synced_dataset(dataset)
            for entrypoint in ENTRYPOINTS:
                result = subprocess.run(
                    [sys.executable] + [str(p) for p in entrypoint] + [
                        "--dataset", str(dataset), "--allow-legacy",
                    ], capture_output=True, text=True, timeout=15,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Sync writer already active", result.stderr)
                self.assertEqual(fixtures.file_bytes(dataset), active)
            separate = self.root / "separate-output"
            sync.build_synced_dataset(dataset, output_dir=separate)
            self.assertTrue((separate / sync.SYNC_LOCK).is_file())
            self.assertTrue((dataset / sync.SYNC_LOCK).is_file())
            self.assertEqual(fixtures.file_bytes(dataset), active)
            stdout, stderr = writer.communicate("\n", timeout=15)
            self.assertEqual(writer.returncode, 0, stdout + stderr)
        finally:
            if writer.poll() is None:
                writer.kill()
            writer.communicate()
        self.assertEqual(json.loads((dataset / "metadata.sync.json").read_text())["sync"]["gps_max_delta_ms"], 25.0)
        self.assert_clean(dataset)

    def test_separate_output_leaves_capture_byte_identical_and_locks_output(self):
        dataset = self.make_complete_capture()
        before = {
            path.relative_to(dataset).as_posix(): path.read_bytes()
            for path in dataset.rglob("*") if path.is_file()
        }
        output = self.root / "portable-output"

        with redirect_stdout(io.StringIO()):
            sync.main(["--dataset", str(dataset), "--output-dir", str(output)])

        after = {
            path.relative_to(dataset).as_posix(): path.read_bytes()
            for path in dataset.rglob("*") if path.is_file()
        }
        self.assertEqual(after, before)
        self.assertFalse((dataset / sync.SYNC_LOCK).exists())
        self.assertTrue((output / sync.SYNC_LOCK).is_file())

    def test_rejected_capture_does_not_create_output_folder_or_lock(self):
        for state in ("recording", "failed"):
            with self.subTest(state=state):
                dataset = self.make_complete_capture(state)
                fixtures.write_capture_manifest(dataset, state)
                output = self.root / (state + "-output")

                with redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, "not complete"):
                        sync.main([
                            "--dataset", str(dataset), "--output-dir", str(output), "--allow-legacy",
                        ])

                self.assertFalse(output.exists())
                self.assertFalse((dataset / sync.SYNC_LOCK).exists())

    def check_unsafe_links(self, kind):
        names = list(sync.SYNC_OUTPUT_FILES)
        names += [name + ".prev" for name in sync.SYNC_OUTPUT_FILES]
        names += [sync.SYNC_INCOMPLETE, sync.SYNC_LOCK, ".timestamps.csv.crashed.tmp"]
        for interrupted in (False, True):
            for index, name in enumerate(names):
                with self.subTest(kind=kind, interrupted=interrupted, name=name):
                    dataset = self.make_raw("{}-{}-{}".format(kind, interrupted, index))
                    if interrupted:
                        (dataset / sync.SYNC_INCOMPLETE).write_text("interrupted\n")
                        for output in sync.SYNC_OUTPUT_FILES:
                            (dataset / output).write_bytes(b"mixed output")
                            (dataset / (output + ".prev")).write_bytes(b"previous output")
                    path = dataset / name
                    path.unlink(missing_ok=True)
                    capture = dataset / "imu_events.csv"
                    if kind == "symlink":
                        try:
                            path.symlink_to(capture)
                        except OSError as error:
                            self.skipTest("Symlinks unavailable: " + str(error))
                    else:
                        os.link(str(capture), str(path))
                    original = fixtures.file_bytes(dataset)
                    entries = set(dataset.iterdir())
                    capture_bytes = capture.read_bytes()
                    with mock.patch.object(sync.os, "open", side_effect=AssertionError("Write attempted")):
                        with self.assertRaisesRegex(ValueError, "Unsafe sync path"):
                            sync.build_synced_dataset(dataset)
                    for entrypoint in ENTRYPOINTS:
                        result = subprocess.run(
                            [sys.executable] + [str(p) for p in entrypoint] + [
                                "--dataset", str(dataset), "--allow-legacy",
                            ], capture_output=True, text=True, timeout=15,
                        )
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("Unsafe sync path", result.stderr)
                    self.assertEqual(capture.read_bytes(), capture_bytes)
                    self.assertEqual(fixtures.file_bytes(dataset), original)
                    self.assertEqual(set(dataset.iterdir()), entries)

    def test_symlinked_transaction_paths_are_refused_before_writes(self):
        self.check_unsafe_links("symlink")

    def test_hardlinked_transaction_paths_are_refused_before_writes(self):
        self.check_unsafe_links("hardlink")

    def test_non_regular_outputs_and_backups_are_refused(self):
        dataset = self.make_raw()
        for name in list(sync.SYNC_OUTPUT_FILES) + [name + ".prev" for name in sync.SYNC_OUTPUT_FILES]:
            with self.subTest(name=name):
                path = dataset / name
                path.mkdir()
                with self.assertRaisesRegex(ValueError, "Unsafe sync path"):
                    sync.build_synced_dataset(dataset)
                self.assertFalse((dataset / sync.SYNC_LOCK).exists())
                path.rmdir()

    def test_cli_validates_capture_paths_before_recovery(self):
        dataset = self.make_raw()
        sync.build_synced_dataset(dataset)
        for name in sync.SYNC_OUTPUT_FILES:
            (dataset / (name + ".prev")).write_bytes((dataset / name).read_bytes())
            (dataset / name).write_bytes(b"mixed output")
        (dataset / sync.SYNC_INCOMPLETE).write_text("interrupted\n")
        fixtures.write_rows(dataset / "rgb_events.csv", [{"file": "../outside.png"}])
        original = fixtures.file_bytes(dataset)
        for entrypoint in ENTRYPOINTS:
            result = subprocess.run(
                [sys.executable] + [str(p) for p in entrypoint] + [
                    "--dataset", str(dataset), "--allow-legacy",
                ], capture_output=True, text=True, timeout=15,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Invalid dataset relative path", result.stderr)
            self.assertEqual(fixtures.file_bytes(dataset), original)

    def test_identical_publications_get_distinct_uuid4_generations(self):
        dataset = self.make_raw()
        generations = []
        for _ in range(3):
            sync.build_synced_dataset(dataset)
            generation = json.loads((dataset / "metadata.sync.json").read_text())["generation"]
            self.assertEqual(UUID(generation).version, 4)
            self.assertEqual(UUID(generation).hex, generation)
            generations.append(generation)
        self.assertEqual(len(set(generations)), 3)

    def test_durable_order_and_lock_through_cleanup(self):
        dataset = self.make_raw()
        sync.build_synced_dataset(dataset)
        events = []
        synced_inodes = set()
        fsync, replace = sync.os.fsync, sync.os.replace
        directory_sync, state_write = sync.fsync_directory, sync.write_sync_state
        unlink = Path.unlink

        def synced(fd):
            info = os.fstat(fd)
            fsync(fd)
            if stat.S_ISREG(info.st_mode):
                synced_inodes.add(info.st_ino)
                events.append("file-sync")

        def synced_directory(path):
            directory_sync(path)
            events.append("directory-sync")

        def state(file, value, previous=None):
            state_write(file, value, previous)
            events.append(value)

        def replaced(source, destination):
            self.assertIn(source.stat().st_ino, synced_inodes)
            self.assertIn("publishing", events)
            replace(source, destination)
            events.append(destination.name)

        def unlinked(path, *args, **kwargs):
            existed = path.exists()
            if existed and path.suffix == ".prev":
                self.assertIn("complete", events)
                self.assertTrue((dataset / sync.SYNC_INCOMPLETE).exists())
                with self.assertRaisesRegex(RuntimeError, "Sync writer already active"):
                    sync.build_synced_dataset(dataset)
            result = unlink(path, *args, **kwargs)
            if existed:
                events.append("unlink:" + path.name)
            return result

        with mock.patch.object(sync.os, "fsync", side_effect=synced):
            with mock.patch.object(sync, "fsync_directory", side_effect=synced_directory):
                with mock.patch.object(sync, "write_sync_state", side_effect=state):
                    with mock.patch.object(sync.os, "replace", side_effect=replaced):
                        with mock.patch.object(Path, "unlink", unlinked):
                            sync.build_synced_dataset(dataset)
        self.assertEqual(events[:3], ["file-sync", "preparing", "directory-sync"])
        swaps = [event for event in events if event in sync.SYNC_OUTPUT_FILES]
        self.assertEqual(swaps, ["timestamps.csv", "imu.csv", "quality_report.json", "metadata.sync.json"])
        last_swap = events.index("metadata.sync.json")
        first_delete = next(i for i, event in enumerate(events) if event.endswith(".prev"))
        self.assertIn("directory-sync", events[last_swap + 1:first_delete])
        self.assertEqual(events[-3:], ["directory-sync", "unlink:sync.incomplete", "directory-sync"])
        self.assert_clean(dataset)

    def test_directory_sync_skips_only_unsupported_operations(self):
        directory = self.root
        # Exercise POSIX behavior on Windows too, with fake descriptors.
        with mock.patch.object(sync.os, "name", "posix"):
            with mock.patch.object(sync.os, "open", return_value=123):
                with mock.patch.object(sync.os, "close") as close:
                    with mock.patch.object(sync.os, "fsync", side_effect=OSError(errno.EINVAL, "Unsupported")):
                        sync.fsync_directory(directory)
                    close.assert_called_once_with(123)
                    with mock.patch.object(sync.os, "fsync", side_effect=OSError(errno.EIO, "Disk error")):
                        with self.assertRaises(OSError):
                            sync.fsync_directory(directory)

    def test_crash_at_every_publication_boundary_recovers_one_generation(self):
        dataset = self.make_raw()
        fixtures.write_rows(dataset / "gps.csv", [{"host_monotonic_ns": 1_100_000_000}])
        sync.build_synced_dataset(dataset)
        old = self.outputs(dataset)
        raw = {path: data for path, data in fixtures.file_bytes(dataset).items()
               if path.name not in sync.SYNC_OUTPUT_FILES}
        baseline = self.child(dataset, 0, "publish")
        self.assertEqual(baseline.returncode, 0, baseline.stdout + baseline.stderr)
        boundaries = int(baseline.stdout)
        self.assertGreater(boundaries, 20)
        for stop in range(1, boundaries + 1):
            with self.subTest(boundary=stop):
                for name, data in old.items():
                    (dataset / name).write_bytes(data)
                crashed = self.child(dataset, stop, "publish")
                self.assertEqual(crashed.returncode, 87, crashed.stdout + crashed.stderr)
                self.recover(dataset)
                recovered = self.outputs(dataset)
                metadata = json.loads(recovered["metadata.sync.json"])
                if metadata["generation"] == json.loads(old["metadata.sync.json"])["generation"]:
                    self.assertEqual(recovered, old)
                else:
                    self.assertEqual(UUID(metadata["generation"]).version, 4)
                    self.assertEqual(metadata["sync"]["gps_max_delta_ms"], 25.0)
                    self.assertEqual(metadata["quality_report"], json.loads(recovered["quality_report.json"]))
                    self.assertEqual(fixtures.read_timestamps(dataset)[0]["gps_frame_delta_ms"], "")
                    self.assertEqual(recovered["imu.csv"], old["imu.csv"])
                for path, data in raw.items():
                    self.assertEqual(path.read_bytes(), data)
                self.assert_clean(dataset)

    def test_crash_at_every_recovery_boundary_is_restartable(self):
        dataset = self.make_raw()
        sync.build_synced_dataset(dataset)
        old = self.outputs(dataset)

        def interrupted():
            for name, data in old.items():
                (dataset / (name + ".prev")).write_bytes(data)
                (dataset / name).write_bytes(b"mixed output")
            # Includes a torn last record as left by an interrupted append.
            (dataset / sync.SYNC_INCOMPLETE).write_text(
                json.dumps({"state": "preparing", "previous": list(sync.SYNC_OUTPUT_FILES)}) +
                '\n{"state": "publishing"}\n{"state": "com', encoding="utf-8",
            )

        interrupted()
        baseline = self.child(dataset, 0, "recover")
        self.assertEqual(baseline.returncode, 0, baseline.stdout + baseline.stderr)
        boundaries = int(baseline.stdout)
        for stop in range(1, boundaries + 1):
            with self.subTest(boundary=stop):
                interrupted()
                crashed = self.child(dataset, stop, "recover")
                self.assertEqual(crashed.returncode, 87, crashed.stdout + crashed.stderr)
                self.recover(dataset)
                self.assertEqual(self.outputs(dataset), old)
                self.assert_clean(dataset)

    def test_failed_restoration_keeps_marker_and_backups_for_retry(self):
        dataset = self.make_raw()
        sync.build_synced_dataset(dataset)
        old = self.outputs(dataset)
        for name, data in old.items():
            (dataset / (name + ".prev")).write_bytes(data)
            (dataset / name).write_bytes(b"mixed output")
        (dataset / sync.SYNC_INCOMPLETE).write_text("interrupted\n")
        with mock.patch.object(sync.os, "replace", side_effect=OSError("Restore failed")):
            with self.assertRaisesRegex(OSError, "Restore failed"):
                self.recover(dataset)
        self.assertTrue((dataset / sync.SYNC_INCOMPLETE).exists())
        for name, data in old.items():
            self.assertEqual((dataset / (name + ".prev")).read_bytes(), data)
        copied, replaced, synced_inodes = [], [], set()
        copy, replace, fsync = sync.shutil.copy2, sync.os.replace, sync.os.fsync

        def copy_fresh(source, destination):
            self.assertTrue(source.name.endswith(".prev"))
            self.assertTrue(destination.name.startswith("." + source.name[:-5] + "."))
            self.assertEqual(destination.stat().st_nlink, 1)
            copied.append(source.name)
            return copy(source, destination)

        def synced(fd):
            fsync(fd)
            synced_inodes.add(os.fstat(fd).st_ino)

        def replace_synced(source, destination):
            self.assertIn(source.stat().st_ino, synced_inodes)
            self.assertEqual(destination.read_bytes(), b"mixed output")
            with self.assertRaisesRegex(RuntimeError, "Sync writer already active"):
                sync.build_synced_dataset(dataset)
            replace(source, destination)
            replaced.append(destination.name)

        with mock.patch.object(sync.shutil, "copy2", side_effect=copy_fresh):
            with mock.patch.object(sync.os, "fsync", side_effect=synced):
                with mock.patch.object(sync.os, "replace", side_effect=replace_synced):
                    self.recover(dataset)
        self.assertEqual(copied, [name + ".prev" for name in sync.SYNC_OUTPUT_FILES])
        self.assertEqual(replaced, ["timestamps.csv", "imu.csv", "quality_report.json", "metadata.sync.json"])
        self.assertEqual(self.outputs(dataset), old)
        self.assert_clean(dataset)
