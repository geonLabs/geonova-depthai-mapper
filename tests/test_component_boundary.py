"""Exercise the exported deployment, file transfer and shutdown boundaries."""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from geonova_common.run_contract import MANIFEST, validate_capture, write_capture_manifest
from geonova_postprocess.sync_builder import main as sync_main

ROOT = Path(__file__).resolve().parents[1]


def csv_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def raw_run(root):
    root.mkdir()
    (root / "metadata.json").write_text(json.dumps({"format_version": "raw_events_v1", "camera_model": {"calibration": "preserve"}}))
    for stream, folder in (("rgb", "rgb"), ("depth", "depth_mm")):
        (root / folder).mkdir()
        (root / folder / "frame.png").write_bytes(b"synthetic-transfer-payload")
        csv_rows(root / (stream + "_events.csv"), [{"file": folder + "/frame.png", "device_ts_ns": 1_000_000_000,
                 "capture_monotonic_ns": 2_000_000_000, "sequence": 1}])
    csv_rows(root / "imu_events.csv", [{"message_index": 1, "packet_index": 0, "message_device_ts_ns": 1_000_000_000}])
    write_capture_manifest(root, "complete", {"rgb": 1, "depth": 1, "imu": 1})
    return root


def hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def test_transfer_sync_preserves_raw_and_copies_portable_output(tmp_path):
    raw = raw_run(tmp_path / "raw")
    before = hashes(raw)
    out = tmp_path / "processed"
    sync_main(["--dataset", str(raw), "--output-dir", str(out)])
    assert hashes(raw) == before
    assert not any(p.is_symlink() for p in out.rglob("*"))
    assert (out / "depth_mm/frame.png").read_bytes() == (raw / "depth_mm/frame.png").read_bytes()
    assert json.loads((out / "metadata.json").read_text())["camera_model"] == {"calibration": "preserve"}
    assert json.loads((out / "processing_manifest.json").read_text())["state"] == "complete"
    with pytest.raises(ValueError, match="new or empty"):
        sync_main(["--dataset", str(raw), "--output-dir", str(out)])


@pytest.mark.parametrize("state", ["recording", "failed"])
def test_reject_unfinished_capture_even_with_legacy_flag(tmp_path, state):
    raw = raw_run(tmp_path / "raw")
    write_capture_manifest(raw, state)
    with pytest.raises(ValueError, match="not complete"):
        sync_main(["--dataset", str(raw), "--output-dir", str(tmp_path / "out"), "--allow-legacy"])
    assert not (tmp_path / "out").exists()


def test_reject_incomplete_or_corrupt_transfer(tmp_path):
    raw = raw_run(tmp_path / "raw")
    (raw / "rgb/frame.png").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="size mismatch"):
        validate_capture(raw)
    write_capture_manifest(raw, "complete")
    contents = (raw / "metadata.json").read_text().replace("preserve", "corrupt!")
    (raw / "metadata.json").write_text(contents)
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_capture(raw)


def test_legacy_requires_explicit_opt_in_and_rejects_unsafe_paths(tmp_path):
    raw = raw_run(tmp_path / "raw")
    (raw / MANIFEST).unlink()
    with pytest.raises(ValueError, match="allow-legacy"):
        validate_capture(raw)
    assert validate_capture(raw, allow_legacy=True) is None
    csv_rows(raw / "rgb_events.csv", [{"file": "../outside.png"}])
    with pytest.raises(ValueError, match="relative path"):
        validate_capture(raw, allow_legacy=True)


@pytest.mark.parametrize("relative", [".", "nested"])
def test_output_cannot_modify_input(tmp_path, relative):
    raw = raw_run(tmp_path / "raw")
    with pytest.raises(ValueError, match="separate"):
        sync_main(["--dataset", str(raw), "--output-dir", str(raw / relative)])


def export(component, destination):
    subprocess.run([sys.executable, str(ROOT / "tools/export_component.py"), component,
                    "--destination", str(destination), "--init-git"], check=True, capture_output=True, text=True)


def test_exports_have_independent_imports_and_entrypoints(tmp_path):
    capture = tmp_path / "geonova-capture"
    server = tmp_path / "geonova-postprocess"
    export("capture", capture)
    export("postprocess", server)
    assert (capture / ".git").is_dir()
    assert not (capture / "src/geonova_postprocess").exists()
    assert not (server / "src/geonova_depthai").exists()
    assert not (capture / "model").exists()
    # -I discards the monorepo PYTHONPATH; exports must resolve their own source.
    for entry, args in ((capture / "main.py", ["--help"]), (server / "main.py", ["sync", "--help"]),
                        (server / "main.py", ["yolo", "--help"]), (server / "main.py", ["inspect", "--help"])):
        # Use an import blocker to verify forbidden dependencies even though the
        # development environment may have camera/numerical packages installed.
        blocked = ["torch", "ultralytics", "pyproj", "scipy", "geonova_postprocess"] if entry.parent == capture else ["depthai", "serial", "geonova_depthai", "ultralytics"]
        script = f'''
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {blocked!r}:
            raise RuntimeError('Forbidden dependency: ' + fullname)
sys.meta_path.insert(0, Block())
sys.argv = {[str(entry)] + args!r}
runpy.run_path({str(entry)!r}, run_name='__main__')
'''
        subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env={**os.environ, "PYTHONPATH": ""},
                       check=True, capture_output=True, text=True)
    raw = raw_run(tmp_path / "raw")
    # Sync runs with stdlib only, with all site packages disabled.
    subprocess.run([sys.executable, "-I", "-S", str(server / "main.py"), "sync", "--dataset", str(raw),
                    "--output-dir", str(tmp_path / "synced")], check=True, capture_output=True, text=True)


def test_cleanup_failure_does_not_publish_complete(tmp_path, monkeypatch):
    from geonova_depthai.capture.cli import parse_args
    from geonova_depthai.capture.raw_writer import RawEventDataset
    from geonova_depthai.capture.raw_event_recorder import _close_recording_resources
    args = parse_args(["--no-gps", "--no-external-imu"])
    dataset = RawEventDataset(tmp_path, args)
    class BadPool:
        def close(self):
            raise RuntimeError("disk full")
    with pytest.raises(RuntimeError, match="disk full"):
        _close_recording_resources(None, {}, BadPool(), dataset, None)
    assert json.loads((dataset.root / MANIFEST).read_text())["state"] == "failed"


def test_sigterm_drains_and_marks_run_complete(tmp_path):
    script = '''
import sys, time
from pathlib import Path
from geonova_depthai.capture import raw_event_recorder as recorder
from geonova_depthai.capture.raw_writer import RawEventDataset

def fake_record(args):
    dataset = RawEventDataset(Path(sys.argv[1]), args)
    (Path(sys.argv[1]) / 'ready').write_text(str(dataset.root))
    try:
        while not recorder._stop_requested:
            time.sleep(0.01)
    finally:
        recorder._close_recording_resources(None, {}, None, dataset, None)
recorder.record_raw_events = fake_record
recorder.main(['--no-gps', '--no-external-imu'])
'''
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "capture/src"), str(ROOT / "shared/src")])}
    process = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "ready").exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (tmp_path / "ready").exists()
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, (stdout, stderr)
        dataset = Path((tmp_path / "ready").read_text())
        assert json.loads((dataset / MANIFEST).read_text())["state"] == "complete"
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_dataset_close_attempts_all_csv_files_after_flush_error(tmp_path):
    from geonova_depthai.capture.cli import parse_args
    from geonova_depthai.capture.raw_writer import RawEventDataset
    dataset = RawEventDataset(tmp_path, parse_args(["--no-gps", "--no-external-imu"]))
    closed = []
    class Writer:
        def __init__(self, name, failure=False):
            self.name, self.failure = name, failure
        def close(self):
            closed.append(self.name)
            if self.failure:
                raise OSError("flush failed")
    dataset.close()  # Close the actual fixture files before installing failure probes.
    dataset.rgb_events, dataset.depth_events, dataset.imu_events = Writer("rgb", True), Writer("depth"), Writer("imu")
    with pytest.raises(OSError, match="flush failed"):
        dataset.close()
    assert closed == ["rgb", "depth", "imu"]


def test_installer_does_not_ignore_real_dependency_errors(monkeypatch):
    spec = importlib.util.spec_from_file_location("installer", ROOT / "tools/install_component.py")
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    result = subprocess.CompletedProcess([], 1, "depthai 3.1.0 is not supported on this platform\nnumpy is missing", "")
    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **kw: result)
    with pytest.raises(RuntimeError, match="numpy is missing"):
        installer.check_dependencies(Path("python"), "capture")
    result.stdout = "depthai 3.1.0 is not supported on this platform\n"
    assert installer.check_dependencies(Path("python"), "capture")
    with pytest.raises(RuntimeError):
        installer.check_dependencies(Path("python"), "postprocess")
