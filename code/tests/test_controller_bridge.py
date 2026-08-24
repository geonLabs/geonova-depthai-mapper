import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from geonova_depthai.controller_bridge import ControllerBridge


def bridge_args(root: Path, preview_fps=0.0, status_interval_s=1.0):
    return SimpleNamespace(
        controller_bridge_enabled=True,
        controller_bridge_dir=str(root),
        controller_status_interval_s=status_interval_s,
        controller_sensor_stale_after_s=3.0,
        controller_preview_fps=preview_fps,
        controller_preview_max_width=640,
        controller_preview_jpeg_quality=75,
        enable_gps=True,
        enable_external_imu=True,
        rotate_180=False,
        flip=False,
    )


class FakeNtrip:
    connected = True
    current_mountpoint = "TEST-RTCM31"
    bytes_received = 2048


class FakeReader:
    started = True
    error = None
    rtk_client = FakeNtrip()

    def __init__(self, sample):
        self.sample = sample

    def latest_sample(self):
        return self.sample


def test_bridge_publishes_live_rtk_and_sensor_state(tmp_path):
    bridge = ControllerBridge(bridge_args(tmp_path))
    bridge.mark_device_connected()
    bridge.observe_imu()
    gps = FakeReader(
        {
            "host_monotonic_ns": time.monotonic_ns(),
            "fix_quality": "4",
            "latitude_deg": 37.501,
            "longitude_deg": 127.039,
            "altitude_m": "42.5",
            "satellites": "18",
            "hdop": "0.7",
        }
    )

    external_imu = FakeReader({"host_monotonic_ns": time.monotonic_ns()})
    assert bridge.publish({"gps": gps, "external_imu": external_imu}, force=True)
    payload = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))

    assert payload["pipeline"]["active"] is True
    assert payload["camera"]["connected"] is True
    assert payload["gnss"]["active"] is True
    assert payload["gnss"]["fixType"] == "rtk_fixed"
    assert payload["gnss"]["ntripConnected"] is True
    assert payload["gnss"]["satellites"] == 18
    assert payload["imu"]["active"] is True
    assert payload["imu"]["source"] == "oak+external"
    bridge.close({"gps": gps})
    stopped = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert stopped["pipeline"]["active"] is False
    assert stopped["camera"]["active"] is False
    assert stopped["gnss"]["active"] is False


def test_bridge_writes_resized_jpeg_preview(tmp_path):
    bridge = ControllerBridge(bridge_args(tmp_path))
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :, 1] = 180

    bridge.write_preview_frame(frame)

    encoded = np.fromfile(tmp_path / "camera-preview.jpg", dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (360, 640)
    assert bridge.snapshot()["camera"]["previewAvailable"] is True
    bridge.close()


def test_bridge_default_preserves_maximum_capture_width(tmp_path):
    args = bridge_args(tmp_path)
    del args.controller_preview_max_width
    bridge = ControllerBridge(args)
    frame = np.zeros((1200, 1920, 3), dtype=np.uint8)

    bridge.write_preview_frame(frame)

    encoded = np.fromfile(tmp_path / "camera-preview.jpg", dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (1200, 1920)
    bridge.close()


def test_bridge_heartbeat_continues_during_transient_camera_failure(tmp_path):
    bridge = ControllerBridge(bridge_args(tmp_path, status_interval_s=0.1))
    gps = FakeReader({"host_monotonic_ns": time.monotonic_ns(), "fix_quality": "1"})
    assert bridge.publish({"gps": gps}, force=True)
    first = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))

    bridge.mark_device_disconnected("camera temporarily missing")
    deadline = time.monotonic() + 1.0
    current = first
    while time.monotonic() < deadline:
        time.sleep(0.03)
        current = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
        if current["updatedAtEpochMillis"] > first["updatedAtEpochMillis"]:
            break

    assert current["updatedAtEpochMillis"] > first["updatedAtEpochMillis"]
    assert current["pipeline"]["active"] is True
    assert current["pipeline"]["error"] == "camera temporarily missing"
    assert current["gnss"]["connected"] is True
    bridge.close({"gps": gps})
