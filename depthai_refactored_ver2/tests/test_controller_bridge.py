import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from geonova_depthai.controller_bridge import ControllerBridge


def bridge_args(root: Path, preview_fps=0.0):
    return SimpleNamespace(
        controller_bridge_enabled=True,
        controller_bridge_dir=str(root),
        controller_status_interval_s=1.0,
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
