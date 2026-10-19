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
        imu_enabled=True,
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
    bridge.observe_imu(SimpleNamespace(packets=[object()]))
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

    assert bridge.publish({"gps": gps}, force=True)
    payload = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))

    assert payload["pipeline"]["active"] is True
    assert payload["camera"]["connected"] is True
    assert payload["gnss"]["active"] is True
    assert payload["gnss"]["fixType"] == "rtk_fixed"
    assert payload["gnss"]["ntripConnected"] is True
    assert payload["gnss"]["satellites"] == 18
    assert payload["imu"]["active"] is True
    assert payload["imu"]["source"] == "oak"
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
    assert bridge.snapshot()["camera"]["frameWidth"] == 1280
    assert bridge.snapshot()["camera"]["frameHeight"] == 720
    bridge.close()


def test_bridge_encoded_frame_dimensions_come_from_decoded_pixels(tmp_path):
    bridge = ControllerBridge(bridge_args(tmp_path))
    encoded = SimpleNamespace(getWidth=lambda: 1030985, getHeight=lambda: 1030992)
    bridge.offer_rgb(encoded)
    assert bridge.snapshot()["camera"]["frameWidth"] is None
    bridge.write_preview_frame(np.zeros((1200, 1920, 3), dtype=np.uint8))
    bridge.offer_rgb(encoded)
    assert bridge.snapshot()["camera"]["frameWidth"] == 1920
    assert bridge.snapshot()["camera"]["frameHeight"] == 1200
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


def test_imu_status_requires_oak_packets_and_expires_at_freshness_window(tmp_path, monkeypatch):
    args = bridge_args(tmp_path)
    args.controller_bridge_enabled = False
    bridge = ControllerBridge(args)
    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    bridge.mark_device_connected()
    waiting = bridge.snapshot()["imu"]
    assert waiting == {
        "configured": True, "connected": False, "active": False,
        "lastSampleAtEpochMillis": None, "source": "oak",
        "error": "Waiting for OAK IMU packets",
    }
    bridge.observe_imu(SimpleNamespace(packets=[]))
    assert bridge.snapshot()["imu"] == waiting
    bridge.observe_imu(SimpleNamespace(packets=[object()]))
    live = bridge.snapshot()["imu"]
    assert live["configured"] is True
    assert live["connected"] is True
    assert live["active"] is True
    assert live["error"] is None
    assert live["lastSampleAtEpochMillis"] is not None
    clock[0] += bridge.sensor_stale_after_s
    assert bridge.snapshot()["imu"]["connected"] is True
    clock[0] += 0.001
    stale = bridge.snapshot()["imu"]
    assert stale["connected"] is False
    assert stale["active"] is False
    assert stale["lastSampleAtEpochMillis"] == live["lastSampleAtEpochMillis"]
    assert stale["error"] == "OAK IMU packets stale"


def test_imu_status_disconnect_and_reconnect_require_new_packets(tmp_path):
    args = bridge_args(tmp_path)
    args.controller_bridge_enabled = False
    bridge = ControllerBridge(args)
    bridge.mark_device_connected()
    bridge.observe_imu(SimpleNamespace(packets=[object()]))
    assert bridge.snapshot()["imu"]["connected"] is True
    bridge.mark_device_disconnected("camera unavailable")
    disconnected = bridge.snapshot()["imu"]
    assert disconnected["configured"] is True
    assert disconnected["connected"] is False
    assert disconnected["active"] is False
    assert disconnected["lastSampleAtEpochMillis"] is None
    assert disconnected["source"] == "oak"
    assert disconnected["error"] == "OAK camera disconnected"
    bridge.mark_device_connected()
    assert bridge.snapshot()["imu"]["connected"] is False
    bridge.configure_imu(False, "No OAK IMU")
    bridge.observe_imu(SimpleNamespace(packets=[object()]))
    unavailable = bridge.snapshot()["imu"]
    assert unavailable["configured"] is False
    assert unavailable["active"] is False
    assert unavailable["connected"] is False
    assert unavailable["error"] == "No OAK IMU"
    bridge.configure_imu(True)
    assert bridge.snapshot()["imu"]["lastSampleAtEpochMillis"] is None
    bridge.observe_imu(SimpleNamespace(packets=[object()]))
    assert bridge.snapshot()["imu"]["connected"] is True
