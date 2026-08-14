"""Publish live capture telemetry and a bounded-rate camera preview."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path

import cv2

from geonova_depthai import runtime


FIX_TYPES = {
    "0": "none",
    "1": "gps",
    "2": "dgps",
    "3": "pps",
    "4": "rtk_fixed",
    "5": "rtk_float",
    "6": "estimated",
    "7": "manual",
    "8": "simulation",
}


def _number(value, converter=float):
    if value in (None, ""):
        return None
    try:
        result = converter(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if isinstance(result, float) and not math.isfinite(result):
        return None
    return result


def _sample_age_seconds(sample, now_monotonic_ns):
    sample_ns = _number((sample or {}).get("host_monotonic_ns"), int)
    if sample_ns is None:
        return None
    return max(0.0, (now_monotonic_ns - sample_ns) / 1_000_000_000.0)


def _atomic_write(path, content):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
    os.replace(str(temporary), str(path))


class ControllerBridge:
    """Write the latest sensor state for the local Jetson control service."""

    def __init__(self, args):
        self.enabled = bool(getattr(args, "controller_bridge_enabled", True))
        self.root = Path(
            getattr(args, "controller_bridge_dir", "/var/lib/jetson-sensors")
        ).expanduser()
        self.status_path = self.root / "status.json"
        self.preview_path = self.root / "camera-preview.jpg"
        self.status_interval_s = max(
            0.1, float(getattr(args, "controller_status_interval_s", 1.0))
        )
        self.sensor_stale_after_s = max(
            0.5, float(getattr(args, "controller_sensor_stale_after_s", 3.0))
        )
        self.preview_fps = max(
            0.0, float(getattr(args, "controller_preview_fps", 4.0))
        )
        self.preview_max_width = max(
            320, int(getattr(args, "controller_preview_max_width", 1280))
        )
        self.preview_jpeg_quality = max(
            1, min(100, int(getattr(args, "controller_preview_jpeg_quality", 78)))
        )
        self.rotate_180 = bool(getattr(args, "rotate_180", False))
        self.flip = bool(getattr(args, "flip", False))
        self.camera_configured = True
        self.gnss_configured = bool(getattr(args, "enable_gps", False))
        self.imu_configured = True
        self.external_imu_configured = bool(
            getattr(args, "enable_external_imu", False)
        )

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._stop = False
        self._pipeline_active = True
        self._pipeline_error = None
        self._device_connected = False
        self._last_camera_monotonic = None
        self._last_camera_epoch_ms = None
        self._last_imu_monotonic = None
        self._last_imu_epoch_ms = None
        self._camera_width = None
        self._camera_height = None
        self._last_preview_epoch_ms = None
        self._preview_error = None
        self._next_preview_monotonic = 0.0
        self._last_status_monotonic = 0.0
        self._pending_rgb = None
        self._preview_thread = None

        if not self.enabled:
            return
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self.enabled = False
            print(f"Controller bridge disabled: {error}")
            return
        if self.preview_fps > 0:
            self._preview_thread = threading.Thread(
                target=self._preview_worker,
                name="controller-preview",
                daemon=True,
            )
            self._preview_thread.start()

    def mark_device_connected(self):
        with self._lock:
            self._device_connected = True

    def offer_rgb(self, message):
        now_monotonic = time.monotonic()
        now_epoch_ms = int(time.time() * 1000)
        with self._condition:
            self._last_camera_monotonic = now_monotonic
            self._last_camera_epoch_ms = now_epoch_ms
            try:
                self._camera_width = int(message.getWidth())
                self._camera_height = int(message.getHeight())
            except Exception:
                pass
            if (
                self._preview_thread is not None
                and now_monotonic >= self._next_preview_monotonic
            ):
                self._pending_rgb = message
                self._next_preview_monotonic = now_monotonic + 1.0 / self.preview_fps
                self._condition.notify()

    def observe_imu(self):
        with self._lock:
            self._last_imu_monotonic = time.monotonic()
            self._last_imu_epoch_ms = int(time.time() * 1000)

    def _preview_worker(self):
        while True:
            with self._condition:
                while self._pending_rgb is None and not self._stop:
                    self._condition.wait()
                if self._stop and self._pending_rgb is None:
                    return
                message = self._pending_rgb
                self._pending_rgb = None
            try:
                frame = runtime.get_color_cv_frame(message)
                self.write_preview_frame(frame)
            except Exception as error:  # Preview failures must not stop recording.
                with self._lock:
                    self._preview_error = str(error)

    def write_preview_frame(self, frame):
        if self.flip:
            frame = cv2.flip(frame, 0)
        if self.rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        height, width = frame.shape[:2]
        if width > self.preview_max_width:
            scale = self.preview_max_width / float(width)
            frame = cv2.resize(
                frame,
                (self.preview_max_width, max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.preview_jpeg_quality],
        )
        if not ok:
            raise RuntimeError("Failed to encode controller camera preview")
        _atomic_write(self.preview_path, encoded.tobytes())
        with self._lock:
            self._last_preview_epoch_ms = int(time.time() * 1000)
            self._preview_error = None

    @staticmethod
    def _latest_reader_sample(reader):
        if reader is None:
            return {}
        try:
            return dict(reader.latest_sample() or {})
        except Exception:
            return {}

    def _gnss_state(self, reader, now_epoch_ms, now_monotonic_ns):
        sample = self._latest_reader_sample(reader)
        age_s = _sample_age_seconds(sample, now_monotonic_ns)
        active = age_s is not None and age_s <= self.sensor_stale_after_s
        quality = str(sample.get("fix_quality") or "")
        details = runtime.gps_fix_details(quality)
        ntrip = getattr(reader, "rtk_client", None) if reader is not None else None
        last_sample_epoch_ms = (
            now_epoch_ms - int(age_s * 1000) if age_s is not None else None
        )
        return {
            "configured": self.gnss_configured,
            "connected": bool(reader and getattr(reader, "started", False) and not getattr(reader, "error", None)),
            "active": active,
            "lastSampleAtEpochMillis": last_sample_epoch_ms,
            "fixQuality": _number(quality, int),
            "fixType": FIX_TYPES.get(quality, "unknown" if quality else "none"),
            "fixName": details["fix_quality_name"],
            "rtkStatus": details["rtk_status"],
            "latitude": _number(sample.get("latitude_deg")),
            "longitude": _number(sample.get("longitude_deg")),
            "altitudeM": _number(sample.get("altitude_m")),
            "satellites": _number(sample.get("satellites"), int),
            "hdop": _number(sample.get("hdop")),
            "differentialAgeS": _number(sample.get("differential_age_s")),
            "referenceStationId": sample.get("reference_station_id") or None,
            "ntripConnected": bool(ntrip and getattr(ntrip, "connected", False)),
            "ntripMountpoint": getattr(ntrip, "current_mountpoint", None) if ntrip else None,
            "rtcmBytes": int(getattr(ntrip, "bytes_received", 0) or 0) if ntrip else 0,
            "error": getattr(reader, "error", None) if reader is not None else None,
        }

    def _imu_state(self, reader, now_epoch_ms, now_monotonic):
        serial_sample = self._latest_reader_sample(reader)
        serial_age_s = _sample_age_seconds(serial_sample, int(now_monotonic * 1_000_000_000))
        with self._lock:
            oak_age_s = (
                now_monotonic - self._last_imu_monotonic
                if self._last_imu_monotonic is not None
                else None
            )
            oak_epoch_ms = self._last_imu_epoch_ms
        oak_active = oak_age_s is not None and oak_age_s <= self.sensor_stale_after_s
        serial_active = serial_age_s is not None and serial_age_s <= self.sensor_stale_after_s
        source = None
        age_s = None
        last_sample_epoch_ms = None
        if oak_active and serial_active:
            source = "oak+external"
            age_s = min(oak_age_s, serial_age_s)
            last_sample_epoch_ms = max(
                oak_epoch_ms or 0,
                now_epoch_ms - int(serial_age_s * 1000),
            )
        elif oak_age_s is not None and (serial_age_s is None or oak_age_s <= serial_age_s):
            source = "oak"
            age_s = oak_age_s
            last_sample_epoch_ms = oak_epoch_ms
        elif serial_age_s is not None:
            source = "external"
            age_s = serial_age_s
            last_sample_epoch_ms = now_epoch_ms - int(serial_age_s * 1000)
        return {
            "configured": self.imu_configured or self.external_imu_configured,
            "connected": bool(
                self._device_connected
                or (reader and getattr(reader, "started", False) and not getattr(reader, "error", None))
            ),
            "active": age_s is not None and age_s <= self.sensor_stale_after_s,
            "lastSampleAtEpochMillis": last_sample_epoch_ms,
            "source": source,
            "error": getattr(reader, "error", None) if reader is not None else None,
        }

    def snapshot(self, serial_readers=None):
        now_epoch_ms = int(time.time() * 1000)
        now_monotonic = time.monotonic()
        with self._lock:
            camera_age_s = (
                now_monotonic - self._last_camera_monotonic
                if self._last_camera_monotonic is not None
                else None
            )
            camera = {
                "configured": self.camera_configured,
                "connected": self._device_connected,
                "active": camera_age_s is not None and camera_age_s <= self.sensor_stale_after_s,
                "lastFrameAtEpochMillis": self._last_camera_epoch_ms,
                "frameWidth": self._camera_width,
                "frameHeight": self._camera_height,
                "previewAvailable": self._last_preview_epoch_ms is not None,
                "previewUpdatedAtEpochMillis": self._last_preview_epoch_ms,
                "previewError": self._preview_error,
            }
            pipeline = {
                "active": self._pipeline_active,
                "error": self._pipeline_error,
            }
        readers = serial_readers or {}
        gnss = self._gnss_state(
            readers.get("gps"), now_epoch_ms, time.monotonic_ns()
        )
        imu = self._imu_state(
            readers.get("external_imu"), now_epoch_ms, now_monotonic
        )
        if not pipeline["active"]:
            camera["connected"] = False
            camera["active"] = False
            camera["previewAvailable"] = False
            gnss["connected"] = False
            gnss["active"] = False
            gnss["ntripConnected"] = False
            imu["connected"] = False
            imu["active"] = False
        return {
            "schemaVersion": 1,
            "updatedAtEpochMillis": now_epoch_ms,
            "pipeline": pipeline,
            "camera": camera,
            "gnss": gnss,
            "imu": imu,
        }

    def publish(self, serial_readers=None, force=False):
        if not self.enabled:
            return False
        now = time.monotonic()
        if not force and now - self._last_status_monotonic < self.status_interval_s:
            return False
        payload = json.dumps(
            self.snapshot(serial_readers),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self._last_status_monotonic = now
        try:
            _atomic_write(self.status_path, payload)
        except OSError as error:
            print(f"Controller bridge status write failed: {error}")
            return False
        return True

    def close(self, serial_readers=None, error=None):
        if not self.enabled:
            return
        with self._condition:
            self._pipeline_active = False
            if error:
                self._pipeline_error = str(error)
            self._stop = True
            self._pending_rgb = None
            self._condition.notify_all()
        if self._preview_thread is not None:
            self._preview_thread.join(timeout=3.0)
        self.publish(serial_readers, force=True)
