import csv
import json
import queue
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from geonova_depthai import runtime


RGB_EVENT_FIELDS = [
    "event_index", "stem", "file", "sequence", "device_ts_ns", "frame_type",
    "capture_wall_time", "capture_monotonic_ns", "dequeue_wall_time",
    "dequeue_monotonic_ns", "queue_lag_ms", "width", "height",
]

DEPTH_EVENT_FIELDS = [
    "event_index", "stem", "file", "sequence", "device_ts_ns", "frame_type",
    "capture_wall_time", "capture_monotonic_ns", "dequeue_wall_time",
    "dequeue_monotonic_ns", "queue_lag_ms", "width", "height", "units",
]

CONFIDENCE_EVENT_FIELDS = [
    "event_index", "stem", "file", "sequence", "device_ts_ns", "frame_type",
    "capture_wall_time", "capture_monotonic_ns", "dequeue_wall_time",
    "dequeue_monotonic_ns", "queue_lag_ms", "width", "height",
]

IMU_EVENT_FIELDS = [
    "message_index", "packet_index", "message_device_ts_ns", "message_sequence",
    "message_capture_wall_time", "message_capture_monotonic_ns",
    "message_dequeue_wall_time", "message_dequeue_monotonic_ns", "message_queue_lag_ms",
    "accel_device_ts_ns", "gyro_device_ts_ns",
    "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",
    "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
]

SERIAL_FIELDS = [
    "sample_index", "source", "device", "host_wall_time", "host_monotonic_ns", "raw",
    "nmea_type", "gps_time_utc", "date_utc", "latitude_deg", "longitude_deg",
    "altitude_m", "fix_quality", "fix_quality_name", "rtk_status", "rtk_fixed",
    "rtk_corrected", "position_valid", "satellites", "hdop", "status",
    "speed_knots", "course_deg", "geoid_separation_m", "differential_age_s",
    "reference_station_id", "measurement_wall_time_utc", "measurement_host_monotonic_ns",
    "receive_latency_ms", "external_imu_format", "orientation_format", "q_x", "q_y", "q_z", "q_w",
    "roll_deg", "pitch_deg", "yaw_deg", "gyro_x", "gyro_y", "gyro_z", "accel_x",
    "accel_y", "accel_z", "mag_x", "mag_y", "mag_z", "ebimu_timestamp_ms",
]


def _target_size(args):
    return runtime.rgb_size_from_args(args)


def _ensure_size(frame, args, interpolation):
    width, height = _target_size(args)
    if frame.shape[:2] != (height, width):
        return cv2.resize(frame, (width, height), interpolation=interpolation)
    return frame


def _apply_saved_image_transform(frame, args, interpolation):
    frame = _ensure_size(frame, args, interpolation)
    if getattr(args, "flip", False):
        frame = cv2.flip(frame, 0)
    if getattr(args, "rotate_180", False):
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    return frame


class ThreadSafeCsv:
    def __init__(self, path: Path, fieldnames):
        self.path = path
        self.file = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames, extrasaction="ignore")
        self.writer.writeheader()
        self.lock = threading.Lock()

    def writerow(self, row: Dict[str, Any]):
        with self.lock:
            self.writer.writerow({key: row.get(key, "") for key in self.writer.fieldnames})

    def close(self):
        with self.lock:
            self.file.flush()
            self.file.close()


class ImageWritePool:
    def __init__(self, dataset, worker_count=4, max_queue=256):
        self.dataset = dataset
        self.jobs = queue.Queue(maxsize=max_queue)
        self.stop_token = object()
        self.errors = []
        self.workers = [
            threading.Thread(target=self._run, name=f"image-writer-{idx}", daemon=True)
            for idx in range(max(1, int(worker_count)))
        ]
        for worker in self.workers:
            worker.start()

    def submit(self, job: Dict[str, Any]):
        self.jobs.put(job)

    def close(self):
        self.jobs.join()
        for _ in self.workers:
            self.jobs.put(self.stop_token)
        for worker in self.workers:
            worker.join(timeout=5.0)
        if self.errors:
            raise RuntimeError("Image writer failed: " + "; ".join(self.errors[:3]))

    def _run(self):
        while True:
            job = self.jobs.get()
            try:
                if job is self.stop_token:
                    return
                self.dataset.write_image_event(job)
            except Exception as exc:  # noqa: BLE001 - record and continue draining
                self.errors.append(str(exc))
            finally:
                self.jobs.task_done()


class RawEventDataset:
    """A dataset root containing unsynchronised per-stream event files.

    The post-processor can later create timestamps.csv and synced imu.csv in the same root.
    """

    def __init__(
        self,
        output_dir,
        args,
        camera_model: Optional[Dict[str, Any]] = None,
        stereo_depth_model: Optional[Dict[str, Any]] = None,
    ):
        self.args = args
        self.started_wall = datetime.now()
        self.root = Path(output_dir) / self.started_wall.strftime("%Y-%m-%d_%H-%M-%S_raw")
        self.root.mkdir(parents=True, exist_ok=True)
        self.rgb_dir = self.root / "rgb"
        self.depth_dir = self.root / "depth_mm"
        self.confidence_dir = self.root / "confidence"
        self.rgb_dir.mkdir(exist_ok=True)
        self.depth_dir.mkdir(exist_ok=True)
        if args.save_confidence_map:
            self.confidence_dir.mkdir(exist_ok=True)

        self.rgb_events = ThreadSafeCsv(self.root / "rgb_events.csv", RGB_EVENT_FIELDS)
        self.depth_events = ThreadSafeCsv(self.root / "depth_events.csv", DEPTH_EVENT_FIELDS)
        self.confidence_events = ThreadSafeCsv(self.root / "confidence_events.csv", CONFIDENCE_EVENT_FIELDS) if args.save_confidence_map else None
        self.imu_events = ThreadSafeCsv(self.root / "imu_events.csv", IMU_EVENT_FIELDS)
        self.gps_events = ThreadSafeCsv(self.root / "gps.csv", SERIAL_FIELDS) if args.enable_gps else None
        self.external_imu_events = ThreadSafeCsv(self.root / "external_imu.csv", SERIAL_FIELDS) if args.enable_external_imu else None

        self.counters = {"rgb": 0, "depth": 0, "confidence": 0, "imu": 0, "gps": 0, "external_imu": 0}
        self.counter_lock = threading.Lock()
        self.metadata_lock = threading.Lock()
        self.camera_model_updated_from_rgb = False
        self.metadata = self._build_metadata(camera_model or {}, stereo_depth_model or {})
        self.write_metadata(self.metadata)

    def next_index(self, stream):
        with self.counter_lock:
            value = self.counters[stream]
            self.counters[stream] += 1
            return value

    def _build_metadata(self, camera_model, stereo_depth_model):
        args = self.args
        width, height = _target_size(args)
        camera_model = dict(camera_model)
        stereo_depth_model = dict(stereo_depth_model)
        camera_model["image_stream_undistorted"] = True
        if camera_model.get("image_stream_undistorted"):
            camera_model["coordinate_unprojection"] = "pinhole on already-undistorted saved RGB/depth pixels"
            camera_model["notes"] = "RGB was requested with enableUndistortion=True; aligned depth uses the same saved pixel geometry. Do not apply cv2.undistortPoints again."
        return {
            "format_version": "raw_events_v1",
            "created_wall_time": self.started_wall.isoformat(timespec="milliseconds"),
            "capture_mode": "per_stream_fast_events_then_postprocess_sync",
            "software_versions": {
                "python": sys.version.split()[0],
                "depthai": getattr(runtime.dai, "__version__", ""),
                "opencv": getattr(cv2, "__version__", ""),
                "numpy": getattr(np, "__version__", ""),
            },
            "depthai_device": {
                "name": getattr(args, "depthai_device_name", ""),
                "id": getattr(args, "depthai_device_id", ""),
                "platform": getattr(args, "depthai_platform", "unknown"),
            },
            "camera_model": camera_model,
            "stereo_depth_model": stereo_depth_model,
            "image_size": {"width": width, "height": height},
            "camera_sockets": {
                "rgb": getattr(args, "rgb_socket_name", "CAM_A"),
                "stereo_left": getattr(args, "left_socket_name", "CAM_B"),
                "stereo_right": getattr(args, "right_socket_name", "CAM_C"),
            },
            "rgb_sensor": {
                "name": getattr(args, "rgb_sensor_name", ""),
                "width": int(getattr(args, "rgb_sensor_width", 0) or 0),
                "height": int(getattr(args, "rgb_sensor_height", 0) or 0),
                "types": list(getattr(args, "rgb_sensor_types", []) or []),
                "resolution_source": getattr(args, "rgb_resolution_source", "unknown"),
            },
            "stereo_sensors": {
                "left": dict(getattr(args, "left_sensor", {}) or {}),
                "right": dict(getattr(args, "right_sensor", {}) or {}),
                "stereo_matching_input_size": {
                    "width": int(getattr(args, "depth_input_width", 0) or 0),
                    "height": int(getattr(args, "depth_input_height", 0) or 0),
                },
                "aligned_depth_output_size": {"width": width, "height": height},
                "input_size": {
                    "width": int(getattr(args, "depth_input_width", 0) or 0),
                    "height": int(getattr(args, "depth_input_height", 0) or 0),
                },
                "resolution_source": getattr(args, "depth_input_resolution_source", "unknown"),
            },
            "image_transform": {
                "flip_vertical": bool(getattr(args, "flip", False)),
                "rotate_180": bool(getattr(args, "rotate_180", False)),
                "operation_order": ["flip_vertical", "rotate_180"],
            },
            "rgb_format": args.rgb_format,
            "rgb_jpeg_quality": args.rgb_jpeg_quality if args.rgb_format == "jpg" else None,
            "requested_fps": float(args.fps),
            "requested_depth_fps": float(getattr(args, "depth_fps", 0.0) or 0.0),
            "depth_fps_effective": float(getattr(args, "depth_fps_effective", args.fps)),
            "host_transport": {
                "usb_speed": getattr(args, "usb_speed", "UNKNOWN"),
                "rgb_transport": getattr(args, "rgb_transport_effective", getattr(args, "rgb_transport", "")),
                "confidence_transport": getattr(
                    args,
                    "confidence_transport_effective",
                    getattr(args, "confidence_transport", ""),
                ),
                "queue_size": int(getattr(args, "queue_size", 0) or 0),
                "writer_threads": int(getattr(args, "writer_threads", 0) or 0),
            },
            "depth_format": "uint16_png_mm",
            "confidence_map": {"saved": bool(args.save_confidence_map), "directory": "confidence" if args.save_confidence_map else ""},
            "stereo_config": {
                "depth_preset": getattr(args, "depth_preset", ""),
                "lr_check": bool(getattr(args, "lr_check", False)),
                "subpixel": bool(getattr(args, "subpixel", False)),
                "subpixel_fractional_bits": int(getattr(args, "subpixel_fractional_bits", 0) or 0),
                "median_filter": getattr(args, "stereo_median_filter_effective", getattr(args, "stereo_median_filter", "")),
                "extended_disparity": False,
            },
            "depth_alignment": {
                "enabled": True,
                "mode": getattr(args, "depth_alignment_effective", args.depth_alignment_mode),
                "requested_mode": args.depth_alignment_mode,
                "platform": getattr(args, "depthai_platform", "unknown"),
                "aligned_to": "RGB",
                "aligned_to_socket": getattr(args, "rgb_socket_name", "CAM_A"),
                "method": getattr(args, "depth_alignment_effective", args.depth_alignment_mode),
                "depth_output_size": {"width": width, "height": height},
                "stereo_matching_input_size": {
                    "width": int(getattr(args, "depth_input_width", 0) or 0),
                    "height": int(getattr(args, "depth_input_height", 0) or 0),
                },
                "rgb_undistort": True,
                "rgb_geometry": f"OAK factory-undistorted {getattr(args, 'rgb_socket_name', 'CAM_A')} output",
                "depth_pixel_coordinates_match_rgb": True,
                "notes": (
                    "Production capture requires factory-undistorted RGB. "
                    "DepthAI v3 platform-specific alignment is selected automatically. "
                    "RVC2/RVC3 link the exact RGB output to StereoDepth.inputAlignTo; "
                    "RVC4 uses ImageAlign. Postprocess pairs events by device timestamp."
                ),
            },
            "sync": {
                "mode": "postprocess",
                "threshold_ms": float(args.sync_threshold_ms),
                "strategy": "RGB anchored nearest depth/IMU/GPS/external-IMU event",
            },
        }

    def write_metadata(self, metadata):
        with open(self.root / "metadata.json", "w") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)

    def update_camera_model_from_rgb_frame(self, msg):
        if self.camera_model_updated_from_rgb:
            return
        frame_model = runtime.imgframe_camera_model_metadata(msg, self.args)
        if not frame_model:
            return
        with self.metadata_lock:
            if self.camera_model_updated_from_rgb:
                return
            camera_model = dict(self.metadata.get("camera_model") or {})
            camera_model.update(frame_model)
            self.metadata["camera_model"] = camera_model
            self.camera_model_updated_from_rgb = True
            self.write_metadata(self.metadata)

    def write_image_event(self, job):
        stream = job["stream"]
        msg = job["message"]
        index = job["event_index"]
        stamp = job["stamp"]
        try:
            capture_dt = datetime.fromisoformat(stamp.get("capture_wall_time"))
        except Exception:  # noqa: BLE001
            capture_dt = datetime.now()
        stem = f"{capture_dt.strftime('%Y-%m-%d-%H-%M-%S')}-{stream}{index:07d}"

        if stream == "rgb":
            self.update_camera_model_from_rgb_frame(msg)
            frame = runtime.get_color_cv_frame(msg)
            frame = _apply_saved_image_transform(frame, self.args, cv2.INTER_AREA)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            filename = f"{stem}_rgb.{self.args.rgb_format}"
            path = self.rgb_dir / filename
            if self.args.rgb_format == "jpg":
                ok = cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, self.args.rgb_jpeg_quality])
            else:
                ok = cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, self.args.rgb_png_compression])
            if not ok:
                raise RuntimeError(f"Failed to write RGB: {path}")
            self.rgb_events.writerow(self._image_row(index, stem, f"rgb/{filename}", msg, stamp, frame, stream))
            return

        if stream == "depth":
            frame = msg.getFrame()
            frame = _apply_saved_image_transform(frame, self.args, cv2.INTER_NEAREST)
            if frame.dtype != np.uint16:
                frame = frame.astype(np.uint16)
            filename = f"{stem}_depth_mm.png"
            path = self.depth_dir / filename
            ok = cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, self.args.depth_png_compression])
            if not ok:
                raise RuntimeError(f"Failed to write depth: {path}")
            row = self._image_row(index, stem, f"depth_mm/{filename}", msg, stamp, frame, stream)
            row["units"] = "millimeters"
            self.depth_events.writerow(row)
            return

        if stream == "confidence":
            frame = runtime.get_confidence_cv_frame(msg)
            frame = _apply_saved_image_transform(frame, self.args, cv2.INTER_NEAREST)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            filename = f"{stem}_confidence.png"
            path = self.confidence_dir / filename
            ok = cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, self.args.confidence_png_compression])
            if not ok:
                raise RuntimeError(f"Failed to write confidence: {path}")
            if self.confidence_events is not None:
                self.confidence_events.writerow(self._image_row(index, stem, f"confidence/{filename}", msg, stamp, frame, stream))
            return

    def _image_row(self, index, stem, filename, msg, stamp, frame, stream):
        width, height = _target_size(self.args)
        return {
            "event_index": index,
            "stem": stem,
            "file": filename,
            "sequence": runtime.get_sequence_num(msg),
            "device_ts_ns": runtime.get_device_ts_ns(msg),
            "frame_type": runtime.imgframe_type_name(msg),
            "capture_wall_time": stamp.get("capture_wall_time", ""),
            "capture_monotonic_ns": stamp.get("capture_monotonic_ns", ""),
            "dequeue_wall_time": stamp.get("dequeue_wall_time", ""),
            "dequeue_monotonic_ns": stamp.get("dequeue_monotonic_ns", ""),
            "queue_lag_ms": stamp.get("queue_lag_ms", ""),
            "width": frame.shape[1] if hasattr(frame, "shape") else width,
            "height": frame.shape[0] if hasattr(frame, "shape") else height,
        }

    def write_imu_message(self, msg, stamp):
        message_index = self.next_index("imu")
        imu_ts_ns = runtime.get_device_ts_ns(msg)
        packets = getattr(msg, "packets", []) or []
        for packet_index, packet in enumerate(packets):
            accel = getattr(packet, "acceleroMeter", None)
            gyro = getattr(packet, "gyroscope", None)
            self.imu_events.writerow({
                "message_index": message_index,
                "packet_index": packet_index,
                "message_device_ts_ns": imu_ts_ns,
                "message_sequence": runtime.get_sequence_num(msg),
                "message_capture_wall_time": stamp.get("capture_wall_time", ""),
                "message_capture_monotonic_ns": stamp.get("capture_monotonic_ns", ""),
                "message_dequeue_wall_time": stamp.get("dequeue_wall_time", ""),
                "message_dequeue_monotonic_ns": stamp.get("dequeue_monotonic_ns", ""),
                "message_queue_lag_ms": stamp.get("queue_lag_ms", ""),
                "accel_device_ts_ns": runtime.get_device_ts_ns(accel) if accel is not None else "",
                "gyro_device_ts_ns": runtime.get_device_ts_ns(gyro) if gyro is not None else "",
                "accel_x_m_s2": getattr(accel, "x", "") if accel is not None else "",
                "accel_y_m_s2": getattr(accel, "y", "") if accel is not None else "",
                "accel_z_m_s2": getattr(accel, "z", "") if accel is not None else "",
                "gyro_x_rad_s": getattr(gyro, "x", "") if gyro is not None else "",
                "gyro_y_rad_s": getattr(gyro, "y", "") if gyro is not None else "",
                "gyro_z_rad_s": getattr(gyro, "z", "") if gyro is not None else "",
            })

    def write_serial_samples(self, name, samples):
        writer = self.gps_events if name == "gps" else self.external_imu_events
        if writer is None:
            return
        for sample in samples:
            writer.writerow(sample)
            self.counters[name] += 1

    def close(self):
        for writer in [self.rgb_events, self.depth_events, self.confidence_events, self.imu_events, self.gps_events, self.external_imu_events]:
            if writer is not None:
                writer.close()
        self.metadata["finished_wall_time"] = datetime.now().isoformat(timespec="milliseconds")
        self.metadata["raw_event_counts"] = dict(self.counters)
        self.write_metadata(self.metadata)
