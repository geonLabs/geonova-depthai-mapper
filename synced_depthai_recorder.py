#!/usr/bin/env python3

import argparse
import csv
import json
import queue
import signal
import shutil
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import depthai as dai
import numpy as np


WIDTH = 1280
HEIGHT = 720
MIN_DEPTHAI_MAJOR = 3
stop_requested = False


def request_stop(signum, frame):
    global stop_requested
    stop_requested = True


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Value must be >= 1.")
    return number


def png_compression_level(value):
    number = int(value)
    if number < 0 or number > 9:
        raise argparse.ArgumentTypeError("PNG compression must be between 0 and 9.")
    return number


def time_to_ns(value):
    if value is None:
        return ""
    if hasattr(value, "total_seconds"):
        return int(value.total_seconds() * 1_000_000_000)
    return ""


def ns_delta_ms(a_ns, b_ns):
    if a_ns == "" or b_ns == "":
        return ""
    return (a_ns - b_ns) / 1_000_000.0


def get_sequence_num(message):
    if hasattr(message, "getSequenceNum"):
        return message.getSequenceNum()
    return ""


def get_device_ts_ns(message):
    if hasattr(message, "getTimestampDevice"):
        return time_to_ns(message.getTimestampDevice())
    if hasattr(message, "getTimestamp"):
        return time_to_ns(message.getTimestamp())
    return ""


def get_group_item(group, name):
    try:
        return group[name]
    except KeyError:
        pass
    raise KeyError(f"Sync group did not contain '{name}'")


def require_depthai_v3():
    major = int(dai.__version__.split(".", 1)[0])
    if major < MIN_DEPTHAI_MAJOR:
        raise RuntimeError(
            f"This recorder is DepthAI v3-only. Found depthai=={dai.__version__}; "
            "activate the new .venv or install depthai>=3."
        )


def enum_by_name(enum_cls, name):
    try:
        return getattr(enum_cls, name)
    except AttributeError as exc:
        valid_names = ", ".join(item for item in dir(enum_cls) if item.isupper())
        raise argparse.ArgumentTypeError(f"Invalid value '{name}'. Valid values: {valid_names}") from exc


def open_video_writer(path, fps, size, preferred_codec):
    codecs = [preferred_codec, "mp4v", "avc1", "H264"]
    tried = []
    for codec in codecs:
        if codec in tried or len(codec) != 4:
            continue
        tried.append(codec)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError(f"Could not open video writer for {path}. Tried: {', '.join(tried)}")


def request_camera_output(camera, fps, size=(WIDTH, HEIGHT)):
    capability = dai.ImgFrameCapability()
    capability.size.fixed(size)
    capability.fps.fixed(fps)
    return camera.requestOutput(capability, True)


def update_metadata(metadata_path, updates):
    try:
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)
    except FileNotFoundError:
        metadata = {}

    metadata.update(updates)
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def compress_depth_raw(depth_path, metadata_path):
    if not depth_path.exists():
        return
    if shutil.which("gzip") is None:
        print(f"gzip not found; leaving uncompressed depth raw: {depth_path}")
        return

    print(f"Compressing depth raw: {depth_path}")
    result = subprocess.run(["gzip", "-f", str(depth_path)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Depth compression failed for {depth_path}: {result.stderr.strip()}")
        return

    gz_path = depth_path.with_name(depth_path.name + ".gz")
    update_metadata(metadata_path, {
        "depth_file": gz_path.name,
        "depth_compressed": True,
        "depth_compression": "gzip",
    })
    print(f"Compressed depth raw: {gz_path}")


def depth_compression_worker(work_queue):
    while True:
        item = work_queue.get()
        try:
            if item is None:
                return
            compress_depth_raw(*item)
        finally:
            work_queue.task_done()


def enqueue_depth_compression(work_queue, writer, args):
    if (
        work_queue is not None
        and args.compress_depth_raw
        and writer.depth_format in ("raw", "both")
        and writer.depth_frame_count > 0
    ):
        work_queue.put((writer.depth_path, writer.metadata_path))


def configure_pipeline(pipeline, args):
    require_depthai_v3()

    color_socket = dai.CameraBoardSocket.CAM_A
    left_socket = dai.CameraBoardSocket.CAM_B
    right_socket = dai.CameraBoardSocket.CAM_C

    cam_rgb = pipeline.create(dai.node.Camera).build(color_socket)
    mono_left = pipeline.create(dai.node.Camera).build(left_socket)
    mono_right = pipeline.create(dai.node.Camera).build(right_socket)
    stereo = pipeline.create(dai.node.StereoDepth)
    imu = pipeline.create(dai.node.IMU)
    sync = pipeline.create(dai.node.Sync)

    rgb_output = request_camera_output(cam_rgb, args.fps)
    left_output = request_camera_output(mono_left, args.fps)
    right_output = request_camera_output(mono_right, args.fps)

    stereo.setDefaultProfilePreset(enum_by_name(dai.node.StereoDepth.PresetMode, args.depth_preset))
    stereo.setLeftRightCheck(args.lr_check)
    stereo.setSubpixel(args.subpixel)
    stereo.setExtendedDisparity(False)
    if hasattr(stereo, "setDepthAlign"):
        stereo.setDepthAlign(color_socket)
    if hasattr(stereo, "setOutputSize"):
        stereo.setOutputSize(WIDTH, HEIGHT)

    imu.enableIMUSensor([
        dai.IMUSensor.ACCELEROMETER_RAW,
        dai.IMUSensor.GYROSCOPE_RAW,
    ], args.imu_rate)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(args.imu_batch)

    sync.setSyncThreshold(timedelta(milliseconds=args.sync_threshold_ms))
    sync.setSyncAttempts(args.sync_attempts)

    left_output.link(stereo.left)
    right_output.link(stereo.right)
    rgb_output.link(sync.inputs["rgb"])
    stereo.depth.link(sync.inputs["depth"])
    imu.out.link(sync.inputs["imu"])

    return sync.out, imu.out, rgb_output, stereo.depth


class SegmentWriter:
    def __init__(
        self,
        output_dir,
        index,
        fps,
        flip,
        rgb_codec,
        save_depth_preview,
        depth_preview_max_mm,
        depth_format,
        depth_png_compression,
        depth_save_every,
    ):
        self.index = index
        self.fps = fps
        self.flip = flip
        self.save_depth_preview = save_depth_preview
        self.depth_preview_max_mm = depth_preview_max_mm
        self.depth_format = depth_format
        self.depth_png_compression = depth_png_compression
        self.depth_save_every = depth_save_every
        self.started_monotonic = time.monotonic()
        self.started_wall = datetime.now()
        self.name = self.started_wall.strftime("%Y-%m-%d_%H-%M-%S")
        self.segment_dir = Path(output_dir) / self.name
        self.segment_dir.mkdir(parents=True, exist_ok=True)

        self.rgb_path = self.segment_dir / "rgb.mp4"
        self.depth_path = self.segment_dir / "depth_mm.raw"
        self.depth_dir = self.segment_dir / "depth_mm_png"
        self.imu_path = self.segment_dir / "imu.csv"
        self.synced_imu_path = self.segment_dir / "synced_imu.csv"
        self.timestamps_path = self.segment_dir / "timestamps.csv"
        self.metadata_path = self.segment_dir / "metadata.json"
        self.depth_preview_path = self.segment_dir / "depth_preview.mp4"

        self.rgb_writer, self.rgb_codec = open_video_writer(
            self.rgb_path, fps, (WIDTH, HEIGHT), rgb_codec
        )
        self.depth_file = None
        if self.depth_format in ("raw", "both"):
            self.depth_file = open(self.depth_path, "wb")
        if self.depth_format in ("png", "both"):
            self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.imu_file = open(self.imu_path, "w", newline="")
        self.synced_imu_file = open(self.synced_imu_path, "w", newline="")
        self.timestamps_file = open(self.timestamps_path, "w", newline="")

        self.imu_writer = csv.writer(self.imu_file)
        self.imu_writer.writerow([
            "packet_index",
            "imu_message_device_ts_ns",
            "accel_device_ts_ns",
            "accel_x_m_s2",
            "accel_y_m_s2",
            "accel_z_m_s2",
            "gyro_device_ts_ns",
            "gyro_x_rad_s",
            "gyro_y_rad_s",
            "gyro_z_rad_s",
        ])

        self.synced_imu_writer = csv.writer(self.synced_imu_file)
        self.synced_imu_writer.writerow([
            "frame_index",
            "packet_index",
            "imu_message_device_ts_ns",
            "accel_device_ts_ns",
            "accel_x_m_s2",
            "accel_y_m_s2",
            "accel_z_m_s2",
            "gyro_device_ts_ns",
            "gyro_x_rad_s",
            "gyro_y_rad_s",
            "gyro_z_rad_s",
        ])

        self.timestamps_writer = csv.writer(self.timestamps_file)
        self.timestamps_writer.writerow([
            "frame_index",
            "rgb_sequence",
            "depth_sequence",
            "rgb_device_ts_ns",
            "depth_device_ts_ns",
            "imu_message_device_ts_ns",
            "rgb_depth_delta_ms",
            "rgb_imu_delta_ms",
            "depth_imu_delta_ms",
            "imu_packets",
            "depth_saved",
            "depth_saved_index",
            "depth_storage",
        ])

        self.depth_preview_writer = None
        self.depth_preview_codec = None
        if save_depth_preview:
            self.depth_preview_writer, self.depth_preview_codec = open_video_writer(
                self.depth_preview_path, fps, (WIDTH, HEIGHT), rgb_codec
            )

        self.frame_count = 0
        self.depth_frame_count = 0
        self.imu_packet_count = 0
        self.synced_imu_packet_count = 0
        print(f"Segment opened: {self.segment_dir}")

    def write_group(self, group):
        rgb_msg = get_group_item(group, "rgb")
        depth_msg = get_group_item(group, "depth")
        imu_msg = get_group_item(group, "imu")

        rgb_frame = rgb_msg.getCvFrame()
        depth_frame = depth_msg.getFrame()

        if rgb_frame.shape[:2] != (HEIGHT, WIDTH):
            rgb_frame = cv2.resize(rgb_frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        if depth_frame.shape[:2] != (HEIGHT, WIDTH):
            depth_frame = cv2.resize(depth_frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

        if self.flip:
            rgb_frame = cv2.flip(rgb_frame, 0)
            depth_frame = cv2.flip(depth_frame, 0)

        if rgb_frame.dtype != np.uint8:
            rgb_frame = rgb_frame.astype(np.uint8)
        if depth_frame.dtype != np.uint16:
            depth_frame = depth_frame.astype(np.uint16)

        self.rgb_writer.write(rgb_frame)
        depth_saved, depth_saved_index, depth_storage = self.write_depth_frame(depth_frame)
        if self.depth_preview_writer is not None:
            self.depth_preview_writer.write(self.make_depth_preview(depth_frame))

        rgb_ts_ns = get_device_ts_ns(rgb_msg)
        depth_ts_ns = get_device_ts_ns(depth_msg)
        imu_ts_ns = get_device_ts_ns(imu_msg)
        imu_packets = getattr(imu_msg, "packets", [])

        self.timestamps_writer.writerow([
            self.frame_count,
            get_sequence_num(rgb_msg),
            get_sequence_num(depth_msg),
            rgb_ts_ns,
            depth_ts_ns,
            imu_ts_ns,
            ns_delta_ms(rgb_ts_ns, depth_ts_ns),
            ns_delta_ms(rgb_ts_ns, imu_ts_ns),
            ns_delta_ms(depth_ts_ns, imu_ts_ns),
            len(imu_packets),
            depth_saved,
            depth_saved_index,
            depth_storage,
        ])

        for packet_index, packet in enumerate(imu_packets):
            self.write_synced_imu_packet(self.frame_count, packet_index, imu_ts_ns, packet)

        self.frame_count += 1

    def write_depth_frame(self, depth_frame):
        if self.depth_format == "none" or self.frame_count % self.depth_save_every != 0:
            return False, "", ""

        saved_index = self.depth_frame_count
        storage_refs = []

        if self.depth_file is not None:
            depth_frame.tofile(self.depth_file)
            storage_refs.append(self.depth_path.name)

        if self.depth_format in ("png", "both"):
            png_name = f"{saved_index:06d}.png"
            png_path = self.depth_dir / png_name
            ok = cv2.imwrite(
                str(png_path),
                depth_frame,
                [cv2.IMWRITE_PNG_COMPRESSION, self.depth_png_compression],
            )
            if not ok:
                raise RuntimeError(f"Failed to write depth PNG: {png_path}")
            storage_refs.append(f"{self.depth_dir.name}/{png_name}")

        self.depth_frame_count += 1
        return True, saved_index, ";".join(storage_refs)

    def write_imu_message(self, imu_msg):
        imu_ts_ns = get_device_ts_ns(imu_msg)
        imu_packets = getattr(imu_msg, "packets", [])
        for packet in imu_packets:
            self.write_imu_packet(imu_ts_ns, packet)

    def write_imu_packet(self, imu_message_ts_ns, packet):
        self.imu_writer.writerow(self.imu_packet_row(
            self.imu_packet_count,
            imu_message_ts_ns,
            packet,
        ))
        self.imu_packet_count += 1

    def write_synced_imu_packet(self, frame_index, packet_index, imu_message_ts_ns, packet):
        self.synced_imu_writer.writerow([
            frame_index,
            packet_index,
            *self.imu_packet_row(None, imu_message_ts_ns, packet)[1:],
        ])
        self.synced_imu_packet_count += 1

    def imu_packet_row(self, packet_index, imu_message_ts_ns, packet):
        accel = getattr(packet, "acceleroMeter", None)
        gyro = getattr(packet, "gyroscope", None)

        accel_ts_ns = get_device_ts_ns(accel)
        gyro_ts_ns = get_device_ts_ns(gyro)

        return [
            packet_index,
            imu_message_ts_ns,
            accel_ts_ns,
            getattr(accel, "x", ""),
            getattr(accel, "y", ""),
            getattr(accel, "z", ""),
            gyro_ts_ns,
            getattr(gyro, "x", ""),
            getattr(gyro, "y", ""),
            getattr(gyro, "z", ""),
        ]

    def make_depth_preview(self, depth_frame):
        clipped = np.clip(depth_frame, 0, self.depth_preview_max_mm)
        scaled = (clipped * (255.0 / self.depth_preview_max_mm)).astype(np.uint8)
        return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)

    def should_rotate(self, segment_duration):
        return time.monotonic() - self.started_monotonic >= segment_duration

    def close(self):
        self.rgb_writer.release()
        if self.depth_preview_writer is not None:
            self.depth_preview_writer.release()
        if self.depth_file is not None:
            self.depth_file.close()
        self.imu_file.close()
        self.synced_imu_file.close()
        self.timestamps_file.close()

        metadata = {
            "segment_index": self.index,
            "started_wall_time": self.started_wall.isoformat(timespec="milliseconds"),
            "closed_wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "width": WIDTH,
            "height": HEIGHT,
            "fps": self.fps,
            "rgb_file": self.rgb_path.name,
            "rgb_codec": self.rgb_codec,
            "depth_format": self.depth_format,
            "depth_file": self.depth_path.name if self.depth_format in ("raw", "both") else None,
            "depth_compressed": False,
            "depth_compression": None,
            "depth_png_dir": self.depth_dir.name if self.depth_format in ("png", "both") else None,
            "depth_dtype": "uint16",
            "depth_units": "millimeters",
            "depth_shape_per_frame": [HEIGHT, WIDTH],
            "depth_frame_count": self.depth_frame_count,
            "depth_save_every": self.depth_save_every,
            "depth_png_compression": self.depth_png_compression if self.depth_format in ("png", "both") else None,
            "imu_file": self.imu_path.name,
            "synced_imu_file": self.synced_imu_path.name,
            "timestamps_file": self.timestamps_path.name,
            "frame_count": self.frame_count,
            "imu_packet_count": self.imu_packet_count,
            "synced_imu_packet_count": self.synced_imu_packet_count,
            "flipped_vertical": self.flip,
            "depth_preview_file": self.depth_preview_path.name if self.save_depth_preview else None,
            "depth_preview_codec": self.depth_preview_codec,
        }
        with open(self.metadata_path, "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

        print(
            f"Segment saved: {self.segment_dir} "
            f"({self.frame_count} frames, {self.depth_frame_count} depth frames, "
            f"{self.imu_packet_count} IMU packets)"
        )

    def average_fps(self, now=None):
        if now is None:
            now = time.monotonic()
        elapsed = max(now - self.started_monotonic, 1e-6)
        return self.frame_count / elapsed


def record_segmented(args):
    global stop_requested
    stop_requested = False
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    compression_queue = None
    compression_thread = None
    if args.compress_depth_raw:
        compression_queue = queue.Queue()
        compression_thread = threading.Thread(
            target=depth_compression_worker,
            args=(compression_queue,),
            daemon=True,
        )
        compression_thread.start()

    print("Starting DepthAI pipeline...")
    try:
        with dai.Pipeline(dai.Device()) as pipeline:
            synced_output, imu_output, _, _ = configure_pipeline(pipeline, args)
            sync_queue = synced_output.createOutputQueue(maxSize=args.queue_size, blocking=False)
            imu_queue = imu_output.createOutputQueue(maxSize=args.imu_queue_size, blocking=False)
            pipeline.start()

            device = pipeline.getDefaultDevice()
            try:
                imu_type = device.getConnectedIMU()
                imu_fw = device.getIMUFirmwareVersion()
                print(f"Connected IMU: {imu_type}, firmware: {imu_fw}")
            except Exception:
                print("Connected IMU information is unavailable, continuing.")

            writer = None
            segment_index = 0
            last_status = time.monotonic()
            last_status_frame_count = 0

            print("Recording synced RGB/depth/IMU segments. Press Ctrl+C to stop.")
            try:
                while not stop_requested:
                    if writer is None:
                        writer = SegmentWriter(
                            args.output_dir,
                            segment_index,
                            args.fps,
                            args.flip,
                            args.rgb_codec,
                            args.save_depth_preview,
                            args.depth_preview_max_mm,
                            args.depth_format,
                            args.depth_png_compression,
                            args.depth_save_every,
                        )
                        last_status = time.monotonic()
                        last_status_frame_count = 0

                    if writer.should_rotate(args.segment_duration):
                        writer.close()
                        enqueue_depth_compression(compression_queue, writer, args)
                        segment_index += 1
                        writer = SegmentWriter(
                            args.output_dir,
                            segment_index,
                            args.fps,
                            args.flip,
                            args.rgb_codec,
                            args.save_depth_preview,
                            args.depth_preview_max_mm,
                            args.depth_format,
                            args.depth_png_compression,
                            args.depth_save_every,
                        )
                        last_status = time.monotonic()
                        last_status_frame_count = 0

                    for imu_msg in imu_queue.tryGetAll():
                        writer.write_imu_message(imu_msg)

                    group = sync_queue.tryGet()
                    if group is None:
                        now = time.monotonic()
                        if now - last_status >= 5.0:
                            print(
                                f"Waiting for synced RGB/depth/IMU group... "
                                f"segment={segment_index}, frames={writer.frame_count}, "
                                f"fps=0.0, avg_fps={writer.average_fps(now):.1f}, "
                                f"imu_packets={writer.imu_packet_count}"
                            )
                            last_status = now
                            last_status_frame_count = writer.frame_count
                        time.sleep(0.001)
                        continue

                    writer.write_group(group)

                    now = time.monotonic()
                    if now - last_status >= 5.0:
                        current_fps = (writer.frame_count - last_status_frame_count) / max(now - last_status, 1e-6)
                        print(
                            f"Recording: segment={segment_index}, "
                            f"frames={writer.frame_count}, depth_frames={writer.depth_frame_count}, "
                            f"fps={current_fps:.1f}, avg_fps={writer.average_fps(now):.1f}, "
                            f"imu_packets={writer.imu_packet_count}"
                        )
                        last_status = now
                        last_status_frame_count = writer.frame_count

            except KeyboardInterrupt:
                stop_requested = True
                print("Recording stopped.")
            finally:
                if writer is not None:
                    for imu_msg in imu_queue.tryGetAll():
                        writer.write_imu_message(imu_msg)
                    writer.close()
                    enqueue_depth_compression(compression_queue, writer, args)
    finally:
        if compression_queue is not None:
            compression_queue.put(None)
            compression_queue.join()
        if compression_thread is not None:
            compression_thread.join()


def diagnose(args):
    print(f"Starting {args.diagnose_seconds:.1f}s DepthAI stream diagnosis...")
    with dai.Pipeline(dai.Device()) as pipeline:
        synced_output, imu_output, rgb_output, depth_output = configure_pipeline(pipeline, args)
        sync_queue = synced_output.createOutputQueue(maxSize=args.queue_size, blocking=False)
        imu_queue = imu_output.createOutputQueue(maxSize=args.imu_queue_size, blocking=False)
        rgb_queue = rgb_output.createOutputQueue(maxSize=4, blocking=False)
        depth_queue = depth_output.createOutputQueue(maxSize=4, blocking=False)
        pipeline.start()

        counts = {"sync": 0, "rgb": 0, "depth": 0, "imu_messages": 0, "imu_packets": 0}
        first_sync_reported = False
        deadline = time.monotonic() + args.diagnose_seconds

        while time.monotonic() < deadline:
            for msg in rgb_queue.tryGetAll():
                counts["rgb"] += 1
            for msg in depth_queue.tryGetAll():
                counts["depth"] += 1
            for msg in imu_queue.tryGetAll():
                counts["imu_messages"] += 1
                counts["imu_packets"] += len(getattr(msg, "packets", []))
            for group in sync_queue.tryGetAll():
                counts["sync"] += 1
                if not first_sync_reported:
                    rgb_msg = get_group_item(group, "rgb")
                    depth_msg = get_group_item(group, "depth")
                    imu_msg = get_group_item(group, "imu")
                    rgb_ts_ns = get_device_ts_ns(rgb_msg)
                    depth_ts_ns = get_device_ts_ns(depth_msg)
                    imu_ts_ns = get_device_ts_ns(imu_msg)
                    print(
                        "First sync delta ms: "
                        f"rgb-depth={ns_delta_ms(rgb_ts_ns, depth_ts_ns)}, "
                        f"rgb-imu={ns_delta_ms(rgb_ts_ns, imu_ts_ns)}, "
                        f"depth-imu={ns_delta_ms(depth_ts_ns, imu_ts_ns)}"
                    )
                    first_sync_reported = True
            time.sleep(0.002)

        print(
            "Diagnosis counts: "
            f"rgb={counts['rgb']}, depth={counts['depth']}, "
            f"imu_messages={counts['imu_messages']}, imu_packets={counts['imu_packets']}, "
            f"sync_groups={counts['sync']}"
        )
        if counts["rgb"] == 0 or counts["depth"] == 0:
            print("RGB/depth direct stream is missing. Check camera sockets/CAM_A,CAM_B,CAM_C and OAK model.")
        elif counts["sync"] == 0:
            print("RGB/depth/IMU streams exist, but Sync produced no group. Increase --sync-threshold-ms.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record synchronized Luxonis RGB/depth/IMU data in time-sliced segments."
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Segment output directory")
    parser.add_argument("--segment-duration", type=float, default=300.0, help="Seconds per segment")
    parser.add_argument("--fps", type=int, default=30, help="RGB/depth FPS")
    parser.add_argument("--imu-rate", type=int, default=200, help="IMU report rate in Hz")
    parser.add_argument("--imu-batch", type=int, default=1, help="Maximum IMU packets per batch")
    parser.add_argument(
        "--depth-preset",
        type=str,
        default="FAST_ACCURACY",
        choices=[item for item in dir(dai.node.StereoDepth.PresetMode) if item.isupper()],
        help="DepthAI v3 StereoDepth preset",
    )
    parser.add_argument(
        "--sync-threshold-ms",
        type=float,
        default=15.0,
        help="Maximum timestamp gap accepted by the DepthAI Sync node",
    )
    parser.add_argument(
        "--sync-attempts",
        type=int,
        default=-1,
        help="DepthAI Sync attempts; -1 waits for best synchronized messages",
    )
    parser.add_argument(
        "--flip",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Vertically flip RGB/depth frames. Supports old style: --flip True",
    )
    parser.add_argument("--rgb-codec", type=str, default="mp4v", help="FourCC for RGB/depth preview MP4")
    parser.add_argument("--queue-size", type=int, default=8, help="Host queue size for synced groups")
    parser.add_argument("--imu-queue-size", type=int, default=64, help="Host queue size for full-rate IMU packets")
    parser.add_argument("--lr-check", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--subpixel", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--save-depth-preview", action="store_true", help="Also save colorized depth_preview.mp4")
    parser.add_argument("--diagnose-seconds", type=float, default=0.0, help="Count raw streams without saving")
    parser.add_argument(
        "--depth-format",
        choices=["png", "raw", "both", "none"],
        default="raw",
        help="Depth storage: png keeps uint16 millimeter values with lossless compression; raw is uncompressed",
    )
    parser.add_argument(
        "--compress-depth-raw",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Gzip raw depth after each segment closes, then remove the uncompressed raw file",
    )
    parser.add_argument(
        "--depth-png-compression",
        type=png_compression_level,
        default=1,
        help="PNG compression level for depth frames, 0 fastest/largest, 9 smallest/slowest",
    )
    parser.add_argument(
        "--depth-save-every",
        type=positive_int,
        default=1,
        help="Save one depth frame every N synced frames",
    )
    parser.add_argument(
        "--depth-preview-max-mm",
        type=int,
        default=8000,
        help="Depth value mapped to 255 in depth_preview.mp4",
    )
    return parser.parse_args()


def main():
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    args = parse_args()
    if args.diagnose_seconds > 0:
        diagnose(args)
    else:
        record_segmented(args)


if __name__ == "__main__":
    main()
