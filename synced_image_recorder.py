#!/usr/bin/env python3

import argparse
import csv
import json
import signal
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


def jpeg_quality(value):
    number = int(value)
    if number < 1 or number > 100:
        raise argparse.ArgumentTypeError("JPEG quality must be between 1 and 100.")
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
            f"This script is DepthAI v3-only. Found depthai=={dai.__version__}; "
            "activate .venv or install depthai>=3."
        )


def enum_by_name(enum_cls, name):
    try:
        return getattr(enum_cls, name)
    except AttributeError as exc:
        valid_names = ", ".join(item for item in dir(enum_cls) if item.isupper())
        raise argparse.ArgumentTypeError(f"Invalid value '{name}'. Valid values: {valid_names}") from exc


def request_camera_output(camera, fps, size=(WIDTH, HEIGHT)):
    capability = dai.ImgFrameCapability()
    capability.size.fixed(size)
    capability.fps.fixed(fps)
    return camera.requestOutput(capability, True)


def make_file_stem(frame_index, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now()
    return f"{timestamp.strftime('%Y-%m-%d-%H-%M-%S')}-frame{frame_index:07d}"


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
    stereo.setDepthAlign(color_socket)
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

    return sync.out


class ImageDatasetWriter:
    def __init__(self, output_dir, args):
        self.args = args
        self.started_wall = datetime.now()
        self.started_monotonic = time.monotonic()
        self.output_dir = Path(output_dir) / self.started_wall.strftime("%Y-%m-%d_%H-%M-%S")
        self.rgb_dir = self.output_dir / "rgb"
        self.depth_dir = self.output_dir / "depth_mm"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)

        self.timestamps_file = open(self.output_dir / "timestamps.csv", "w", newline="")
        self.imu_file = open(self.output_dir / "imu.csv", "w", newline="")

        self.timestamps_writer = csv.writer(self.timestamps_file)
        self.timestamps_writer.writerow([
            "frame_index",
            "stem",
            "rgb_file",
            "depth_file",
            "rgb_sequence",
            "depth_sequence",
            "rgb_device_ts_ns",
            "depth_device_ts_ns",
            "imu_message_device_ts_ns",
            "rgb_depth_delta_ms",
            "rgb_imu_delta_ms",
            "depth_imu_delta_ms",
            "imu_packets",
        ])

        self.imu_writer = csv.writer(self.imu_file)
        self.imu_writer.writerow([
            "frame_index",
            "stem",
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

        self.frame_count = 0
        self.imu_packet_count = 0
        print(f"Dataset opened: {self.output_dir}")

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

        if self.args.flip:
            rgb_frame = cv2.flip(rgb_frame, 0)
            depth_frame = cv2.flip(depth_frame, 0)

        if rgb_frame.dtype != np.uint8:
            rgb_frame = rgb_frame.astype(np.uint8)
        if depth_frame.dtype != np.uint16:
            depth_frame = depth_frame.astype(np.uint16)

        stem = make_file_stem(self.frame_count)
        rgb_path = self.rgb_dir / f"{stem}_rgb.{self.args.rgb_format}"
        depth_path = self.depth_dir / f"{stem}_depth_mm.png"

        self.write_rgb(rgb_path, rgb_frame)
        ok = cv2.imwrite(
            str(depth_path),
            depth_frame,
            [cv2.IMWRITE_PNG_COMPRESSION, self.args.depth_png_compression],
        )
        if not ok:
            raise RuntimeError(f"Failed to write depth image: {depth_path}")

        rgb_ts_ns = get_device_ts_ns(rgb_msg)
        depth_ts_ns = get_device_ts_ns(depth_msg)
        imu_ts_ns = get_device_ts_ns(imu_msg)
        imu_packets = getattr(imu_msg, "packets", [])

        self.timestamps_writer.writerow([
            self.frame_count,
            stem,
            rgb_path.relative_to(self.output_dir),
            depth_path.relative_to(self.output_dir),
            get_sequence_num(rgb_msg),
            get_sequence_num(depth_msg),
            rgb_ts_ns,
            depth_ts_ns,
            imu_ts_ns,
            ns_delta_ms(rgb_ts_ns, depth_ts_ns),
            ns_delta_ms(rgb_ts_ns, imu_ts_ns),
            ns_delta_ms(depth_ts_ns, imu_ts_ns),
            len(imu_packets),
        ])

        for packet_index, packet in enumerate(imu_packets):
            self.write_imu_packet(self.frame_count, stem, packet_index, imu_ts_ns, packet)

        self.frame_count += 1

    def write_rgb(self, path, rgb_frame):
        if self.args.rgb_format == "jpg":
            params = [cv2.IMWRITE_JPEG_QUALITY, self.args.rgb_jpeg_quality]
        else:
            params = [cv2.IMWRITE_PNG_COMPRESSION, self.args.rgb_png_compression]
        ok = cv2.imwrite(str(path), rgb_frame, params)
        if not ok:
            raise RuntimeError(f"Failed to write RGB image: {path}")

    def write_imu_packet(self, frame_index, stem, packet_index, imu_message_ts_ns, packet):
        accel = getattr(packet, "acceleroMeter", None)
        gyro = getattr(packet, "gyroscope", None)
        self.imu_writer.writerow([
            frame_index,
            stem,
            packet_index,
            imu_message_ts_ns,
            get_device_ts_ns(accel),
            getattr(accel, "x", ""),
            getattr(accel, "y", ""),
            getattr(accel, "z", ""),
            get_device_ts_ns(gyro),
            getattr(gyro, "x", ""),
            getattr(gyro, "y", ""),
            getattr(gyro, "z", ""),
        ])
        self.imu_packet_count += 1

    def average_fps(self, now=None):
        if now is None:
            now = time.monotonic()
        elapsed = max(now - self.started_monotonic, 1e-6)
        return self.frame_count / elapsed

    def close(self):
        self.timestamps_file.close()
        self.imu_file.close()
        metadata = {
            "started_wall_time": self.started_wall.isoformat(timespec="milliseconds"),
            "closed_wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "width": WIDTH,
            "height": HEIGHT,
            "camera_sockets": {
                "rgb": "CAM_A",
                "stereo_left": "CAM_B",
                "stereo_right": "CAM_C",
            },
            "depth_alignment": {
                "enabled": True,
                "aligned_to": "rgb",
                "aligned_to_socket": "CAM_A",
                "method": "StereoDepth.setDepthAlign(CAM_A)",
                "output_size": [WIDTH, HEIGHT],
                "rgb_size": [WIDTH, HEIGHT],
                "depth_size": [WIDTH, HEIGHT],
                "depth_pixel_coordinates_match_rgb": True,
                "uses_device_calibration": True,
            },
            "requested_fps": self.args.fps,
            "average_saved_fps": self.average_fps(),
            "frame_count": self.frame_count,
            "imu_packet_count": self.imu_packet_count,
            "rgb_format": self.args.rgb_format,
            "depth_format": "uint16_png",
            "depth_units": "millimeters",
            "depth_png_compression": self.args.depth_png_compression,
            "sync_threshold_ms": self.args.sync_threshold_ms,
            "depth_preset": self.args.depth_preset,
            "timestamps_file": "timestamps.csv",
            "imu_file": "imu.csv",
        }
        with open(self.output_dir / "metadata.json", "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)
        print(
            f"Dataset saved: {self.output_dir} "
            f"({self.frame_count} frames, avg_fps={metadata['average_saved_fps']:.1f})"
        )


def record_images(args):
    global stop_requested
    stop_requested = False
    writer = None

    print("Starting DepthAI synced image pipeline...")
    with dai.Pipeline(dai.Device()) as pipeline:
        synced_output = configure_pipeline(pipeline, args)
        sync_queue = synced_output.createOutputQueue(maxSize=args.queue_size, blocking=False)
        pipeline.start()

        device = pipeline.getDefaultDevice()
        try:
            print(f"Connected IMU: {device.getConnectedIMU()}, firmware: {device.getIMUFirmwareVersion()}")
        except Exception:
            print("Connected IMU information is unavailable, continuing.")

        writer = ImageDatasetWriter(args.output_dir, args)
        last_status = time.monotonic()
        last_frame_count = 0
        print("Saving synced RGB/depth/IMU images. Press Ctrl+C to stop.")

        try:
            while not stop_requested:
                if args.max_frames and writer.frame_count >= args.max_frames:
                    break
                if args.duration and time.monotonic() - writer.started_monotonic >= args.duration:
                    break

                group = sync_queue.tryGet()
                if group is None:
                    now = time.monotonic()
                    if now - last_status >= args.status_interval:
                        print(
                            f"Waiting for synced group... frames={writer.frame_count}, "
                            f"fps=0.0, avg_fps={writer.average_fps(now):.1f}"
                        )
                        last_status = now
                        last_frame_count = writer.frame_count
                    time.sleep(0.001)
                    continue

                writer.write_group(group)

                now = time.monotonic()
                if now - last_status >= args.status_interval:
                    current_fps = (writer.frame_count - last_frame_count) / max(now - last_status, 1e-6)
                    print(
                        f"Saving: frames={writer.frame_count}, "
                        f"fps={current_fps:.1f}, avg_fps={writer.average_fps(now):.1f}, "
                        f"imu_packets={writer.imu_packet_count}"
                    )
                    last_status = now
                    last_frame_count = writer.frame_count
        except KeyboardInterrupt:
            print("Recording stopped.")
        finally:
            if writer is not None:
                writer.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save synchronized Luxonis RGB images, uint16 depth-mm images, and IMU CSV."
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output root directory")
    parser.add_argument("--fps", type=int, default=30, help="Requested RGB/depth FPS")
    parser.add_argument("--imu-rate", type=int, default=200, help="IMU report rate in Hz")
    parser.add_argument("--imu-batch", type=int, default=1, help="Maximum IMU packets per batch")
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds; 0 means run until Ctrl+C")
    parser.add_argument("--max-frames", type=positive_int, default=0, help="Stop after N frames; 0 means no limit")
    parser.add_argument("--status-interval", type=float, default=5.0, help="Seconds between FPS prints")
    parser.add_argument("--queue-size", type=int, default=8, help="Host queue size for synced groups")
    parser.add_argument(
        "--depth-preset",
        type=str,
        default="FAST_ACCURACY",
        choices=[item for item in dir(dai.node.StereoDepth.PresetMode) if item.isupper()],
        help="DepthAI v3 StereoDepth preset",
    )
    parser.add_argument("--sync-threshold-ms", type=float, default=15.0)
    parser.add_argument("--sync-attempts", type=int, default=-1)
    parser.add_argument("--lr-check", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--subpixel", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--flip", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--rgb-format", choices=["png", "jpg"], default="png")
    parser.add_argument("--rgb-png-compression", type=png_compression_level, default=1)
    parser.add_argument("--rgb-jpeg-quality", type=jpeg_quality, default=95)
    parser.add_argument("--depth-png-compression", type=png_compression_level, default=1)
    return parser.parse_args()


def main():
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    record_images(parse_args())


if __name__ == "__main__":
    main()
