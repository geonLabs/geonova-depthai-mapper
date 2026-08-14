#!/usr/bin/env python3
"""Refresh saved dataset camera intrinsics from a connected DepthAI camera."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import depthai as dai

from geonova_depthai import runtime
from geonova_depthai.capture.defaults import DEFAULTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update metadata.json camera_model.intrinsics using the actual "
            "DepthAI ImgFrame transformation for the dataset image size."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Raw or synced dataset folder")
    parser.add_argument("--fps", type=float, default=5.0, help="Temporary probe RGB FPS")
    parser.add_argument("--allow-usb2", action="store_true", help="Allow reduced-bandwidth USB2 probe")
    parser.add_argument("--usb3-retries", type=int, default=2, help="USB3 negotiation retries")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing metadata.json")
    return parser.parse_args()


def metadata_image_size(metadata: dict) -> tuple[int, int]:
    size = metadata.get("image_size") or {}
    camera_model = metadata.get("camera_model") or {}
    width = int(size.get("width") or camera_model.get("width") or runtime.WIDTH)
    height = int(size.get("height") or camera_model.get("height") or runtime.HEIGHT)
    return width, height


def socket_from_metadata(metadata: dict):
    sockets = metadata.get("camera_sockets") or {}
    camera_model = metadata.get("camera_model") or {}
    socket_name = sockets.get("rgb") or camera_model.get("socket") or "CAM_A"
    return runtime.enum_by_name(dai.CameraBoardSocket, socket_name)


def probe_args(metadata: dict, args: argparse.Namespace) -> SimpleNamespace:
    width, height = metadata_image_size(metadata)
    transform = metadata.get("image_transform") or {}
    values = dict(DEFAULTS)
    values.update(
        fps=args.fps,
        rgb_width=width,
        rgb_height=height,
        flip=bool(transform.get("flip_vertical", False)),
        rotate_180=bool(transform.get("rotate_180", False)),
        allow_usb2=bool(args.allow_usb2),
        usb3_retries=int(args.usb3_retries),
        rgb_socket=socket_from_metadata(metadata),
        rgb_socket_name=runtime.enum_name(socket_from_metadata(metadata)),
        rgb_undistort=True,
        rgb_undistort_effective=True,
        rgb_transport="raw",
        rgb_transport_effective="raw",
        enable_gps=False,
        enable_external_imu=False,
    )
    return SimpleNamespace(**values)


def read_frame_camera_model(device, args: SimpleNamespace) -> dict:
    with dai.Pipeline(device) as pipeline:
        device = pipeline.getDefaultDevice()
        runtime.resolve_rgb_output_size(device, args, getattr(args, "rgb_socket", dai.CameraBoardSocket.CAM_A))
        factory_model = runtime.read_camera_model_metadata(device, args)
        cam_rgb = pipeline.create(dai.node.Camera).build(args.rgb_socket)
        rgb_output = runtime.request_camera_output(
            cam_rgb,
            args.fps,
            size=runtime.rgb_size_from_args(args),
            frame_type=dai.ImgFrame.Type.NV12,
            enable_undistortion=True,
        )
        queue = rgb_output.createOutputQueue(maxSize=2, blocking=True)
        pipeline.start()
        frame_model = runtime.imgframe_camera_model_metadata(queue.get(), args)
    factory_model.update(frame_model)
    return factory_model


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    metadata_path = dataset / "metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"metadata.json not found: {metadata_path}")

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    probe = probe_args(metadata, args)
    device = runtime.connect_depthai_device(probe)
    try:
        camera_model = read_frame_camera_model(device, probe)
    finally:
        device.close()

    previous = (metadata.get("camera_model") or {}).get("intrinsics")
    metadata["camera_model"] = {
        **(metadata.get("camera_model") or {}),
        **camera_model,
    }
    metadata["image_size"] = {
        "width": int(probe.rgb_width),
        "height": int(probe.rgb_height),
    }

    print(f"Dataset: {dataset}")
    print(f"Previous intrinsics: {previous}")
    print(f"Updated intrinsics:  {metadata['camera_model'].get('intrinsics')}")
    print(f"Source: {metadata['camera_model'].get('intrinsics_source')}")
    if args.dry_run:
        return

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    print(f"Updated: {metadata_path}")


if __name__ == "__main__":
    main()
