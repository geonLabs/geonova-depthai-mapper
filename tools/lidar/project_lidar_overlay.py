#!/usr/bin/env python3
"""Project a ROS1 PointCloud2 frame onto its nearest camera image.

This utility is intentionally isolated from the Jetson DepthAI runtime. Run it
inside a ROS1 environment that provides rosbag, sensor_msgs, and cv_bridge.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import rosbag as _rosbag
    import sensor_msgs.point_cloud2 as _pc2
    from cv_bridge import CvBridge as _CvBridge
except ImportError as exc:  # The converter remains usable without ROS1 installed.
    _rosbag = None
    _pc2 = None
    _CvBridge = None
    _ROS_IMPORT_ERROR: ImportError | None = exc
else:
    _ROS_IMPORT_ERROR = None


def require_ros1() -> tuple[Any, Any, Any]:
    """Return ROS1 modules or raise an actionable environment error."""
    if _ROS_IMPORT_ERROR is not None or _rosbag is None or _pc2 is None or _CvBridge is None:
        raise RuntimeError(
            "ROS1 Python modules are unavailable. Source /opt/ros/noetic/setup.bash "
            "and install ros-noetic-rosbag, ros-noetic-cv-bridge, and "
            "ros-noetic-sensor-msgs before running this command."
        ) from _ROS_IMPORT_ERROR
    return _rosbag, _pc2, _CvBridge


def stamp_to_sec(stamp: Any) -> float:
    return float(stamp.secs) + float(stamp.nsecs) * 1e-9


def resolve_bag(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        bags = sorted(candidate.glob("*.bag"))
        if not bags:
            raise FileNotFoundError(f"no .bag files in {candidate}")
        return bags[0].resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"bag file does not exist: {candidate}")
    if candidate.suffix.lower() != ".bag":
        raise ValueError(f"expected a .bag file: {candidate}")
    return candidate.resolve()


def quat_xyzw_to_rot(q: np.ndarray | list[float]) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64).reshape(4)
    norm_squared = x * x + y * y + z * z + w * w
    if norm_squared == 0.0:
        return np.eye(3, dtype=np.float64)
    scale = 2.0 / norm_squared
    xx, yy, zz = x * x * scale, y * y * scale, z * z * scale
    xy, xz, yz = x * y * scale, x * z * scale, y * z * scale
    wx, wy, wz = w * x * scale, w * y * scale, w * z * scale
    return np.array(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=np.float64,
    )


def load_calib(path: str | os.PathLike[str]):
    calibration_path = Path(path).expanduser().resolve()
    with calibration_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    results = config.get("results", {})
    transform_key = next(
        (
            candidate
            for candidate in (
                "T_lidar_camera",
                "init_T_lidar_camera",
                "init_T_lidar_camera_auto",
            )
            if candidate in results
        ),
        None,
    )
    if transform_key is None:
        raise KeyError("calib.json has no T_lidar_camera or initial guess transform")

    values = np.asarray(results[transform_key], dtype=np.float64).reshape(-1)
    if values.size < 7:
        raise ValueError(
            f"{transform_key} must contain translation xyz and quaternion xyzw; got {values.size} values"
        )
    translation_lidar_camera = values[:3]
    rotation_lidar_camera = quat_xyzw_to_rot(values[3:7])

    rotation_camera_lidar = rotation_lidar_camera.T
    translation_camera_lidar = -rotation_camera_lidar @ translation_lidar_camera

    intrinsics = np.asarray(config["camera"]["intrinsics"], dtype=np.float64).reshape(-1)
    if intrinsics.size != 4:
        raise ValueError("camera.intrinsics must contain [fx, fy, cx, cy]")
    fx, fy, cx, cy = intrinsics
    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.asarray(
        config["camera"].get("distortion_coeffs", []),
        dtype=np.float64,
    ).reshape(-1)
    return (
        transform_key,
        rotation_camera_lidar,
        translation_camera_lidar,
        camera_matrix,
        distortion,
    )


def read_first_pair(bag_path: Path, image_topic: str, points_topic: str):
    rosbag, _, CvBridge = require_ros1()
    bridge = CvBridge()
    image_msg = None
    image_stamp = None
    clouds = []

    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, message, _ in bag.read_messages(topics=[image_topic, points_topic]):
            if topic == image_topic and image_msg is None:
                image_msg = message
                image_stamp = stamp_to_sec(message.header.stamp)
            elif topic == points_topic:
                clouds.append(message)

            if image_msg is not None and len(clouds) >= 3:
                break

    if image_msg is None or image_stamp is None:
        raise RuntimeError(f"no image message on {image_topic}")
    if not clouds:
        raise RuntimeError(f"no point cloud message on {points_topic}")

    cloud_msg = min(
        clouds,
        key=lambda message: abs(stamp_to_sec(message.header.stamp) - image_stamp),
    )
    image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
    return image, image_stamp, cloud_msg, stamp_to_sec(cloud_msg.header.stamp)


def cloud_to_array(message: Any, max_points: int):
    _, point_cloud2, _ = require_ros1()
    field_names = [field.name for field in message.fields]
    fields = ["x", "y", "z"]
    has_intensity = "intensity" in field_names
    if has_intensity:
        fields.append("intensity")

    rows = list(
        point_cloud2.read_points(
            message,
            field_names=fields,
            skip_nans=True,
        )
    )
    if not rows:
        raise RuntimeError("point cloud has no finite points")

    points = np.asarray(rows, dtype=np.float64)
    if max_points > 0 and points.shape[0] > max_points:
        indices = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
        points = points[indices]
    return points[:, :3], points[:, 3] if has_intensity else None


def draw_overlay(
    image: np.ndarray,
    points_lidar: np.ndarray,
    intensities: np.ndarray | None,
    rotation_camera_lidar: np.ndarray,
    translation_camera_lidar: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    point_radius: int,
    image_alpha: float,
    point_alpha: float,
) -> tuple[np.ndarray, int]:
    points_camera = (
        rotation_camera_lidar @ points_lidar.T
    ).T + translation_camera_lidar.reshape(1, 3)
    in_front = points_camera[:, 2] > 0.1
    points_lidar = points_lidar[in_front]
    points_camera = points_camera[in_front]
    if intensities is not None:
        intensities = intensities[in_front]
    if points_lidar.size == 0:
        return image.copy(), 0

    rotation_vector, _ = cv2.Rodrigues(rotation_camera_lidar)
    translation_vector = translation_camera_lidar.reshape(3, 1)
    image_points, _ = cv2.projectPoints(
        points_lidar.astype(np.float64),
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion_coefficients,
    )
    image_points = image_points.reshape(-1, 2)
    finite = np.isfinite(image_points).all(axis=1)
    image_points = image_points[finite]
    points_camera = points_camera[finite]
    if intensities is not None:
        intensities = intensities[finite]

    height, width = image.shape[:2]
    inside = (
        (image_points[:, 0] >= 0.0)
        & (image_points[:, 0] < width)
        & (image_points[:, 1] >= 0.0)
        & (image_points[:, 1] < height)
    )
    u = np.round(image_points[:, 0][inside]).astype(np.int32)
    v = np.round(image_points[:, 1][inside]).astype(np.int32)
    depths = points_camera[:, 2][inside]
    color_values = intensities[inside] if intensities is not None else depths

    if color_values.size == 0:
        return image.copy(), 0

    low, high = np.percentile(color_values, [2, 98])
    normalized = np.clip(
        (color_values - low) / max(high - low, 1e-9),
        0.0,
        1.0,
    )
    colors = cv2.applyColorMap(
        (normalized * 255.0).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    ).reshape(-1, 3)

    overlay = np.zeros_like(image)
    for index in np.argsort(depths)[::-1]:
        cv2.circle(
            overlay,
            (int(u[index]), int(v[index])),
            max(1, int(point_radius)),
            colors[index].tolist(),
            -1,
            lineType=cv2.LINE_AA,
        )

    background = cv2.addWeighted(
        image,
        float(image_alpha),
        np.zeros_like(image),
        1.0 - float(image_alpha),
        0.0,
    )
    blended = cv2.addWeighted(
        background,
        1.0,
        overlay,
        float(point_alpha),
        0.0,
    )
    return blended, int(len(u))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project the nearest ROS1 LiDAR cloud onto the first camera frame."
    )
    parser.add_argument("--bag", required=True, help="ROS1 .bag file or a directory containing bags")
    parser.add_argument("--calib", required=True, help="Calibration JSON containing T_lidar_camera")
    parser.add_argument("--output", required=True, help="Output overlay image path")
    parser.add_argument("--image-topic", default="/roof_clpe_ros/roof_cam_1/image_raw")
    parser.add_argument("--points-topic", default="/lidar0/velodyne_points")
    parser.add_argument("--max-points", type=int, default=80000)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--image-alpha", type=float, default=0.35)
    parser.add_argument("--point-alpha", type=float, default=1.0)
    arguments = parser.parse_args(argv)
    for name in ("image_alpha", "point_alpha"):
        value = float(getattr(arguments, name))
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if arguments.max_points < 0:
        parser.error("--max-points must be >= 0")
    if arguments.point_radius < 1:
        parser.error("--point-radius must be >= 1")
    return arguments


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    bag_path = resolve_bag(arguments.bag)
    (
        transform_key,
        rotation_camera_lidar,
        translation_camera_lidar,
        camera_matrix,
        distortion_coefficients,
    ) = load_calib(arguments.calib)
    image, image_stamp, cloud_msg, cloud_stamp = read_first_pair(
        bag_path,
        arguments.image_topic,
        arguments.points_topic,
    )
    points, intensities = cloud_to_array(cloud_msg, arguments.max_points)
    overlay, count = draw_overlay(
        image,
        points,
        intensities,
        rotation_camera_lidar,
        translation_camera_lidar,
        camera_matrix,
        distortion_coefficients,
        arguments.point_radius,
        arguments.image_alpha,
        arguments.point_alpha,
    )

    output_path = Path(arguments.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise RuntimeError(f"OpenCV failed to write overlay image: {output_path}")

    print(f"bag: {bag_path}")
    print(f"transform: {transform_key}")
    print(f"image_stamp: {image_stamp:.6f}")
    print(f"cloud_stamp: {cloud_stamp:.6f}")
    print(f"projected_points: {count}")
    print(f"output: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        raise SystemExit(f"error: {error}") from error
