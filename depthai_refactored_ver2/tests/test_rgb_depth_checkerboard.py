#!/usr/bin/env python3
"""Validate RGB-to-depth calibration with a planar checkerboard.

The test uses the RGB calibration produced by test_checkerboard_calibration.py,
captures synchronized RGB and RGB-aligned depth, and compares the checkerboard
plane predicted from RGB pose with the measured depth plane.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import depthai as dai
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geonova_depthai import runtime  # noqa: E402
from geonova_depthai.capture.defaults import DEFAULTS  # noqa: E402
from geonova_depthai.config_cli import parse_args_with_yaml  # noqa: E402
from tests.test_checkerboard_calibration import (  # noqa: E402
    detect_corners,
    ensure_opencv_qt_fonts,
    sufficiently_different,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("test_output/checkerboard/calibration.json"),
        help="RGB calibration JSON from test_checkerboard_calibration.py",
    )
    parser.add_argument(
        "--captures",
        type=Path,
        help="Re-analyze an existing captures folder instead of opening the camera",
    )
    parser.add_argument(
        "--factory-calibration",
        type=Path,
        default=Path("test_output/rgb_depth_checkerboard/factory_calibration.json"),
        help="Cached OAK factory RGB calibration; read from the device when missing",
    )
    parser.add_argument(
        "--captures-are-undistorted",
        action="store_true",
        help="Do not factory-undistort RGB files supplied with --captures",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("test_output/rgb_depth_checkerboard"), help="Capture overlays and report directory")
    parser.add_argument("--warmup-frames", type=int, default=200, help="Synchronized frames discarded before capture")
    parser.add_argument("--samples", type=int, default=8, help="Target accepted RGB-Depth views")
    parser.add_argument("--min-samples", type=int, default=5, help="Minimum valid views required for evaluation")
    parser.add_argument("--fps", type=float, default=15.0, help="Live camera frame rate")
    parser.add_argument("--auto-capture", action="store_true", help="Automatically accept stable checkerboard views")
    parser.add_argument("--auto-interval-s", type=float, default=1.0, help="Minimum seconds between automatic captures")
    parser.add_argument("--detection-hold-s", type=float, default=1.5, help="Seconds to retain the latest successful detection")
    parser.add_argument("--live-detection-scale", type=float, default=0.75, help="Live corner-detection image scale")
    parser.add_argument("--no-rotate-180", action="store_true", help="Keep native camera orientation")
    parser.add_argument("--depth-radius-px", type=int, default=4, help="Depth sampling radius around checkerboard cells")
    parser.add_argument("--min-valid-ratio", type=float, default=0.50, help="Minimum valid Depth ratio inside the board")
    parser.add_argument("--max-median-error-mm", type=float, default=80.0, help="Maximum median RGB-predicted versus measured Depth error")
    parser.add_argument("--max-p95-error-mm", type=float, default=250.0, help="Maximum per-view p95 Depth error")
    parser.add_argument("--max-normal-angle-deg", type=float, default=5.0, help="Maximum RGB-to-Depth plane-normal angle")
    parser.add_argument("--max-plane-rmse-mm", type=float, default=30.0, help="Maximum fitted Depth-plane RMSE")
    parser.add_argument("--edge-search-px", type=int, default=35, help="Search radius for measured Depth discontinuities")
    parser.add_argument("--min-edge-views", type=int, default=3, help="Minimum views containing usable board/background edges")
    parser.add_argument("--max-edge-median-px", type=float, default=12.0, help="Maximum median RGB-to-Depth edge displacement")
    parser.add_argument("--max-edge-p95-px", type=float, default=30.0, help="Maximum p95 RGB-to-Depth edge displacement")
    return parse_args_with_yaml(parser)


def load_calibration(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"RGB calibration not found: {path}\n"
            "Run: python tests/test_checkerboard_calibration.py"
        )
    calibration = json.loads(path.read_text(encoding="utf-8"))
    required = ("camera_matrix", "distortion_coefficients", "board", "image_size")
    missing = [key for key in required if key not in calibration]
    if missing:
        raise ValueError(f"Calibration JSON is missing: {', '.join(missing)}")
    return calibration


def board_parameters(calibration: dict):
    board = calibration["board"]
    pattern = (
        int(board["inner_corners_horizontal"]),
        int(board["inner_corners_vertical"]),
    )
    square_size_mm = float(board["square_size_mm"])
    return pattern, square_size_mm


def load_factory_calibration(path: Path, image_size: tuple[int, int], socket_name: str) -> dict:
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        cached_size = tuple(int(value) for value in cached.get("image_size", []))
        if cached_size == tuple(image_size) and cached.get("socket", "CAM_A") == socket_name:
            return cached
        print(
            "Cached factory RGB calibration uses "
            f"{cached.get('socket', 'CAM_A')} {cached.get('image_size')}, "
            f"refreshing for {socket_name} {list(image_size)}..."
        )
    print("Reading factory RGB calibration from the OAK device...")
    with dai.Device() as device:
        calibration = device.readCalibration()
        width, height = (int(value) for value in image_size)
        socket = getattr(dai.CameraBoardSocket, socket_name, dai.CameraBoardSocket.CAM_A)
        intrinsics = calibration.getCameraIntrinsics(
            socket, width, height
        )
        distortion = calibration.getDistortionCoefficients(
            socket
        )
        result = {
            "device": device.getDeviceName(),
            "socket": socket_name,
            "image_size": [width, height],
            "camera_matrix": [[float(value) for value in row] for row in intrinsics],
            "distortion_coefficients": [float(value) for value in distortion],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def saved_undistorted_camera_model(factory: dict, rotate_180: bool):
    camera_matrix = np.asarray(factory["camera_matrix"], dtype=np.float64).copy()
    width, height = (int(value) for value in factory["image_size"])
    if rotate_180:
        camera_matrix[0, 2] = (width - 1) - camera_matrix[0, 2]
        camera_matrix[1, 2] = (height - 1) - camera_matrix[1, 2]
    # RGB is transformed to the factory-undistorted geometry before analysis.
    return camera_matrix, np.zeros(5, dtype=np.float64)


def factory_undistort_saved_rgb(image: np.ndarray, factory: dict, rotate_180: bool):
    camera_matrix = np.asarray(factory["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(factory["distortion_coefficients"], dtype=np.float64)
    source = cv2.rotate(image, cv2.ROTATE_180) if rotate_180 else image
    corrected = cv2.undistort(source, camera_matrix, distortion, None, camera_matrix)
    return cv2.rotate(corrected, cv2.ROTATE_180) if rotate_180 else corrected


def object_points(pattern, square_size_mm):
    cols, rows = pattern
    points = np.zeros((cols * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm
    return points


def recorder_args(args: argparse.Namespace) -> SimpleNamespace:
    values = dict(DEFAULTS)
    values.update(
        fps=args.fps,
        depth_alignment_mode="auto",
        # The official RVC2 RGB-D example aligns depth to an undistorted RGB
        # output. Raw distorted RGB and aligned depth do not share one geometry.
        rgb_undistort=True,
        sync_mode="device",
        sync_attempts=-1,
        save_confidence_map=False,
        enable_gps=False,
        enable_external_imu=False,
        rgb_transport="raw",
        confidence_transport="raw",
        rgb_transport_effective="raw",
        confidence_transport_effective="raw",
    )
    return SimpleNamespace(**values)


def colorize_depth(depth: np.ndarray, max_mm=5000) -> np.ndarray:
    valid = depth > 0
    clipped = np.clip(depth, 0, max_mm).astype(np.float32)
    preview = np.uint8(255.0 * (1.0 - clipped / float(max_mm)))
    preview[~valid] = 0
    color = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def capture_live(args, calibration, pattern):
    ensure_opencv_qt_fonts()
    config = recorder_args(args)
    rotate_180 = bool(calibration.get("saved_image_rotated_180", True))
    if args.no_rotate_180:
        rotate_180 = False

    capture_dir = args.output_dir / "captures"
    capture_dir.mkdir(parents=True, exist_ok=True)
    accepted_rgb = []
    accepted_depth = []
    accepted_corners = []
    candidate = None
    candidate_time = 0.0
    last_auto_capture = 0.0

    device = runtime.connect_depthai_device(config)
    try:
        with dai.Pipeline(device) as pipeline:
            device = pipeline.getDefaultDevice()
            runtime.resolve_transport_options(config, device)
            outputs = runtime.configure_pipeline(pipeline, config, sync_imu=False)
            queue = outputs["sync"].createOutputQueue(maxSize=4, blocking=True)
            pipeline.start()
            print(
                f"Camera: {device.getDeviceName()} ({config.depthai_platform}), "
                f"alignment={config.depth_alignment_effective}"
            )
            print(f"Warming up {args.warmup_frames} synchronized RGB-D frames...")
            for frame_index in range(args.warmup_frames):
                queue.get()
                if (frame_index + 1) % 50 == 0:
                    print(f"  warm-up {frame_index + 1}/{args.warmup_frames}")

            print("Show the full checkerboard. LOCKED=Space accept, Q=finish.")
            while len(accepted_rgb) < args.samples:
                group = queue.get()
                rgb_message = runtime.get_group_item(group, "rgb")
                depth_message = runtime.get_group_item(group, "depth")
                rgb = runtime.get_color_cv_frame(rgb_message)
                depth = depth_message.getFrame()
                if rotate_180:
                    rgb = cv2.rotate(rgb, cv2.ROTATE_180)
                    depth = cv2.rotate(depth, cv2.ROTATE_180)

                found, corners = detect_corners(
                    rgb,
                    pattern,
                    scale=args.live_detection_scale,
                    exhaustive=False,
                )
                now = time.monotonic()
                if found:
                    candidate = (rgb.copy(), depth.copy(), corners.copy())
                    candidate_time = now
                locked = candidate is not None and now - candidate_time <= args.detection_hold_s

                preview_rgb = candidate[0].copy() if locked else rgb.copy()
                preview_depth = candidate[1] if locked else depth
                if locked:
                    cv2.drawChessboardCorners(preview_rgb, pattern, candidate[2], True)
                depth_color = colorize_depth(preview_depth)
                overlay = cv2.addWeighted(preview_rgb, 0.65, depth_color, 0.35, 0.0)
                cv2.putText(
                    overlay,
                    (
                        f"accepted {len(accepted_rgb)}/{args.samples} | "
                        f"{'LOCKED - press Space' if locked else 'searching'}"
                    ),
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if locked else (0, 0, 255),
                    2,
                )
                cv2.imshow("RGB-depth checkerboard validation", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                accept = key == ord(" ")
                if (
                    args.auto_capture
                    and locked
                    and now - last_auto_capture >= args.auto_interval_s
                ):
                    accept = sufficiently_different(candidate[2], accepted_corners, candidate[0].shape)

                if accept and locked:
                    refined, refined_corners = detect_corners(
                        candidate[0], pattern, scale=1.0, exhaustive=True
                    )
                    corners_to_save = refined_corners if refined else candidate[2]
                    index = len(accepted_rgb)
                    accepted_rgb.append(candidate[0].copy())
                    accepted_depth.append(candidate[1].copy())
                    accepted_corners.append(corners_to_save.copy())
                    cv2.imwrite(str(capture_dir / f"rgb_{index:02d}.png"), candidate[0])
                    cv2.imwrite(str(capture_dir / f"depth_{index:02d}.png"), candidate[1])
                    print(f"accepted RGB-D view {index + 1}/{args.samples}")
                    candidate = None
                    candidate_time = 0.0
                    last_auto_capture = now
            (capture_dir / "capture_metadata.json").write_text(
                json.dumps(
                    {
                        "rgb_undistorted": True,
                        "rotate_180": rotate_180,
                        "alignment": config.depth_alignment_effective,
                        "platform": config.depthai_platform,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    finally:
        cv2.destroyAllWindows()
        try:
            device.close()
        except Exception:
            pass
    return accepted_rgb, accepted_depth


def load_captures(path: Path):
    rgb_paths = sorted(path.glob("rgb_*.png"))
    images = []
    depths = []
    for rgb_path in rgb_paths:
        suffix = rgb_path.stem[len("rgb_"):]
        depth_path = path / f"depth_{suffix}.png"
        rgb = cv2.imread(str(rgb_path))
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is not None and depth is not None:
            images.append(rgb)
            depths.append(depth)
    metadata_path = path / "capture_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    return images, depths, metadata


def robust_depth(depth: np.ndarray, point, radius: int):
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    y0, y1 = max(0, y - radius), min(depth.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(depth.shape[1], x + radius + 1)
    values = depth[y0:y1, x0:x1]
    values = values[values > 0]
    return float(np.median(values)) if values.size else None


def angle_between_normals(a, b):
    cosine = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def checkerboard_edge_alignment(
    depth,
    pattern,
    square_size_mm,
    camera_matrix,
    distortion,
    rotation_vector,
    translation,
    search_px,
):
    """Compare projected RGB board edges with depth discontinuities.

    The board must have free space behind its outer edge. A board placed flat on
    a wall can validate the plane but cannot expose RGB-depth edge displacement.
    """
    cols, rows = pattern
    outer_object = np.array(
        [
            [-square_size_mm, -square_size_mm, 0.0],
            [cols * square_size_mm, -square_size_mm, 0.0],
            [cols * square_size_mm, rows * square_size_mm, 0.0],
            [-square_size_mm, rows * square_size_mm, 0.0],
        ],
        dtype=np.float32,
    )
    projected, _ = cv2.projectPoints(
        outer_object, rotation_vector, translation, camera_matrix, distortion
    )
    outer = projected.reshape(4, 2)
    board_center = outer.mean(axis=0)
    rotation, _ = cv2.Rodrigues(rotation_vector)
    normal_3d = rotation[:, 2]
    normal_3d /= np.linalg.norm(normal_3d)
    plane_point = translation.reshape(3)
    plane_numerator = float(np.dot(plane_point, normal_3d))
    offsets = []
    observed_points = []

    for edge_index in range(4):
        start = outer[edge_index]
        end = outer[(edge_index + 1) % 4]
        edge = end - start
        edge_length = float(np.linalg.norm(edge))
        if edge_length < 20:
            continue
        tangent = edge / edge_length
        normal_2d = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        midpoint = 0.5 * (start + end)
        if np.dot(normal_2d, board_center - midpoint) < 0:
            normal_2d *= -1.0

        for fraction in np.linspace(0.15, 0.85, 20):
            expected_edge = start + fraction * edge
            distances = np.arange(-search_px, search_px + 1, dtype=np.float64)
            search_points = expected_edge[None, :] + distances[:, None] * normal_2d[None, :]
            normalized = cv2.undistortPoints(
                search_points.reshape(-1, 1, 2), camera_matrix, distortion
            ).reshape(-1, 2)
            rays = np.column_stack([normalized, np.ones(len(normalized))])
            expected_z = plane_numerator / (rays @ normal_3d)
            board_like = []
            for point, predicted_z in zip(search_points, expected_z):
                measured_z = robust_depth(depth, point, radius=1)
                tolerance_mm = max(120.0, abs(float(predicted_z)) * 0.06)
                board_like.append(
                    measured_z is not None
                    and abs(measured_z - predicted_z) <= tolerance_mm
                )
            board_like = np.asarray(board_like, dtype=np.uint8)
            sustained = np.convolve(board_like, np.ones(3, dtype=np.uint8), mode="same") >= 2
            transitions = np.flatnonzero((~sustained[:-1]) & sustained[1:])
            if transitions.size == 0:
                continue
            transition_index = min(
                transitions,
                key=lambda index: abs(float(distances[index + 1])),
            ) + 1
            offset = float(distances[transition_index])
            offsets.append(offset)
            observed_points.append(expected_edge + offset * normal_2d)
    return np.asarray(offsets, dtype=np.float64), outer, observed_points


def analyze_view(
    rgb,
    depth,
    pattern,
    square_size_mm,
    camera_matrix,
    distortion,
    radius,
    edge_search_px,
):
    found, corners = detect_corners(rgb, pattern, scale=1.0, exhaustive=True)
    if not found:
        return None, None
    obj = object_points(pattern, square_size_mm)
    solved, rotation_vector, translation = cv2.solvePnP(
        obj, corners, camera_matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not solved:
        return None, None

    cols, rows = pattern
    corner_grid = corners.reshape(rows, cols, 2)
    centers = 0.25 * (
        corner_grid[:-1, :-1]
        + corner_grid[1:, :-1]
        + corner_grid[:-1, 1:]
        + corner_grid[1:, 1:]
    )
    centers = centers.reshape(-1, 2)
    observed = [robust_depth(depth, point, radius) for point in centers]
    valid_mask = np.array([value is not None for value in observed], dtype=bool)
    valid_centers = centers[valid_mask]
    observed_z = np.array([value for value in observed if value is not None], dtype=np.float64)
    valid_ratio = float(valid_mask.mean())
    if observed_z.size < 6:
        return None, None

    normalized = cv2.undistortPoints(
        valid_centers.reshape(-1, 1, 2).astype(np.float64),
        camera_matrix,
        distortion,
    ).reshape(-1, 2)
    rays = np.column_stack([normalized, np.ones(len(normalized))])
    rotation, _ = cv2.Rodrigues(rotation_vector)
    rgb_normal = rotation[:, 2]
    rgb_normal /= np.linalg.norm(rgb_normal)
    plane_point = translation.reshape(3)
    denominator = rays @ rgb_normal
    numerator = float(np.dot(plane_point, rgb_normal))
    predicted_z = numerator / denominator
    signed_errors = observed_z - predicted_z

    points_3d = rays * observed_z[:, None]
    pnp_plane_residual = (points_3d - plane_point) @ rgb_normal
    median_residual = float(np.median(pnp_plane_residual))
    mad = float(np.median(np.abs(pnp_plane_residual - median_residual)))
    inlier_limit = max(50.0, 4.0 * 1.4826 * mad)
    inliers = np.abs(pnp_plane_residual - median_residual) <= inlier_limit
    fit_points = points_3d[inliers]
    if len(fit_points) < 6:
        fit_points = points_3d
    centroid = fit_points.mean(axis=0)
    _, _, vh = np.linalg.svd(fit_points - centroid, full_matrices=False)
    depth_normal = vh[-1]
    depth_normal /= np.linalg.norm(depth_normal)
    fitted_residual = (fit_points - centroid) @ depth_normal

    projected, _ = cv2.projectPoints(
        obj, rotation_vector, translation, camera_matrix, distortion
    )
    reprojection_rmse = float(
        np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, axis=1)))
    )
    metrics = {
        "valid_depth_ratio": valid_ratio,
        "valid_points": int(len(observed_z)),
        "depth_error_bias_mm": float(np.median(signed_errors)),
        "median_abs_depth_error_mm": float(np.median(np.abs(signed_errors))),
        "p95_abs_depth_error_mm": float(np.percentile(np.abs(signed_errors), 95)),
        "rgb_depth_normal_angle_deg": angle_between_normals(rgb_normal, depth_normal),
        "depth_plane_rmse_mm": float(np.sqrt(np.mean(fitted_residual ** 2))),
        "rgb_pnp_reprojection_rmse_px": reprojection_rmse,
        "board_distance_mm": float(translation[2, 0]),
    }

    edge_offsets, outer_corners, observed_edge_points = checkerboard_edge_alignment(
        depth,
        pattern,
        square_size_mm,
        camera_matrix,
        distortion,
        rotation_vector,
        translation,
        edge_search_px,
    )
    metrics["edge_samples"] = int(len(edge_offsets))
    metrics["edge_alignment_bias_px"] = (
        float(np.median(edge_offsets)) if len(edge_offsets) else None
    )
    metrics["median_abs_edge_alignment_px"] = (
        float(np.median(np.abs(edge_offsets))) if len(edge_offsets) else None
    )
    metrics["p95_abs_edge_alignment_px"] = (
        float(np.percentile(np.abs(edge_offsets), 95)) if len(edge_offsets) else None
    )

    overlay = rgb.copy()
    cv2.drawChessboardCorners(overlay, pattern, corners, True)
    cv2.polylines(
        overlay,
        [np.round(outer_corners).astype(np.int32)],
        True,
        (255, 255, 0),
        2,
    )
    for point in observed_edge_points:
        cv2.circle(overlay, tuple(np.round(point).astype(int)), 2, (255, 0, 255), -1)
    for point, error in zip(valid_centers, signed_errors):
        color = (0, 255, 0) if abs(error) <= 80.0 else (0, 0, 255)
        cv2.circle(overlay, tuple(np.round(point).astype(int)), 3, color, -1)
    cv2.putText(
        overlay,
        (
            f"median={metrics['median_abs_depth_error_mm']:.1f}mm "
            f"p95={metrics['p95_abs_depth_error_mm']:.1f}mm "
            f"angle={metrics['rgb_depth_normal_angle_deg']:.2f}deg "
            f"edge={metrics['median_abs_edge_alignment_px'] if metrics['median_abs_edge_alignment_px'] is not None else -1:.1f}px"
        ),
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return metrics, overlay


def aggregate_results(args, view_results):
    def median(key):
        return float(np.median([row[key] for row in view_results]))

    all_p95 = max(row["p95_abs_depth_error_mm"] for row in view_results)
    edge_views = [
        row for row in view_results
        if row["median_abs_edge_alignment_px"] is not None
    ]
    result = {
        "usable_views": len(view_results),
        "median_valid_depth_ratio": median("valid_depth_ratio"),
        "median_abs_depth_error_mm": median("median_abs_depth_error_mm"),
        "worst_view_p95_abs_depth_error_mm": float(all_p95),
        "median_rgb_depth_normal_angle_deg": median("rgb_depth_normal_angle_deg"),
        "median_depth_plane_rmse_mm": median("depth_plane_rmse_mm"),
        "edge_usable_views": len(edge_views),
        "median_abs_edge_alignment_px": (
            float(np.median([row["median_abs_edge_alignment_px"] for row in edge_views]))
            if edge_views else None
        ),
        "worst_view_p95_abs_edge_alignment_px": (
            float(max(row["p95_abs_edge_alignment_px"] for row in edge_views))
            if edge_views else None
        ),
        "views": view_results,
        "thresholds": {
            "min_valid_ratio": args.min_valid_ratio,
            "max_median_error_mm": args.max_median_error_mm,
            "max_p95_error_mm": args.max_p95_error_mm,
            "max_normal_angle_deg": args.max_normal_angle_deg,
            "max_plane_rmse_mm": args.max_plane_rmse_mm,
            "min_edge_views": args.min_edge_views,
            "max_edge_median_px": args.max_edge_median_px,
            "max_edge_p95_px": args.max_edge_p95_px,
        },
    }
    failures = []
    if len(view_results) < args.min_samples:
        failures.append(f"usable views {len(view_results)} < {args.min_samples}")
    if result["median_valid_depth_ratio"] < args.min_valid_ratio:
        failures.append("depth valid ratio is too low inside the checkerboard")
    if result["median_abs_depth_error_mm"] > args.max_median_error_mm:
        failures.append("RGB-predicted plane and depth distance disagree")
    if result["worst_view_p95_abs_depth_error_mm"] > args.max_p95_error_mm:
        failures.append("one or more views have large depth errors")
    if result["median_rgb_depth_normal_angle_deg"] > args.max_normal_angle_deg:
        failures.append("RGB and depth plane normals disagree")
    if result["median_depth_plane_rmse_mm"] > args.max_plane_rmse_mm:
        failures.append("measured checkerboard depth is not planar enough")
    if len(edge_views) < args.min_edge_views:
        failures.append(
            "not enough board/background depth edges; hold the board away from a wall"
        )
    elif result["median_abs_edge_alignment_px"] > args.max_edge_median_px:
        failures.append("RGB and depth checkerboard outer edges are displaced")
    elif result["worst_view_p95_abs_edge_alignment_px"] > args.max_edge_p95_px:
        failures.append("one or more views have large RGB-depth edge displacement")
    result["passed"] = not failures
    result["failures"] = failures
    return result


def main() -> None:
    args = parse_args()
    calibration = load_calibration(args.calibration)
    pattern, square_size_mm = board_parameters(calibration)
    expected_size = tuple(int(value) for value in calibration["image_size"])
    factory = load_factory_calibration(
        args.factory_calibration,
        expected_size,
        calibration.get("socket", "CAM_A"),
    )
    rotate_180 = bool(calibration.get("saved_image_rotated_180", True))
    if args.no_rotate_180:
        rotate_180 = False
    camera_matrix, distortion = saved_undistorted_camera_model(
        factory, rotate_180
    )
    print(
        f"RGB calibration: {args.calibration}\n"
        f"Board: {pattern[0]}x{pattern[1]} inner corners, {square_size_mm:g} mm"
    )

    try:
        if args.captures is None:
            images, depths = capture_live(args, calibration, pattern)
            capture_metadata = {"rgb_undistorted": True}
        else:
            images, depths, capture_metadata = load_captures(args.captures)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print(f"\nStopped. Captures remain in {args.output_dir / 'captures'}.")
        return
    if len(images) < args.min_samples:
        raise RuntimeError(f"Only {len(images)} RGB-D views; need at least {args.min_samples}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    captures_are_undistorted = bool(
        args.captures_are_undistorted
        or capture_metadata.get("rgb_undistorted", False)
    )
    if not captures_are_undistorted:
        print(
            "Saved RGB is distorted while depth uses aligned geometry; "
            "applying OAK factory RGB undistortion."
        )
        corrected_dir = args.output_dir / "corrected_captures"
        corrected_dir.mkdir(exist_ok=True)
        corrected_images = []
        for index, (rgb, depth) in enumerate(zip(images, depths)):
            corrected = factory_undistort_saved_rgb(rgb, factory, rotate_180)
            corrected_images.append(corrected)
            cv2.imwrite(str(corrected_dir / f"rgb_{index:02d}.png"), corrected)
            cv2.imwrite(str(corrected_dir / f"depth_{index:02d}.png"), depth)
        (corrected_dir / "capture_metadata.json").write_text(
            json.dumps(
                {
                    "rgb_undistorted": True,
                    "rotate_180": rotate_180,
                    "source": str(args.captures),
                    "method": f"cv2.undistort with OAK factory {factory.get('socket', 'CAM_A')} calibration",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        images = corrected_images
    view_results = []
    for index, (rgb, depth) in enumerate(zip(images, depths)):
        if (rgb.shape[1], rgb.shape[0]) != expected_size or depth.shape[::-1] != expected_size:
            raise ValueError(
                f"View {index} size does not match RGB calibration {expected_size}: "
                f"RGB={rgb.shape[1]}x{rgb.shape[0]}, depth={depth.shape[1]}x{depth.shape[0]}"
            )
        metrics, overlay = analyze_view(
            rgb,
            depth,
            pattern,
            square_size_mm,
            camera_matrix,
            distortion,
            args.depth_radius_px,
            args.edge_search_px,
        )
        if metrics is None:
            print(f"view {index}: skipped (checkerboard/depth unavailable)")
            continue
        metrics["view_index"] = index
        view_results.append(metrics)
        cv2.imwrite(str(args.output_dir / f"overlay_{index:02d}.jpg"), overlay)
        cv2.imwrite(str(args.output_dir / f"depth_color_{index:02d}.png"), colorize_depth(depth))
        print(
            f"view {index}: valid={metrics['valid_depth_ratio']:.1%}, "
            f"median={metrics['median_abs_depth_error_mm']:.1f}mm, "
            f"p95={metrics['p95_abs_depth_error_mm']:.1f}mm, "
            f"normal={metrics['rgb_depth_normal_angle_deg']:.2f}deg, "
            f"plane={metrics['depth_plane_rmse_mm']:.1f}mm, "
            f"edge={metrics['median_abs_edge_alignment_px'] if metrics['median_abs_edge_alignment_px'] is not None else -1:.1f}px"
        )

    if not view_results:
        raise RuntimeError("No usable RGB-D checkerboard views")
    report = aggregate_results(args, view_results)
    report.update(
        calibration_file=str(args.calibration),
        factory_calibration_file=str(args.factory_calibration),
        input_rgb_was_undistorted=captures_are_undistorted,
        applied_factory_rgb_undistortion=not captures_are_undistorted,
        depth_units="millimeters",
        validation=(
            "RGB solvePnP checkerboard plane vs synchronized RGB-aligned StereoDepth plane"
        ),
    )
    (args.output_dir / "rgb_depth_calibration_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
