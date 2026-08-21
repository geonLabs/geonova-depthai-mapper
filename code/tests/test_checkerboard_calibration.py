#!/usr/bin/env python3
"""Capture a checkerboard and verify RGB camera calibration quality.

Board definition requested for this project:
  14 squares horizontally x 10 squares vertically, 30 mm per square
  -> OpenCV inner-corner pattern is 13 x 9.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geonova_depthai import runtime  # noqa: E402
from geonova_depthai.config_cli import parse_args_with_yaml  # noqa: E402


def ensure_opencv_qt_fonts() -> None:
    """Provide the font directory hard-coded by OpenCV's bundled Qt plugin."""
    qt_fonts = Path(cv2.__file__).resolve().parent / "qt" / "fonts"
    system_fonts = Path("/usr/share/fonts/truetype/dejavu")
    if qt_fonts.exists() or not system_fonts.exists():
        return
    try:
        qt_fonts.symlink_to(system_fonts, target_is_directory=True)
    except OSError:
        # The preview still works without this; Qt will only print warnings.
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture checkerboard views and run OpenCV rational-model calibration. "
            "Press Space to accept a detected live view and Q to finish."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--images", type=Path, help="Calibrate existing JPG/PNG images")
    source.add_argument("--camera", action="store_true", help="Use the connected OAK camera (default)")
    parser.add_argument("--square-cols", type=int, default=14, help="Horizontal square count")
    parser.add_argument("--square-rows", type=int, default=10, help="Vertical square count")
    parser.add_argument("--square-size-mm", type=float, default=30.0, help="Measured checkerboard square edge length in millimeters")
    parser.add_argument("--samples", type=int, default=20, help="Target accepted checkerboard views")
    parser.add_argument("--min-samples", type=int, default=12, help="Minimum views required to calibrate")
    parser.add_argument("--fps", type=float, default=15.0, help="Live camera frame rate")
    parser.add_argument("--width", type=int, default=0, help="Calibration image width; 0 selects automatically from the connected color sensor")
    parser.add_argument("--height", type=int, default=0, help="Calibration image height; 0 selects automatically from the connected color sensor")
    parser.add_argument("--output-dir", type=Path, default=Path("test_output/checkerboard"), help="Capture and calibration output directory")
    parser.add_argument("--max-rms-px", type=float, default=1.0, help="Maximum accepted calibration reprojection RMS in pixels")
    parser.add_argument("--auto-capture", action="store_true", help="Accept stable, sufficiently different views automatically")
    parser.add_argument("--auto-interval-s", type=float, default=0.8, help="Minimum seconds between automatic captures")
    parser.add_argument(
        "--detection-hold-s",
        type=float,
        default=1.5,
        help="Keep the latest successful live detection available for Space",
    )
    parser.add_argument(
        "--live-detection-scale",
        type=float,
        default=0.75,
        help="Downscale used for fast live detection; offline calibration remains full resolution",
    )
    parser.add_argument("--no-preview", action="store_true", help="Disable the OpenCV live preview window")
    parser.add_argument("--no-rotate-180", action="store_true", help="Keep native camera orientation during live capture")
    parser.add_argument(
        "--images-rotated-180",
        action="store_true",
        help="Mark --images input as saved after a 180-degree rotation",
    )
    parser.add_argument(
        "--skip-factory-comparison",
        action="store_true",
        help="Skip comparison with the connected OAK factory focal length",
    )
    return parse_args_with_yaml(parser)


def pattern_size(args: argparse.Namespace) -> tuple[int, int]:
    if args.square_cols < 2 or args.square_rows < 2:
        raise ValueError("Checkerboard needs at least 2x2 squares")
    return args.square_cols - 1, args.square_rows - 1


def detect_corners(
    image: np.ndarray,
    pattern: tuple[int, int],
    scale: float = 1.0,
    exhaustive: bool = True,
):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = min(max(float(scale), 0.2), 1.0)
    detection_image = gray
    if scale < 1.0:
        detection_image = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE
    if exhaustive:
        flags |= cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(
        detection_image, pattern, flags=flags
    )
    if found and scale < 1.0:
        corners = corners / scale
    return bool(found), corners


def sufficiently_different(corners: np.ndarray, accepted: list[np.ndarray], image_shape) -> bool:
    if not accepted:
        return True
    height, width = image_shape[:2]
    diagonal = float(np.hypot(width, height))
    current = corners.reshape(-1, 2)
    previous = accepted[-1].reshape(-1, 2)
    motion = float(np.mean(np.linalg.norm(current - previous, axis=1)) / diagonal)
    return motion >= 0.025


def capture_live(args: argparse.Namespace, pattern: tuple[int, int]):
    ensure_opencv_qt_fonts()
    accepted_images: list[np.ndarray] = []
    accepted_corners: list[np.ndarray] = []
    last_auto_capture = 0.0
    candidate_image = None
    candidate_corners = None
    candidate_time = 0.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    capture_dir = args.output_dir / "captures"
    capture_dir.mkdir(exist_ok=True)

    with dai.Pipeline() as pipeline:
        device = pipeline.getDefaultDevice()
        if args.width > 0 and args.height > 0:
            args.rgb_width = args.width
            args.rgb_height = args.height
        else:
            args.rgb_width = 0
            args.rgb_height = 0
        args.width, args.height = runtime.resolve_rgb_output_size(device, args)
        camera = pipeline.create(dai.node.Camera).build(
            getattr(args, "rgb_socket", dai.CameraBoardSocket.CAM_A)
        )
        # Calibration must see the distorted source image. Enabling device-side
        # undistortion here would calibrate an already corrected image.
        output = runtime.request_camera_output(
            camera,
            args.fps,
            size=(args.width, args.height),
            frame_type=dai.ImgFrame.Type.NV12,
            enable_undistortion=False,
        )
        queue = output.createOutputQueue(maxSize=2, blocking=False)
        pipeline.start()
        print(f"Camera: {device.getDeviceName()} ({device.getPlatform()})")
        print("Show the full board at varied positions/tilts. Space=accept, Q=finish.")

        while len(accepted_images) < args.samples:
            message = queue.get()
            image = runtime.get_color_cv_frame(message)
            if not args.no_rotate_180:
                image = cv2.rotate(image, cv2.ROTATE_180)
            found, corners = detect_corners(
                image,
                pattern,
                scale=args.live_detection_scale,
                exhaustive=False,
            )
            now = time.monotonic()
            if found:
                candidate_image = image.copy()
                candidate_corners = corners.copy()
                candidate_time = now

            locked = (
                candidate_image is not None
                and now - candidate_time <= args.detection_hold_s
            )
            # Freeze the latest successfully detected frame briefly. This makes
            # the green overlay and Space action stable when detection flickers.
            preview = candidate_image.copy() if locked else image.copy()
            if locked:
                cv2.drawChessboardCorners(
                    preview, pattern, candidate_corners, True
                )
            cv2.putText(
                preview,
                (
                    f"accepted {len(accepted_images)}/{args.samples} | "
                    f"{'LOCKED - press Space' if locked else 'searching'}"
                ),
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if locked else (0, 0, 255),
                2,
            )

            accept = False
            key = -1
            if not args.no_preview:
                cv2.imshow("checkerboard calibration", preview)
                key = cv2.waitKey(1) & 0xFF
                accept = key == ord(" ")
                if key in (ord("q"), 27):
                    break
            if (
                args.auto_capture
                and locked
                and now - last_auto_capture >= args.auto_interval_s
            ):
                accept = sufficiently_different(
                    candidate_corners, accepted_corners, candidate_image.shape
                )

            if accept and locked:
                refined, refined_corners = detect_corners(
                    candidate_image, pattern, scale=1.0, exhaustive=True
                )
                corners_to_save = (
                    refined_corners if refined else candidate_corners
                )
                accepted_images.append(candidate_image.copy())
                accepted_corners.append(corners_to_save.copy())
                last_auto_capture = now
                cv2.imwrite(
                    str(capture_dir / f"capture_{len(accepted_images) - 1:02d}.png"),
                    candidate_image,
                )
                print(f"accepted view {len(accepted_images)}/{args.samples}")
                candidate_image = None
                candidate_corners = None
                candidate_time = 0.0

    cv2.destroyAllWindows()
    return accepted_images, accepted_corners


def load_images(args: argparse.Namespace, pattern: tuple[int, int]):
    paths = sorted(
        path
        for path in args.images.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )
    images = []
    corners_list = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        found, corners = detect_corners(image, pattern, exhaustive=True)
        print(f"{path.name}: {'detected' if found else 'not detected'}")
        if found:
            images.append(image)
            corners_list.append(corners)
    return images, corners_list


def calibrate(args: argparse.Namespace, images, corners_list) -> dict:
    if len(images) < args.min_samples:
        raise RuntimeError(
            f"Only {len(images)} usable checkerboard views; need at least {args.min_samples}"
        )
    sizes = {(image.shape[1], image.shape[0]) for image in images}
    if len(sizes) != 1:
        raise RuntimeError(f"All calibration images must have the same size: {sorted(sizes)}")
    image_size = next(iter(sizes))
    inner_cols, inner_rows = pattern_size(args)

    object_template = np.zeros((inner_cols * inner_rows, 3), dtype=np.float32)
    grid = np.mgrid[0:inner_cols, 0:inner_rows].T.reshape(-1, 2)
    object_template[:, :2] = grid * float(args.square_size_mm)
    object_points = [object_template.copy() for _ in corners_list]
    image_points = [corners.astype(np.float32) for corners in corners_list]

    flags = cv2.CALIB_RATIONAL_MODEL
    rms, camera_matrix, distortion, rotations, translations = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None, flags=flags
    )
    per_view_errors = []
    for object_point, image_point, rotation, translation in zip(
        object_points, image_points, rotations, translations
    ):
        projected, _ = cv2.projectPoints(
            object_point, rotation, translation, camera_matrix, distortion
        )
        error = cv2.norm(image_point, projected, cv2.NORM_L2) / len(projected)
        per_view_errors.append(float(error))

    return {
        "passed": bool(rms <= args.max_rms_px),
        "failures": [] if rms <= args.max_rms_px else ["RMS reprojection error is too high"],
        "rms_reprojection_error_px": float(rms),
        "max_allowed_rms_px": float(args.max_rms_px),
        "per_view_error_px": per_view_errors,
        "image_size": list(image_size),
        "socket": getattr(args, "rgb_socket_name", "CAM_A"),
        "sensor_name": getattr(args, "rgb_sensor_name", ""),
        "sensor_size": [
            int(getattr(args, "rgb_sensor_width", 0) or 0),
            int(getattr(args, "rgb_sensor_height", 0) or 0),
        ],
        "resolution_source": getattr(args, "rgb_resolution_source", "unknown"),
        "usable_views": len(images),
        "board": {
            "squares_horizontal": args.square_cols,
            "squares_vertical": args.square_rows,
            "inner_corners_horizontal": inner_cols,
            "inner_corners_vertical": inner_rows,
            "square_size_mm": args.square_size_mm,
        },
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
        "model": "OpenCV pinhole + rational distortion",
        "saved_image_rotated_180": (
            not args.no_rotate_180
            if args.images is None
            else bool(args.images_rotated_180)
        ),
    }


def compare_with_factory_intrinsics(args: argparse.Namespace, result: dict) -> None:
    if args.skip_factory_comparison:
        return
    try:
        socket_name = result.get("socket", "CAM_A")
        socket = getattr(dai.CameraBoardSocket, socket_name, dai.CameraBoardSocket.CAM_A)
        with dai.Device() as device:
            calibration = device.readCalibration()
            factory = np.asarray(
                calibration.getCameraIntrinsics(
                    socket,
                    int(result["image_size"][0]),
                    int(result["image_size"][1]),
                ),
                dtype=np.float64,
            )
        estimated = np.asarray(result["camera_matrix"], dtype=np.float64)
        difference = {
            "fx_relative": float(abs(estimated[0, 0] - factory[0, 0]) / factory[0, 0]),
            "fy_relative": float(abs(estimated[1, 1] - factory[1, 1]) / factory[1, 1]),
        }
        reliable = max(difference.values()) <= 0.25
        result["factory_comparison"] = {
            "factory_camera_matrix": factory.tolist(),
            "estimated_to_factory_focal_difference": difference,
            "within_25_percent": reliable,
        }
        result["intrinsics_reliable"] = reliable
        if not reliable:
            result["passed"] = False
            result["failures"].append(
                "Estimated focal length differs from OAK factory calibration by more than 25%; "
                "collect wider position/tilt coverage or use factory intrinsics"
            )
    except Exception as exc:
        result["factory_comparison"] = {"available": False, "error": str(exc)}


def main() -> None:
    args = parse_args()
    pattern = pattern_size(args)
    print(
        f"Board: {args.square_cols}x{args.square_rows} squares, "
        f"{pattern[0]}x{pattern[1]} inner corners, {args.square_size_mm:g} mm squares"
    )
    try:
        if args.images is None:
            images, corners = capture_live(args, pattern)
        else:
            images, corners = load_images(args, pattern)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print(
            f"\nStopped. Already accepted images remain in "
            f"{args.output_dir / 'captures'}."
        )
        return

    result = calibrate(args, images, corners)
    compare_with_factory_intrinsics(args, result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, (image, detected) in enumerate(zip(images, corners)):
        annotated = image.copy()
        cv2.drawChessboardCorners(annotated, pattern, detected, True)
        cv2.imwrite(str(args.output_dir / f"view_{index:02d}.jpg"), annotated)
    (args.output_dir / "calibration.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
