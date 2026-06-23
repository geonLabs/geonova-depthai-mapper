"""YOLO segmentation post-processing and point/Shapefile export.

Each mask is reduced to three ordered points on its principal (usually
vertical/slanted) axis: upper endpoint, midpoint, and lower endpoint.  Pixel
coordinates are always exported.  WGS84 POINTZ output is added when the saved
RGB-D/GPS data can be projected by the dataset geometry helpers.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import shapefile
from ultralytics import YOLO

from .debug_ui import (
    Dataset,
    compute_world_coordinate_for_source,
    get_depth_frame,
)


POINT_ROLES = ("endpoint_a", "midpoint", "endpoint_b")


def _duration_text(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _progress_text(
    completed: int,
    total: int,
    batch_detections: int,
    total_detections: int,
    elapsed_s: float,
) -> str:
    ratio = completed / total if total else 1.0
    terminal_width = shutil.get_terminal_size((120, 20)).columns
    progress_width = 12 if terminal_width < 96 else 20
    filled = min(progress_width, int(ratio * progress_width))
    progress_bar = "█" * filled + "·" * (progress_width - filled)

    fps = completed / elapsed_s if elapsed_s > 0 else 0.0
    remaining = total - completed
    eta_s = remaining / fps if fps > 0 else 0.0
    if terminal_width < 96:
        return (
            f"[{progress_bar}] {ratio * 100:5.1f}% {completed}/{total} "
            f"ETA {_duration_text(eta_s)} DET +{batch_detections}/{total_detections}"
        )

    detection_width = 5
    detection_level = min(detection_width, batch_detections)
    detection_bar = "█" * detection_level + "·" * (detection_width - detection_level)
    return (
        f"YOLO [{progress_bar}] {ratio * 100:5.1f}%  {completed}/{total}  "
        f"{fps:4.1f}fps  ETA {_duration_text(eta_s)}  "
        f"DET [{detection_bar}] +{batch_detections}  total {total_detections}"
    )


def _draw_progress(line: str, finished: bool = False) -> None:
    if not sys.stdout.isatty():
        if finished:
            print(line, flush=True)
        return
    terminal_width = max(20, shutil.get_terminal_size((120, 20)).columns)
    visible_line = line[: terminal_width - 1]
    sys.stdout.write("\r\033[2K" + visible_line)
    if finished:
        sys.stdout.write("\n")
    sys.stdout.flush()


@dataclass(frozen=True)
class AxisPoint:
    role: str
    x: int
    y: int


def _snap_to_mask(point: np.ndarray, xy: np.ndarray) -> np.ndarray:
    distances = np.sum((xy - point) ** 2, axis=1)
    return xy[int(np.argmin(distances))]


def mask_axis_points(mask: np.ndarray) -> list[AxisPoint]:
    """Return two endpoints and a midpoint along a binary mask's PCA axis."""
    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        raise ValueError("Segmentation mask needs at least three pixels.")

    xy = np.column_stack((xs, ys)).astype(np.float64)
    # Large masks do not need every pixel for a stable covariance estimate.
    sample = xy[:: max(1, len(xy) // 50_000)]
    center = np.mean(sample, axis=0)
    covariance = np.cov(sample - center, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if axis[1] < 0 or (abs(axis[1]) < 1e-9 and axis[0] < 0):
        axis = -axis

    projection = (xy - center) @ axis
    low, high = np.percentile(projection, (2.0, 98.0))
    span = max(float(high - low), 1.0)
    band = max(2.0, span * 0.025)

    def cross_section(target: float) -> np.ndarray:
        selected = xy[np.abs(projection - target) <= band]
        estimate = np.median(selected, axis=0) if len(selected) else center + target * axis
        return _snap_to_mask(estimate, xy)

    upper = cross_section(float(low))
    lower = cross_section(float(high))
    middle = _snap_to_mask((upper + lower) * 0.5, xy)
    ordered = (upper, middle, lower)
    return [
        AxisPoint(role, int(round(point[0])), int(round(point[1])))
        for role, point in zip(POINT_ROLES, ordered)
    ]


def median_depth_mm(
    depth: np.ndarray,
    x: int,
    y: int,
    radius: int = 5,
    max_depth_mm: int = 20_000,
) -> tuple[int, int]:
    """Return a robust local depth and the number of valid samples used."""
    y0, y1 = max(0, y - radius), min(depth.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(depth.shape[1], x + radius + 1)
    patch = depth[y0:y1, x0:x1]
    valid = patch[(patch > 0) & (patch <= max_depth_mm)]
    if not valid.size:
        return 0, 0
    return int(np.median(valid)), int(valid.size)


def _binary_masks(result, image_shape: tuple[int, int]) -> Iterable[tuple[int, np.ndarray]]:
    if result.masks is None:
        return []
    output = []
    height, width = image_shape
    # masks.data can include inference letterbox padding. masks.xy is already
    # scaled back to the original RGB geometry, which must match aligned depth.
    for index, polygon in enumerate(result.masks.xy):
        mask = np.zeros((height, width), dtype=np.uint8)
        points = np.rint(polygon).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 1)
        output.append((index, mask.astype(bool)))
    return output


def _prepare_writer(path: Path, shape_type: int) -> shapefile.Writer:
    writer = shapefile.Writer(str(path), shapeType=shape_type, encoding="utf-8")
    writer.field("frame", "N", 10, 0)
    writer.field("detect_id", "N", 10, 0)
    writer.field("role", "C", 10)
    writer.field("class_id", "N", 6, 0)
    writer.field("class", "C", 40)
    writer.field("conf", "F", 8, 5)
    writer.field("pixel_x", "N", 8, 0)
    writer.field("pixel_y", "N", 8, 0)
    writer.field("depth_mm", "N", 10, 0)
    writer.field("depth_n", "N", 8, 0)
    writer.field("coord_q", "C", 16)
    return writer


def _record(writer: shapefile.Writer, point: dict) -> None:
    writer.record(
        point["frame_index"],
        point["detection_id"],
        point["role"],
        point["class_id"],
        point["class_name"],
        point["confidence"],
        point["pixel_x"],
        point["pixel_y"],
        point["depth_mm"],
        point["depth_sample_count"],
        point["coordinate_quality"],
    )


def _write_prj(path: Path) -> None:
    path.with_suffix(".prj").write_text(
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]',
        encoding="ascii",
    )


def run_dataset(
    dataset_root: Path,
    model_path: Path,
    output_dir: Path | None = None,
    start_frame: int = 200,
    max_frames: int = 100,
    stride: int = 1,
    confidence: float = 0.25,
    batch_size: int = 4,
    image_size: int = 1280,
    device: str | None = None,
    classes: list[int] | None = None,
    depth_radius: int = 5,
    max_depth_mm: int = 20_000,
    orientation_source: str = "gps-course-level",
) -> dict:
    dataset_root = dataset_root.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    output_dir = (output_dir or dataset_root / "yolo_seg").expanduser().resolve()
    detected_rgb_dir = output_dir / "detected_rgb"
    shutil.rmtree(detected_rgb_dir, ignore_errors=True)
    detected_rgb_dir.mkdir(parents=True, exist_ok=True)
    # Remove overlays produced by older versions; current output keeps the
    # original RGB only for frames where a detection exists.
    shutil.rmtree(output_dir / "overlays", ignore_errors=True)

    print("[YOLO] Loading dataset and model...", flush=True)
    dataset = Dataset(dataset_root)
    if start_frame < 200:
        raise ValueError("start_frame must be at least 200 for post-warm-up comparison.")
    if start_frame >= dataset.frame_count:
        raise ValueError(f"start_frame {start_frame} exceeds {dataset.frame_count} frames.")
    if max_frames <= 0 or stride <= 0 or batch_size <= 0:
        raise ValueError("max_frames, stride, and batch_size must be positive.")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0.")

    model = YOLO(str(model_path))
    if model.task != "segment":
        raise ValueError(f"Expected a YOLO segmentation model, got task={model.task!r}.")

    pixel_writer = _prepare_writer(output_dir / "yolo_seg_points_pixels", shapefile.POINT)
    world_writer = _prepare_writer(output_dir / "yolo_seg_points_wgs84", shapefile.POINTZ)
    point_rows: list[dict] = []
    detection_rows: list[dict] = []
    world_count = 0

    stop = min(dataset.frame_count, start_frame + max_frames * stride)
    indices = list(range(start_frame, stop, stride))
    print(
        f"[YOLO] {len(indices)} frames ({indices[0]}..{indices[-1]}) | "
        f"stride={stride} batch={batch_size} conf={confidence} size={image_size}",
        flush=True,
    )
    print(f"[YOLO] Save: {output_dir}", flush=True)
    progress_started_at = time.monotonic()
    _draw_progress(
        _progress_text(0, len(indices), 0, 0, 0.0)
    )

    def batched_predictions():
        for offset in range(0, len(indices), batch_size):
            batch_indices = indices[offset : offset + batch_size]
            batch_items = []
            for frame_index in batch_indices:
                frame = dataset.frame(frame_index)
                image = cv2.imread(str(frame["rgb_path"]), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Failed to read RGB image: {frame['rgb_path']}")
                batch_items.append((frame_index, frame, image))

            predict_args = {
                "source": [item[2] for item in batch_items],
                "batch": len(batch_items),
                "conf": confidence,
                "imgsz": image_size,
                "classes": classes,
                "verbose": False,
            }
            if device:
                predict_args["device"] = device
            results = model.predict(**predict_args)
            if len(results) != len(batch_items):
                raise RuntimeError(
                    f"YOLO returned {len(results)} results for a batch of {len(batch_items)} images."
                )
            batch_detection_count = sum(
                len(result.boxes) if result.boxes is not None else 0
                for result in results
            )
            yield from (
                (frame_index, frame, image, result)
                for (frame_index, frame, image), result in zip(batch_items, results)
            )
            completed = offset + len(batch_items)
            total_detections = sum(row["detection_count"] for row in detection_rows)
            _draw_progress(
                _progress_text(
                    completed,
                    len(indices),
                    batch_detection_count,
                    total_detections,
                    time.monotonic() - progress_started_at,
                ),
                finished=completed == len(indices),
            )

    for frame_index, frame, image, result in batched_predictions():
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            continue
        depth = get_depth_frame(dataset, frame_index)
        frame_detections = []

        for detection_id, mask in _binary_masks(result, image.shape[:2]):
            if detection_id >= len(result.boxes):
                continue
            box = result.boxes[detection_id]
            class_id = int(box.cls.item())
            class_name = str(model.names.get(class_id, class_id))
            det_confidence = float(box.conf.item())
            axis_points = mask_axis_points(mask)

            serialized_points = []
            for order, axis_point in enumerate(axis_points):
                depth_mm, depth_count = median_depth_mm(
                    depth, axis_point.x, axis_point.y, depth_radius, max_depth_mm
                )
                world = compute_world_coordinate_for_source(
                    dataset,
                    frame,
                    axis_point.x,
                    axis_point.y,
                    depth_mm,
                    orientation_source,
                )
                quality = "unavailable"
                if world.get("status") == "ok":
                    trusted = world.get("gps", {}).get("position_quality", {}).get("trusted", False)
                    quality = "trusted" if trusted else "approximate"

                point = {
                    "frame_index": frame_index,
                    "detection_id": detection_id,
                    "point_order": order,
                    "role": axis_point.role,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(det_confidence, 6),
                    "pixel_x": axis_point.x,
                    "pixel_y": axis_point.y,
                    "depth_mm": depth_mm,
                    "depth_sample_count": depth_count,
                    "coordinate_quality": quality,
                    "orientation_source": orientation_source,
                    "longitude_deg": world.get("longitude_deg"),
                    "latitude_deg": world.get("latitude_deg"),
                    "altitude_m": world.get("altitude_m"),
                    "world_status": world.get("status"),
                    "world_reason": world.get("reason"),
                }
                point_rows.append(point)
                serialized_points.append(point)
                pixel_writer.point(axis_point.x, axis_point.y)
                _record(pixel_writer, point)
                if world.get("status") == "ok":
                    world_writer.pointz(
                        float(world["longitude_deg"]),
                        float(world["latitude_deg"]),
                        float(world["altitude_m"]),
                    )
                    _record(world_writer, point)
                    world_count += 1

            frame_detections.append({
                "detection_id": detection_id,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(det_confidence, 6),
                "points": serialized_points,
            })

        if not frame_detections:
            continue
        detected_rgb_path = detected_rgb_dir / f"{frame_index:010d}_{frame['rgb_path'].name}"
        shutil.copy2(frame["rgb_path"], detected_rgb_path)
        try:
            detected_rgb_file = detected_rgb_path.relative_to(dataset_root).as_posix()
        except ValueError:
            detected_rgb_file = str(detected_rgb_path)
        detection_rows.append({
            "frame_index": frame_index,
            "rgb_file": frame["row"].get("rgb_file"),
            "detected_rgb_file": detected_rgb_file,
            "detection_count": len(frame_detections),
            "detections": frame_detections,
        })

    pixel_writer.close()
    world_writer.close()
    _write_prj(output_dir / "yolo_seg_points_wgs84")

    point_fields = list(point_rows[0]) if point_rows else [
        "frame_index", "detection_id", "point_order", "role", "class_id",
        "class_name", "confidence", "pixel_x", "pixel_y", "depth_mm",
        "depth_sample_count", "coordinate_quality", "orientation_source",
        "longitude_deg", "latitude_deg", "altitude_m", "world_status", "world_reason",
    ]
    with (output_dir / "points.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=point_fields)
        writer.writeheader()
        writer.writerows(point_rows)
    with (output_dir / "detections.jsonl").open("w", encoding="utf-8") as file:
        for row in detection_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "dataset": str(dataset_root),
        "model": str(model_path),
        "model_task": model.task,
        "model_classes": model.names,
        "start_frame": start_frame,
        "processed_frames": len(indices),
        "detected_frames": len(detection_rows),
        "stride": stride,
        "confidence": confidence,
        "batch_size": batch_size,
        "image_size": image_size,
        "detections": sum(row["detection_count"] for row in detection_rows),
        "points": len(point_rows),
        "world_points": world_count,
        "orientation_source": orientation_source,
        "coordinate_note": (
            "WGS84 coordinates depend on saved GPS, camera intrinsics, camera mount/extrinsics, "
            "and the selected orientation source. gps-course-level is an approximate test mode."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[YOLO] Done | frames={summary['processed_frames']}, "
        f"detected_frames={summary['detected_frames']}, "
        f"detections={summary['detections']}, points={summary['points']}, "
        f"world_points={summary['world_points']}",
        flush=True,
    )
    print(f"[YOLO] Saved: {output_dir}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO-seg after frame 200 and export mask-axis points to Shapefiles."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("best.pt"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start-frame", type=int, default=200)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--confidence", "--conf", dest="confidence", type=float, default=0.25)
    parser.add_argument("--batch-size", "--batch", dest="batch_size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--device")
    parser.add_argument("--classes", type=int, nargs="+")
    parser.add_argument("--depth-radius", type=int, default=5)
    parser.add_argument("--max-depth-mm", type=int, default=20_000)
    parser.add_argument(
        "--orientation-source",
        choices=("gps-course-level", "gps-course", "ebimu"),
        default="gps-course-level",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dataset(
        dataset_root=args.dataset,
        model_path=args.model,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        stride=args.stride,
        confidence=args.confidence,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=args.device,
        classes=args.classes,
        depth_radius=args.depth_radius,
        max_depth_mm=args.max_depth_mm,
        orientation_source=args.orientation_source,
    )


if __name__ == "__main__":
    main()
