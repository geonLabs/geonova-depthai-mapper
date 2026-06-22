import argparse
import csv
import math
import random
import statistics
from pathlib import Path

import numpy as np

import dataset_debug_ui as ui


def percentile(values, pct):
    clean = sorted(
        value for value in values
        if value is not None and not (isinstance(value, float) and math.isnan(value))
    )
    if not clean:
        return None
    return clean[int((len(clean) - 1) * pct / 100)]


def median(values):
    clean = [
        value for value in values
        if value is not None and not (isinstance(value, float) and math.isnan(value))
    ]
    return statistics.median(clean) if clean else None


def fmt(value, digits=3):
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def angle_diff_deg(a, b):
    return ((a - b + 180.0) % 360.0) - 180.0


def orientation_axes(dataset, frame, source):
    world_cfg = dataset.metadata.get("world_coordinates") or {}
    imu_from_camera = world_cfg.get("imu_from_camera_rpy_deg") or [0.0, 0.0, 0.0]
    camera_mount_rpy = world_cfg.get("camera_mount_rpy_deg") or world_cfg.get("vehicle_to_camera_rpy_deg") or [0.0, 0.0, 0.0]
    camera_mount_rpy = ui.normalized_rpy(camera_mount_rpy)
    declination_deg = float(world_cfg.get("magnetic_declination_deg") or 0.0)
    gps = frame.get("gps")
    ebimu = ui.parse_ebimu_row(frame.get("external_imu"))

    r_imu_from_camera = ui.rpy_matrix_deg(*[float(value) for value in imu_from_camera])
    r_declination = ui.rotation_z(np.deg2rad(declination_deg))

    if source == "ebimu":
        r_enu_from_imu = ui.orientation_matrix_from_ebimu(ebimu)
        if r_enu_from_imu is None:
            return None
        r_enu_from_camera = r_declination @ r_enu_from_imu @ r_imu_from_camera
        course_deg = ui.safe_float(gps.get("course_deg")) if gps else None
    else:
        course_deg = ui.safe_float(gps.get("course_deg")) if gps else None
        if course_deg is None:
            return None
        reference = None
        if source == "gps-course":
            r_enu_from_imu = ui.orientation_matrix_from_ebimu(ebimu)
            if r_enu_from_imu is not None:
                reference = r_declination @ r_enu_from_imu @ r_imu_from_camera
        if source == "gps-course-level":
            r_enu_from_vehicle = ui.orientation_matrix_from_gps_course(course_deg, None)
            r_enu_from_camera = r_enu_from_vehicle @ ui.camera_mount_matrix_deg(*camera_mount_rpy)
        else:
            r_enu_from_camera = ui.orientation_matrix_from_gps_course(course_deg + camera_mount_rpy[2], reference)

    forward = r_enu_from_camera[:, 2]
    heading = math.degrees(math.atan2(forward[0], forward[1])) % 360.0
    elevation = math.degrees(math.atan2(forward[2], math.hypot(forward[0], forward[1])))
    return {
        "heading_deg": heading,
        "elevation_deg": elevation,
        "down_up_component": float(r_enu_from_camera[2, 1]),
        "course_deg": course_deg,
    }


def print_depth_stats(dataset, frame_indices):
    valid_ratios = []
    max_like_ratios = []
    median_depths = []
    p90_depths = []

    for index in frame_indices:
        depth = ui.get_depth_frame(dataset, index)
        valid = depth[depth > 0]
        valid_ratios.append(valid.size / depth.size)
        max_like_ratios.append(float((depth >= 42000).sum()) / depth.size)
        if valid.size:
            median_depths.append(float(np.percentile(valid, 50)) / 1000.0)
            p90_depths.append(float(np.percentile(valid, 90)) / 1000.0)

    print("Depth")
    print(
        "  valid pixel ratio p10/median/p90:",
        fmt(percentile(valid_ratios, 10)),
        fmt(median(valid_ratios)),
        fmt(percentile(valid_ratios, 90)),
    )
    print(
        "  >=42m pixel ratio p10/median/p90:",
        fmt(percentile(max_like_ratios, 10)),
        fmt(median(max_like_ratios)),
        fmt(percentile(max_like_ratios, 90)),
    )
    print(
        "  frame median depth m p10/median/p90:",
        fmt(percentile(median_depths, 10), 2),
        fmt(median(median_depths), 2),
        fmt(percentile(median_depths, 90), 2),
    )
    print(
        "  frame p90 depth m p10/median/p90:",
        fmt(percentile(p90_depths, 10), 2),
        fmt(median(p90_depths), 2),
        fmt(percentile(p90_depths, 90), 2),
    )


def print_sync_stats(dataset, row_sample):
    rgb_depth = []
    gps_delta = []
    external_imu_delta = []
    speed = []
    hdop = []

    for row in row_sample:
        rgb_depth.append(abs(ui.safe_float(row.get("rgb_depth_delta_ms"), float("nan"))))
        gps_delta.append(abs(ui.safe_float(row.get("gps_frame_delta_ms"), float("nan"))))
        external_imu_delta.append(abs(ui.safe_float(row.get("external_imu_frame_delta_ms"), float("nan"))))
        gps = dataset.gps_by_sample_index.get(ui.safe_int(row.get("gps_sample_index"), -1))
        if gps:
            speed_knots = ui.safe_float(gps.get("speed_knots"))
            if speed_knots is not None:
                speed.append(speed_knots * 0.514444)
            gps_hdop = ui.safe_float(gps.get("hdop"))
            if gps_hdop is not None:
                hdop.append(gps_hdop)

    print("Sync")
    print("  abs rgb-depth ms median/p95:", fmt(median(rgb_depth)), fmt(percentile(rgb_depth, 95)))
    print("  abs gps-frame ms median/p95:", fmt(median(gps_delta)), fmt(percentile(gps_delta, 95)))
    print("  abs external-imu-frame ms median/p95:", fmt(median(external_imu_delta)), fmt(percentile(external_imu_delta, 95)))
    print("GPS")
    print("  speed m/s p10/median/p90:", fmt(percentile(speed, 10)), fmt(median(speed)), fmt(percentile(speed, 90)))
    print(
        "  speed < 2m/s fraction:",
        fmt(sum(1 for value in speed if value < 2.0) / len(speed) if speed else None),
    )
    print("  hdop median/p90/p95:", fmt(median(hdop)), fmt(percentile(hdop, 90)), fmt(percentile(hdop, 95)))


def print_orientation_stats(dataset, moving_indices):
    sources = [
        ("ebimu", "EBIMU"),
        ("gps-course", "GPS Course + Tilt"),
        ("gps-course-level", "GPS Course Level"),
    ]
    print("Orientation")
    for source, label in sources:
        heading_diffs = []
        elevations = []
        down_up = []
        for index in moving_indices:
            frame = dataset.frame(index)
            axes = orientation_axes(dataset, frame, source)
            if not axes or axes["course_deg"] is None:
                continue
            heading_diffs.append(abs(angle_diff_deg(axes["heading_deg"], axes["course_deg"])))
            elevations.append(axes["elevation_deg"])
            down_up.append(axes["down_up_component"])
        print(f"  {label}")
        print(
            "    heading diff deg median/p90/p95:",
            fmt(median(heading_diffs), 2),
            fmt(percentile(heading_diffs, 90), 2),
            fmt(percentile(heading_diffs, 95), 2),
        )
        print(
            "    optical elevation deg p10/median/p90:",
            fmt(percentile(elevations, 10), 2),
            fmt(median(elevations), 2),
            fmt(percentile(elevations, 90), 2),
        )
        print(
            "    camera down ENU-up component p10/median/p90:",
            fmt(percentile(down_up, 10)),
            fmt(median(down_up)),
            fmt(percentile(down_up, 90)),
        )


def write_point_samples(dataset, moving_indices, args):
    if not args.output_csv:
        return None

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sources = ["ebimu", "gps-course", "gps-course-level"]
    fieldnames = [
        "frame_index", "x", "y", "depth_mm", "depth_source", "orientation_source",
        "status", "east_m", "north_m", "up_m", "lat_deg", "lon_deg", "alt_m",
        "optical_heading_deg", "optical_elevation_deg", "rgb_file", "depth_file",
    ]
    rng = np.random.default_rng(args.seed)
    written = 0

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index in moving_indices[:args.point_frames]:
            frame = dataset.frame(index)
            depth = ui.get_depth_frame(dataset, index)
            row = frame["row"]
            for _ in range(args.points_per_frame):
                x = int(rng.integers(args.x_min, args.x_max + 1))
                y = int(rng.integers(args.y_min, args.y_max + 1))
                depth_value = ui.robust_depth_value(depth, x, y, args.radius)
                if depth_value["depth_mm"] <= 0 or depth_value["depth_mm"] >= args.max_depth_mm:
                    continue
                for source in sources:
                    world = ui.compute_world_coordinate_for_source(
                        dataset, frame, x, y, depth_value["depth_mm"], source
                    )
                    orientation = world.get("orientation") or {}
                    enu = world.get("enu_offset_m") or {}
                    writer.writerow({
                        "frame_index": index,
                        "x": x,
                        "y": y,
                        "depth_mm": depth_value["depth_mm"],
                        "depth_source": depth_value["source"],
                        "orientation_source": source,
                        "status": world.get("status"),
                        "east_m": enu.get("east"),
                        "north_m": enu.get("north"),
                        "up_m": enu.get("up"),
                        "lat_deg": world.get("latitude_deg"),
                        "lon_deg": world.get("longitude_deg"),
                        "alt_m": world.get("altitude_m"),
                        "optical_heading_deg": orientation.get("optical_heading_deg"),
                        "optical_elevation_deg": orientation.get("optical_elevation_deg"),
                        "rgb_file": row.get("rgb_file"),
                        "depth_file": row.get("depth_file"),
                    })
                    written += 1

    return output_path, written


def moving_frame_indices(dataset, min_speed_m_s, max_hdop):
    indices = []
    step = max(1, dataset.frame_count // 20000)
    for index in range(0, dataset.frame_count, step):
        frame = dataset.frame(index)
        gps = frame.get("gps")
        if not gps:
            continue
        speed_knots = ui.safe_float(gps.get("speed_knots"))
        course = ui.safe_float(gps.get("course_deg"))
        hdop = ui.safe_float(gps.get("hdop"))
        if speed_knots is None or course is None:
            continue
        if speed_knots * 0.514444 < min_speed_m_s:
            continue
        if hdop is not None and hdop > max_hdop:
            continue
        indices.append(index)
    return indices


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose DepthAI dataset geometry and coordinate projection quality.")
    parser.add_argument("dataset", type=Path, help="Dataset folder with metadata.json/timestamps.csv")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--frame-samples", type=int, default=600)
    parser.add_argument("--row-samples", type=int, default=20000)
    parser.add_argument("--point-frames", type=int, default=120)
    parser.add_argument("--points-per-frame", type=int, default=20)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--min-speed-m-s", type=float, default=2.0)
    parser.add_argument("--max-hdop", type=float, default=2.5)
    parser.add_argument("--max-depth-mm", type=int, default=40000)
    parser.add_argument("--x-min", type=int, default=160)
    parser.add_argument("--x-max", type=int, default=1120)
    parser.add_argument("--y-min", type=int, default=240)
    parser.add_argument("--y-max", type=int, default=680)
    parser.add_argument("--output-csv", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    dataset = ui.Dataset(args.dataset)

    frame_indices = random.sample(
        range(dataset.frame_count),
        min(args.frame_samples, dataset.frame_count),
    )
    row_sample = random.sample(
        dataset.timestamps,
        min(args.row_samples, len(dataset.timestamps)),
    )
    moving_indices = moving_frame_indices(dataset, args.min_speed_m_s, args.max_hdop)
    random.shuffle(moving_indices)

    print(f"Dataset: {dataset.root}")
    print(f"Frames: {dataset.frame_count}")
    print(f"Moving frames for orientation/point sampling: {len(moving_indices)}")
    print()
    print_depth_stats(dataset, frame_indices)
    print()
    print_sync_stats(dataset, row_sample)
    print()
    print_orientation_stats(dataset, moving_indices[:min(3000, len(moving_indices))])
    print()
    sample_output = write_point_samples(dataset, moving_indices, args)
    if sample_output:
        path, count = sample_output
        print(f"Point samples CSV: {path} ({count} rows)")


if __name__ == "__main__":
    main()
