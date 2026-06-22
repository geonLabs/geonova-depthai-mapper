#!/usr/bin/env python3
"""Delete unsynchronized frames from the latest image_records session.

The recorder's ``gps_frame_delta_ms`` alone cannot detect a backed-up camera
queue, because it compares GPS receipt time with the time the delayed frame was
dequeued.  This script reconstructs the camera capture time from the RGB device
timestamp, checks the queue delay, rematches GPS by its NMEA UTC epoch, and
rematches the external IMU by host monotonic time.

Nothing is changed unless ``--apply`` is supplied.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median


SESSION_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
CAMERA_DELTA_COLUMNS = (
    "rgb_depth_delta_ms",
    "depth_confidence_delta_ms",
    "rgb_imu_delta_ms",
    "depth_imu_delta_ms",
)
GPS_TIMESTAMP_COLUMNS = {
    "gps_sample_index": "sample_index",
    "gps_host_monotonic_ns": "host_monotonic_ns",
    "gps_nmea_type": "nmea_type",
    "gps_latitude_deg": "latitude_deg",
    "gps_longitude_deg": "longitude_deg",
    "gps_altitude_m": "altitude_m",
    "gps_fix_quality": "fix_quality",
    "gps_fix_quality_name": "fix_quality_name",
    "gps_rtk_status": "rtk_status",
    "gps_rtk_fixed": "rtk_fixed",
    "gps_rtk_corrected": "rtk_corrected",
    "gps_position_valid": "position_valid",
    "gps_satellites": "satellites",
    "gps_hdop": "hdop",
    "gps_differential_age_s": "differential_age_s",
    "gps_reference_station_id": "reference_station_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep only frames whose camera/GPS/external-IMU synchronization is "
            "within the requested threshold. The latest timestamp-named dataset "
            "is selected unless --dataset is given."
        )
    )
    parser.add_argument("--records-dir", type=Path, default=Path("image_records"))
    parser.add_argument("--dataset", type=Path, help="Explicit dataset directory")
    parser.add_argument("--max-delta-ms", type=float, default=50.0)
    parser.add_argument(
        "--baseline-seconds",
        type=float,
        default=600.0,
        help="Initial interval used to estimate camera device-to-host clock offset",
    )
    parser.add_argument(
        "--local-utc-offset-hours",
        type=float,
        default=9.0,
        help="Timezone of frame_host_wall_time (default: Korea, UTC+9)",
    )
    parser.add_argument(
        "--require-rtk-fixed",
        action="store_true",
        help="Additionally keep only GGA fix_quality=4 samples",
    )
    parser.add_argument(
        "--max-correction-age-s",
        type=float,
        default=2.0,
        help="Maximum differential age when --require-rtk-fixed is used",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rewrite CSV files and delete rejected image files",
    )
    parser.add_argument(
        "--no-csv-backup",
        action="store_true",
        help="Do not copy the original CSV/metadata files to a backup directory",
    )
    return parser.parse_args()


def latest_dataset(records_dir: Path) -> Path:
    candidates = [
        path
        for path in records_dir.iterdir()
        if path.is_dir() and SESSION_NAME.fullmatch(path.name)
    ]
    if not candidates:
        raise RuntimeError(f"No timestamp-named dataset found under {records_dir}")
    return max(candidates, key=lambda path: datetime.strptime(path.name, "%Y-%m-%d_%H-%M-%S"))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise RuntimeError("Cannot calculate a clock baseline from an empty list")
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def parse_local_wall(value: str, local_timezone: timezone) -> float:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=local_timezone)
    return parsed.timestamp()


def parse_gps_epoch(row: dict[str, str]) -> float | None:
    date_text = row.get("date_utc", "")
    time_text = row.get("gps_time_utc", "")
    if not date_text or not time_text:
        return None
    formats = ("%d%m%y%H%M%S.%f", "%d%m%y%H%M%S")
    for fmt in formats:
        try:
            parsed = datetime.strptime(date_text + time_text, fmt)
            return parsed.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def nearest_row(
    sorted_times: list[float],
    rows: list[dict[str, str]],
    target: float,
) -> tuple[dict[str, str] | None, float | None]:
    if not sorted_times:
        return None, None
    index = bisect.bisect_left(sorted_times, target)
    candidates = []
    if index < len(sorted_times):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    best = min(candidates, key=lambda item: abs(sorted_times[item] - target))
    return rows[best], sorted_times[best] - target


def atomic_write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as file:
        temporary = Path(file.name)
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as file:
        temporary = Path(file.name)
        json.dump(value, file, indent=2)
    os.replace(temporary, path)


def validate_required_files(dataset: Path) -> None:
    required = ("timestamps.csv", "gps.csv", "external_imu.csv", "imu.csv", "metadata.json")
    missing = [name for name in required if not (dataset / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing required files in {dataset}: {', '.join(missing)}")


def main() -> None:
    args = parse_args()
    if args.max_delta_ms <= 0:
        raise RuntimeError("--max-delta-ms must be greater than zero")

    dataset = args.dataset or latest_dataset(args.records_dir)
    dataset = dataset.resolve()
    validate_required_files(dataset)
    local_timezone = timezone(timedelta(hours=args.local_utc_offset_hours))
    limit_s = args.max_delta_ms / 1000.0

    timestamp_fields, timestamp_rows = read_csv(dataset / "timestamps.csv")
    gps_fields, all_gps_rows = read_csv(dataset / "gps.csv")
    external_fields, all_external_rows = read_csv(dataset / "external_imu.csv")
    imu_fields, all_imu_rows = read_csv(dataset / "imu.csv")
    if not timestamp_rows:
        raise RuntimeError(f"No frame rows in {dataset / 'timestamps.csv'}")

    first_host_ns = int(timestamp_rows[0]["frame_host_monotonic_ns"])
    calibration_rows = [
        row
        for row in timestamp_rows
        if int(row["frame_host_monotonic_ns"]) - first_host_ns
        <= args.baseline_seconds * 1_000_000_000
    ]
    # A low percentile estimates the normal dequeue latency and is not pulled up
    # by a queue that starts filling during the calibration interval.
    monotonic_offsets_ns = [
        int(row["frame_host_monotonic_ns"]) - int(row["rgb_device_ts_ns"])
        for row in calibration_rows
    ]
    wall_offsets_s = [
        parse_local_wall(row["frame_host_wall_time"], local_timezone)
        - int(row["rgb_device_ts_ns"]) / 1_000_000_000.0
        for row in calibration_rows
    ]
    monotonic_offset_ns = int(percentile(monotonic_offsets_ns, 0.01))
    wall_offset_s = percentile(wall_offsets_s, 0.01)

    gps_pairs = []
    for row in all_gps_rows:
        if row.get("nmea_type") != "GGA":
            continue
        epoch = parse_gps_epoch(row)
        if epoch is not None and row.get("latitude_deg") and row.get("longitude_deg"):
            gps_pairs.append((epoch, row))
    gps_pairs.sort(key=lambda item: item[0])
    gps_times = [item[0] for item in gps_pairs]
    gps_rows = [item[1] for item in gps_pairs]

    external_pairs = []
    for row in all_external_rows:
        value = row.get("host_monotonic_ns", "")
        if value:
            external_pairs.append((int(value) / 1_000_000_000.0, row))
    external_pairs.sort(key=lambda item: item[0])
    external_times = [item[0] for item in external_pairs]
    external_rows = [item[1] for item in external_pairs]

    kept_rows: list[dict[str, str]] = []
    rejected_reasons: Counter[str] = Counter()
    kept_maxima: Counter[str] = Counter()
    capture_mono_values: list[int] = []

    for original in timestamp_rows:
        row = dict(original)
        rgb_device_ns = int(row["rgb_device_ts_ns"])
        capture_mono_ns = rgb_device_ns + monotonic_offset_ns
        capture_epoch = rgb_device_ns / 1_000_000_000.0 + wall_offset_s
        original_host_ns = int(row["frame_host_monotonic_ns"])
        queue_lag_ms = (original_host_ns - capture_mono_ns) / 1_000_000.0

        reasons = []
        if abs(queue_lag_ms) > args.max_delta_ms:
            reasons.append("camera_queue_lag")

        for column in CAMERA_DELTA_COLUMNS:
            value = float_or_none(row.get(column))
            if value is not None and abs(value) > args.max_delta_ms:
                reasons.append(column)

        gps_row, gps_delta_s = nearest_row(gps_times, gps_rows, capture_epoch)
        if gps_row is None or gps_delta_s is None or abs(gps_delta_s) > limit_s + 1e-6:
            reasons.append("gps_measurement_delta")
        elif args.require_rtk_fixed:
            if gps_row.get("fix_quality") != "4":
                reasons.append("rtk_not_fixed")
            correction_age = float_or_none(gps_row.get("differential_age_s"))
            if correction_age is None or correction_age > args.max_correction_age_s:
                reasons.append("rtk_correction_age")

        external_row, external_delta_s = nearest_row(
            external_times,
            external_rows,
            capture_mono_ns / 1_000_000_000.0,
        )
        if (
            external_row is None
            or external_delta_s is None
            or abs(external_delta_s) > limit_s + 1e-6
        ):
            reasons.append("external_imu_delta")

        if reasons:
            for reason in set(reasons):
                rejected_reasons[reason] += 1
            continue

        # Store estimated capture time instead of delayed host dequeue time.
        capture_wall = datetime.fromtimestamp(capture_epoch, local_timezone).replace(tzinfo=None)
        row["frame_host_monotonic_ns"] = str(capture_mono_ns)
        row["frame_host_wall_time"] = capture_wall.isoformat(timespec="milliseconds")

        assert gps_row is not None and gps_delta_s is not None
        for timestamp_column, gps_column in GPS_TIMESTAMP_COLUMNS.items():
            row[timestamp_column] = gps_row.get(gps_column, "")
        row["gps_frame_delta_ms"] = f"{gps_delta_s * 1000.0:.6f}"

        assert external_row is not None and external_delta_s is not None
        row["external_imu_sample_index"] = external_row.get("sample_index", "")
        row["external_imu_host_monotonic_ns"] = external_row.get("host_monotonic_ns", "")
        row["external_imu_frame_delta_ms"] = f"{external_delta_s * 1000.0:.6f}"

        kept_rows.append(row)
        capture_mono_values.append(capture_mono_ns)
        kept_maxima["camera_queue_lag_ms"] = max(
            kept_maxima["camera_queue_lag_ms"], abs(queue_lag_ms)
        )
        kept_maxima["gps_measurement_delta_ms"] = max(
            kept_maxima["gps_measurement_delta_ms"], abs(gps_delta_s * 1000.0)
        )
        kept_maxima["external_imu_delta_ms"] = max(
            kept_maxima["external_imu_delta_ms"], abs(external_delta_s * 1000.0)
        )
        for column in CAMERA_DELTA_COLUMNS:
            value = float_or_none(row.get(column))
            if value is not None:
                kept_maxima[column] = max(kept_maxima[column], abs(value))

    if not kept_rows:
        raise RuntimeError("The sync criteria rejected every frame; nothing was changed")

    kept_indices = {row["frame_index"] for row in kept_rows}
    kept_imu_rows = [row for row in all_imu_rows if row.get("frame_index") in kept_indices]
    min_mono_ns = min(capture_mono_values) - int(args.max_delta_ms * 1_000_000)
    max_mono_ns = max(capture_mono_values) + int(args.max_delta_ms * 1_000_000)
    kept_gps_rows = [
        row
        for row in all_gps_rows
        if row.get("host_monotonic_ns")
        and min_mono_ns <= int(row["host_monotonic_ns"]) <= max_mono_ns + 150_000_000
    ]
    kept_external_rows = [
        row
        for row in all_external_rows
        if row.get("host_monotonic_ns")
        and min_mono_ns <= int(row["host_monotonic_ns"]) <= max_mono_ns
    ]

    referenced_images = set()
    for row in kept_rows:
        for column in ("rgb_file", "depth_file", "confidence_file"):
            relative = row.get(column, "")
            if relative:
                referenced_images.add(Path(relative))
    existing_images = set()
    for directory in ("rgb", "depth_mm", "confidence"):
        image_dir = dataset / directory
        if image_dir.is_dir():
            existing_images.update(path.relative_to(dataset) for path in image_dir.iterdir() if path.is_file())
    images_to_delete = existing_images - referenced_images

    print(f"Dataset: {dataset}")
    print(f"Threshold: {args.max_delta_ms:.3f} ms")
    print(f"Frames: {len(timestamp_rows)} -> {len(kept_rows)} (reject {len(timestamp_rows) - len(kept_rows)})")
    print(f"Image files to delete: {len(images_to_delete)}")
    print("Reject reasons:")
    for reason, count in rejected_reasons.most_common():
        print(f"  {reason}: {count}")
    print("Maximum absolute deltas among kept frames:")
    for name, value in sorted(kept_maxima.items()):
        print(f"  {name}: {value:.6f} ms")

    if not args.apply:
        print("Dry run only. Re-run with --apply to rewrite CSVs and delete rejected images.")
        return

    backup_dir = None
    if not args.no_csv_backup:
        backup_dir = dataset / f"sync_prune_csv_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir()
        for name in ("timestamps.csv", "imu.csv", "gps.csv", "external_imu.csv", "metadata.json"):
            shutil.copy2(dataset / name, backup_dir / name)

    atomic_write_csv(dataset / "timestamps.csv", timestamp_fields, kept_rows)
    atomic_write_csv(dataset / "imu.csv", imu_fields, kept_imu_rows)
    atomic_write_csv(dataset / "gps.csv", gps_fields, kept_gps_rows)
    atomic_write_csv(dataset / "external_imu.csv", external_fields, kept_external_rows)

    with (dataset / "metadata.json").open() as file:
        metadata = json.load(file)
    gga_counts = Counter(
        str(row.get("fix_quality") or "unknown")
        for row in kept_gps_rows
        if row.get("nmea_type") == "GGA"
    )
    first_wall = datetime.fromisoformat(kept_rows[0]["frame_host_wall_time"])
    last_wall = datetime.fromisoformat(kept_rows[-1]["frame_host_wall_time"])
    duration_s = max((last_wall - first_wall).total_seconds(), 1e-9)
    metadata["closed_wall_time"] = kept_rows[-1]["frame_host_wall_time"]
    metadata["frame_count"] = len(kept_rows)
    metadata["average_saved_fps"] = len(kept_rows) / duration_s
    metadata["imu_packet_count"] = len(kept_imu_rows)
    metadata["confidence_frame_count"] = sum(bool(row.get("confidence_file")) for row in kept_rows)
    metadata["gps_sample_count"] = len(kept_gps_rows)
    metadata["gps_gga_fix_quality_counts"] = dict(gga_counts)
    metadata["external_imu_sample_count"] = len(kept_external_rows)
    metadata["sync_pruning"] = {
        "applied_wall_time": datetime.now().isoformat(timespec="milliseconds"),
        "maximum_delta_ms": args.max_delta_ms,
        "original_frame_count": len(timestamp_rows),
        "kept_frame_count": len(kept_rows),
        "deleted_frame_count": len(timestamp_rows) - len(kept_rows),
        "deleted_image_file_count": len(images_to_delete),
        "camera_device_to_host_monotonic_offset_ns": monotonic_offset_ns,
        "camera_device_to_wall_offset_s": wall_offset_s,
        "gps_matching": "nearest GGA gps_time_utc/date_utc to reconstructed RGB capture time",
        "external_imu_matching": "nearest host_monotonic_ns to reconstructed RGB capture time",
        "require_rtk_fixed": args.require_rtk_fixed,
        "maximum_rtk_correction_age_s": (
            args.max_correction_age_s if args.require_rtk_fixed else None
        ),
        "maximum_kept_absolute_deltas_ms": dict(kept_maxima),
        "csv_backup_directory": backup_dir.name if backup_dir else None,
    }
    atomic_write_json(dataset / "metadata.json", metadata)

    for relative in images_to_delete:
        (dataset / relative).unlink()

    print(f"Applied successfully. Kept {len(kept_rows)} synchronized frames.")
    if backup_dir is not None:
        print(f"CSV backup: {backup_dir}")


if __name__ == "__main__":
    main()
