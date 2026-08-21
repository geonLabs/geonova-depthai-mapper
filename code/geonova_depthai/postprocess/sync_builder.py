import argparse
import bisect
import csv
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Tuple

from geonova_depthai.config_cli import parse_args_with_yaml


TIMESTAMP_FIELDS = [
    "frame_index", "stem", "frame_host_wall_time", "frame_host_monotonic_ns",
    "frame_dequeue_host_wall_time", "frame_dequeue_host_monotonic_ns", "frame_queue_lag_ms",
    "rgb_file", "depth_file", "confidence_file",
    "rgb_sequence", "depth_sequence", "confidence_sequence",
    "rgb_device_ts_ns", "depth_device_ts_ns", "confidence_device_ts_ns", "imu_message_device_ts_ns",
    "rgb_depth_delta_ms", "depth_confidence_delta_ms", "rgb_imu_delta_ms", "depth_imu_delta_ms", "imu_packets",
    "gps_sample_index", "gps_host_monotonic_ns", "gps_measurement_host_monotonic_ns", "gps_receive_latency_ms", "gps_frame_delta_ms",
    "gps_nmea_type", "gps_latitude_deg", "gps_longitude_deg", "gps_altitude_m", "gps_fix_quality", "gps_fix_quality_name",
    "gps_rtk_status", "gps_rtk_fixed", "gps_rtk_corrected", "gps_position_valid", "gps_satellites", "gps_hdop",
    "gps_differential_age_s", "gps_reference_station_id",
    "external_imu_sample_index", "external_imu_host_monotonic_ns", "external_imu_frame_delta_ms",
]

SYNCED_IMU_FIELDS = [
    "frame_index", "stem", "packet_index", "imu_message_device_ts_ns",
    "accel_device_ts_ns", "gyro_device_ts_ns",
    "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",
    "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
    "message_index", "message_sequence", "message_capture_wall_time", "message_capture_monotonic_ns",
]

DEFAULT_SYNC_THRESHOLD_MS = 50.0
DEFAULT_RGB_DEPTH_THRESHOLD_MS = 10.0


def safe_int(value, default=None):
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value, default=None):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def sorted_by_int(rows: List[Dict[str, str]], key: str) -> List[Dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            safe_int(row.get(key)) is None,
            safe_int(row.get(key), 0),
        ),
    )


def count_order_inversions(rows: List[Dict[str, str]], key: str) -> int:
    previous = None
    inversions = 0
    for row in rows:
        value = safe_int(row.get(key))
        if value is None:
            continue
        if previous is not None and value < previous:
            inversions += 1
        previous = value
    return inversions


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, object]]):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


class NearestIndex:
    def __init__(self, rows: List[Dict[str, str]], key: str):
        pairs = []
        for row in rows:
            value = safe_int(row.get(key))
            if value is not None:
                pairs.append((value, row))
        pairs.sort(key=lambda item: item[0])
        self.times = [item[0] for item in pairs]
        self.rows = [item[1] for item in pairs]
        self.key = key

    def nearest(self, target_ns, max_delta_ms: Optional[float]) -> Tuple[Optional[Dict[str, str]], Optional[float]]:
        if target_ns is None or not self.times:
            return None, None
        pos = bisect.bisect_left(self.times, target_ns)
        candidates = []
        if pos < len(self.times):
            candidates.append((self.times[pos], self.rows[pos]))
        if pos > 0:
            candidates.append((self.times[pos - 1], self.rows[pos - 1]))
        if not candidates:
            return None, None
        best_ns, best_row = min(candidates, key=lambda item: abs(item[0] - target_ns))
        delta_ms = (best_ns - target_ns) / 1_000_000.0
        if max_delta_ms is not None and abs(delta_ms) > max_delta_ms:
            return None, delta_ms
        return best_row, delta_ms


def choose_gps_key(row):
    return safe_int(row.get("measurement_host_monotonic_ns"), safe_int(row.get("host_monotonic_ns")))


def build_gps_index(rows):
    indexed = []
    for row in rows:
        key = choose_gps_key(row)
        if key is not None:
            copy = dict(row)
            copy["_sync_key_ns"] = str(key)
            indexed.append(copy)
    return NearestIndex(indexed, "_sync_key_ns")


def group_rows(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get(key, "")].append(row)
    return grouped


def stats(values):
    values = [abs(float(v)) for v in values if v not in (None, "") and not math.isnan(float(v))]
    if not values:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    values.sort()
    p95_index = min(len(values) - 1, int(round((len(values) - 1) * 0.95)))
    return {"count": len(values), "p50": values[len(values)//2], "p95": values[p95_index], "max": values[-1]}


def build_synced_dataset(
    dataset: Path,
    threshold_ms: float = DEFAULT_SYNC_THRESHOLD_MS,
    output_dir: Optional[Path] = None,
    copy_images: bool = False,
    rgb_depth_threshold_ms: float = DEFAULT_RGB_DEPTH_THRESHOLD_MS,
):
    dataset = dataset.expanduser().resolve()
    if output_dir is None:
        output_dir = dataset
    else:
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    rgb_rows = read_csv(dataset / "rgb_events.csv")
    depth_rows = read_csv(dataset / "depth_events.csv")
    confidence_rows = read_csv(dataset / "confidence_events.csv")
    imu_rows = read_csv(dataset / "imu_events.csv")
    gps_rows = read_csv(dataset / "gps.csv")
    external_rows = read_csv(dataset / "external_imu.csv")

    if not rgb_rows:
        raise ValueError(f"No rgb_events.csv rows found in {dataset}")
    if not depth_rows:
        raise ValueError(f"No depth_events.csv rows found in {dataset}")

    rgb_rows_for_sync = sorted_by_int(rgb_rows, "device_ts_ns")
    depth_index = NearestIndex(depth_rows, "device_ts_ns")
    confidence_index = NearestIndex(confidence_rows, "device_ts_ns") if confidence_rows else None

    # One IMU message has multiple packet rows; choose a representative row by message_device_ts_ns.
    imu_by_message = group_rows(imu_rows, "message_index")
    imu_rep_rows = []
    for message_index, rows in imu_by_message.items():
        if rows:
            rep = dict(rows[0])
            rep["_message_index"] = message_index
            imu_rep_rows.append(rep)
    imu_index = NearestIndex(imu_rep_rows, "message_device_ts_ns")
    gps_index = build_gps_index(gps_rows)
    external_index = NearestIndex(external_rows, "host_monotonic_ns")

    timestamp_rows = []
    synced_imu_rows = []
    dropped = {"no_depth": 0, "no_imu": 0}
    deltas = {"rgb_depth_ms": [], "rgb_imu_ms": [], "gps_frame_ms": [], "external_imu_frame_ms": [], "depth_confidence_ms": []}

    frame_index = 0
    for rgb in rgb_rows_for_sync:
        rgb_device_ns = safe_int(rgb.get("device_ts_ns"))
        rgb_capture_ns = safe_int(rgb.get("capture_monotonic_ns"))
        if rgb_device_ns is None:
            dropped["no_depth"] += 1
            continue
        depth, rgb_depth_delta_ms = depth_index.nearest(rgb_device_ns, rgb_depth_threshold_ms)
        if depth is None:
            dropped["no_depth"] += 1
            continue
        imu, rgb_imu_delta_ms = imu_index.nearest(rgb_device_ns, threshold_ms)
        if imu is None:
            dropped["no_imu"] += 1
            continue

        confidence = None
        depth_conf_delta_ms = ""
        if confidence_index is not None:
            confidence, depth_conf_delta_ms = confidence_index.nearest(safe_int(depth.get("device_ts_ns")), threshold_ms)
            if confidence is not None:
                deltas["depth_confidence_ms"].append(depth_conf_delta_ms)

        gps, gps_delta_ms = gps_index.nearest(rgb_capture_ns, None)
        external, external_delta_ms = external_index.nearest(rgb_capture_ns, None)

        stem = f"sync-frame{frame_index:07d}"
        imu_message_index = imu.get("message_index") or imu.get("_message_index") or ""
        selected_imu_packets = imu_by_message.get(imu_message_index, [])
        for packet in selected_imu_packets:
            synced_imu_rows.append({
                "frame_index": frame_index,
                "stem": stem,
                "packet_index": packet.get("packet_index", ""),
                "imu_message_device_ts_ns": packet.get("message_device_ts_ns", ""),
                "accel_device_ts_ns": packet.get("accel_device_ts_ns", ""),
                "gyro_device_ts_ns": packet.get("gyro_device_ts_ns", ""),
                "accel_x_m_s2": packet.get("accel_x_m_s2", ""),
                "accel_y_m_s2": packet.get("accel_y_m_s2", ""),
                "accel_z_m_s2": packet.get("accel_z_m_s2", ""),
                "gyro_x_rad_s": packet.get("gyro_x_rad_s", ""),
                "gyro_y_rad_s": packet.get("gyro_y_rad_s", ""),
                "gyro_z_rad_s": packet.get("gyro_z_rad_s", ""),
                "message_index": packet.get("message_index", ""),
                "message_sequence": packet.get("message_sequence", ""),
                "message_capture_wall_time": packet.get("message_capture_wall_time", ""),
                "message_capture_monotonic_ns": packet.get("message_capture_monotonic_ns", ""),
            })

        row = {
            "frame_index": frame_index,
            "stem": stem,
            "frame_host_wall_time": rgb.get("capture_wall_time", ""),
            "frame_host_monotonic_ns": rgb.get("capture_monotonic_ns", ""),
            "frame_dequeue_host_wall_time": rgb.get("dequeue_wall_time", ""),
            "frame_dequeue_host_monotonic_ns": rgb.get("dequeue_monotonic_ns", ""),
            "frame_queue_lag_ms": rgb.get("queue_lag_ms", ""),
            "rgb_file": rgb.get("file", ""),
            "depth_file": depth.get("file", ""),
            "confidence_file": confidence.get("file", "") if confidence else "",
            "rgb_sequence": rgb.get("sequence", ""),
            "depth_sequence": depth.get("sequence", ""),
            "confidence_sequence": confidence.get("sequence", "") if confidence else "",
            "rgb_device_ts_ns": rgb.get("device_ts_ns", ""),
            "depth_device_ts_ns": depth.get("device_ts_ns", ""),
            "confidence_device_ts_ns": confidence.get("device_ts_ns", "") if confidence else "",
            "imu_message_device_ts_ns": imu.get("message_device_ts_ns", ""),
            "rgb_depth_delta_ms": rgb_depth_delta_ms,
            "depth_confidence_delta_ms": depth_conf_delta_ms,
            "rgb_imu_delta_ms": rgb_imu_delta_ms,
            "depth_imu_delta_ms": (safe_int(imu.get("message_device_ts_ns"), 0) - safe_int(depth.get("device_ts_ns"), 0)) / 1_000_000.0 if imu and depth else "",
            "imu_packets": len(selected_imu_packets),
        }

        if gps:
            gps_key = choose_gps_key(gps)
            row.update({
                "gps_sample_index": gps.get("sample_index", ""),
                "gps_host_monotonic_ns": gps.get("host_monotonic_ns", ""),
                "gps_measurement_host_monotonic_ns": gps.get("measurement_host_monotonic_ns", ""),
                "gps_receive_latency_ms": gps.get("receive_latency_ms", ""),
                "gps_frame_delta_ms": gps_delta_ms,
                "gps_nmea_type": gps.get("nmea_type", ""),
                "gps_latitude_deg": gps.get("latitude_deg", ""),
                "gps_longitude_deg": gps.get("longitude_deg", ""),
                "gps_altitude_m": gps.get("altitude_m", ""),
                "gps_fix_quality": gps.get("fix_quality", ""),
                "gps_fix_quality_name": gps.get("fix_quality_name", ""),
                "gps_rtk_status": gps.get("rtk_status", ""),
                "gps_rtk_fixed": gps.get("rtk_fixed", ""),
                "gps_rtk_corrected": gps.get("rtk_corrected", ""),
                "gps_position_valid": gps.get("position_valid", ""),
                "gps_satellites": gps.get("satellites", ""),
                "gps_hdop": gps.get("hdop", ""),
                "gps_differential_age_s": gps.get("differential_age_s", ""),
                "gps_reference_station_id": gps.get("reference_station_id", ""),
            })
            deltas["gps_frame_ms"].append(gps_delta_ms)
        if external:
            row.update({
                "external_imu_sample_index": external.get("sample_index", ""),
                "external_imu_host_monotonic_ns": external.get("host_monotonic_ns", ""),
                "external_imu_frame_delta_ms": external_delta_ms,
            })
            deltas["external_imu_frame_ms"].append(external_delta_ms)

        deltas["rgb_depth_ms"].append(rgb_depth_delta_ms)
        deltas["rgb_imu_ms"].append(rgb_imu_delta_ms)
        timestamp_rows.append(row)
        frame_index += 1

    if output_dir != dataset:
        for dirname in ["rgb", "depth_mm", "confidence"]:
            src = dataset / dirname
            dst = output_dir / dirname
            if not src.exists():
                continue
            if dst.exists():
                continue
            if copy_images:
                shutil.copytree(src, dst)
            else:
                try:
                    dst.symlink_to(src, target_is_directory=True)
                except OSError:
                    shutil.copytree(src, dst)
        for filename in ["gps.csv", "external_imu.csv", "rgb_events.csv", "depth_events.csv", "confidence_events.csv", "imu_events.csv", "metadata.json"]:
            src = dataset / filename
            if src.exists():
                shutil.copy2(src, output_dir / filename)

    write_csv(output_dir / "timestamps.csv", TIMESTAMP_FIELDS, timestamp_rows)
    write_csv(output_dir / "imu.csv", SYNCED_IMU_FIELDS, synced_imu_rows)

    metadata_path = output_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as file:
            metadata = json.load(file)
    metadata["format_version"] = "synced_from_raw_events_v1"
    metadata["synced_created_wall_time"] = datetime.now().isoformat(timespec="milliseconds")
    metadata.setdefault("sync", {})
    metadata["sync"].update({
        "mode": "postprocess",
        "threshold_ms": threshold_ms,
        "rgb_depth_threshold_ms": rgb_depth_threshold_ms,
        "source_dataset": str(dataset),
        "kept_frame_count": len(timestamp_rows),
        "dropped": dropped,
    })
    metadata["quality_report"] = {
        "raw_counts": {
            "rgb": len(rgb_rows),
            "depth": len(depth_rows),
            "confidence": len(confidence_rows),
            "imu_packets": len(imu_rows),
            "imu_messages": len(imu_by_message),
            "gps": len(gps_rows),
            "external_imu": len(external_rows),
        },
        "kept_frame_count": len(timestamp_rows),
        "dropped": dropped,
        "order_inversions": {
            "rgb_device_ts": count_order_inversions(rgb_rows, "device_ts_ns"),
            "depth_device_ts": count_order_inversions(depth_rows, "device_ts_ns"),
        },
        "deltas_ms": {name: stats(values) for name, values in deltas.items()},
    }
    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    with open(output_dir / "quality_report.json", "w") as file:
        json.dump(metadata["quality_report"], file, indent=2, ensure_ascii=False)

    return output_dir, metadata["quality_report"]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build synchronized dataset files from raw per-stream event manifests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="Raw event dataset directory")
    parser.add_argument("--output-dir", default="", help="Default: build synced files in-place in the raw dataset directory")
    parser.add_argument("--sync-threshold-ms", type=float, default=DEFAULT_SYNC_THRESHOLD_MS, help="Maximum absolute device-time difference used for IMU pairing, in milliseconds")
    parser.add_argument("--rgb-depth-threshold-ms", type=float, default=DEFAULT_RGB_DEPTH_THRESHOLD_MS, help="Maximum absolute device-time difference used for RGB/depth pairing, in milliseconds")
    parser.add_argument("--copy-images", action="store_true", help="When output-dir is separate, copy images instead of symlinking")
    return parser


def main(argv=None):
    args = parse_args_with_yaml(build_parser(), argv)
    output_dir = Path(args.output_dir) if args.output_dir else None
    out, report = build_synced_dataset(
        Path(args.dataset),
        args.sync_threshold_ms,
        output_dir,
        args.copy_images,
        args.rgb_depth_threshold_ms,
    )
    print(f"Synced dataset ready: {out}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
