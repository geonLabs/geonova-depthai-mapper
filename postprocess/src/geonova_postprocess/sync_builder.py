import argparse
import bisect
import csv
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from geonova_common.config_cli import parse_args_with_yaml
from geonova_common.run_contract import atomic_json, relative_file, separate_output, validate_capture, utc_now


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
DEFAULT_GPS_MAX_DELTA_MS = 150.0
SYNC_OUTPUT_FILES = ("timestamps.csv", "imu.csv", "metadata.sync.json", "quality_report.json")
SYNC_INCOMPLETE = "sync.incomplete"
RAW_EVENT_FORMATS = ("raw_events_v1", "synced_from_raw_events_v1")


def validate_thresholds(*values):
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Synchronization thresholds must be finite and nonnegative")


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


def write_csv(file, fieldnames: List[str], rows: Iterable[Dict[str, object]]):
    writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def finish_sync_generation(output_dir):
    """Remove transaction files, exposing the generation only at the last step."""
    for name in SYNC_OUTPUT_FILES:
        (output_dir / (name + ".prev")).unlink(missing_ok=True)
        for temporary in output_dir.glob("." + name + ".*.tmp"):
            temporary.unlink(missing_ok=True)
    (output_dir / SYNC_INCOMPLETE).unlink()


def recover_sync_generation(output_dir):
    """Recover a crash before reading or publishing another sync generation."""
    if not (output_dir / SYNC_INCOMPLETE).exists():
        return
    backups = [output_dir / (name + ".prev") for name in SYNC_OUTPUT_FILES]
    if all(backup.is_file() for backup in backups):
        # Keep every backup until restoration finishes, including on a second
        # interruption. Dataset readers refuse the folder while marked.
        for name, backup in zip(SYNC_OUTPUT_FILES, backups):
            shutil.copy2(backup, output_dir / name)
    else:
        # Without a complete previous generation no subset is safe to publish.
        for name in SYNC_OUTPUT_FILES:
            (output_dir / name).unlink(missing_ok=True)
    finish_sync_generation(output_dir)


def backup_sync_output(path):
    # Rename only a fully copied backup so crash recovery cannot mistake a
    # half-written fourth backup for a complete previous generation.
    with tempfile.NamedTemporaryFile(
        dir=str(path.parent), prefix="." + path.name + ".", suffix=".tmp", delete=False,
    ) as file:
        temporary = Path(file.name)
    try:
        shutil.copy2(path, temporary)
        with temporary.open("rb+") as file:
            os.fsync(file.fileno())
        temporary.rename(path.with_name(path.name + ".prev"))
    finally:
        temporary.unlink(missing_ok=True)


def write_sync_outputs(output_dir, timestamp_rows, synced_imu_rows, metadata):
    """Stage all four outputs and publish a recoverable generation."""
    payloads = (
        ("timestamps.csv", TIMESTAMP_FIELDS, timestamp_rows),
        ("imu.csv", SYNCED_IMU_FIELDS, synced_imu_rows),
        ("metadata.sync.json", None, metadata),
        ("quality_report.json", None, metadata["quality_report"]),
    )
    staged = []
    try:
        for filename, fieldnames, payload in payloads:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=str(output_dir),
                prefix="." + filename + ".", suffix=".tmp", delete=False,
            ) as file:
                temporary = Path(file.name)
                staged.append((temporary, output_dir / filename))
                if fieldnames is None:
                    json.dump(payload, file, indent=2, ensure_ascii=False, allow_nan=False)
                    file.write("\n")
                else:
                    write_csv(file, fieldnames, payload)
                file.flush()
                os.fsync(file.fileno())
        marker_created = False
        publishing = False
        previous = set()
        try:
            with (output_dir / SYNC_INCOMPLETE).open("x", encoding="utf-8") as file:
                marker_created = True
                file.write("Sync generation incomplete\n")
                file.flush()
                os.fsync(file.fileno())
            for _, path in staged:
                if path.exists():
                    backup_sync_output(path)
                    previous.add(path.name)
            publishing = True
            for temporary, path in staged:
                os.replace(temporary, path)
        except BaseException:
            if publishing:
                for _, path in staged:
                    if path.name in previous:
                        shutil.copy2(path.with_name(path.name + ".prev"), path)
                    else:
                        path.unlink(missing_ok=True)
            if marker_created:
                finish_sync_generation(output_dir)
            raise
        finish_sync_generation(output_dir)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def check_existing_outputs(dataset, output_dir):
    existing = [output_dir / name for name in SYNC_OUTPUT_FILES
                if (output_dir / name).exists() or (output_dir / name).is_symlink()]
    if not existing:
        return
    if output_dir == dataset:
        try:
            metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            metadata = None
        if isinstance(metadata, dict) and metadata.get("format_version") in RAW_EVENT_FORMATS:
            return
    raise FileExistsError(
        "Refusing to replace sync outputs: metadata.json must identify a "
        "raw_events_v1 or synced_from_raw_events_v1 capture; use a new output directory"
    )


def validate_sync_capture(dataset, allow_legacy=False):
    """Also accept the historical format marker rewritten by in-place sync."""
    try:
        return validate_capture(dataset, allow_legacy=allow_legacy)
    except ValueError as error:
        # The shared validator checks completion, inventory and checksums before
        # rejecting this old marker. Finish its remaining path checks below.
        if str(error) != "Expected a raw_events_v1 dataset":
            raise
        metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format_version") != "synced_from_raw_events_v1":
            raise
    for name in ("rgb_events.csv", "depth_events.csv", "confidence_events.csv"):
        if (dataset / name).exists():
            for row in read_csv(relative_file(dataset, name)):
                relative_file(dataset, row.get("file", ""))
    for path in dataset.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Dataset symlink is not portable: {path}")
    manifest_path = dataset / "capture_manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None


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
    gps_max_delta_ms: float = DEFAULT_GPS_MAX_DELTA_MS,
):
    """Build or replace sync outputs; leave capture metadata and streams intact."""
    validate_thresholds(threshold_ms, rgb_depth_threshold_ms, gps_max_delta_ms)
    dataset = dataset.expanduser().resolve()
    if (dataset / SYNC_INCOMPLETE).exists():
        check_existing_outputs(dataset, dataset)
        recover_sync_generation(dataset)
    if output_dir is None:
        output_dir = dataset
    else:
        output_dir = separate_output(dataset, output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    check_existing_outputs(dataset, output_dir)

    rgb_rows = read_csv(dataset / "rgb_events.csv")
    depth_rows = read_csv(dataset / "depth_events.csv")
    confidence_rows = read_csv(dataset / "confidence_events.csv")
    imu_rows = read_csv(dataset / "imu_events.csv")
    gps_rows = read_csv(dataset / "gps.csv")

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

    timestamp_rows = []
    synced_imu_rows = []
    dropped = {"no_device_ts": 0, "no_depth": 0}
    no_imu_match = 0
    no_gps_within_threshold = 0
    deltas = {"rgb_depth_ms": [], "rgb_imu_ms": [], "gps_frame_ms": [], "depth_confidence_ms": []}

    frame_index = 0
    for rgb in rgb_rows_for_sync:
        rgb_device_ns = safe_int(rgb.get("device_ts_ns"))
        rgb_capture_ns = safe_int(rgb.get("capture_monotonic_ns"))
        if rgb_device_ns is None:
            dropped["no_device_ts"] += 1
            continue
        depth, rgb_depth_delta_ms = depth_index.nearest(rgb_device_ns, rgb_depth_threshold_ms)
        if depth is None:
            dropped["no_depth"] += 1
            continue
        imu, rgb_imu_delta_ms = imu_index.nearest(rgb_device_ns, threshold_ms)
        if imu is None:
            no_imu_match += 1

        confidence = None
        depth_conf_delta_ms = ""
        if confidence_index is not None:
            confidence, depth_conf_delta_ms = confidence_index.nearest(safe_int(depth.get("device_ts_ns")), threshold_ms)
            if confidence is not None:
                deltas["depth_confidence_ms"].append(depth_conf_delta_ms)

        gps, gps_delta_ms = gps_index.nearest(rgb_capture_ns, gps_max_delta_ms)
        if gps is None:
            no_gps_within_threshold += 1

        stem = f"sync-frame{frame_index:07d}"
        imu_message_index = (imu.get("message_index") or imu.get("_message_index") or "") if imu else None
        selected_imu_packets = imu_by_message.get(imu_message_index, []) if imu else []
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
            "imu_message_device_ts_ns": imu.get("message_device_ts_ns", "") if imu else "",
            "rgb_depth_delta_ms": rgb_depth_delta_ms,
            "depth_confidence_delta_ms": depth_conf_delta_ms,
            "rgb_imu_delta_ms": rgb_imu_delta_ms if imu else "",
            "depth_imu_delta_ms": (safe_int(imu.get("message_device_ts_ns"), 0) - safe_int(depth.get("device_ts_ns"), 0)) / 1_000_000.0 if imu and depth else "",
            "imu_packets": len(selected_imu_packets) if imu else "",
        }

        if gps:
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
        deltas["rgb_depth_ms"].append(rgb_depth_delta_ms)
        if imu is not None:
            deltas["rgb_imu_ms"].append(rgb_imu_delta_ms)
        timestamp_rows.append(row)
        frame_index += 1

    if not timestamp_rows:
        raise ValueError(
            "No synchronized frames survived: kept=0, rgb={}, depth={}, "
            "no_device_ts={}, no_depth={}".format(
                len(rgb_rows), len(depth_rows), dropped["no_device_ts"], dropped["no_depth"],
            )
        )

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
        for filename in ["gps.csv", "rgb_events.csv", "depth_events.csv", "confidence_events.csv",
                         "imu_events.csv", "metadata.json", "metadata.overrides.json"]:
            src = dataset / filename
            if src.exists():
                shutil.copy2(src, output_dir / filename)

    # Sync owns only this sidecar. metadata.json remains the exact capture
    # document, including its format_version, IMU calibration and sync settings.
    metadata = {}
    metadata["format_version"] = "synced_from_raw_events_v1"
    metadata["synced_created_wall_time"] = datetime.now().isoformat(timespec="milliseconds")
    metadata.setdefault("sync", {})
    metadata["sync"].update({
        "mode": "postprocess",
        "threshold_ms": threshold_ms,
        "rgb_depth_threshold_ms": rgb_depth_threshold_ms,
        "gps_max_delta_ms": gps_max_delta_ms,
        "source_dataset": str(dataset),
        "kept_frame_count": len(timestamp_rows),
        "dropped": dropped,
        "no_imu_match": no_imu_match,
    })
    metadata["quality_report"] = {
        "raw_counts": {
            "rgb": len(rgb_rows),
            "depth": len(depth_rows),
            "confidence": len(confidence_rows),
            "imu_packets": len(imu_rows),
            "imu_messages": len(imu_by_message),
            "gps": len(gps_rows),
        },
        "kept_frame_count": len(timestamp_rows),
        "dropped": dropped,
        "no_imu_match": no_imu_match,
        "gps_max_delta_ms": gps_max_delta_ms,
        "no_gps_within_threshold": no_gps_within_threshold,
        "order_inversions": {
            "rgb_device_ts": count_order_inversions(rgb_rows, "device_ts_ns"),
            "depth_device_ts": count_order_inversions(depth_rows, "device_ts_ns"),
        },
        "deltas_ms": {name: stats(values) for name, values in deltas.items()},
    }
    write_sync_outputs(output_dir, timestamp_rows, synced_imu_rows, metadata)

    return output_dir, metadata["quality_report"]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build synchronized dataset files from raw per-stream event manifests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="Raw event dataset directory")
    parser.add_argument("--output-dir", default="", help="Empty writes sync outputs into the capture; otherwise use a new or empty separate directory")
    parser.add_argument("--allow-legacy", action="store_true", help="Accept a stopped historical raw-event run without capture_manifest.json")
    parser.add_argument("--sync-threshold-ms", type=float, default=DEFAULT_SYNC_THRESHOLD_MS, help="Maximum absolute device-time difference used for IMU pairing, in milliseconds")
    parser.add_argument("--rgb-depth-threshold-ms", type=float, default=DEFAULT_RGB_DEPTH_THRESHOLD_MS, help="Maximum absolute device-time difference used for RGB/depth pairing, in milliseconds")
    parser.add_argument("--gps-max-delta-ms", type=float, default=DEFAULT_GPS_MAX_DELTA_MS, help="Maximum absolute host-time difference used for RGB/GPS pairing, in milliseconds")
    parser.add_argument("--copy-images", action="store_true", help="Compatibility flag; server CLI always copies images for portable output")
    return parser


def main(argv=None):
    args = parse_args_with_yaml(build_parser(), argv)
    validate_thresholds(args.sync_threshold_ms, args.rgb_depth_threshold_ms, args.gps_max_delta_ms)
    config_root = Path(args.config).parent if args.config else Path.cwd()
    def configured_path(value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else config_root / path
    dataset = configured_path(args.dataset).resolve()
    if (dataset / SYNC_INCOMPLETE).exists():
        check_existing_outputs(dataset, dataset)
        recover_sync_generation(dataset)
    output = separate_output(dataset, configured_path(args.output_dir)) if args.output_dir else None
    check_existing_outputs(dataset, output if output is not None else dataset)
    capture = validate_sync_capture(dataset, allow_legacy=args.allow_legacy)
    try:
        out, report = build_synced_dataset(
            dataset, args.sync_threshold_ms, output, True, args.rgb_depth_threshold_ms,
            gps_max_delta_ms=args.gps_max_delta_ms,
        )
    except BaseException as error:
        if output is not None and output.is_dir():
            atomic_json(output / "processing_manifest.json", {
                "schema": "geonova.processing-run/v1", "state": "failed", "stage": "sync",
                "source_run_id": capture["run_id"] if capture else dataset.name,
                "updated_at_utc": utc_now(), "error_type": type(error).__name__,
            })
        raise
    if output is not None:
        atomic_json(out / "processing_manifest.json", {
            "schema": "geonova.processing-run/v1", "state": "complete", "stage": "sync",
            "source_run_id": capture["run_id"] if capture else dataset.name,
            "source_capture": capture, "updated_at_utc": utc_now(),
        })
    print(f"Synced dataset ready: {out}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
