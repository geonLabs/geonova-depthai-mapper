"""Summarise aligned-depth coverage and cadence of a raw capture run.

Reads ``metadata.json``, ``depth_events.csv`` and an evenly spaced sample of
``depth_mm`` PNGs; nothing in the run folder is modified. Use it to compare
short field recordings made with different StereoDepth settings or mounts.
It needs only NumPy and OpenCV, so it also runs as a plain script::

    python capture/src/geonova_depthai/capture/depth_density.py \
        /data/collections/<run>_raw --json /tmp/density.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

# Road seen from a vehicle mount is at least ~1.5 m away, so rows whose
# median depth is closer than this are most likely the vehicle itself.
NEAR_DEPTH_MM = 1200
# The road rectangle the server-side deterioration gate scores, as fractions
# of the image height and width, so field checks report the same number.
ROAD_ROWS = (0.58, 0.80)
ROAD_COLUMNS = (0.08, 0.92)


def parse_band(text):
    try:
        start, end = (float(value) for value in str(text).split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected 'start,end' fractions of the image size") from error
    if not 0.0 <= start < end <= 1.0:
        raise argparse.ArgumentTypeError("fractions need 0 <= start < end <= 1")
    return start, end


def read_depth_events(run):
    with (run / "depth_events.csv").open(newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row.get("file")]


def depth_cadence(events):
    stamps = sorted(int(row["device_ts_ns"]) for row in events if row.get("device_ts_ns"))
    result = {"frames": len(stamps), "duration_s": None, "fps": None,
              "median_interval_ms": None, "gaps_over_1_5_frames": None, "max_gap_ms": None}
    if len(stamps) < 2:
        return result
    intervals_ms = np.diff(np.asarray(stamps, dtype=np.int64)) / 1_000_000.0
    median_ms = float(np.median(intervals_ms))
    duration_s = (stamps[-1] - stamps[0]) / 1_000_000_000.0
    result.update(
        duration_s=round(duration_s, 2),
        fps=round((len(stamps) - 1) / duration_s, 2) if duration_s > 0 else None,
        median_interval_ms=round(median_ms, 2),
        gaps_over_1_5_frames=int(np.count_nonzero(intervals_ms > 1.5 * median_ms)),
        max_gap_ms=round(float(intervals_ms.max()), 1),
    )
    return result


def evenly_spaced(rows, count):
    if count <= 0 or len(rows) <= count:
        return list(rows)
    indices = np.linspace(0, len(rows) - 1, count).round().astype(int)
    return [rows[index] for index in sorted(set(indices.tolist()))]


def _depth_path(run, relative):
    path = (run / relative).resolve()
    path.relative_to(run)  # refuse CSV entries that leave the run folder
    return path


def _percentile(values, q):
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), q)), 3) if values else None


def depth_coverage(run, rows, band, row_step, columns=ROAD_COLUMNS):
    frame_ratios, band_ratios = [], []
    bins = {}
    missing = 0
    band_rows = band_columns = None
    for row in rows:
        depth = cv2.imread(str(_depth_path(run, row["file"])), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.ndim != 2:
            missing += 1
            continue
        valid = depth > 0
        height, width = depth.shape
        band_rows = (int(band[0] * height), int(band[1] * height))
        band_columns = (int(columns[0] * width), int(columns[1] * width))
        frame_ratios.append(float(valid.mean()))
        band_ratios.append(float(
            valid[band_rows[0]:band_rows[1], band_columns[0]:band_columns[1]].mean()
        ))
        for start in range(0, height, row_step):
            stop = min(start + row_step, height)
            ratios, depths = bins.setdefault((start, stop), ([], []))
            bin_valid = valid[start:stop]
            ratios.append(float(bin_valid.mean()))
            if bin_valid.any():
                depths.append(float(np.median(depth[start:stop][bin_valid])))
    profile = []
    for (start, stop), (ratios, depths) in sorted(bins.items()):
        median_depth = round(float(np.median(depths))) if depths else None
        profile.append({
            "rows": [start, stop],
            "valid_ratio": round(float(np.median(ratios)), 3),
            "median_depth_mm": median_depth,
            "near": median_depth is not None and median_depth < NEAR_DEPTH_MM,
        })
    # The vehicle body fills the bottom of the frame; tolerate the odd far bin
    # (a reflection or the mat's edge) inside it.
    near_from = None
    for index, entry in enumerate(profile):
        below = profile[index:]
        if entry["near"] and sum(item["near"] for item in below) >= 0.75 * len(below):
            near_from = entry["rows"][0]
            break
    return {
        "sampled_frames": len(frame_ratios),
        "unreadable_frames": missing,
        "band": {
            "from": band[0],
            "to": band[1],
            "rows": list(band_rows) if band_rows else None,
            "columns": list(band_columns) if band_columns else None,
        },
        "valid_ratio": {
            "frame_median": _percentile(frame_ratios, 50),
            "frame_p10": _percentile(frame_ratios, 10),
            "band_median": _percentile(band_ratios, 50),
            "band_p10": _percentile(band_ratios, 10),
            "band_min": round(min(band_ratios), 3) if band_ratios else None,
        },
        "near_rows_from": near_from,
        "row_profile": profile,
    }


def gnss_fix_summary(run):
    """Count GGA fix qualities so a test run shows whether RTK was fixed."""
    path = run / "gps.csv"
    if not path.is_file():
        return None
    counts = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("nmea_type") == "GGA":
                name = row.get("fix_quality_name") or row.get("fix_quality") or "unknown"
                counts[name] = counts.get(name, 0) + 1
    total = sum(counts.values())
    return {
        "gga_rows": total,
        "fix_quality": counts,
        "rtk_fixed_ratio": round(counts.get("RTK fixed", 0) / total, 3) if total else None,
    }


def summarise_run(run, band=ROAD_ROWS, samples=60, row_step=50, columns=ROAD_COLUMNS):
    run = Path(run).resolve()
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    events = read_depth_events(run)
    report = {
        "run": run.name,
        "device": (metadata.get("depthai_device") or {}).get("name"),
        "stereo_config": metadata.get("stereo_config"),
        "depth_cadence": depth_cadence(events),
        "gnss": gnss_fix_summary(run),
    }
    report.update(depth_coverage(run, evenly_spaced(events, samples), band, row_step, columns))
    return report


def format_report(report):
    config = report.get("stereo_config") or {}
    cadence = report["depth_cadence"]
    ratio = report["valid_ratio"]
    lines = [
        f"run {report['run']} ({report.get('device') or 'unknown device'})",
        f"stereo preset={config.get('depth_preset')} confidence={config.get('confidence_threshold', 'n/a')} "
        f"lr={config.get('lr_check_threshold', 'n/a')} subpixel_bits={config.get('subpixel_fractional_bits')}",
        f"depth fps={cadence['fps']} frames={cadence['frames']} gaps>1.5f={cadence['gaps_over_1_5_frames']} "
        f"max_gap_ms={cadence['max_gap_ms']}",
        f"valid depth: frame median={ratio['frame_median']} road band rows {report['band']['rows']} "
        f"cols {report['band']['columns']} median={ratio['band_median']} p10={ratio['band_p10']} min={ratio['band_min']} "
        f"({report['sampled_frames']} frames)",
    ]
    gnss = report.get("gnss")
    if gnss is not None:
        lines.append(f"gnss GGA={gnss['gga_rows']} rtk_fixed_ratio={gnss['rtk_fixed_ratio']} {gnss['fix_quality']}")
    if report["near_rows_from"] is not None:
        lines.append(
            f"rows from y={report['near_rows_from']} to the bottom are closer than "
            f"{NEAR_DEPTH_MM / 1000:g} m: probably the vehicle body"
        )
    lines.append("rows        valid  median_depth_mm")
    for entry in report["row_profile"]:
        start, stop = entry["rows"]
        mark = "  near" if entry["near"] else ""
        lines.append(f"{start:4d}-{stop:<5d} {entry['valid_ratio']:6.3f}  {entry['median_depth_mm']}{mark}")
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(description="Summarise aligned-depth coverage of a raw capture run.")
    parser.add_argument("run", type=Path, help="Raw capture folder containing metadata.json and depth_events.csv")
    parser.add_argument("--band", type=parse_band, default=ROAD_ROWS, help="Road band rows as start,end fractions of the image height")
    parser.add_argument("--columns", type=parse_band, default=ROAD_COLUMNS, help="Road band columns as start,end fractions of the image width")
    parser.add_argument("--samples", type=int, default=60, help="Depth frames sampled evenly across the run; 0 reads all")
    parser.add_argument("--row-step", type=int, default=50, help="Row-profile bin height in pixels")
    parser.add_argument("--json", type=Path, help="Also write the full report as JSON")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = summarise_run(
        args.run, band=args.band, samples=args.samples, row_step=max(args.row_step, 1), columns=args.columns
    )
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(format_report(report))
    return report


if __name__ == "__main__":
    main()
