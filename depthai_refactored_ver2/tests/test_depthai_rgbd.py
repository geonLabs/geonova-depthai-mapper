#!/usr/bin/env python3
"""Run a real OAK RGB-D health test after a 200-frame warm-up.

This is a hardware test, not a pytest unit test. It exits non-zero when the
post-warm-up RGB/depth stream is empty, badly synchronized, saturated, or has
almost no valid depth.
"""

from __future__ import annotations

import argparse
import json
import sys
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate synchronized RGB-D output after camera warm-up.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--warmup-frames", type=int, default=200, help="Frames discarded before measurement")
    parser.add_argument("--sample-frames", type=int, default=30, help="Post-warm-up frames used for health metrics")
    parser.add_argument("--fps", type=float, default=15.0, help="Requested camera frame rate")
    parser.add_argument("--output-dir", type=Path, default=Path("test_output/rgbd"), help="Diagnostic image and report directory")
    parser.add_argument("--min-valid-depth-ratio", type=float, default=0.10, help="Minimum nonzero Depth-pixel ratio required to pass")
    parser.add_argument("--max-saturated-rgb-ratio", type=float, default=0.50, help="Maximum near-white RGB-pixel ratio allowed")
    parser.add_argument("--max-sync-delta-ms", type=float, default=20.0, help="Maximum RGB-to-Depth device timestamp difference")
    return parse_args_with_yaml(parser)


def recorder_args(args: argparse.Namespace) -> SimpleNamespace:
    values = dict(DEFAULTS)
    values.update(
        fps=args.fps,
        depth_alignment_mode="auto",
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


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def save_diagnostic(output_dir: Path, rgb: np.ndarray, depth: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "rgb.png"), rgb)
    valid = depth > 0
    clipped = np.clip(depth, 0, 15_000).astype(np.float32)
    preview = np.uint8(255.0 * (1.0 - clipped / 15_000.0))
    preview[~valid] = 0
    color = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    cv2.imwrite(str(output_dir / "depth_color.png"), color)
    overlay = cv2.addWeighted(rgb, 0.65, color, 0.35, 0.0)
    overlay[~valid] = rgb[~valid]
    cv2.imwrite(str(output_dir / "rgb_depth_overlay.png"), overlay)
    cv2.imwrite(str(output_dir / "depth_mm.png"), depth)


def run_test(args: argparse.Namespace) -> dict:
    if args.warmup_frames < 200:
        raise ValueError("--warmup-frames must be at least 200 for this health test")
    if args.sample_frames < 1:
        raise ValueError("--sample-frames must be positive")

    config = recorder_args(args)
    total_frames = args.warmup_frames + args.sample_frames
    valid_ratios: list[float] = []
    saturated_ratios: list[float] = []
    sync_deltas_ms: list[float] = []
    last_rgb = None
    last_depth = None

    device = runtime.connect_depthai_device(config)
    device_name = device.getDeviceName()
    try:
        with dai.Pipeline(device) as pipeline:
            device = pipeline.getDefaultDevice()
            runtime.resolve_transport_options(config, device)
            outputs = runtime.configure_pipeline(pipeline, config, sync_imu=False)
            queue = outputs["sync"].createOutputQueue(maxSize=4, blocking=True)
            pipeline.start()

            for frame_number in range(total_frames):
                group = queue.get()
                rgb_message = runtime.get_group_item(group, "rgb")
                depth_message = runtime.get_group_item(group, "depth")
                if frame_number < args.warmup_frames:
                    continue

                rgb = runtime.get_color_cv_frame(rgb_message)
                depth = depth_message.getFrame()
                if rgb.shape[:2] != depth.shape[:2]:
                    raise RuntimeError(
                        f"RGB/depth shape mismatch: {rgb.shape[:2]} vs {depth.shape[:2]}"
                    )
                rgb_ts = runtime.get_device_ts_ns(rgb_message)
                depth_ts = runtime.get_device_ts_ns(depth_message)
                delta_ms = abs(rgb_ts - depth_ts) / 1_000_000.0
                gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

                valid_ratios.append(float(np.count_nonzero(depth) / depth.size))
                saturated_ratios.append(float(np.count_nonzero(gray >= 250) / gray.size))
                sync_deltas_ms.append(delta_ms)
                last_rgb = rgb
                last_depth = depth
    finally:
        try:
            device.close()
        except Exception:
            pass

    if last_rgb is None or last_depth is None:
        raise RuntimeError("No post-warm-up RGB-D frames were received")

    report = {
        "device": device_name,
        "platform": config.depthai_platform,
        "alignment_requested": config.depth_alignment_mode,
        "alignment_effective": config.depth_alignment_effective,
        "rgb_undistorted": config.rgb_undistort,
        "warmup_frames": args.warmup_frames,
        "sample_frames": args.sample_frames,
        "valid_depth_ratio": {
            "min": min(valid_ratios),
            "median": percentile(valid_ratios, 50),
        },
        "saturated_rgb_ratio": {
            "median": percentile(saturated_ratios, 50),
            "max": max(saturated_ratios),
        },
        "rgb_depth_delta_ms": {
            "median": percentile(sync_deltas_ms, 50),
            "p95": percentile(sync_deltas_ms, 95),
            "max": max(sync_deltas_ms),
        },
    }
    failures = []
    if report["valid_depth_ratio"]["median"] < args.min_valid_depth_ratio:
        failures.append("median valid depth ratio is too low")
    if report["saturated_rgb_ratio"]["median"] > args.max_saturated_rgb_ratio:
        failures.append("median RGB saturation ratio is too high")
    if report["rgb_depth_delta_ms"]["max"] > args.max_sync_delta_ms:
        failures.append("RGB-depth timestamp delta is too high")
    report["passed"] = not failures
    report["failures"] = failures

    save_diagnostic(args.output_dir, last_rgb, last_depth)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    report = run_test(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
