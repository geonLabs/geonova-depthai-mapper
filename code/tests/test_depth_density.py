"""Depth coverage report for recorded raw runs."""

from __future__ import annotations

import csv
import json

import cv2
import numpy as np
import pytest

from geonova_depthai.capture import depth_density

HEIGHT, WIDTH = 40, 20


def write_run(root, frames, stamps_ms):
    root.mkdir()
    (root / "depth_mm").mkdir()
    (root / "metadata.json").write_text(json.dumps({
        "depthai_device": {"name": "OAK-D-PRO-W"},
        "stereo_config": {"depth_preset": "FAST_DENSITY", "confidence_threshold": 15,
                          "lr_check_threshold": 10, "subpixel_fractional_bits": 5},
    }))
    rows = []
    for index, (frame, stamp_ms) in enumerate(zip(frames, stamps_ms)):
        name = f"depth_mm/run-depth{index:07d}_depth_mm.png"
        cv2.imwrite(str(root / name), frame)
        rows.append({"event_index": index, "file": name, "device_ts_ns": int(stamp_ms * 1_000_000)})
    with (root / "depth_events.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def road_frame(valid_road_rows):
    """Far scene on top, road band in the middle, vehicle body (0.5 m) at the bottom."""
    frame = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
    frame[:20] = 9000
    frame[20:20 + valid_road_rows] = 4000
    frame[30:] = 500
    return frame


def test_report_measures_band_cadence_and_vehicle_rows(tmp_path):
    frames = [road_frame(10), road_frame(5), road_frame(0), road_frame(5)]
    run = write_run(tmp_path / "run_raw", frames, [0.0, 50.0, 100.0, 200.0])

    report = depth_density.summarise_run(run, band=(0.5, 0.75), samples=0, row_step=10)

    assert report["device"] == "OAK-D-PRO-W"
    assert report["stereo_config"]["depth_preset"] == "FAST_DENSITY"
    assert report["band"]["rows"] == [20, 30]
    assert report["valid_ratio"]["band_median"] == pytest.approx(0.5)
    assert report["valid_ratio"]["band_min"] == pytest.approx(0.0)
    assert report["depth_cadence"]["frames"] == 4
    assert report["depth_cadence"]["gaps_over_1_5_frames"] == 1
    assert report["depth_cadence"]["fps"] == pytest.approx(15.0)
    assert report["near_rows_from"] == 30
    assert [entry["near"] for entry in report["row_profile"]] == [False, False, False, True]
    assert "probably the vehicle body" in depth_density.format_report(report)


def test_vehicle_rows_tolerate_one_far_bin_at_the_bottom(tmp_path):
    frame = road_frame(10)
    frame[38:] = 1500  # far-reading strip inside the vehicle body
    run = write_run(tmp_path / "run_raw", [frame], [0.0])

    report = depth_density.summarise_run(run, samples=0, row_step=2)

    assert report["near_rows_from"] == 30


def test_sampling_spreads_over_the_run(tmp_path):
    frames = [road_frame(10)] * 7
    run = write_run(tmp_path / "run_raw", frames, [index * 66.7 for index in range(7)])

    report = depth_density.summarise_run(run, samples=3)

    assert report["sampled_frames"] == 3
    assert report["depth_cadence"]["frames"] == 7


def test_events_outside_the_run_are_refused(tmp_path):
    run = write_run(tmp_path / "run_raw", [road_frame(10)], [0.0])
    with (run / "depth_events.csv").open("a", newline="") as stream:
        stream.write("1,../outside.png,1000000\n")
    (tmp_path / "outside.png").write_bytes(b"not a depth frame")

    with pytest.raises(ValueError):
        depth_density.summarise_run(run, samples=0)


def test_cli_writes_json(tmp_path, capsys):
    run = write_run(tmp_path / "run_raw", [road_frame(10), road_frame(10)], [0.0, 66.7])
    output = tmp_path / "report.json"

    depth_density.main([str(run), "--band", "0.5,0.75", "--json", str(output)])

    assert json.loads(output.read_text())["valid_ratio"]["band_median"] == pytest.approx(1.0)
    assert "depth fps=" in capsys.readouterr().out
