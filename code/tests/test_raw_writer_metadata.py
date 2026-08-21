from __future__ import annotations

import json
from types import SimpleNamespace

from geonova_depthai.capture.raw_writer import RawEventDataset


def recorder_args() -> SimpleNamespace:
    return SimpleNamespace(
        rgb_width=1920,
        rgb_height=1200,
        flip=False,
        rotate_180=False,
        save_confidence_map=False,
        enable_gps=False,
        enable_external_imu=False,
        rgb_format="jpg",
        rgb_jpeg_quality=100,
        fps=30.0,
        depth_fps=30.0,
        depth_fps_effective=30.0,
        usb_speed="SUPER",
        rgb_transport="raw",
        rgb_transport_effective="raw",
        confidence_transport="raw",
        confidence_transport_effective="raw",
        queue_size=16,
        writer_threads=4,
        depth_preset="FAST_ACCURACY",
        lr_check=True,
        subpixel=True,
        subpixel_fractional_bits=5,
        stereo_median_filter="off",
        stereo_median_filter_effective="off",
        depth_alignment_mode="auto",
        depth_alignment_effective="stereo",
        depthai_platform="RVC2",
        depthai_device_name="OAK-D-LR",
        depthai_device_id="mxid",
        rgb_socket_name="CAM_A",
        left_socket_name="CAM_B",
        right_socket_name="CAM_C",
        rgb_sensor_name="AR0234",
        rgb_sensor_width=1920,
        rgb_sensor_height=1200,
        rgb_sensor_types=["COLOR"],
        rgb_resolution_source="sensor_aspect",
        left_sensor={"socket": "CAM_B", "name": "AR0234"},
        right_sensor={"socket": "CAM_C", "name": "AR0234"},
        depth_input_width=1280,
        depth_input_height=800,
        depth_input_resolution_source="sensor_aspect_platform_limit",
        sync_threshold_ms=50.0,
    )


def test_raw_metadata_records_versions_and_stereo_config(tmp_path) -> None:
    dataset = RawEventDataset(tmp_path, recorder_args())
    try:
        metadata = json.loads((dataset.root / "metadata.json").read_text())
    finally:
        dataset.close()

    assert metadata["software_versions"]["depthai"]
    assert metadata["stereo_config"]["depth_preset"] == "FAST_ACCURACY"
    assert metadata["stereo_config"]["subpixel_fractional_bits"] == 5
    assert metadata["host_transport"]["usb_speed"] == "SUPER"
    assert metadata["depth_alignment"]["stereo_matching_input_size"] == {
        "width": 1280,
        "height": 800,
    }
