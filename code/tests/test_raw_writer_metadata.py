from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from geonova_depthai import runtime
from geonova_depthai.capture import raw_writer
from geonova_depthai.capture.raw_writer import RawEventDataset


def recorder_args() -> SimpleNamespace:
    return SimpleNamespace(
        rgb_width=1920,
        rgb_height=1200,
        flip=False,
        rotate_180=False,
        save_confidence_map=False,
        enable_gps=False,
        rgb_format="jpg",
        rgb_jpeg_quality=100,
        fps=30.0,
        imu_rate=200,
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


def test_dataset_roots_are_unique_when_wall_clock_is_identical(
    tmp_path, monkeypatch
) -> None:
    fixed_wall_time = datetime(2026, 8, 24, 15, 30, 45, 123456)

    class FrozenDateTime:
        @classmethod
        def now(cls):
            return fixed_wall_time

    monkeypatch.setattr(raw_writer, "datetime", FrozenDateTime)
    first = RawEventDataset(tmp_path, recorder_args())
    first_root = first.root
    first.close()
    marker = first_root / "first-dataset-marker"
    marker.write_text("preserve", encoding="utf-8")

    second = RawEventDataset(tmp_path, recorder_args())
    try:
        assert first_root != second.root
        assert first_root.name == "2026-08-24-15-30-45_raw"
        assert second.root.name == "2026-08-24-15-30-46_raw"
        assert marker.read_text(encoding="utf-8") == "preserve"
    finally:
        second.close()


class FakeImuDevice:
    def __init__(self, chip="BNO086", calibration_error=None, firmware_error=False):
        self.chip = chip
        self.calibration_error = calibration_error
        self.firmware_error = firmware_error
        self.detections = 0
        self.extrinsics_calls = []
        self.matrix = [[1, 0, 0, 1.5], [0, 1, 0, -2.5], [0, 0, 1, 3.5], [0, 0, 0, 1]]

    def getConnectedIMU(self):
        self.detections += 1
        return self.chip

    def getIMUFirmwareVersion(self):
        if self.firmware_error:
            raise RuntimeError("Firmware unavailable")
        return "3.9.9"

    def readCalibration(self):
        return self

    def getImuToCameraExtrinsics(self, socket, use_spec_translation):
        self.extrinsics_calls.append((socket, use_spec_translation))
        if self.calibration_error:
            raise RuntimeError(self.calibration_error)
        return self.matrix


class FakeImuNode:
    def enableIMUSensor(self, sensors, rate):
        self.sensors, self.rate = sensors, rate

    def setBatchReportThreshold(self, threshold):
        self.threshold = threshold

    def setMaxBatchReports(self, maximum):
        self.maximum = maximum


class FakeImuPipeline:
    def __init__(self, device):
        self.device, self.created = device, []

    def getDefaultDevice(self):
        return self.device

    def create(self, node_type):
        assert node_type == runtime.dai.node.IMU
        node = FakeImuNode()
        self.created.append(node)
        return node


@pytest.mark.parametrize("chip,expected", [
    ("BNO086", ["ACCELEROMETER_RAW", "GYROSCOPE_RAW", "GAME_ROTATION_VECTOR"]),
    ("BNO085", ["ACCELEROMETER_RAW", "GYROSCOPE_RAW", "GAME_ROTATION_VECTOR"]),
    ("BMI270", ["ACCELEROMETER_RAW", "GYROSCOPE_RAW"]),
    ("NONE", []),
    ("NEW_UNKNOWN_CHIP", []),
])
def test_oak_imu_reports_and_requested_rate_by_chip(chip, expected, capsys):
    args = recorder_args()
    args.imu_batch = 10
    device = FakeImuDevice(chip)
    pipeline = FakeImuPipeline(device)
    node = runtime.configure_oak_imu(pipeline, args)
    assert device.detections == 1
    assert args.imu_sensors == expected
    assert args.imu_enabled == bool(expected)
    assert args.imu_metadata["type"] == chip
    assert args.imu_metadata["sensors"] == expected
    assert args.imu_metadata["rate_hz"] == args.imu_rate == 200
    if expected:
        assert len(pipeline.created) == 1
        assert node.sensors == [getattr(runtime.dai.IMUSensor, name) for name in expected]
        assert node.rate == 200
        assert node.threshold == 1
        assert node.maximum == 10
    else:
        assert node is None
        assert not pipeline.created
        assert args.imu_metadata["imu_to_camera"]["matrix"] is None
        assert args.imu_metadata["imu_to_camera"]["error"]
        assert "OAK IMU disabled" in capsys.readouterr().out


@pytest.mark.parametrize("failure", [None, "IMU calibration missing"])
def test_raw_metadata_imu_calibration_available_or_failing(tmp_path, failure):
    args = recorder_args()
    args.rgb_socket = runtime.dai.CameraBoardSocket.CAM_C
    args.rgb_socket_name = "CAM_C"
    args.imu_type = "BNO086"
    args.imu_sensors = runtime.imu_sensors_for_chip(args.imu_type)
    device = FakeImuDevice(calibration_error=failure)
    args.imu_metadata = runtime.read_imu_metadata(device, args)
    dataset = RawEventDataset(tmp_path, args)
    try:
        metadata = json.loads((dataset.root / "metadata.json").read_text())
    finally:
        dataset.close()
    assert device.extrinsics_calls == [(args.rgb_socket, True)]
    expected_extrinsics = {
        "camera_socket": metadata["camera_sockets"]["rgb"],
        "matrix": None if failure else device.matrix,
        "source": None if failure else "device_calibration",
    }
    if failure:
        expected_extrinsics["error"] = failure
    else:
        expected_extrinsics["translation_unit"] = "cm"
    assert metadata["imu"] == {
        "source": "oak", "type": "BNO086", "firmware_version": "3.9.9",
        "sensors": ["ACCELEROMETER_RAW", "GYROSCOPE_RAW", "GAME_ROTATION_VECTOR"],
        "rate_hz": 200, "imu_to_camera": expected_extrinsics,
    }


def test_imu_metadata_firmware_failure_does_not_abort_calibration():
    args = recorder_args()
    args.imu_type, args.imu_sensors = "BMI270", runtime.imu_sensors_for_chip("BMI270")
    metadata = runtime.read_imu_metadata(FakeImuDevice(firmware_error=True), args)
    assert metadata["firmware_version"] is None
    assert metadata["imu_to_camera"]["source"] == "device_calibration"


@pytest.mark.parametrize("chip,report", [("BNO086", "present"), ("BNO086", "missing"), ("BNO086", "default"), ("BMI270", "present")])
def test_imu_event_columns_with_and_without_rotation_vector(tmp_path, chip, report):
    args = recorder_args()
    args.imu_type, args.imu_sensors = chip, runtime.imu_sensors_for_chip(chip)
    args.imu_metadata = runtime.read_imu_metadata(FakeImuDevice(chip), args)
    accel = SimpleNamespace(x=1, y=2, z=3, getTimestampDevice=lambda: timedelta(seconds=1))
    gyro = SimpleNamespace(x=4, y=5, z=6, getTimestampDevice=lambda: timedelta(seconds=2))
    packet = SimpleNamespace(acceleroMeter=accel, gyroscope=gyro)
    if report != "missing":
        packet.rotationVector = SimpleNamespace(
            i=0.1 if report == "present" else 0, j=0, k=0,
            real=0.9 if report == "present" else 0, rotationVectorAccuracy=0.05,
            getTimestampDevice=lambda: timedelta(seconds=3),
        )
    message = SimpleNamespace(packets=[packet], getTimestampDevice=lambda: timedelta(seconds=4), getSequenceNum=lambda: 7)
    dataset = RawEventDataset(tmp_path, args)
    dataset.write_imu_message(message, {"capture_monotonic_ns": 99})
    dataset.close()
    with (dataset.root / "imu_events.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    rv_fields = ["rv_device_ts_ns", "rv_i", "rv_j", "rv_k", "rv_real", "rv_accuracy_rad"]
    assert fields == raw_writer.IMU_EVENT_FIELDS
    assert fields[-6:] == rv_fields
    assert rows[0]["accel_device_ts_ns"] == "1000000000"
    assert rows[0]["gyro_device_ts_ns"] == "2000000000"
    assert rows[0]["accel_z_m_s2"] == "3"
    assert rows[0]["gyro_z_rad_s"] == "6"
    if chip == "BNO086" and report == "present":
        assert [rows[0][field] for field in rv_fields] == ["3000000000", "0.1", "0", "0", "0.9", "0.05"]
    else:
        assert all(rows[0][field] == "" for field in rv_fields)
    assert sorted(dataset.counters) == ["confidence", "depth", "gps", "imu", "rgb"]
    assert sorted(path.name for path in dataset.root.glob("*.csv")) == ["depth_events.csv", "imu_events.csv", "rgb_events.csv"]
