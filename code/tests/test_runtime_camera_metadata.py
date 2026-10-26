#!/usr/bin/env python3
"""Camera metadata helpers that do not need an attached OAK device."""

from __future__ import annotations

from types import SimpleNamespace

import depthai as dai

from geonova_depthai import runtime


class FakePoint:
    x = 1012.0
    y = 760.0


class FakeSize:
    width = 2024.0
    height = 1265.0


class FakeRect:
    angle = 0.0
    center = FakePoint()
    size = FakeSize()

    def isNormalized(self):
        return False


class FakeTransformation:
    def getIntrinsicMatrix(self):
        return [
            [1000.0, 0.0, 500.0],
            [0.0, 900.0, 300.0],
            [0.0, 0.0, 1.0],
        ]

    def getSourceIntrinsicMatrix(self):
        return [
            [1100.0, 0.0, 550.0],
            [0.0, 1000.0, 400.0],
            [0.0, 0.0, 1.0],
        ]

    def getMatrix(self):
        return [
            [0.9, 0.0, 1.0],
            [0.0, 0.9, -2.0],
            [0.0, 0.0, 1.0],
        ]

    def getSourceSize(self):
        return (2024, 1520)

    def getDistortionCoefficients(self):
        return [0.0] * 14

    def getDistortionModel(self):
        return "Perspective"

    def getSrcCrops(self):
        return [FakeRect()]


class FakeImgFrame:
    def getTransformation(self):
        return FakeTransformation()

    def getLensPosition(self):
        return 42

    def getLensPositionRaw(self):
        return 42.5

    def getSourceWidth(self):
        return 2024

    def getSourceHeight(self):
        return 1520

    def getSourceHFov(self):
        return 82.7

    def getSourceVFov(self):
        return 67.0

    def getSourceDFov(self):
        return 95.5


def test_imgframe_camera_model_uses_frame_transformation_intrinsics():
    args = SimpleNamespace(
        rgb_width=1000,
        rgb_height=600,
        flip=True,
        rotate_180=False,
    )

    metadata = runtime.imgframe_camera_model_metadata(FakeImgFrame(), args)

    assert metadata["intrinsics_source"] == "DepthAI ImgFrame.getTransformation().getIntrinsicMatrix"
    assert metadata["intrinsics_frame"][0][0] == 1000.0
    assert metadata["intrinsics"][0][2] == 500.0
    assert metadata["intrinsics"][1][2] == 299.0
    assert metadata["distortion_coefficients"] == [0.0] * 14
    assert metadata["depthai_frame_transformation"]["source_size"] == {"width": 2024, "height": 1520}
    assert metadata["depthai_frame_transformation"]["source_crops"][0]["size"]["width"] == 2024.0
    assert metadata["lens_position"] == 42


def test_oak_d_lr_like_stereo_sensor_uses_rvc2_width_limit():
    width, height, source = runtime.choose_stereo_input_size(
        1920, 1200, dai.Platform.RVC2
    )

    assert (width, height) == (1280, 800)
    assert source == "scaled_to_platform_limit"


def test_ov9282_stereo_sensor_prefers_full_800p_height():
    width, height, source = runtime.choose_stereo_input_size(
        1280, 800, dai.Platform.RVC2
    )

    assert (width, height) == (1280, 800)
    assert source == "sensor_max"


def test_ov9782_rgb_uses_full_sensor_before_uniform_1920x1200_upscale():
    args = SimpleNamespace(rgb_sensor_width=1280, rgb_sensor_height=800)

    size = runtime.resolve_rgb_camera_output_size(args, (1920, 1200))

    assert size == (1280, 800)
    assert args.rgb_camera_resolution_source == "sensor_max_then_uniform_upscale"


def test_monitor_pipeline_resizes_physical_sensor_frame_to_preview_geometry(monkeypatch):
    requested_sizes = []
    resized_sizes = []

    class FakeDevice:
        def getPlatform(self):
            return dai.Platform.RVC2

    class FakeCamera:
        def build(self, socket):
            assert socket == dai.CameraBoardSocket.CAM_A
            return self

    class FakeImu:
        out = object()

        def enableIMUSensor(self, sensors, rate):  # noqa: ARG002
            return None

        def setBatchReportThreshold(self, threshold):  # noqa: ARG002
            return None

        def setMaxBatchReports(self, reports):  # noqa: ARG002
            return None

    class FakePipeline:
        def __init__(self):
            self.nodes = iter((FakeCamera(), FakeImu()))

        def getDefaultDevice(self):
            return FakeDevice()

        def create(self, node_type):  # noqa: ARG002
            return next(self.nodes)

    args = SimpleNamespace(
        fps=15.0,
        imu_rate=100,
        imu_batch=5,
        rgb_transport_effective="raw",
        rgb_transport_quality=80,
        rgb_width=1920,
        rgb_height=1200,
        rgb_sensor_width=1280,
        rgb_sensor_height=800,
        rgb_socket=dai.CameraBoardSocket.CAM_A,
    )

    monkeypatch.setattr(runtime, "require_depthai_v3", lambda: None)
    monkeypatch.setattr(runtime, "set_device_identity_metadata", lambda device, args: None)
    monkeypatch.setattr(runtime, "resolve_rgb_output_size", lambda device, args: (1920, 1200))
    monkeypatch.setattr(
        runtime,
        "resolve_rgb_camera_output_size",
        lambda args, requested: (1280, 800),
    )

    def request_output(camera, fps, size, frame_type, enable_undistortion):  # noqa: ARG001
        requested_sizes.append(size)
        return "camera-output"

    def resize_output(pipeline, output, source_size, target_size):  # noqa: ARG001
        resized_sizes.append((source_size, target_size))
        return "resized-output"

    monkeypatch.setattr(runtime, "request_camera_output", request_output)
    monkeypatch.setattr(runtime, "resize_camera_output", resize_output)

    outputs = runtime.configure_monitor_pipeline(FakePipeline(), args)

    assert requested_sizes == [(1280, 800)]
    assert resized_sizes == [((1280, 800), (1920, 1200))]
    assert outputs["rgb"] == "resized-output"


def test_rgb_upscale_rejects_aspect_ratio_change():
    args = SimpleNamespace(rgb_sensor_width=1280, rgb_sensor_height=800)

    try:
        runtime.resolve_rgb_camera_output_size(args, (1920, 1080))
    except RuntimeError as error:
        assert "different aspect ratio" in str(error)
    else:
        raise AssertionError("geometry-changing RGB resize was accepted")


def test_geo_resolution_runtime_records_all_roles_and_preserves_legacy_metadata(tmp_path, capsys):
    from geonova_depthai.capture.cli import parse_args
    from geonova_depthai.capture.raw_writer import RawEventDataset
    import json

    def feature(socket, name, sensor_type):
        return SimpleNamespace(
            socket=socket, sensorName=name, width=1280, height=800,
            supportedTypes=[sensor_type], configs=[SimpleNamespace(
                width=1280, height=800, minFps=1, maxFps=120, type=sensor_type,
            )],
        )

    cameras = [feature(dai.CameraBoardSocket.CAM_A, "OV9782", dai.CameraSensorType.COLOR),
               feature(dai.CameraBoardSocket.CAM_B, "OV9282", dai.CameraSensorType.MONO),
               feature(dai.CameraBoardSocket.CAM_C, "OV9282", dai.CameraSensorType.MONO),
               feature(dai.CameraBoardSocket.CAM_D, "extra", dai.CameraSensorType.MONO)]
    device = SimpleNamespace(getConnectedCameraFeatures=lambda: cameras,
                             getUsbSpeed=lambda: dai.UsbSpeed.SUPER)
    args = parse_args(["--no-gps"])
    args.depthai_platform = "RVC2"
    runtime.resolve_transport_options(args, device)
    size = runtime.resolve_rgb_output_size(device, args)
    assert size == (1280, 800)
    assert runtime.resolve_rgb_camera_output_size(args, size) == size
    assert runtime.resolve_stereo_input_size(device, args, dai.Platform.RVC2,
                                             dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C) == size
    runtime.record_camera_resolution(device, args)
    with_dataset = RawEventDataset(tmp_path, args, camera_model={"width": 1280, "height": 800},
                                   stereo_depth_model={"width": 1280, "height": 800})
    try:
        metadata = json.loads((with_dataset.root / "metadata.json").read_text())
    finally:
        with_dataset.close()
    resolution = metadata["camera_resolution"]
    assert [c["role"] for c in resolution["cameras"]] == ["rgb", "stereo_left", "stereo_right", "unused"]
    assert resolution["rgb"]["source"] == "sensor_max"
    assert resolution["rgb"]["camera_output_size"] == {"width": 1280, "height": 800}
    assert resolution["rgb"]["upscaled"] is False
    assert resolution["stereo"]["width"] == 1280
    assert resolution["aligned_depth_output"] == {"width": 1280, "height": 800}
    assert resolution["estimated_stream_mb_s"] == 53.76
    assert metadata["camera_model"]["width"] == metadata["stereo_depth_model"]["width"] == 1280
    assert metadata["rgb_sensor"]["width"] == 1280
    assert metadata["stereo_sensors"]["input_size"] == {"width": 1280, "height": 800}
    assert metadata["depth_alignment"]["stereo_matching_input_size"] == {"width": 1280, "height": 800}
    log = capsys.readouterr().out
    assert "Camera CAM_A OV9782 max 1280x800 configs=1280x800@120 role=rgb" in log
    assert "53.8 MB/s of 150 MB/s budget (SUPER)" in log


def test_monitor_auto_resolves_usb_transport_before_size_and_never_upscales(monkeypatch):
    from geonova_depthai.capture.cli import parse_args

    feature = SimpleNamespace(socket=dai.CameraBoardSocket.CAM_A, sensorName="OV9782",
                              width=1280, height=800, supportedTypes=[dai.CameraSensorType.COLOR])
    device = SimpleNamespace(getConnectedCameraFeatures=lambda: [feature],
                             getUsbSpeed=lambda: dai.UsbSpeed.HIGH, getPlatform=lambda: dai.Platform.RVC2)
    camera = SimpleNamespace(build=lambda socket: "camera")
    created = []
    requested = []

    def create(node):
        created.append(node)
        assert node == dai.node.Camera
        return camera

    def request(camera, fps, size, **kwargs):
        requested.append(size)
        return "rgb"

    pipeline = SimpleNamespace(getDefaultDevice=lambda: device, create=create)
    monkeypatch.setattr(runtime, "configure_oak_imu", lambda pipeline, args: None)
    monkeypatch.setattr(runtime, "request_camera_output", request)
    monkeypatch.setattr(runtime, "create_mjpeg_output", lambda pipeline, output, fps, quality: "mjpeg")
    args = parse_args(["--monitor-only"])
    outputs = runtime.configure_monitor_pipeline(pipeline, args)
    assert outputs["rgb"] == "mjpeg"
    assert requested == [(1280, 800)]
    assert created == [dai.node.Camera]
    assert args.rgb_transport_effective == "mjpeg"
    assert args.usb_speed == "HIGH"
    assert args.camera_resolution["rgb"]["upscaled"] is False
    assert args.camera_resolution["estimated_stream_mb_s"] == 5.76
    assert args.camera_resolution["aligned_depth_output"] is None
