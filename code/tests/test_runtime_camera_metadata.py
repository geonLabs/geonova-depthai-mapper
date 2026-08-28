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
    assert source == "sensor_aspect_platform_limit"


def test_ov9282_stereo_sensor_prefers_full_800p_height():
    width, height, source = runtime.choose_stereo_input_size(
        1280, 800, dai.Platform.RVC2
    )

    assert (width, height) == (1280, 800)
    assert source == "sensor_aspect_platform_limit"


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
