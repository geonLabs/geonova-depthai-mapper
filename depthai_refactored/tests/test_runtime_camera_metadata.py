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
