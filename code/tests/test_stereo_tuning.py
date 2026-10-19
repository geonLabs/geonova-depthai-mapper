"""StereoDepth tuning options, checked without an attached OAK device."""

from __future__ import annotations

from types import SimpleNamespace

import depthai as dai
import pytest

from geonova_depthai import runtime
from geonova_depthai.capture.cli import parse_args
from geonova_depthai.stereo_settings import parse_ae_region, sensor_ae_region, validate_stereo_args


class FakeStereo:
    """Node-level calls are recorded; ``initialConfig`` is the real host-side message."""

    def __init__(self):
        self.initialConfig = dai.StereoDepthConfig()
        self.calls = []

    def setDefaultProfilePreset(self, preset):
        self.calls.append(("preset", preset))

    def setLeftRightCheck(self, enable):
        self.calls.append(("lr_check", enable))

    def setSubpixel(self, enable):
        self.calls.append(("subpixel", enable))

    def setSubpixelFractionalBits(self, bits):
        self.calls.append(("subpixel_bits", bits))

    def setExtendedDisparity(self, enable):
        self.calls.append(("extended", enable))

    def setPostProcessingHardwareResources(self, shaves, memory_slices):
        self.calls.append(("resources", shaves, memory_slices))


class FakeCamera:
    def __init__(self):
        self.initialControl = self
        self.calls = []

    def setAutoExposureLimit(self, value):
        self.calls.append(("limit", value))

    def setAutoExposureRegion(self, *region):
        self.calls.append(("region", region))


def stereo_args(**overrides):
    values = dict(
        depth_preset="FAST_DENSITY",
        lr_check=True,
        subpixel=True,
        subpixel_fractional_bits=5,
        stereo_median_filter="off",
        stereo_confidence_threshold=-1,
        stereo_lr_check_threshold=-1,
        stereo_speckle_filter=False,
        stereo_speckle_range=50,
        stereo_speckle_difference_threshold=2,
        stereo_spatial_filter=False,
        stereo_spatial_hole_filling_radius=2,
        stereo_spatial_alpha=0.5,
        stereo_spatial_delta=0,
        stereo_spatial_iterations=1,
        stereo_postprocessing_shaves=0,
        stereo_postprocessing_memory_slices=0,
        stereo_ae_max_exposure_us=0,
        stereo_ae_region="",
        rotate_180=False,
        flip=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_default_tuning_only_applies_preset_and_fixed_choices():
    stereo = FakeStereo()
    args = stereo_args()

    runtime.configure_stereo_depth(stereo, args)

    assert stereo.calls == [
        ("preset", dai.node.StereoDepth.PresetMode.FAST_DENSITY),
        ("lr_check", True),
        ("subpixel", True),
        ("subpixel_bits", 5),
        ("extended", False),
    ]
    effective = args.stereo_config_effective
    # A real node would hold the preset's thresholds; the fake keeps message defaults.
    assert effective["confidence_threshold"] == 55
    assert effective["lr_check_threshold"] == 10
    assert effective["speckle_filter"]["enable"] is False
    assert effective["spatial_filter"]["enable"] is False
    assert effective["decimation_factor"] == 1
    assert effective["postprocessing_resources"] is None
    assert args.stereo_median_filter_effective == "off"


def test_overrides_and_filters_reach_the_stereo_config():
    stereo = FakeStereo()
    args = stereo_args(
        stereo_confidence_threshold=25,
        stereo_lr_check_threshold=8,
        stereo_speckle_filter=True,
        stereo_speckle_range=60,
        stereo_spatial_filter=True,
        stereo_spatial_hole_filling_radius=3,
        stereo_spatial_alpha=0.4,
        stereo_postprocessing_shaves=3,
        stereo_postprocessing_memory_slices=3,
    )

    runtime.configure_stereo_depth(stereo, args)

    config = stereo.initialConfig
    assert config.getConfidenceThreshold() == 25
    assert config.getLeftRightCheckThreshold() == 8
    assert config.postProcessing.speckleFilter.enable is True
    assert config.postProcessing.speckleFilter.speckleRange == 60
    assert config.postProcessing.spatialFilter.enable is True
    assert config.postProcessing.spatialFilter.holeFillingRadius == 3
    assert config.postProcessing.spatialFilter.alpha == pytest.approx(0.4)
    assert config.postProcessing.spatialFilter.delta == 0
    assert ("resources", 3, 3) in stereo.calls
    effective = args.stereo_config_effective
    assert effective["confidence_threshold"] == 25
    assert effective["spatial_filter"]["hole_filling_radius"] == 3
    assert effective["postprocessing_resources"] == {"shaves": 3, "memory_slices": 3}


def test_median_is_dropped_for_fine_subpixel():
    stereo = FakeStereo()
    args = stereo_args(stereo_median_filter="7x7", subpixel_fractional_bits=5)

    runtime.configure_stereo_depth(stereo, args)

    assert args.stereo_median_filter_effective == "off"
    assert stereo.initialConfig.getMedianFilter() == dai.MedianFilter.MEDIAN_OFF


def test_exposure_settings_reach_both_stereo_cameras():
    left, right = FakeCamera(), FakeCamera()
    args = stereo_args(stereo_ae_max_exposure_us=2000, stereo_ae_region="0,0.55,1,0.9")

    runtime.configure_stereo_exposure((left, right), args, (1280, 800))

    expected = [("limit", 2000), ("region", (0, 440, 1280, 280))]
    assert left.calls == expected
    assert right.calls == expected
    assert args.stereo_auto_exposure_effective == {
        "max_exposure_us": 2000,
        "region": [0.0, 0.55, 1.0, 0.9],
        "sensor_region": [0, 440, 1280, 280],
    }


def test_exposure_defaults_leave_cameras_untouched():
    left = FakeCamera()
    args = stereo_args()

    runtime.configure_stereo_exposure((left,), args, (1280, 800))

    assert left.calls == []
    assert args.stereo_auto_exposure_effective == {
        "max_exposure_us": None, "region": None, "sensor_region": None,
    }


def test_region_follows_saved_image_orientation():
    region = parse_ae_region("0.1,0.6,0.5,0.9")

    assert sensor_ae_region(region, (1000, 1000)) == (100, 600, 400, 300)
    assert sensor_ae_region(region, (1000, 1000), flip=True) == (100, 100, 400, 300)
    assert sensor_ae_region(region, (1000, 1000), rotate_180=True) == (500, 100, 400, 300)
    assert sensor_ae_region(region, (1000, 1000), rotate_180=True, flip=True) == (500, 600, 400, 300)


@pytest.mark.parametrize("text", ["0,0.5,1", "0,0.9,1,0.5", "a,b,c,d", "0,0,1.2,1"])
def test_invalid_regions_are_rejected(text):
    with pytest.raises(ValueError):
        parse_ae_region(text)


def test_blank_region_meters_the_whole_frame():
    assert parse_ae_region("") is None
    assert parse_ae_region("  ") is None


def test_postprocessing_resources_are_set_together():
    with pytest.raises(ValueError, match="together"):
        validate_stereo_args(stereo_args(stereo_postprocessing_shaves=3))


def test_cli_accepts_tuning_values_and_rejects_bad_ones(tmp_path):
    args = parse_args([
        "--output-dir", str(tmp_path),
        "--stereo-confidence-threshold", "20",
        "--stereo-speckle-filter", "true",
        "--stereo-ae-region", "0,0.55,1,0.9",
    ])
    assert args.stereo_confidence_threshold == 20
    assert args.stereo_speckle_filter is True
    assert args.stereo_ae_region == "0,0.55,1,0.9"

    for bad in (
        ["--stereo-confidence-threshold", "300"],
        ["--stereo-ae-region", "0,0.9,1,0.5"],
        ["--stereo-postprocessing-shaves", "3"],
    ):
        with pytest.raises(SystemExit):
            parse_args(["--output-dir", str(tmp_path), *bad])
