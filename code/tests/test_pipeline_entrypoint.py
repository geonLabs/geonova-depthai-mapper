from __future__ import annotations

import importlib.util
import os
import warnings
from pathlib import Path

from geonova_depthai.capture.cli import parse_args


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def load_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "geonova_pipeline_entrypoint",
        REPOSITORY_ROOT / "main.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_controller_results_override_yaml_and_cli_paths() -> None:
    entrypoint = load_entrypoint()
    arguments = entrypoint.recorder_arguments(
        ["--config", "/release/config.yaml", "--output-dir", "/old"],
        {"JETSON_PIPELINE_RESULTS_DIR": "/jobs/camera/results"},
    )

    assert arguments[-4:] == [
        "--output-dir",
        "/jobs/camera/results",
        "--controller-bridge-dir",
        str(Path("/jobs/camera/results") / "controller-bridge"),
    ]


def test_default_arguments_select_root_config_and_results_bridge() -> None:
    entrypoint = load_entrypoint()
    arguments = entrypoint.recorder_arguments([], {})

    assert arguments[:2] == [
        "--config",
        str(REPOSITORY_ROOT / "config.yaml"),
    ]
    assert "--controller-bridge-dir" not in arguments


def test_direct_bridge_argument_is_preserved_without_controller_environment() -> None:
    entrypoint = load_entrypoint()
    arguments = entrypoint.recorder_arguments(
        ["--controller-bridge-dir", "/var/lib/jetson-sensors"],
        {},
    )

    assert arguments[-2:] == [
        "--controller-bridge-dir",
        "/var/lib/jetson-sensors",
    ]


def test_controller_can_supply_shared_sensor_bridge() -> None:
    entrypoint = load_entrypoint()
    arguments = entrypoint.recorder_arguments(
        [],
        {
            "JETSON_PIPELINE_RESULTS_DIR": "/jobs/camera/results",
            "JETSON_PIPELINE_SENSOR_BRIDGE_DIR": "/var/lib/jetson-sensors",
        },
    )

    assert arguments[-2:] == [
        "--controller-bridge-dir",
        "/var/lib/jetson-sensors",
    ]


def test_root_config_is_a_valid_capture_config() -> None:
    args = parse_args(["--config", str(REPOSITORY_ROOT / "config.yaml")])

    assert args.output_dir == "results"
    assert args.controller_bridge_dir == "results/controller-bridge"
    assert args.max_runtime_s == 0.0
    assert args.monitor_only is False
    assert args.allow_usb2 is False
    assert args.fps == 15.0
    assert args.depth_fps == 15.0
    assert (args.rgb_width, args.rgb_height) == (1920, 1200)
    # Density-first preset; filters and exposure limits stay opt-in until
    # a field run confirms depth still keeps 15 FPS.
    assert args.depth_preset == "FAST_DENSITY"
    assert (args.lr_check, args.subpixel, args.subpixel_fractional_bits) == (True, True, 5)
    assert args.stereo_confidence_threshold == -1
    assert args.stereo_speckle_filter is False
    assert args.stereo_spatial_filter is False
    assert args.stereo_ae_max_exposure_us == 0
    assert args.stereo_ae_region == ""


def test_removed_imu_config_and_environment_keys_are_ignored_once(tmp_path) -> None:
    config = tmp_path / "legacy.yaml"
    config.write_text(
        "fps: 12\n"
        "enable_external_imu: true\n"
        "external-imu-device: /old/device\n"
        "external_imu_baudrate: no-longer-validated\n"
        "external_imu_format: ebimu\n"
        "external_imu_max_hz: invalid\n",
        encoding="utf-8",
    )
    old_value = os.environ.get("EXTERNAL_IMU_DEVICE")
    os.environ["EXTERNAL_IMU_DEVICE"] = "obsolete-value"
    try:
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            args = parse_args(["--config", str(config), "--fps", "13"])
    finally:
        if old_value is None:
            os.environ.pop("EXTERNAL_IMU_DEVICE", None)
        else:
            os.environ["EXTERNAL_IMU_DEVICE"] = old_value
    assert args.fps == 13
    assert len(emitted) == 1
    assert "Ignoring removed configuration keys" in str(emitted[0].message)
    assert "obsolete-value" not in str(emitted[0].message)
    assert not any("external_imu" in key for key in vars(args))


def test_unrelated_unknown_config_keys_still_fail(tmp_path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("typo_setting: 1\n", encoding="utf-8")
    try:
        parse_args(["--config", str(config)])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("Unknown YAML key was accepted")


def test_removed_imu_cli_options_are_not_accepted() -> None:
    try:
        parse_args(["--external-imu-device", "/old/device"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("Removed IMU CLI option was accepted")


def test_generated_capture_config_has_no_removed_keys(tmp_path) -> None:
    config = tmp_path / "defaults.yaml"
    try:
        parse_args(["--write-default-config", str(config)])
    except SystemExit as error:
        assert error.code == 0
    else:
        raise AssertionError("Default config writer did not exit")
    assert "external_imu" not in config.read_text(encoding="utf-8")
    assert parse_args(["--config", str(config)]).imu_rate == 200
