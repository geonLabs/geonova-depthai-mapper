from __future__ import annotations

import importlib.util
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
        "/jobs/camera/results/controller-bridge",
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
    assert args.fps == 30.0
    assert args.depth_fps == 30.0
