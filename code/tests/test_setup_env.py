from __future__ import annotations

from pathlib import Path

import setup_env


def test_cuda_build_selection(monkeypatch) -> None:
    for version, expected in (
        (None, "cpu"),
        ((11, 7), "cpu"),
        ((11, 8), "cu118"),
        ((12, 5), "cu118"),
        ((12, 6), "cu126"),
        ((12, 8), "cu128"),
    ):
        monkeypatch.setattr(setup_env, "cuda_version", lambda value=version: value)
        assert setup_env.select_torch_build("auto")[0] == expected


def test_explicit_cuda_build_wins(monkeypatch) -> None:
    monkeypatch.setattr(setup_env, "cuda_version", lambda: (11, 8))
    assert setup_env.select_torch_build("cu128")[0] == "cu128"


def test_platform_and_python_auto_selection(monkeypatch) -> None:
    monkeypatch.setattr(setup_env, "jetson_release", lambda: (35, 6))
    assert setup_env.resolve_platform("auto") == "jetson"
    assert setup_env.resolve_python("auto", "jetson") == setup_env.sys.executable
    assert setup_env.resolve_python("auto", "desktop") == "3.11"


def test_torchvision_matches_nvidia_torch() -> None:
    assert setup_env.torchvision_for_torch("2.1.0a0+nv23.06") == "0.16.0"


def test_torch_location_inside_venv_can_be_identified() -> None:
    venv = Path("/repo/.venv")
    location = Path("/repo/.venv/lib/python3.8/site-packages/torch/__init__.py")
    assert venv in location.parents


def test_jetpack_5_uses_nvidia_v511_index() -> None:
    assert "jp/v511/pytorch" in setup_env.jetson_index_candidates((35, 6))[0]


def test_jetson_fallback_wheel_is_python_abi_specific(monkeypatch) -> None:
    monkeypatch.setattr(setup_env.sys, "version_info", (3, 8, 10))
    monkeypatch.setattr(setup_env, "jetson_index_candidates", lambda release: [])
    wheel = setup_env.discover_jetson_torch_wheel((35, 6), Path("/usr/bin/python3"))
    assert "cp38-cp38-linux_aarch64.whl" in wheel
