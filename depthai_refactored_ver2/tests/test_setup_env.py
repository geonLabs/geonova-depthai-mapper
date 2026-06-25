from __future__ import annotations

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
        monkeypatch.setattr(setup_env, "nvcc_version", lambda value=version: value)
        assert setup_env.select_torch_build("auto")[0] == expected


def test_explicit_cuda_build_wins(monkeypatch) -> None:
    monkeypatch.setattr(setup_env, "nvcc_version", lambda: (11, 8))
    assert setup_env.select_torch_build("cu128")[0] == "cu128"
