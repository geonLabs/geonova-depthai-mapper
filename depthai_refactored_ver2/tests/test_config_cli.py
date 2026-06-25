from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from geonova_depthai.config_cli import parse_args_with_yaml, write_default_yaml


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--dataset", type=Path, required=True, help="Input dataset")
    result.add_argument("--count", type=int, default=10, help="Item count")
    result.add_argument("--enabled", action="store_true", help="Enable processing")
    result.add_argument("--password", default="secret", help=argparse.SUPPRESS)
    return result


def test_yaml_supplies_required_values_and_keeps_defaults(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("dataset: data/run\nenabled: true\n", encoding="utf-8")
    args = parse_args_with_yaml(parser(), ["--config", str(config)])
    assert args.dataset == Path("data/run")
    assert args.count == 10
    assert args.enabled is True


def test_explicit_cli_value_overrides_yaml(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("dataset: data/run\ncount: 20\n", encoding="utf-8")
    args = parse_args_with_yaml(
        parser(), ["--config", str(config), "--count", "30"]
    )
    assert args.count == 30


def test_unknown_yaml_key_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("dataset: data/run\nunknown: 1\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        parse_args_with_yaml(parser(), ["--config", str(config)])


def test_default_yaml_redacts_password(tmp_path: Path) -> None:
    output = tmp_path / "defaults.yaml"
    write_default_yaml(parser(), output)
    text = output.read_text(encoding="utf-8")
    assert "password: null" in text
    assert "secret" not in text
