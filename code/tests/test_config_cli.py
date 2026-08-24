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


def test_default_config_is_loaded_when_config_flag_is_omitted(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "dataset: data/default-run\ncount: 20\nenabled: true\n",
        encoding="utf-8",
    )

    args = parse_args_with_yaml(parser(), [], default_config=config)

    assert args.config == config.resolve()
    assert args.dataset == Path("data/default-run")
    assert args.count == 20
    assert args.enabled is True


def test_explicit_config_takes_priority_over_default_config(tmp_path: Path) -> None:
    default_config = tmp_path / "default.yaml"
    default_config.write_text(
        "dataset: data/default-run\ncount: 20\n",
        encoding="utf-8",
    )
    explicit_config = tmp_path / "explicit.yaml"
    explicit_config.write_text(
        "dataset: data/explicit-run\ncount: 30\n",
        encoding="utf-8",
    )

    args = parse_args_with_yaml(
        parser(),
        ["--config", str(explicit_config)],
        default_config=default_config,
    )

    assert args.config == explicit_config.resolve()
    assert args.dataset == Path("data/explicit-run")
    assert args.count == 30


def test_explicit_cli_value_overrides_default_config(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "dataset: data/default-run\ncount: 20\n",
        encoding="utf-8",
    )

    args = parse_args_with_yaml(
        parser(),
        ["--dataset", "data/cli-run", "--count", "40"],
        default_config=config,
    )

    assert args.dataset == Path("data/cli-run")
    assert args.count == 40


def test_missing_default_config_is_reported_as_argument_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(SystemExit) as error:
        parse_args_with_yaml(parser(), [], default_config=missing)

    assert error.value.code == 2
    assert "could not load YAML configuration file" in capsys.readouterr().err


def test_write_default_config_does_not_require_default_config(tmp_path: Path) -> None:
    output = tmp_path / "generated.yaml"

    with pytest.raises(SystemExit) as error:
        parse_args_with_yaml(
            parser(),
            ["--write-default-config", str(output)],
            default_config=tmp_path / "missing.yaml",
        )

    assert error.value.code == 0
    assert output.is_file()


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
