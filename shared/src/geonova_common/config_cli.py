"""Shared YAML-backed argparse support with stdlib-only bootstrap fallback."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Sequence


class SafeDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show ordinary defaults but never echo credential defaults."""

    def _get_help_string(self, action: argparse.Action) -> str:
        if any(token in action.dest.lower() for token in ("password", "username")):
            return action.help or "Credential value"
        return super()._get_help_string(action)


def _fallback_scalar(text: str) -> Any:
    value = text.strip()
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value.strip("'\"")


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a flat YAML mapping; use PyYAML when available."""
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        data = {}
        for number, source_line in enumerate(text.splitlines(), 1):
            line = source_line.strip()
            if not line or line.startswith("---") or line.startswith("#"):
                continue
            if line.startswith(("-", " ")) or ":" not in line:
                raise ValueError(
                    f"{path}:{number}: bootstrap YAML supports a flat key: value mapping only"
                )
            key, value = line.split(":", 1)
            data[key.strip()] = _fallback_scalar(value.split(" #", 1)[0])
    else:
        data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return {str(key).replace("-", "_"): value for key, value in data.items()}


def _argument_value(action: argparse.Action, value: Any) -> Any:
    if value is None:
        return None
    values = value if isinstance(value, list) else [value]
    if action.type is not None:
        values = [action.type(item) for item in values]
    if action.choices is not None:
        invalid = [item for item in values if item not in action.choices]
        if invalid:
            raise ValueError(
                f"Invalid YAML value for {action.dest}: {invalid}; choices={list(action.choices)}"
            )
    if isinstance(value, list) or action.nargs in ("+", "*"):
        return values
    return values[0]


def _serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def write_default_yaml(parser: argparse.ArgumentParser, path: Path) -> None:
    rows = []
    seen = set()
    for action in parser._actions:
        if action.dest in {"help", "config", "write_default_config"}:
            continue
        if action.dest in seen:
            continue
        seen.add(action.dest)
        sensitive = any(token in action.dest.lower() for token in ("password", "username"))
        effective_default = parser.get_default(action.dest)
        default = None if effective_default is argparse.SUPPRESS or sensitive else _serializable(effective_default)
        if sensitive:
            rows.append("# Sensitive value intentionally omitted; use an environment variable when available.")
        elif action.help and action.help is not argparse.SUPPRESS:
            rows.append(f"# {action.help.replace('%(default)s', str(default))}")
        rows.append(f"{action.dest}: {json.dumps(default, ensure_ascii=False)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def parse_args_with_yaml(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
    *,
    default_config: Path | None = None,
) -> argparse.Namespace:
    """Apply YAML values as parser defaults, then let explicit CLI options win.

    ``default_config`` is opt-in so unrelated commands do not accidentally load
    a neighbouring ``config.yaml``. An explicit ``--config`` always takes
    precedence over it.
    """
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML configuration file. Missing keys keep the built-in defaults.",
    )
    parser.add_argument(
        "--write-default-config",
        type=Path,
        help="Write this command's current defaults as YAML and exit.",
    )
    arguments = list(sys.argv[1:] if argv is None else argv)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    bootstrap.add_argument("--write-default-config", type=Path)
    known, _ = bootstrap.parse_known_args(arguments)
    if known.write_default_config:
        write_default_yaml(parser, known.write_default_config)
        parser.exit(0, f"Wrote default YAML: {known.write_default_config}\n")
    config_path = known.config if known.config is not None else default_config
    if config_path is not None:
        config_path = Path(config_path).expanduser().resolve()
        try:
            data = load_yaml(config_path)
        except Exception as exc:
            parser.error(f"could not load YAML configuration file {config_path}: {exc}")
        actions = {action.dest: action for action in parser._actions}
        unknown = sorted(set(data) - set(actions))
        if unknown:
            parser.error(f"unknown YAML keys: {', '.join(unknown)}")
        converted = {}
        for key, value in data.items():
            action = actions[key]
            try:
                converted[key] = _argument_value(action, value)
            except (TypeError, ValueError) as exc:
                parser.error(str(exc))
            if value is not None:
                action.required = False
        converted["config"] = config_path
        parser.set_defaults(**converted)
    return parser.parse_args(arguments)
