#!/usr/bin/env python3
"""Create a cross-platform uv environment and install the matching PyTorch build.

This bootstrap intentionally uses Python 3.8-compatible syntax and only the standard
library.  It can therefore install the project's Python 3.11 environment on older
Ubuntu/Jetson hosts before any project dependencies are available.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import site
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TORCH_VERSION = "2.7.1"
TORCHVISION_VERSION = "0.22.1"
TORCHAUDIO_VERSION = "2.7.1"
SUPPORTED_BUILDS = ("cpu", "cu118", "cu126", "cu128")


def run(command: List[str], cwd: Optional[Path] = None, capture: bool = False) -> str:
    print("+", subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return completed.stdout or ""


def uv_candidates() -> List[Path]:
    """Return common uv locations, including user installs not present in PATH."""
    executable_name = "uv.exe" if os.name == "nt" else "uv"
    candidates = [Path(sysconfig.get_path("scripts")) / executable_name]
    candidates.append(Path(site.getuserbase()) / ("Scripts" if os.name == "nt" else "bin") / executable_name)
    candidates.append(Path.home() / ".local" / "bin" / executable_name)
    candidates.append(Path.home() / ".cargo" / "bin" / executable_name)
    return candidates


def locate_uv() -> Optional[str]:
    executable = shutil.which("uv")
    if executable:
        return executable
    for candidate in uv_candidates():
        if candidate.is_file():
            return str(candidate)
    return None


def install_uv_officially() -> None:
    """Use Astral's installer when pip is unavailable or externally managed."""
    if os.name == "nt":
        shell = shutil.which("pwsh") or shutil.which("powershell")
        if not shell:
            raise RuntimeError("PowerShell is required to bootstrap uv on Windows.")
        url = "https://astral.sh/uv/install.ps1"
        suffix = ".ps1"
    else:
        shell = shutil.which("sh")
        if not shell:
            raise RuntimeError("A POSIX sh executable is required to bootstrap uv.")
        url = "https://astral.sh/uv/install.sh"
        suffix = ".sh"

    print(f"Downloading the official uv installer from {url}", flush=True)
    with tempfile.TemporaryDirectory(prefix="depthai-uv-") as temporary:
        installer = Path(temporary) / ("install" + suffix)
        urllib.request.urlretrieve(url, str(installer))
        if os.name == "nt":
            run([shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(installer)])
        else:
            run([shell, str(installer)])


def find_uv() -> str:
    executable = locate_uv()
    if executable:
        return executable

    print("uv was not found; bootstrapping it for the current user.", flush=True)
    try:
        run([sys.executable, "-m", "pip", "--version"], capture=True)
    except subprocess.CalledProcessError:
        try:
            run([sys.executable, "-m", "ensurepip", "--upgrade"])
        except subprocess.CalledProcessError:
            pass

    pip_command = [sys.executable, "-m", "pip", "install"]
    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        pip_command.append("--user")
    try:
        run(pip_command + ["uv"])
    except subprocess.CalledProcessError:
        install_uv_officially()

    executable = locate_uv()
    if executable:
        return executable
    raise RuntimeError("uv installation completed but the uv executable was not found.")


def nvcc_version() -> Optional[Tuple[int, int]]:
    executable = shutil.which("nvcc")
    if not executable:
        return None
    output = run([executable, "--version"], capture=True)
    print(output, end="")
    match = re.search(r"release\s+(\d+)\.(\d+)", output)
    return (int(match.group(1)), int(match.group(2))) if match else None


def select_torch_build(requested: str) -> Tuple[str, Optional[Tuple[int, int]]]:
    version = nvcc_version()
    if requested != "auto":
        return requested, version
    if version is None:
        return "cpu", None
    if version >= (12, 8):
        return "cu128", version
    if version >= (12, 6):
        return "cu126", version
    if version >= (11, 8):
        return "cu118", version
    print(f"CUDA Toolkit {version[0]}.{version[1]} has no pinned PyTorch 2.7.1 wheel; using CPU.")
    return "cpu", version


def environment_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def parse_scalar(text: str) -> Any:
    value = text.strip()
    lowered = value.lower()
    if lowered in ("null", "none", "~"):
        return None
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value.strip("'\"")


def load_setup_config(path: Path) -> Dict[str, Any]:
    """Read the installer's flat YAML file without requiring PyYAML."""
    data = {}  # type: Dict[str, Any]
    for number, source_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = source_line.strip()
        if not line or line.startswith("#") or line == "---":
            continue
        if source_line[:1].isspace() or line.startswith("-") or ":" not in line:
            raise ValueError(f"{path}:{number}: setup config must use flat key: value entries")
        key, value = line.split(":", 1)
        data[key.strip().replace("-", "_")] = parse_scalar(value.split(" #", 1)[0])
    return data


def setup_defaults(config_path: Optional[Path]) -> Dict[str, Any]:
    defaults = {
        "venv": Path(".venv"),
        "python": "3.11",
        "cuda": "auto",
        "recreate": False,
        "dev": False,
    }  # type: Dict[str, Any]
    if config_path is None:
        return defaults

    loaded = load_setup_config(config_path.expanduser().resolve())
    unknown = sorted(set(loaded) - set(defaults))
    if unknown:
        raise ValueError("Unknown setup config keys: " + ", ".join(unknown))
    defaults.update(loaded)
    defaults["venv"] = Path(str(defaults["venv"]))
    defaults["python"] = str(defaults["python"])
    if defaults["cuda"] not in ("auto",) + SUPPORTED_BUILDS:
        raise ValueError("cuda must be one of: auto, " + ", ".join(SUPPORTED_BUILDS))
    for name in ("recreate", "dev"):
        if not isinstance(defaults[name], bool):
            raise ValueError(f"{name} must be true or false")
    return defaults


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    known, _ = bootstrap.parse_known_args()
    try:
        defaults = setup_defaults(known.config)
    except (OSError, ValueError) as exc:
        bootstrap.error(str(exc))

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="Flat YAML setup configuration")
    parser.add_argument("--venv", type=Path, default=defaults["venv"], help="Virtual environment directory")
    parser.add_argument("--python", default=defaults["python"], help="Python version passed to uv venv")
    parser.add_argument(
        "--cuda", choices=("auto",) + SUPPORTED_BUILDS, default=defaults["cuda"],
        help="PyTorch wheel build; auto detects nvcc",
    )
    parser.add_argument("--recreate", dest="recreate", action="store_true", help="Delete and rebuild the selected virtual environment")
    parser.add_argument("--no-recreate", dest="recreate", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--dev", dest="dev", action="store_true", help="Also install pytest")
    parser.add_argument("--no-dev", dest="dev", action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(recreate=defaults["recreate"], dev=defaults["dev"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parent
    venv = (root / args.venv).resolve() if not args.venv.is_absolute() else args.venv.resolve()
    uv = find_uv()
    build, detected = select_torch_build(args.cuda)
    detected_text = "not found" if detected is None else f"{detected[0]}.{detected[1]}"
    print(f"Platform: {sys.platform}; nvcc CUDA: {detected_text}; PyTorch build: {build}")

    if args.recreate and venv.exists():
        shutil.rmtree(venv)
    if not environment_python(venv).exists():
        run([uv, "python", "install", args.python], cwd=root)
        run([uv, "venv", "--python", args.python, str(venv)], cwd=root)

    python = environment_python(venv)
    index_url = (
        "https://download.pytorch.org/whl/cpu"
        if build == "cpu"
        else f"https://download.pytorch.org/whl/{build}"
    )
    run([
        uv, "pip", "install", "--python", str(python),
        f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}",
        f"torchaudio=={TORCHAUDIO_VERSION}", "--index-url", index_url,
    ], cwd=root)
    run([uv, "pip", "install", "--python", str(python), "-r", str(root / "requirements.txt")], cwd=root)
    try:
        run([uv, "pip", "install", "--python", str(python), "depthai==3.1.0"], cwd=root)
    except subprocess.CalledProcessError:
        print("uv rejected the DepthAI wheel metadata; using pip for this package only.")
        run([uv, "pip", "install", "--python", str(python), "pip"], cwd=root)
        run([str(python), "-m", "pip", "install", "--force-reinstall", "--no-deps", "depthai==3.1.0"], cwd=root)
    if args.dev:
        run([uv, "pip", "install", "--python", str(python), "pytest"], cwd=root)
    run([
        str(python), "-c",
        "import torch, torchvision, torchaudio; "
        "print('torch=', torch.__version__); "
        "print('torchvision=', torchvision.__version__); "
        "print('torchaudio=', torchaudio.__version__); "
        "print('torch CUDA build=', torch.version.cuda); "
        "print('CUDA available=', torch.cuda.is_available())",
    ], cwd=root)
    activate = venv / ("Scripts/Activate.ps1" if os.name == "nt" else "bin/activate")
    print(f"Environment ready. Activate it with: {activate}")


if __name__ == "__main__":
    main()
