#!/usr/bin/env python3
"""Create a cross-platform uv environment and install the matching PyTorch build."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from geonova_depthai.config_cli import parse_args_with_yaml


TORCH_VERSION = "2.7.1"
TORCHVISION_VERSION = "0.22.1"
TORCHAUDIO_VERSION = "2.7.1"
SUPPORTED_BUILDS = ("cpu", "cu118", "cu126", "cu128")


def run(command: list[str], cwd: Path | None = None, capture: bool = False) -> str:
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


def find_uv() -> str:
    executable = shutil.which("uv")
    if executable:
        return executable
    print("uv was not found; installing it with the current Python.", flush=True)
    run([sys.executable, "-m", "pip", "install", "uv"])
    executable = shutil.which("uv")
    if executable:
        return executable
    scripts_dir = Path(sysconfig.get_path("scripts"))
    candidate = scripts_dir / ("uv.exe" if os.name == "nt" else "uv")
    if candidate.exists():
        return str(candidate)
    raise RuntimeError("uv installation completed but the uv executable was not found.")


def nvcc_version() -> tuple[int, int] | None:
    executable = shutil.which("nvcc")
    if not executable:
        return None
    output = run([executable, "--version"], capture=True)
    print(output, end="")
    match = re.search(r"release\s+(\d+)\.(\d+)", output)
    return (int(match.group(1)), int(match.group(2))) if match else None


def select_torch_build(requested: str) -> tuple[str, tuple[int, int] | None]:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--venv", type=Path, default=Path(".venv"), help="Virtual environment directory")
    parser.add_argument("--python", default="3.11", help="Python version passed to uv venv")
    parser.add_argument("--cuda", choices=("auto",) + SUPPORTED_BUILDS, default="auto", help="PyTorch wheel build; auto detects nvcc")
    parser.add_argument("--recreate", action="store_true", help="Delete and rebuild the selected virtual environment")
    parser.add_argument("--dev", action="store_true", help="Also install pytest")
    args = parse_args_with_yaml(parser)

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
