#!/usr/bin/env python3
"""Create a portable environment for Jetson, Ubuntu, and Windows hosts.

The bootstrap only uses the Python 3.8 standard library. Desktop hosts get a uv
managed Python 3.11 environment and official PyTorch wheels. Jetson hosts keep the
JetPack Python ABI and NVIDIA CUDA-enabled PyTorch instead of replacing it with an
incompatible desktop wheel.
"""

from __future__ import annotations

import argparse
import ast
import html
import os
import platform
import re
import site
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TORCH_VERSION = "2.7.1"
TORCHVISION_VERSION = "0.22.1"
TORCHAUDIO_VERSION = "2.7.1"
SUPPORTED_BUILDS = ("cpu", "cu118", "cu126", "cu128")
SUPPORTED_PLATFORMS = ("auto", "desktop", "jetson")
TORCHVISION_BY_TORCH = {
    "1.13": "0.14.1",
    "2.0": "0.15.1",
    "2.1": "0.16.0",
    "2.2": "0.17.0",
    "2.3": "0.18.0",
    "2.4": "0.19.0",
    "2.5": "0.20.0",
    "2.6": "0.21.0",
    "2.7": "0.22.1",
    "2.8": "0.23.0",
}
JETSON_TORCH_FALLBACKS = {
    (35, 3, 8): (
        "https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/"
        "torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
    ),
}


def run(
    command: List[str],
    cwd: Optional[Path] = None,
    capture: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> str:
    print("+", subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        env=env,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return completed.stdout or ""


def uv_candidates() -> List[Path]:
    """Return common uv locations, including user installs not present in PATH."""
    executable_name = "uv.exe" if os.name == "nt" else "uv"
    candidates = [Path(sysconfig.get_path("scripts")) / executable_name]
    candidates.append(
        Path(site.getuserbase()) / ("Scripts" if os.name == "nt" else "bin") / executable_name
    )
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


def command_cuda_version(command: List[str], pattern: str) -> Optional[Tuple[int, int]]:
    try:
        output = run(command, capture=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(pattern, output, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def cuda_version() -> Optional[Tuple[int, int]]:
    """Detect a desktop CUDA runtime conservatively from nvcc or nvidia-smi."""
    nvcc = shutil.which("nvcc")
    if nvcc:
        version = command_cuda_version([nvcc, "--version"], r"release\s+(\d+)\.(\d+)")
        if version:
            return version
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        return command_cuda_version([nvidia_smi], r"CUDA\s+Version\s*:\s*(\d+)\.(\d+)")
    return None


def select_torch_build(requested: str) -> Tuple[str, Optional[Tuple[int, int]]]:
    version = cuda_version()
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
    print(f"CUDA {version[0]}.{version[1]} has no pinned PyTorch 2.7.1 wheel; using CPU.")
    return "cpu", version


def jetson_release() -> Optional[Tuple[int, int]]:
    release_file = Path("/etc/nv_tegra_release")
    if platform.machine().lower() not in ("aarch64", "arm64") or not release_file.is_file():
        return None
    match = re.search(
        r"#\s*R(\d+).*?REVISION:\s*(\d+)",
        release_file.read_text(encoding="utf-8", errors="replace"),
    )
    return (int(match.group(1)), int(match.group(2))) if match else None


def resolve_platform(requested: str) -> str:
    if requested == "auto":
        return "jetson" if jetson_release() is not None else "desktop"
    if requested == "jetson" and jetson_release() is None:
        raise RuntimeError("--platform jetson requires an aarch64 NVIDIA L4T host.")
    return requested


def resolve_python(requested: str, target_platform: str) -> str:
    if requested != "auto":
        return requested
    return sys.executable if target_platform == "jetson" else "3.11"


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
        "python": "auto",
        "platform": "auto",
        "cuda": "auto",
        "jetson_torch": "auto",
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
    for name in ("python", "platform", "cuda", "jetson_torch"):
        defaults[name] = str(defaults[name])
    if defaults["platform"] not in SUPPORTED_PLATFORMS:
        raise ValueError("platform must be one of: " + ", ".join(SUPPORTED_PLATFORMS))
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
    parser.add_argument("--python", default=defaults["python"], help="Python version/path, or auto")
    parser.add_argument("--platform", choices=SUPPORTED_PLATFORMS, default=defaults["platform"], help="Target platform profile")
    parser.add_argument(
        "--cuda", choices=("auto",) + SUPPORTED_BUILDS, default=defaults["cuda"],
        help="Desktop PyTorch wheel build; auto detects nvcc or nvidia-smi",
    )
    parser.add_argument(
        "--jetson-torch", default=defaults["jetson_torch"],
        help="Jetson PyTorch source: auto, system, wheel path, or NVIDIA wheel URL",
    )
    parser.add_argument("--recreate", dest="recreate", action="store_true", help="Delete and rebuild the selected virtual environment")
    parser.add_argument("--no-recreate", dest="recreate", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--dev", dest="dev", action="store_true", help="Also install pytest")
    parser.add_argument("--no-dev", dest="dev", action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(recreate=defaults["recreate"], dev=defaults["dev"])
    return parser.parse_args()


def create_environment(uv: str, root: Path, venv: Path, python: str, target_platform: str) -> None:
    if environment_python(venv).exists():
        return
    if target_platform == "desktop" and re.fullmatch(r"\d+(?:\.\d+)*", python):
        run([uv, "python", "install", python], cwd=root)
    command = [uv, "venv", "--python", python]
    if target_platform == "jetson":
        command.append("--system-site-packages")
    command.append(str(venv))
    run(command, cwd=root)


def validate_environment_profile(venv: Path, python: Path, target_platform: str) -> None:
    if target_platform != "jetson":
        return
    output = run([
        str(python), "-c",
        "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')",
    ], capture=True).strip().splitlines()[-1]
    expected = f"{sys.version_info[0]}.{sys.version_info[1]}"
    config = (venv / "pyvenv.cfg").read_text(encoding="utf-8", errors="replace")
    has_system_packages = bool(
        re.search(r"^include-system-site-packages\s*=\s*true\s*$", config, re.MULTILINE)
    )
    if output != expected or not has_system_packages:
        raise RuntimeError(
            "The existing Jetson environment has the wrong Python/profile "
            f"(expected Python {expected} with system packages). "
            "Rerun the installer with --recreate."
        )


def python_package_version(python: Path, package: str) -> Optional[str]:
    code = "import importlib.metadata as m; " + f"print(m.version({package!r}))"
    try:
        output = run([str(python), "-c", code], capture=True)
    except subprocess.CalledProcessError:
        return None
    return output.strip().splitlines()[-1]


def torch_details(python: Path) -> Tuple[str, Optional[str], bool]:
    code = (
        "import torch; print(torch.__version__); print(torch.version.cuda or 'none'); "
        "print('yes' if torch.cuda.is_available() else 'no')"
    )
    output = run([str(python), "-c", code], capture=True).strip().splitlines()
    return output[-3], None if output[-2] == "none" else output[-2], output[-1] == "yes"


def torch_location(python: Path) -> Path:
    output = run(
        [str(python), "-c", "import torch; print(torch.__file__)"], capture=True
    ).strip().splitlines()
    return Path(output[-1]).resolve()


def jetson_index_candidates(release: Tuple[int, int]) -> List[str]:
    major, revision = release
    if major == 35:
        versions = ("v511", "v51")
    elif major == 36 and revision >= 4:
        versions = ("v62", "v61", "v60")
    elif major == 36:
        versions = ("v60",)
    else:
        versions = ()
    return [
        f"https://developer.download.nvidia.com/compute/redist/jp/{version}/pytorch/"
        for version in versions
    ]


def discover_jetson_torch_wheel(release: Tuple[int, int], python: Path) -> str:
    version_output = run(
        [str(python), "-c", "import sys; print(f'cp{sys.version_info[0]}{sys.version_info[1]}')"],
        capture=True,
    ).strip()
    tag = version_output.splitlines()[-1]
    for index_url in jetson_index_candidates(release):
        try:
            with urllib.request.urlopen(index_url, timeout=20) as response:
                listing = response.read().decode("utf-8", errors="replace")
        except OSError:
            continue
        links = [html.unescape(link) for link in re.findall(r'href=["\']([^"\']+\.whl)["\']', listing)]
        matches = [
            link for link in links
            if "torch-" in link and f"-{tag}-{tag}-linux_aarch64.whl" in urllib.parse.unquote(link)
        ]
        if matches:
            return urllib.parse.urljoin(index_url, sorted(matches)[-1])

    fallback = JETSON_TORCH_FALLBACKS.get((release[0], sys.version_info[0], sys.version_info[1]))
    if fallback:
        return fallback
    raise RuntimeError(
        "No NVIDIA PyTorch wheel was found automatically for this L4T/Python combination. "
        "Install NVIDIA PyTorch first or pass --jetson-torch <wheel-or-url>."
    )


def ensure_jetson_torch(
    uv: str, root: Path, python: Path, requested: str, release: Tuple[int, int]
) -> Tuple[str, Optional[str], bool]:
    try:
        details = torch_details(python)
        if details[1] and details[2]:
            print(f"Reusing NVIDIA PyTorch {details[0]} (CUDA {details[1]}).")
            return details
        location = torch_location(python)
        venv = python.parent.parent.resolve()
        if venv in location.parents:
            print(f"Removing incompatible Jetson-local PyTorch {details[0]} from {location}.")
            run([uv, "pip", "uninstall", "--python", str(python), "torch"], cwd=root)
            details = torch_details(python)
            if details[1] and details[2]:
                print(f"Reusing NVIDIA PyTorch {details[0]} (CUDA {details[1]}).")
                return details
    except subprocess.CalledProcessError:
        if requested == "system":
            raise RuntimeError(
                "NVIDIA PyTorch is not visible from the Jetson environment. "
                "Use --jetson-torch auto or provide an NVIDIA wheel URL."
            )

    source = discover_jetson_torch_wheel(release, python) if requested == "auto" else requested
    run([uv, "pip", "install", "--python", str(python), "numpy<2"], cwd=root)
    run([uv, "pip", "install", "--python", str(python), source], cwd=root)
    return torch_details(python)


def torchvision_for_torch(torch_version: str) -> str:
    match = re.match(r"(\d+\.\d+)", torch_version)
    if not match or match.group(1) not in TORCHVISION_BY_TORCH:
        raise RuntimeError(f"No torchvision mapping is configured for PyTorch {torch_version}.")
    return TORCHVISION_BY_TORCH[match.group(1)]


def ensure_jetson_torchvision(
    uv: str, root: Path, python: Path, torch_version: str, torch_cuda: Optional[str]
) -> None:
    expected = torchvision_for_torch(torch_version)
    installed = python_package_version(python, "torchvision")
    if installed and installed.startswith(expected):
        try:
            run([
                str(python), "-c",
                "import torchvision; assert torchvision.extension._has_ops()",
            ], capture=True)
            print(f"Reusing torchvision {installed}.")
            return
        except subprocess.CalledProcessError:
            print(f"Rebuilding unusable torchvision {installed}.")

    if not shutil.which("g++"):
        raise RuntimeError("Building Jetson torchvision requires g++. Install build-essential and rerun.")
    print(
        f"Building torchvision {expected} for NVIDIA PyTorch {torch_version}. "
        "The first Jetson setup can take several minutes.",
        flush=True,
    )
    run([
        uv, "pip", "install", "--python", str(python),
        "numpy<2", "pillow", "setuptools", "wheel", "ninja", "importlib-metadata>=6,<9",
    ], cwd=root)
    build_env = os.environ.copy()
    build_env.setdefault("FORCE_CUDA", "1" if torch_cuda else "0")
    build_env.setdefault("MAX_JOBS", "2")
    build_env["BUILD_VERSION"] = expected
    if torch_cuda:
        cuda_home = Path("/usr/local") / ("cuda-" + torch_cuda)
        if cuda_home.is_dir():
            build_env["CUDA_HOME"] = str(cuda_home)
            build_env["PATH"] = str(cuda_home / "bin") + os.pathsep + build_env.get("PATH", "")
    source = f"https://github.com/pytorch/vision/archive/refs/tags/v{expected}.tar.gz"
    run([
        uv, "pip", "install", "--python", str(python),
        "--no-build-isolation", "--no-deps", source,
    ], cwd=root, env=build_env)


def install_depthai(uv: str, root: Path, python: Path) -> None:
    try:
        run([uv, "pip", "install", "--python", str(python), "depthai==3.1.0"], cwd=root)
    except subprocess.CalledProcessError:
        print("uv rejected the DepthAI wheel metadata; using pip for this package only.")
        run([uv, "pip", "install", "--python", str(python), "pip"], cwd=root)
        run([
            str(python), "-m", "pip", "install", "--force-reinstall", "--no-deps", "depthai==3.1.0"
        ], cwd=root)


def install_desktop(uv: str, root: Path, python: Path, requested_build: str) -> None:
    build, detected = select_torch_build(requested_build)
    detected_text = "not found" if detected is None else f"{detected[0]}.{detected[1]}"
    print(f"Platform: desktop; detected CUDA: {detected_text}; PyTorch build: {build}")
    index_url = "https://download.pytorch.org/whl/cpu" if build == "cpu" else f"https://download.pytorch.org/whl/{build}"
    run([
        uv, "pip", "install", "--python", str(python),
        f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}",
        f"torchaudio=={TORCHAUDIO_VERSION}", "--index-url", index_url,
    ], cwd=root)
    run([uv, "pip", "install", "--python", str(python), "-r", str(root / "requirements.txt")], cwd=root)


def install_jetson(
    uv: str, root: Path, python: Path, requested_torch: str, release: Tuple[int, int]
) -> None:
    print(f"Platform: Jetson L4T R{release[0]}.{release[1]}; Python: {python}")
    torch_version, torch_cuda, cuda_available = ensure_jetson_torch(
        uv, root, python, requested_torch, release
    )
    if not torch_cuda or not cuda_available:
        raise RuntimeError(
            f"Jetson PyTorch {torch_version} does not expose CUDA. Install the matching NVIDIA wheel."
        )
    ensure_jetson_torchvision(uv, root, python, torch_version, torch_cuda)
    run([
        uv, "pip", "install", "--python", str(python),
        "-r", str(root / "requirements-jetson.txt"),
    ], cwd=root)
    run([
        uv, "pip", "install", "--python", str(python), "--no-deps",
        "ultralytics-thop==2.1.6", "ultralytics==8.4.75",
    ], cwd=root)


def verify_environment(python: Path, target_platform: str) -> None:
    code = (
        "import cv2, depthai, numpy, torch, torchvision, ultralytics; "
        "print('python=', __import__('sys').version.split()[0]); "
        "print('torch=', torch.__version__); "
        "print('torchvision=', torchvision.__version__); "
        "print('torch CUDA build=', torch.version.cuda); "
        "print('CUDA available=', torch.cuda.is_available()); "
        "print('depthai=', depthai.__version__); "
        "print('ultralytics=', ultralytics.__version__)"
    )
    run([str(python), "-c", code])
    if target_platform == "jetson":
        run([
            str(python), "-c",
            "import torch, torchvision; "
            "boxes=torch.tensor([[0.,0.,1.,1.]]); scores=torch.tensor([1.]); "
            "print('torchvision NMS=', torchvision.ops.nms(boxes, scores, 0.5).tolist())",
        ])


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    venv = (root / args.venv).resolve() if not args.venv.is_absolute() else args.venv.resolve()
    if venv.is_symlink():
        raise RuntimeError(f"Refusing symbolic-link virtualenv: {venv}")

    target_platform = resolve_platform(args.platform)
    python_selector = resolve_python(args.python, target_platform)
    uv = find_uv()

    if args.recreate and venv.exists():
        shutil.rmtree(venv)
    create_environment(uv, root, venv, python_selector, target_platform)
    python = environment_python(venv)
    validate_environment_profile(venv, python, target_platform)

    if target_platform == "jetson":
        release = jetson_release()
        if release is None:
            raise RuntimeError("Unable to read the Jetson L4T release.")
        install_jetson(uv, root, python, args.jetson_torch, release)
    else:
        install_desktop(uv, root, python, args.cuda)

    install_depthai(uv, root, python)
    if args.dev:
        run([uv, "pip", "install", "--python", str(python), "pytest"], cwd=root)
    verify_environment(python, target_platform)

    activate = venv / ("Scripts/Activate.ps1" if os.name == "nt" else "bin/activate")
    print(f"Environment ready. Activate it with: {activate}")


if __name__ == "__main__":
    main()
