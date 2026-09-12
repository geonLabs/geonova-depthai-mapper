#!/usr/bin/env python3
"""Create a local environment for one component without installing the other."""
from __future__ import annotations
import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command, capture=False):
    print("+ " + " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def check_dependencies(python, component):
    """Do not conflate DepthAI 3.1.0's compressed-wheel tag with dependency errors.

    Its downloaded multi-CPython wheel has a single cp311 tag inside WHEEL.
    All actual imports must already have succeeded before calling this helper.
    """
    checked = subprocess.run([str(python), "-m", "pip", "check"], text=True, capture_output=True)
    lines = [line.strip() for line in checked.stdout.splitlines() if line.strip()]
    if checked.returncode and component == "capture" and lines == ["depthai 3.1.0 is not supported on this platform"]:
        note = "DepthAI 3.1.0 wheel metadata tag mismatch; native SDK import passed. Verify sensors on the target Jetson."
        print(note, flush=True)
        return [note]
    if checked.returncode:
        raise RuntimeError("Dependency validation failed: " + checked.stdout + checked.stderr)
    print(checked.stdout.strip(), flush=True)
    return []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("capture", "postprocess"), default="capture")
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--system-site-packages", action="store_true",
                        help="Opt in to JetPack/system Python packages; default is isolated")
    parser.add_argument("--with-yolo", action="store_true", help="Server only; requires installed torch and torchvision")
    args = parser.parse_args(argv)
    if args.component == "capture" and args.with_yolo:
        parser.error("YOLO belongs to the postprocess environment")
    minimum = (3, 8) if args.component == "capture" else (3, 11)
    if not minimum <= sys.version_info[:2] <= (3, 12):
        parser.error(f"{args.component} requires Python {minimum[0]}.{minimum[1]} through 3.12")
    component_root = ROOT / args.component if (ROOT / args.component / "requirements.lock").is_file() else ROOT
    requirements = component_root / "requirements.lock"
    venv = args.venv or ((ROOT if args.component == "capture" else component_root) / ".venv")
    venv = venv.expanduser().absolute()
    if venv.is_symlink():
        parser.error(".venv must be a real directory, not a symlink")
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not venv.exists():
        command = [sys.executable, "-m", "venv", str(venv)]
        if args.system_site_packages:
            command.append("--system-site-packages")
        run(command)
    if not python.is_file():
        parser.error(f"Incomplete environment: {python}")
    version = run([python, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"], True).strip()
    if version != "%d.%d" % sys.version_info[:2]:
        parser.error("Existing environment Python differs from installer; create a new local venv")
    run([python, "-m", "pip", "install", "-r", requirements])
    if args.with_yolo:
        run([python, "-c", "import torch, torchvision; print(torch.__version__, torch.version.cuda)"])
        # Keep the selected CUDA wheels and numerical pins. Ultralytics requires
        # the opencv-python distribution; replace the headless provider instead
        # of leaving two packages that own the same cv2 files.
        pins = run([python, "-c", "import importlib.metadata as m; print('\\n'.join(n+'=='+m.version(n) for n in ('torch','torchvision')))"], True)
        with tempfile.TemporaryDirectory(prefix="geonova-yolo-") as temporary:
            constraints = Path(temporary) / "constraints.txt"
            numerical = "\n".join(line for line in requirements.read_text().splitlines()
                                    if not line.startswith("opencv-python-headless"))
            constraints.write_text(numerical + "\n" + pins, encoding="utf-8")
            run([python, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless"])
            run([python, "-m", "pip", "install", "-c", constraints,
                 "-r", component_root / "requirements-yolo.txt"])
    imports = "cv2, depthai, serial, yaml, numpy" if args.component == "capture" else "cv2, yaml, numpy, pyproj, shapefile, scipy"
    run([python, "-c", "import " + imports])
    validation_notes = check_dependencies(python, args.component)
    freeze = run([python, "-m", "pip", "freeze", "--all"], True)
    (component_root / "environment.freeze.txt").write_text(freeze, encoding="utf-8")
    versions = run([python, "-c", "import json, importlib.metadata as m; print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()}))"], True)
    release = Path("/etc/nv_tegra_release")
    (component_root / "environment.json").write_text(json.dumps({
        "component": args.component, "python": version, "machine": platform.machine(),
        "platform": platform.platform(), "venv": str(venv),
        "l4t": release.read_text().strip() if release.is_file() else None,
        "packages": json.loads(versions),
        "validation_notes": validation_notes,
        "cuda_tensorrt": "capture does not require either; record server model environment separately",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"{args.component} environment ready: {python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
