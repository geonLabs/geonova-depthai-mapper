#!/usr/bin/env python3
"""Export one self-contained component, excluding models, data and virtualenvs."""
from __future__ import annotations
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", ".pytest_cache")


def export_component(component, destination, init_git=False):
    destination = Path(destination).expanduser().absolute()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", destination.name):
        raise ValueError("Destination folder must use the Controller lowercase pipeline ID convention")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Export destination already exists: {destination}")
    if ROOT.resolve() == destination.resolve() or destination.resolve() in ROOT.resolve().parents:
        raise ValueError("Destination must not contain the source repository")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    destination.mkdir(parents=True)
    component_root = ROOT / component
    shutil.copytree(component_root / "src", destination / "src", ignore=IGNORE)
    shutil.copytree(ROOT / "shared/src/geonova_common", destination / "src/geonova_common", ignore=IGNORE)
    for name in ("requirements.lock", "README.md"):
        shutil.copy2(component_root / name, destination / name)
    (destination / "tools").mkdir()
    shutil.copy2(ROOT / "tools/install_component.py", destination / "tools/install_component.py")
    (destination / ".gitignore").write_text(".venv/\nresults/\n__pycache__/\n*.pyc\nenvironment.json\nenvironment.freeze.txt\nmodel/\n", encoding="utf-8")
    if component == "capture":
        for name in ("main.py", "config.yaml", "install.sh", "install.ps1"):
            shutil.copy2(ROOT / name, destination / name)
        (destination / "results").mkdir()
    else:
        for name in ("main.py", "install.sh", "requirements-yolo.txt"):
            shutil.copy2(component_root / name, destination / name)
        shutil.copytree(component_root / "configs", destination / "configs", ignore=IGNORE)
        # Optional offline LiDAR/ROS tools travel with the server component.
        shutil.copytree(ROOT / "tools/lidar", destination / "tools/lidar", ignore=IGNORE)
    (destination / "source_revision.json").write_text(json.dumps({
        "component": component,
        "source_repository": "https://github.com/geonLabs/geonova-depthai-mapper",
        "source_commit": revision,
        "source_worktree_dirty": dirty,
        "contract": "JetsonControllerApp/docs/EXTERNAL_PIPELINE_CONTRACT.md",
    }, indent=2) + "\n", encoding="utf-8")
    if init_git:
        subprocess.run(["git", "init", "-b", "main", str(destination)], check=True)
        subprocess.run(["git", "add", "."], cwd=destination, check=True)
        subprocess.run(["git", "-c", "user.name=Geonova Export", "-c", "user.email=export@localhost",
                        "commit", "-m", f"Export {component} from {revision}"], cwd=destination, check=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("capture", "postprocess"))
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--init-git", action="store_true", help="Create a local Git worktree for Controller registration/revision tracking")
    args = parser.parse_args(argv)
    print(export_component(args.component, args.destination, args.init_git))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
