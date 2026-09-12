"""Versioned, file-based boundary between acquisition and offline workers."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "geonova.raw-run/v1"
MANIFEST = "capture_manifest.json"
REQUIRED = ("metadata.json", "rgb_events.csv", "depth_events.csv", "imu_events.csv")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_file(root, name):
    """Only accept ordinary relative files inside the transferred run."""
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Invalid dataset relative path: {name!r}")
    candidate = root / path
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Dataset symlink is not portable: {name}")
    if not candidate.is_file():
        raise ValueError(f"Dataset file missing: {name}")
    return candidate


def write_capture_manifest(root, state, counts=None, error=None):
    root = Path(root)
    files = []
    if state == "complete":
        for name in REQUIRED:
            relative_file(root, name)
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Unexpected symlink in capture: {path}")
            if not path.is_file() or path.name == MANIFEST or path.name.startswith("."):
                continue
            item = {"path": path.relative_to(root).as_posix(), "size": path.stat().st_size}
            # Hash manifests/calibration; image sizes are checked without a second
            # full read of potentially hundreds of gigabytes on the Jetson.
            if path.suffix in (".csv", ".json"):
                item["sha256"] = sha256_file(path)
            files.append(item)
    payload = {
        "schema": SCHEMA,
        "state": state,
        "run_id": root.name,
        "updated_at_utc": utc_now(),
        "pipeline_id": os.environ.get("JETSON_PIPELINE_ID"),
        "pipeline_release": os.environ.get("JETSON_PIPELINE_RELEASE"),
        "raw_format": "raw_events_v1",
        "counts": dict(counts or {}),
        "files": files,
        # Exception text may contain credentials supplied to third-party SDKs.
        "error_type": type(error).__name__ if error is not None else None,
    }
    atomic_json(root / MANIFEST, payload)
    return payload


def validate_capture(root, allow_legacy=False):
    root = Path(root).expanduser().resolve(strict=True)
    for name in REQUIRED:
        relative_file(root, name)
    manifest_path = root / MANIFEST
    if not manifest_path.exists():
        if not allow_legacy:
            raise ValueError("Missing capture_manifest.json; use --allow-legacy only for stopped historical runs")
        payload = None
    else:
        relative_file(root, MANIFEST)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema") != SCHEMA or payload.get("state") != "complete":
            raise ValueError("Capture is not complete or its schema is unsupported")
        files = payload.get("files")
        if not isinstance(files, list) or not set(REQUIRED).issubset({f.get("path") for f in files}):
            raise ValueError("Capture inventory is incomplete")
        for item in files:
            path = relative_file(root, item["path"])
            if path.stat().st_size != item.get("size"):
                raise ValueError(f"Transferred file size mismatch: {item['path']}")
            if item.get("sha256") and sha256_file(path) != item["sha256"]:
                raise ValueError(f"Transferred file checksum mismatch: {item['path']}")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != "raw_events_v1":
        raise ValueError("Expected a raw_events_v1 dataset")
    # Validate even legacy CSV references before any copying or model loading.
    import csv
    for name in ("rgb_events.csv", "depth_events.csv", "confidence_events.csv"):
        path = root / name
        if path.exists():
            relative_file(root, name)
            with path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    relative_file(root, row.get("file", ""))
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Dataset symlink is not portable: {path}")
    return payload


def separate_output(source, output):
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Processing output must be separate from the input dataset")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Processing output must be new or empty: {output}")
    return output
