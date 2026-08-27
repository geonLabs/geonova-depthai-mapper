#!/usr/bin/env python3
"""Extract matched lidar PCD files and camera images from ROS1 bag files.

Run without arguments to load config.yaml placed beside this script:

    python convert_lidar_bag_to_pcd.py

An optional config path can be passed when needed:

    python convert_lidar_bag_to_pcd.py other_config.yaml
"""

from __future__ import annotations

import bisect
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from ruamel.yaml import YAML


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.yaml"
DEFAULT_LIDAR_TOPIC = "/lidar0/velodyne_points"
DEFAULT_CAMERA_TOPIC = "/roof_clpe_ros/roof_cam_1/image_raw"
NS_PER_SEC = 1_000_000_000
NS_PER_MS = 1_000_000

# sensor_msgs/PointField datatype values:
# https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/PointField.html
POINT_FIELD_TYPES: dict[int, tuple[str, int, str]] = {
    1: ("i1", 1, "I"),  # INT8
    2: ("u1", 1, "U"),  # UINT8
    3: ("i2", 2, "I"),  # INT16
    4: ("u2", 2, "U"),  # UINT16
    5: ("i4", 4, "I"),  # INT32
    6: ("u4", 4, "U"),  # UINT32
    7: ("f4", 4, "F"),  # FLOAT32
    8: ("f8", 8, "F"),  # FLOAT64
}


@dataclass(frozen=True)
class FieldSpec:
    """PointCloud2 field metadata mapped to one PCD field."""

    msg_name: str
    pcd_name: str
    offset: int
    datatype: int
    count: int
    size: int
    pcd_type: str


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from YAML."""

    config_path: Path
    bag_paths: list[Path]
    lidar_topic: str
    camera_topic: str
    output_root: Path
    pcd_subdir: Path
    image_subdir: Path
    group_by_bag: bool
    pcd_output_dir: Path
    image_output_dir: Path
    timestamp_source: str
    timezone: str
    save_all: bool
    max_pairs: int | None
    every_lidar: int
    match_tolerance_ns: int | None
    require_camera_match: bool
    pcd_fields: list[str] | None
    drop_nan: bool
    image_format: str
    jpeg_quality: int


@dataclass(frozen=True)
class PairPlan:
    """One lidar frame and its nearest camera frame to export."""

    base_name: str
    lidar_ns: int
    camera_ns: int
    diff_ns: int


def cli_config_path() -> Path:
    if len(sys.argv) == 1:
        return DEFAULT_CONFIG
    if len(sys.argv) == 2:
        return Path(sys.argv[1]).expanduser().resolve()

    program = Path(sys.argv[0]).name
    raise SystemExit(f"usage: {program} [config.yaml]")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    yaml = YAML(typ="safe")
    loaded = yaml.load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a YAML mapping: {path}")
    return loaded


def nested(config: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def path_list(value: Any) -> list[str | Path]:
    if value is None:
        return ["."]
    if isinstance(value, (str, Path)):
        return [value]
    if isinstance(value, list):
        return value
    raise ValueError("bags.paths must be a path string or a list of paths.")


def resolve_bag_paths(config: dict[str, Any], config_dir: Path) -> list[Path]:
    raw_paths = nested(config, ("bags", "paths"), config.get("bag_paths", ["."]))
    recursive = as_bool(nested(config, ("bags", "recursive"), False), default=False)
    resolved: list[Path] = []

    for raw_path in path_list(raw_paths):
        path = resolve_path(raw_path, config_dir)

        if path.is_dir():
            pattern = "**/*.bag" if recursive else "*.bag"
            resolved.extend(sorted(item.resolve() for item in path.glob(pattern) if item.is_file()))
        elif path.is_file():
            if path.suffix != ".bag":
                raise ValueError(f"Bag path is not a .bag file: {path}")
            resolved.append(path)
        else:
            raise FileNotFoundError(f"Bag path does not exist: {path}")

    unique_paths = list(dict.fromkeys(resolved))
    if not unique_paths:
        raise FileNotFoundError("No .bag files found from bags.paths.")
    return unique_paths


def read_settings(config_path: Path) -> Settings:
    config = load_yaml(config_path)
    config_dir = config_path.parent

    output_root = resolve_path(nested(config, ("output", "root_dir"), "./extracted"), config_dir)
    pcd_subdir = Path(str(nested(config, ("output", "pcd_dir"), "pcd"))).expanduser()
    image_subdir = Path(str(nested(config, ("output", "image_dir"), "images"))).expanduser()
    group_by_bag = as_bool(nested(config, ("output", "group_by_bag"), True), default=True)

    pcd_output_dir = pcd_subdir if pcd_subdir.is_absolute() else output_root / pcd_subdir
    image_output_dir = image_subdir if image_subdir.is_absolute() else output_root / image_subdir

    save_all = as_bool(nested(config, ("extract", "save_all"), False), default=False)
    max_pairs_raw = nested(config, ("extract", "max_pairs"), 1)
    max_pairs = None if save_all or max_pairs_raw is None else int(max_pairs_raw)

    every_lidar = int(nested(config, ("extract", "every_lidar"), nested(config, ("extract", "every"), 1)))
    if every_lidar < 1:
        raise ValueError("extract.every_lidar must be >= 1")
    if max_pairs is not None and max_pairs < 1:
        raise ValueError("extract.max_pairs must be >= 1 or null")

    tolerance_ms = nested(config, ("extract", "match_tolerance_ms"), 100)
    match_tolerance_ns = None if tolerance_ms is None else int(float(tolerance_ms) * NS_PER_MS)

    timestamp_source = str(nested(config, ("extract", "timestamp_source"), "bag")).strip().lower()
    if timestamp_source not in {"bag", "header"}:
        raise ValueError('extract.timestamp_source must be "bag" or "header"')

    image_format = str(nested(config, ("image", "format"), "png")).strip().lower()
    if image_format == "jpeg":
        image_format = "jpg"
    if image_format not in {"png", "jpg"}:
        raise ValueError('image.format must be "png", "jpg", or "jpeg"')

    pcd_fields = nested(config, ("pcd", "fields"), None)
    if pcd_fields is not None:
        if not isinstance(pcd_fields, list):
            raise ValueError("pcd.fields must be null or a list of field names.")
        pcd_fields = [str(field) for field in pcd_fields]

    jpeg_quality = int(nested(config, ("image", "jpeg_quality"), 95))
    jpeg_quality = max(1, min(100, jpeg_quality))

    return Settings(
        config_path=config_path,
        bag_paths=resolve_bag_paths(config, config_dir),
        lidar_topic=str(nested(config, ("topics", "lidar"), DEFAULT_LIDAR_TOPIC)),
        camera_topic=str(nested(config, ("topics", "camera"), DEFAULT_CAMERA_TOPIC)),
        output_root=output_root,
        pcd_subdir=pcd_subdir,
        image_subdir=image_subdir,
        group_by_bag=group_by_bag,
        pcd_output_dir=pcd_output_dir.resolve(),
        image_output_dir=image_output_dir.resolve(),
        timestamp_source=timestamp_source,
        timezone=str(nested(config, ("extract", "timezone"), "Asia/Seoul")),
        save_all=save_all,
        max_pairs=max_pairs,
        every_lidar=every_lidar,
        match_tolerance_ns=match_tolerance_ns,
        require_camera_match=as_bool(
            nested(config, ("extract", "require_camera_match"), True),
            default=True,
        ),
        pcd_fields=pcd_fields,
        drop_nan=as_bool(nested(config, ("pcd", "drop_nan"), False), default=False),
        image_format=image_format,
        jpeg_quality=jpeg_quality,
    )


def sanitize_field_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]", "_", name.strip())
    if not clean:
        clean = "field"
    if clean[0].isdigit():
        clean = f"field_{clean}"
    return clean


def sanitize_path_component(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]", "_", name.strip())
    clean = clean.strip("._")
    return clean or "bag"


def unique_name(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base

    index = 2
    while f"{base}_{index}" in used:
        index += 1
    name = f"{base}_{index}"
    used.add(name)
    return name


def grouped_output_dir(output_root: Path, bag_name: str, subdir: Path) -> Path:
    if subdir.is_absolute():
        return (subdir / bag_name).resolve()
    return (output_root / bag_name / subdir).resolve()


def per_bag_settings(settings: Settings) -> list[Settings]:
    used_names: set[str] = set()
    grouped: list[Settings] = []

    for bag_path in settings.bag_paths:
        bag_name = unique_name(sanitize_path_component(bag_path.stem), used_names)
        grouped.append(
            replace(
                settings,
                bag_paths=[bag_path],
                pcd_output_dir=grouped_output_dir(
                    settings.output_root,
                    bag_name,
                    settings.pcd_subdir,
                ),
                image_output_dir=grouped_output_dir(
                    settings.output_root,
                    bag_name,
                    settings.image_subdir,
                ),
            ),
        )

    return grouped


def build_field_specs(msg_fields: list[Any], selected_fields: list[str] | None) -> list[FieldSpec]:
    selected = set(selected_fields) if selected_fields else None
    used_pcd_names: set[str] = set()
    specs: list[FieldSpec] = []

    for field in sorted(msg_fields, key=lambda item: int(item.offset)):
        msg_name = str(field.name)
        if selected is not None and msg_name not in selected:
            continue

        datatype = int(field.datatype)
        if datatype not in POINT_FIELD_TYPES:
            raise ValueError(f"Unsupported PointField datatype {datatype} for field {msg_name!r}")

        _, size, pcd_type = POINT_FIELD_TYPES[datatype]
        count = int(field.count) if int(field.count) > 0 else 1
        pcd_name = unique_name(sanitize_field_name(msg_name), used_pcd_names)
        specs.append(
            FieldSpec(
                msg_name=msg_name,
                pcd_name=pcd_name,
                offset=int(field.offset),
                datatype=datatype,
                count=count,
                size=size,
                pcd_type=pcd_type,
            ),
        )

    if selected_fields:
        found = {spec.msg_name for spec in specs}
        missing = [name for name in selected_fields if name not in found]
        if missing:
            raise ValueError(f"Selected field(s) not found in cloud: {', '.join(missing)}")

    if not specs:
        raise ValueError("No PointCloud2 fields selected for export.")

    return specs


def numpy_dtype(datatype: int, *, is_bigendian: bool) -> np.dtype:
    code, size, _ = POINT_FIELD_TYPES[datatype]
    if size == 1:
        return np.dtype(code)
    endian = ">" if is_bigendian else "<"
    return np.dtype(endian + code)


def output_numpy_dtype(datatype: int) -> np.dtype:
    code, size, _ = POINT_FIELD_TYPES[datatype]
    if size == 1:
        return np.dtype(code)
    return np.dtype("<" + code)


def make_input_dtype(specs: list[FieldSpec], point_step: int, *, is_bigendian: bool) -> np.dtype:
    names: list[str] = []
    formats: list[np.dtype | tuple[np.dtype, tuple[int, ...]]] = []
    offsets: list[int] = []

    for spec in specs:
        dtype = numpy_dtype(spec.datatype, is_bigendian=is_bigendian)
        field_format: np.dtype | tuple[np.dtype, tuple[int, ...]]
        field_format = dtype if spec.count == 1 else (dtype, (spec.count,))
        names.append(spec.pcd_name)
        formats.append(field_format)
        offsets.append(spec.offset)

    return np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": point_step,
        },
    )


def make_output_dtype(specs: list[FieldSpec]) -> np.dtype:
    fields: list[tuple[str, np.dtype] | tuple[str, np.dtype, tuple[int, ...]]] = []

    for spec in specs:
        dtype = output_numpy_dtype(spec.datatype)
        if spec.count == 1:
            fields.append((spec.pcd_name, dtype))
        else:
            fields.append((spec.pcd_name, dtype, (spec.count,)))

    return np.dtype(fields)


def pointcloud_data_bytes(msg: Any) -> tuple[bytes, int, int, int]:
    width = int(msg.width)
    height = int(msg.height)
    point_step = int(msg.point_step)
    row_step = int(msg.row_step)

    if width <= 0 or height <= 0:
        raise ValueError("PointCloud2 message has empty width/height.")
    if point_step <= 0:
        raise ValueError("PointCloud2 message has invalid point_step.")

    raw = msg.data.tobytes() if isinstance(msg.data, np.ndarray) else bytes(msg.data)
    valid_row_bytes = width * point_step
    expected_bytes = row_step * height

    if len(raw) < expected_bytes:
        raise ValueError(
            f"PointCloud2 data is shorter than expected: {len(raw)} < {expected_bytes}",
        )

    if row_step == valid_row_bytes:
        return raw[: valid_row_bytes * height], width, height, point_step

    rows = []
    for row_index in range(height):
        start = row_index * row_step
        rows.append(raw[start : start + valid_row_bytes])
    return b"".join(rows), width, height, point_step


def cloud_to_structured_array(msg: Any, specs: list[FieldSpec]) -> tuple[np.ndarray, int, int]:
    data, width, height, point_step = pointcloud_data_bytes(msg)
    dtype = make_input_dtype(specs, point_step, is_bigendian=bool(msg.is_bigendian))
    point_count = width * height
    points = np.frombuffer(data, dtype=dtype, count=point_count)
    return points, width, height


def drop_invalid_xyz(points: np.ndarray, specs: list[FieldSpec]) -> np.ndarray:
    name_map = {spec.msg_name: spec.pcd_name for spec in specs}
    required = ("x", "y", "z")
    if not all(name in name_map for name in required):
        raise ValueError("pcd.drop_nan requires x, y, and z fields.")

    mask = np.ones(points.shape[0], dtype=bool)
    for axis in required:
        mask &= np.isfinite(points[name_map[axis]])
    return points[mask]


def write_binary_pcd(
    path: Path,
    points: np.ndarray,
    specs: list[FieldSpec],
    *,
    width: int,
    height: int,
) -> None:
    output_dtype = make_output_dtype(specs)
    output = np.empty(points.shape[0], dtype=output_dtype)

    for spec in specs:
        output[spec.pcd_name] = points[spec.pcd_name]

    point_count = int(output.shape[0])
    if width * height != point_count:
        width = point_count
        height = 1

    header = "\n".join(
        [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            "FIELDS " + " ".join(spec.pcd_name for spec in specs),
            "SIZE " + " ".join(str(spec.size) for spec in specs),
            "TYPE " + " ".join(spec.pcd_type for spec in specs),
            "COUNT " + " ".join(str(spec.count) for spec in specs),
            f"WIDTH {width}",
            f"HEIGHT {height}",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {point_count}",
            "DATA binary",
            "",
        ],
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(header.encode("ascii"))
        file.write(output.tobytes(order="C"))


def raw_image_bytes(msg: Any, bytes_per_pixel: int) -> tuple[bytes, int, int]:
    width = int(msg.width)
    height = int(msg.height)
    step = int(msg.step)

    if width <= 0 or height <= 0:
        raise ValueError("Image message has empty width/height.")

    row_bytes = width * bytes_per_pixel
    if step < row_bytes:
        raise ValueError(f"Image step is too small: {step} < {row_bytes}")

    raw = msg.data.tobytes() if isinstance(msg.data, np.ndarray) else bytes(msg.data)
    expected_bytes = step * height
    if len(raw) < expected_bytes:
        raise ValueError(f"Image data is shorter than expected: {len(raw)} < {expected_bytes}")

    if step == row_bytes:
        return raw[: row_bytes * height], width, height

    rows = []
    for row_index in range(height):
        start = row_index * step
        rows.append(raw[start : start + row_bytes])
    return b"".join(rows), width, height


def image_msg_to_pillow(msg: Any) -> Image.Image:
    encoding = str(msg.encoding).strip().lower()
    is_bigendian = bool(msg.is_bigendian)

    if encoding in {"rgb8", "bgr8", "8uc3"}:
        data, width, height = raw_image_bytes(msg, 3)
        array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        if encoding == "bgr8":
            array = array[:, :, ::-1]
        return Image.fromarray(array, mode="RGB")

    if encoding in {"rgba8", "bgra8", "8uc4"}:
        data, width, height = raw_image_bytes(msg, 4)
        array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
        if encoding == "bgra8":
            array = array[:, :, [2, 1, 0, 3]]
        return Image.fromarray(array, mode="RGBA")

    if encoding in {"mono8", "8uc1"} or encoding.startswith("bayer_"):
        data, width, height = raw_image_bytes(msg, 1)
        array = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
        return Image.fromarray(array, mode="L")

    if encoding in {"mono16", "16uc1"}:
        data, width, height = raw_image_bytes(msg, 2)
        dtype = np.dtype(">u2" if is_bigendian else "<u2")
        array = np.frombuffer(data, dtype=dtype).reshape(height, width).astype("<u2", copy=False)
        return Image.fromarray(array)

    raise ValueError(f"Unsupported image encoding: {msg.encoding!r}")


def save_image_msg(path: Path, msg: Any, *, image_format: str, jpeg_quality: int) -> None:
    image = image_msg_to_pillow(msg)
    path.parent.mkdir(parents=True, exist_ok=True)

    if image_format == "jpg":
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        image.save(path, format="JPEG", quality=jpeg_quality)
    else:
        image.save(path, format="PNG")


def stamp_to_ns(stamp: Any) -> int | None:
    if stamp is None:
        return None
    if not hasattr(stamp, "sec") or not hasattr(stamp, "nanosec"):
        return None
    return int(stamp.sec) * NS_PER_SEC + int(stamp.nanosec)


def header_stamp_ns(msg: Any) -> int | None:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    return stamp_to_ns(stamp)


def message_time_ns(
    reader: AnyReader,
    rawdata: bytes,
    msgtype: str,
    bag_timestamp_ns: int,
    settings: Settings,
) -> tuple[int, Any | None]:
    if settings.timestamp_source == "bag":
        return bag_timestamp_ns, None

    msg = reader.deserialize(rawdata, msgtype)
    return header_stamp_ns(msg) or bag_timestamp_ns, msg


def format_timestamp_ns(timestamp_ns: int, timezone_name: str) -> str:
    sec, nsec = divmod(timestamp_ns, NS_PER_SEC)
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = ZoneInfo("UTC")
    dt = datetime.fromtimestamp(sec, tz=tz)
    return f"{dt:%Y%m%d_%H%M%S}_{nsec:09d}"


def nearest_timestamp(sorted_timestamps: list[int], target_ns: int) -> int | None:
    if not sorted_timestamps:
        return None

    index = bisect.bisect_left(sorted_timestamps, target_ns)
    candidates: list[int] = []
    if index < len(sorted_timestamps):
        candidates.append(sorted_timestamps[index])
    if index > 0:
        candidates.append(sorted_timestamps[index - 1])

    return min(candidates, key=lambda item: abs(item - target_ns))


def print_available_topics(reader: AnyReader) -> None:
    print("Available topics in bag(s):", file=sys.stderr)
    for topic, info in sorted(reader.topics.items()):
        print(f"  {topic}  [{info.msgtype}]  messages={info.msgcount}", file=sys.stderr)


def selected_connections(reader: AnyReader, settings: Settings) -> tuple[list[Any], list[Any]]:
    lidar_connections = [conn for conn in reader.connections if conn.topic == settings.lidar_topic]
    camera_connections = [conn for conn in reader.connections if conn.topic == settings.camera_topic]

    if not lidar_connections or not camera_connections:
        print_available_topics(reader)
    if not lidar_connections:
        raise RuntimeError(f"Lidar topic not found: {settings.lidar_topic}")
    if not camera_connections:
        raise RuntimeError(f"Camera topic not found: {settings.camera_topic}")

    return lidar_connections, camera_connections


def collect_timestamps(settings: Settings) -> tuple[list[int], list[int]]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    lidar_times: list[int] = []
    camera_times: list[int] = []
    seen_lidar = 0

    with AnyReader(settings.bag_paths, default_typestore=typestore) as reader:
        lidar_connections, camera_connections = selected_connections(reader, settings)
        selected = [*lidar_connections, *camera_connections]

        for connection, bag_timestamp_ns, rawdata in reader.messages(connections=selected):
            timestamp_ns, _ = message_time_ns(
                reader,
                rawdata,
                connection.msgtype,
                bag_timestamp_ns,
                settings,
            )

            if connection.topic == settings.lidar_topic:
                seen_lidar += 1
                if (seen_lidar - 1) % settings.every_lidar == 0:
                    lidar_times.append(timestamp_ns)
            elif connection.topic == settings.camera_topic:
                camera_times.append(timestamp_ns)

    return lidar_times, camera_times


def build_pair_plan(settings: Settings, lidar_times: list[int], camera_times: list[int]) -> list[PairPlan]:
    sorted_camera_times = sorted(camera_times)
    used_names: set[str] = set()
    pairs: list[PairPlan] = []
    skipped_unmatched = 0

    for lidar_ns in lidar_times:
        camera_ns = nearest_timestamp(sorted_camera_times, lidar_ns)
        if camera_ns is None:
            skipped_unmatched += 1
            if settings.require_camera_match:
                continue
            raise RuntimeError("No camera frames are available to match.")

        diff_ns = camera_ns - lidar_ns
        if settings.match_tolerance_ns is not None and abs(diff_ns) > settings.match_tolerance_ns:
            skipped_unmatched += 1
            if settings.require_camera_match:
                continue
            raise RuntimeError(
                "Nearest camera frame exceeds extract.match_tolerance_ms: "
                f"{abs(diff_ns) / NS_PER_MS:.3f} ms",
            )

        base = unique_name(format_timestamp_ns(lidar_ns, settings.timezone), used_names)
        pairs.append(PairPlan(base_name=base, lidar_ns=lidar_ns, camera_ns=camera_ns, diff_ns=diff_ns))

        if settings.max_pairs is not None and len(pairs) >= settings.max_pairs:
            break

    if skipped_unmatched:
        print(f"skipped unmatched lidar frame(s): {skipped_unmatched}")
    if not pairs:
        raise RuntimeError("No lidar-camera pairs were selected. Check topics or match_tolerance_ms.")

    return pairs


def export_pairs(settings: Settings, pairs: list[PairPlan]) -> tuple[int, int]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    lidar_by_time: dict[int, list[PairPlan]] = defaultdict(list)
    camera_by_time: dict[int, list[PairPlan]] = defaultdict(list)

    for pair in pairs:
        lidar_by_time[pair.lidar_ns].append(pair)
        camera_by_time[pair.camera_ns].append(pair)

    lidar_written = 0
    image_written = 0

    with AnyReader(settings.bag_paths, default_typestore=typestore) as reader:
        lidar_connections, camera_connections = selected_connections(reader, settings)
        selected = [*lidar_connections, *camera_connections]

        for connection, bag_timestamp_ns, rawdata in reader.messages(connections=selected):
            timestamp_ns, cached_msg = message_time_ns(
                reader,
                rawdata,
                connection.msgtype,
                bag_timestamp_ns,
                settings,
            )

            if connection.topic == settings.lidar_topic and timestamp_ns in lidar_by_time:
                plans = lidar_by_time.pop(timestamp_ns)
                msg = cached_msg or reader.deserialize(rawdata, connection.msgtype)
                specs = build_field_specs(list(msg.fields), settings.pcd_fields)
                points, width, height = cloud_to_structured_array(msg, specs)

                if settings.drop_nan:
                    points = drop_invalid_xyz(points, specs)
                    width = int(points.shape[0])
                    height = 1

                for plan in plans:
                    path = settings.pcd_output_dir / f"{plan.base_name}.pcd"
                    write_binary_pcd(path, points, specs, width=width, height=height)
                    lidar_written += 1

            elif connection.topic == settings.camera_topic and timestamp_ns in camera_by_time:
                plans = camera_by_time.pop(timestamp_ns)
                msg = cached_msg or reader.deserialize(rawdata, connection.msgtype)

                for plan in plans:
                    path = settings.image_output_dir / f"{plan.base_name}.{settings.image_format}"
                    save_image_msg(
                        path,
                        msg,
                        image_format=settings.image_format,
                        jpeg_quality=settings.jpeg_quality,
                    )
                    image_written += 1

            if not lidar_by_time and not camera_by_time:
                break

    return lidar_written, image_written


def print_settings(settings: Settings) -> None:
    print(f"config: {settings.config_path}")
    print("bags:")
    for path in settings.bag_paths:
        print(f"  - {path}")
    print(f"lidar topic: {settings.lidar_topic}")
    print(f"camera topic: {settings.camera_topic}")
    print(f"group by bag: {settings.group_by_bag}")
    print(f"pcd output: {settings.pcd_output_dir}")
    print(f"image output: {settings.image_output_dir}")
    print(f"timestamp source: {settings.timestamp_source}")


def run_extraction(settings: Settings) -> tuple[int, int]:
    print_settings(settings)
    lidar_times, camera_times = collect_timestamps(settings)
    print(f"indexed lidar frames: {len(lidar_times)}")
    print(f"indexed camera frames: {len(camera_times)}")

    pairs = build_pair_plan(settings, lidar_times, camera_times)
    print(f"selected matched pairs: {len(pairs)}")
    if pairs:
        max_diff_ms = max(abs(pair.diff_ns) for pair in pairs) / NS_PER_MS
        print(f"max lidar-camera time diff: {max_diff_ms:.3f} ms")

    lidar_written, image_written = export_pairs(settings, pairs)
    print(f"done: wrote {lidar_written} PCD file(s), {image_written} image file(s)")
    return lidar_written, image_written


def main() -> int:
    config_path = cli_config_path()
    settings = read_settings(config_path)

    if settings.group_by_bag:
        total_lidar = 0
        total_images = 0
        bag_settings = per_bag_settings(settings)

        for index, current_settings in enumerate(bag_settings, start=1):
            bag_name = current_settings.bag_paths[0].name
            print(f"\n[{index}/{len(bag_settings)}] processing bag: {bag_name}")
            lidar_written, image_written = run_extraction(current_settings)
            total_lidar += lidar_written
            total_images += image_written

        if len(bag_settings) > 1:
            print(f"\ntotal: wrote {total_lidar} PCD file(s), {total_images} image file(s)")
        return 0

    run_extraction(settings)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
