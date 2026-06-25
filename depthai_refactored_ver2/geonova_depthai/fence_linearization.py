"""Guard-fence point correction and directional line generation.

The pipeline follows the processing order proposed in the project research
report: validate source points, project once to EPSG:5179, express points in a
trajectory-local tangent/normal frame, correct lateral-side errors, restore
missing detection roles, reject local-fit outliers, and connect representative
points with distance and direction constraints.

Every spatial debug stage is exported as both CSV and Shapefile.  The CSV is
the authoritative audit trail; a Shapefile can contain only rows that have a
geometry at that stage and uses abbreviated DBF field names.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

import numpy as np
import shapefile
from pyproj import CRS, Transformer
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

from .config_cli import parse_args_with_yaml


CRS_WGS84 = "EPSG:4326"
CRS_WORK = "EPSG:5179"
ROLES = ("endpoint_a", "midpoint", "endpoint_b")
ROLE_POSITION = {"endpoint_a": -1.0, "midpoint": 0.0, "endpoint_b": 1.0}


@dataclass(frozen=True)
class LinearizationConfig:
    work_crs: str = CRS_WORK
    depth_sample_min: int = 20
    max_observation_depth_mm: int = 8_000
    gps_smoothing_window: int = 11
    gps_savgol_polyorder: int = 2
    gps_max_hdop: float = 2.5
    side_vote_window: int = 11
    side_window_m: float = 20.0
    side_frame_window: int = 100
    huber_delta_m: float = 0.75
    max_correction_m: float = 3.0
    lateral_mad_k: float = 3.0
    residual_window: int = 9
    residual_mad_k: float = 3.0
    residual_floor_m: float = 0.30
    max_gap_connect_m: float = 6.0
    max_frame_gap: int = 60
    max_heading_deg: float = 35.0
    max_lateral_step_m: float = 2.0
    min_line_support: int = 5
    line_sample_spacing_m: float = 0.5
    spline_smoothing: float = 0.15


def _float(value, default=None):
    try:
        if value in (None, ""):
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _int(value, default=0):
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bool(value, default=True):
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _iter_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        yield from csv.DictReader(file)


def _read_csv(path: Path) -> list[dict]:
    return list(_iter_csv(path))


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (np.bool_, bool)):
        return int(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _ordered_fields(rows: Sequence[dict], preferred: Sequence[str] = ()) -> list[str]:
    discovered: list[str] = []
    seen = set()
    for name in preferred:
        if name not in seen:
            discovered.append(name)
            seen.add(name)
    for row in rows:
        for name in row:
            if name not in seen:
                discovered.append(name)
                seen.add(name)
    return discovered


def _write_csv(path: Path, rows: Sequence[dict], preferred: Sequence[str] = ()) -> None:
    fields = _ordered_fields(rows, preferred)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fields})


def _write_prj(path: Path, crs: str) -> None:
    path.with_suffix(".prj").write_text(CRS.from_user_input(crs).to_wkt("WKT1_ESRI"), encoding="utf-8")


def _point_writer(path: Path, shape_type: int = shapefile.POINTZ) -> shapefile.Writer:
    writer = shapefile.Writer(str(path), shapeType=shape_type, encoding="utf-8")
    writer.field("det_uid", "C", 32)
    writer.field("frame", "N", 10, 0)
    writer.field("detect_id", "N", 10, 0)
    writer.field("role", "C", 12)
    writer.field("src_stat", "C", 16)
    writer.field("corr_type", "C", 32)
    writer.field("disp_m", "F", 12, 4)
    writer.field("resid_m", "F", 12, 4)
    writer.field("side", "N", 3, 0)
    writer.field("qa_score", "F", 8, 4)
    writer.field("accepted", "N", 1, 0)
    return writer


def _write_point_shp(
    path: Path,
    rows: Sequence[dict],
    crs: str,
    x_field: str,
    y_field: str,
    z_field: str | None = None,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = _point_writer(path)
    count = 0
    for row in rows:
        x, y = _float(row.get(x_field)), _float(row.get(y_field))
        if x is None or y is None:
            continue
        z = _float(row.get(z_field), 0.0) if z_field else 0.0
        writer.pointz(x, y, z or 0.0)
        writer.record(
            str(row.get("det_uid", "")),
            _int(row.get("frame_index")),
            _int(row.get("detection_id")),
            str(row.get("role", "")),
            str(row.get("source_status", ""))[:16],
            str(row.get("correction_type", ""))[:32],
            _float(row.get("displacement_m"), 0.0),
            _float(row.get("residual_m"), 0.0),
            _int(row.get("side_sign"), 0),
            _float(row.get("qa_score"), 0.0),
            int(_bool(row.get("accepted"), True)),
        )
        count += 1
    writer.close()
    _write_prj(path, crs)
    return count


def _export_stage(
    output_dir: Path,
    stem: str,
    rows: Sequence[dict],
    crs: str,
    x_field: str,
    y_field: str,
    z_field: str | None = None,
) -> None:
    _write_csv(output_dir / f"{stem}.csv", rows)
    _write_point_shp(output_dir / stem, rows, crs, x_field, y_field, z_field)


def _normalise_points(path: Path, config: LinearizationConfig) -> list[dict]:
    rows = []
    for source in _read_csv(path):
        row = dict(source)
        row["frame_index"] = _int(source.get("frame_index"), -1)
        row["detection_id"] = _int(source.get("detection_id"), -1)
        row["det_uid"] = f"{row['frame_index']}_{row['detection_id']}"
        row["depth_sample_count"] = _int(source.get("depth_sample_count"), 0)
        row["depth_mm"] = _float(source.get("depth_mm"), 0.0)
        row["observation_weight"] = _float(source.get("observation_weight"), 1.0)
        row["longitude_deg"] = _float(source.get("longitude_deg"))
        row["latitude_deg"] = _float(source.get("latitude_deg"))
        row["altitude_m"] = _float(source.get("altitude_m"), 0.0)
        world_ok = str(source.get("world_status", "ok")).lower() == "ok"
        coordinate_ok = row["longitude_deg"] is not None and row["latitude_deg"] is not None
        depth_ok = row["depth_sample_count"] >= config.depth_sample_min
        distance_ok = 0 < row["depth_mm"] <= config.max_observation_depth_mm
        if not world_ok or not coordinate_ok:
            status = "invalid_world"
        elif not depth_ok:
            status = "low_depth"
        elif not distance_ok:
            status = "depth_too_far"
        else:
            status = "observed"
        row["source_status"] = status
        row["source_valid"] = status == "observed"
        row["correction_type"] = "none"
        row["displacement_m"] = 0.0
        row["accepted"] = False
        rows.append(row)
    modern_detections = {
        row["det_uid"] for row in rows
        if row.get("point_usage") == "mapping" or row.get("role") == "representative"
    }
    for row in rows:
        row["mapping_candidate"] = (
            row.get("point_usage") == "mapping" or row.get("role") == "representative"
            if row["det_uid"] in modern_detections
            else True
        )
    return rows


def _smooth_xy(coords: np.ndarray, window: int) -> np.ndarray:
    if len(coords) < 3 or window <= 1:
        return coords.copy()
    width = min(window, len(coords) if len(coords) % 2 else len(coords) - 1)
    width = max(1, width)
    half = width // 2
    output = np.empty_like(coords)
    for index in range(len(coords)):
        section = coords[max(0, index - half): min(len(coords), index + half + 1)]
        output[index] = np.median(section, axis=0)
    if width >= 3:
        polyorder = min(2, width - 1)
        output[:, 0] = savgol_filter(output[:, 0], width, polyorder, mode="interp")
        output[:, 1] = savgol_filter(output[:, 1], width, polyorder, mode="interp")
    return output


def _trajectory_from_csv(
    path: Path | None,
    transformer: Transformer,
    config: LinearizationConfig,
) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    coordinates = []
    seen_epochs = set()
    for row in _iter_csv(path):
        lon = _float(row.get("gps_longitude_deg", row.get("longitude_deg")))
        lat = _float(row.get("gps_latitude_deg", row.get("latitude_deg")))
        if lon is None or lat is None:
            continue
        if not _bool(row.get("gps_position_valid", row.get("position_valid")), True):
            continue
        hdop = _float(row.get("gps_hdop", row.get("hdop")))
        if hdop is not None and hdop > config.gps_max_hdop:
            continue
        epoch = row.get("gps_measurement_host_monotonic_ns") or row.get("measurement_wall_time_utc")
        if epoch and epoch in seen_epochs:
            continue
        if epoch:
            seen_epochs.add(epoch)
        coordinates.append(transformer.transform(lon, lat))
    if len(coordinates) < 2:
        return None
    coords = _smooth_xy(np.asarray(coordinates, dtype=float), config.gps_smoothing_window)
    keep = np.r_[True, np.linalg.norm(np.diff(coords, axis=0), axis=1) > 0.01]
    coords = coords[keep]
    return coords if len(coords) >= 2 else None


def _representative(rows: Sequence[dict], x_field: str = "x", y_field: str = "y") -> dict | None:
    valid = [row for row in rows if _float(row.get(x_field)) is not None and _float(row.get(y_field)) is not None]
    if not valid:
        return None
    mapping = next(
        (row for row in valid if row.get("point_usage") == "mapping" or row.get("role") == "representative"),
        None,
    )
    if mapping is not None:
        return dict(mapping)
    midpoint = next((row for row in valid if row.get("role") == "midpoint"), None)
    if midpoint is not None:
        return dict(midpoint)
    representative = dict(valid[0])
    representative[x_field] = float(np.mean([row[x_field] for row in valid]))
    representative[y_field] = float(np.mean([row[y_field] for row in valid]))
    representative["altitude_m"] = float(np.mean([_float(row.get("altitude_m"), 0.0) for row in valid]))
    representative["role"] = "centroid"
    return representative


def _fallback_trajectory(projected: Sequence[dict]) -> np.ndarray:
    groups = defaultdict(list)
    for row in projected:
        groups[row["det_uid"]].append(row)
    reps = [_representative(group) for group in groups.values()]
    reps = sorted((row for row in reps if row), key=lambda row: (row["frame_index"], row["detection_id"]))
    coords = np.asarray([[row["x"], row["y"]] for row in reps], dtype=float)
    if len(coords) >= 2:
        keep = np.r_[True, np.linalg.norm(np.diff(coords, axis=0), axis=1) > 1e-6]
        coords = coords[keep]
    if len(coords) >= 2:
        return coords
    if len(coords) == 1:
        return np.vstack((coords[0] - [0.5, 0.0], coords[0] + [0.5, 0.0]))
    raise ValueError("At least one valid world point is required for linearization.")


def _polyline_lengths(coords: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))]


class _TrajectoryIndex:
    """Narrow nearest-segment searches to segments beside nearby vertices."""

    def __init__(self, coords: np.ndarray):
        self.coords = coords
        self.tree = cKDTree(coords)

    def candidate_segments(self, point: np.ndarray) -> list[int]:
        k = min(8, len(self.coords))
        _, vertices = self.tree.query(point, k=k)
        candidates = set()
        for vertex in np.atleast_1d(vertices):
            index = int(vertex)
            if index > 0:
                candidates.add(index - 1)
            if index < len(self.coords) - 1:
                candidates.add(index)
        return sorted(candidates)


def _project_to_trajectory(
    point: np.ndarray,
    coords: np.ndarray,
    lengths: np.ndarray,
    trajectory_index: _TrajectoryIndex | None = None,
) -> dict:
    best = None
    segment_indices = (
        trajectory_index.candidate_segments(point)
        if trajectory_index is not None
        else range(len(coords) - 1)
    )
    for index in segment_indices:
        start, end = coords[index], coords[index + 1]
        delta = end - start
        length2 = float(delta @ delta)
        if length2 <= 1e-12:
            continue
        fraction = float(np.clip(((point - start) @ delta) / length2, 0.0, 1.0))
        foot = start + fraction * delta
        distance2 = float((point - foot) @ (point - foot))
        if best is None or distance2 < best[0]:
            tangent = delta / math.sqrt(length2)
            normal = np.array([-tangent[1], tangent[0]])
            best = (
                distance2,
                lengths[index] + fraction * math.sqrt(length2),
                float((point - foot) @ normal),
                foot,
                tangent,
                normal,
            )
    if best is None:
        raise ValueError("Trajectory contains no non-zero segment.")
    return {
        "s_m": best[1], "lateral_m": best[2],
        "trajectory_x": best[3][0], "trajectory_y": best[3][1],
        "tangent_x": best[4][0], "tangent_y": best[4][1],
        "normal_x": best[5][0], "normal_y": best[5][1],
    }


def _append_correction(existing: str, correction: str) -> str:
    if not existing or existing == "none":
        return correction
    parts = existing.split("+")
    return existing if correction in parts else existing + "+" + correction


def _local_median_and_mad(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    center = float(median(values))
    mad = float(median([abs(value - center) for value in values]))
    return center, mad


def _quality_weight(row: dict) -> float:
    observation = max(0.01, _float(row.get("observation_weight"), 1.0))
    coordinate = 1.0 if row.get("coordinate_quality") == "trusted" else 0.35
    sync_delta = abs(_float(row.get("gps_frame_delta_ms"), 0.0))
    sync = max(0.1, 1.0 - min(sync_delta, 250.0) / 300.0)
    return observation * coordinate * sync


def _huber_offset_prediction(
    target: dict,
    neighborhood: Sequence[dict],
    expected_sign: int,
    config: LinearizationConfig,
) -> tuple[float, float]:
    candidates = [
        row for row in neighborhood
        if expected_sign == 0 or float(row["lateral_m"]) * expected_sign > 0
    ]
    if len(candidates) < 3:
        candidates = list(neighborhood)
    if not candidates:
        return float(target["lateral_m"]), 0.0
    s0 = float(target["s_m"])
    x = np.asarray([float(row["s_m"]) - s0 for row in candidates])
    y = np.asarray([float(row["lateral_m"]) for row in candidates])
    design = np.column_stack((np.ones(len(x)), x))
    base_weights = np.asarray([_quality_weight(row) for row in candidates], dtype=float)
    weights = base_weights.copy()
    coefficients = np.array([float(np.median(y)), 0.0])
    for _ in range(8):
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_y = y * np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
        residuals = y - design @ coefficients
        robust = np.ones_like(residuals)
        large = np.abs(residuals) > config.huber_delta_m
        robust[large] = config.huber_delta_m / np.abs(residuals[large])
        new_weights = np.maximum(1e-6, base_weights * robust)
        if np.allclose(weights, new_weights, rtol=1e-3, atol=1e-5):
            break
        weights = new_weights
    residuals = y - design @ coefficients
    rmse = math.sqrt(float(np.average(residuals ** 2, weights=np.maximum(weights, 1e-6))))
    return float(coefficients[0]), rmse


def _side_correct(
    local_rows: Sequence[dict],
    coords: np.ndarray,
    lengths: np.ndarray,
    trajectory_index: _TrajectoryIndex,
    config: LinearizationConfig,
) -> list[dict]:
    groups = defaultdict(list)
    for row in local_rows:
        groups[row["det_uid"]].append(row)
    reps = [_representative(group) for group in groups.values()]
    reps = sorted((row for row in reps if row), key=lambda row: (row["s_m"], row["frame_index"], row["detection_id"]))
    stations = np.asarray([float(row["s_m"]) for row in reps])
    half = max(1, config.side_vote_window // 2)
    shifts: dict[str, tuple[np.ndarray, int, str, bool, float]] = {}
    for index, rep in enumerate(reps):
        left = int(np.searchsorted(stations, float(rep["s_m"]) - config.side_window_m, side="left"))
        right = int(np.searchsorted(stations, float(rep["s_m"]) + config.side_window_m, side="right"))
        metric_neighbors = [
            row for row in reps[left:right]
            if abs(_int(row.get("frame_index")) - _int(rep.get("frame_index"))) <= config.side_frame_window
        ]
        neighborhood = metric_neighbors or reps[max(0, index - half): min(len(reps), index + half + 1)]
        offsets = [float(row["lateral_m"]) for row in neighborhood if abs(float(row["lateral_m"])) > 0.05]
        signs = [1 if value > 0 else -1 for value in offsets]
        expected_sign = 0 if not signs else (1 if sum(signs) >= 0 else -1)
        positive_count = signs.count(1)
        negative_count = signs.count(-1)
        # When both road sides are repeatedly observed, preserve the current
        # side instead of forcing the minority fence onto the majority side.
        if min(positive_count, negative_count) >= 2 and min(positive_count, negative_count) / len(signs) >= 0.30:
            expected_sign = 1 if float(rep["lateral_m"]) >= 0 else -1
        lateral = float(rep["lateral_m"])
        correction = "none"
        target, model_rmse = _huber_offset_prediction(rep, neighborhood, expected_sign, config)
        delta = target - lateral
        pre_reject = False
        if abs(delta) > config.max_correction_m:
            target = lateral
            correction = "gross_lateral_outlier"
            pre_reject = True
        elif abs(delta) > 0.05:
            correction = "side_huber" if expected_sign and lateral * expected_sign < 0 else "offset_huber"
        normal = np.array([rep["normal_x"], rep["normal_y"]], dtype=float)
        shifts[rep["det_uid"]] = ((target - lateral) * normal, expected_sign, correction, pre_reject, model_rmse)

    corrected = []
    for source in local_rows:
        row = dict(source)
        shift, side_sign, correction, pre_reject, model_rmse = shifts.get(
            row["det_uid"], (np.zeros(2), 0, "none", False, 0.0)
        )
        row["x_raw"] = row["x"]
        row["y_raw"] = row["y"]
        row["x"] = float(row["x"] + shift[0])
        row["y"] = float(row["y"] + shift[1])
        row["side_sign"] = side_sign
        row["displacement_m"] = float(np.linalg.norm(shift))
        row["pre_reject"] = pre_reject
        row["offset_model_rmse_m"] = model_rmse
        row["correction_type"] = _append_correction(row.get("correction_type", "none"), correction) if correction != "none" else row.get("correction_type", "none")
        row.update(_project_to_trajectory(np.array([row["x"], row["y"]]), coords, lengths, trajectory_index))
        corrected.append(row)
    return corrected


def _median_role_delta(groups: dict[str, list[dict]]) -> np.ndarray:
    deltas = []
    for rows in groups.values():
        by_role = {row["role"]: row for row in rows}
        if "endpoint_a" in by_role and "endpoint_b" in by_role:
            a, b = by_role["endpoint_a"], by_role["endpoint_b"]
            deltas.append([
                (b["x"] - a["x"]) * 0.5,
                (b["y"] - a["y"]) * 0.5,
                (_float(b.get("altitude_m"), 0.0) - _float(a.get("altitude_m"), 0.0)) * 0.5,
            ])
    if not deltas:
        return np.zeros(3, dtype=float)
    values = np.asarray(deltas, dtype=float)
    reference = values[0]
    for index in range(len(values)):
        if values[index] @ reference < 0:
            values[index] *= -1
    return np.median(values, axis=0)


def _interpolate_roles(raw_rows: Sequence[dict], corrected: Sequence[dict]) -> list[dict]:
    raw_groups = defaultdict(list)
    corrected_groups = defaultdict(list)
    for row in raw_rows:
        raw_groups[row["det_uid"]].append(row)
    for row in corrected:
        corrected_groups[row["det_uid"]].append(row)
    role_delta = _median_role_delta(corrected_groups)
    output = []
    for det_uid, raw_group in raw_groups.items():
        observed = {row["role"]: dict(row) for row in corrected_groups.get(det_uid, [])}
        if not observed:
            continue
        known_positions = np.asarray([ROLE_POSITION[row["role"]] for row in observed.values()], dtype=float)
        known_xyz = np.asarray([
            [row["x"], row["y"], _float(row.get("altitude_m"), 0.0)]
            for row in observed.values()
        ], dtype=float)
        raw_by_role = {row["role"]: row for row in raw_group}
        for role in ROLES:
            if role in observed:
                row = observed[role]
                row["interpolated"] = False
                output.append(row)
                continue
            position = ROLE_POSITION[role]
            if len(known_positions) >= 2:
                design = np.column_stack((known_positions, np.ones(len(known_positions))))
                coefficients, *_ = np.linalg.lstsq(design, known_xyz, rcond=None)
                xyz = np.array([position, 1.0]) @ coefficients
                method = "role_linear"
            else:
                known_role, known_row = next(iter(observed.items()))
                xyz = known_xyz[0] + (position - ROLE_POSITION[known_role]) * role_delta
                method = "role_neighbor_span"
            base = dict(raw_by_role.get(role, raw_group[0]))
            base.update({
                "role": role,
                "x": float(xyz[0]), "y": float(xyz[1]), "altitude_m": float(xyz[2]),
                "source_status": "interpolated", "source_valid": False,
                "correction_type": method, "interpolated": True,
                "displacement_m": 0.0,
                "side_sign": next(iter(observed.values())).get("side_sign", 0),
            })
            output.append(base)
    return sorted(output, key=lambda row: (row["frame_index"], row["detection_id"], ROLE_POSITION.get(row["role"], 9)))


def _tls_residual(point: np.ndarray, neighbors: np.ndarray) -> float:
    if len(neighbors) < 2:
        return 0.0
    center = neighbors.mean(axis=0)
    _, _, vh = np.linalg.svd(neighbors - center, full_matrices=False)
    direction = vh[0]
    delta = point - center
    return float(np.linalg.norm(delta - (delta @ direction) * direction))


def _filter_detections(rows: Sequence[dict], config: LinearizationConfig) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["det_uid"]].append(row)
    reps = [_representative(group) for group in groups.values()]
    reps = sorted((row for row in reps if row), key=lambda row: (row.get("s_m", 0.0), row["frame_index"], row["detection_id"]))
    half = max(1, config.residual_window // 2)
    for index, row in enumerate(reps):
        neighbors = reps[max(0, index - half): min(len(reps), index + half + 1)]
        xy = np.asarray([[item["x"], item["y"]] for item in neighbors], dtype=float)
        row["residual_m"] = _tls_residual(np.array([row["x"], row["y"]]), xy)
    residuals = [row["residual_m"] for row in reps]
    center, mad = _local_median_and_mad(residuals)
    threshold = center + max(config.residual_floor_m, config.residual_mad_k * 1.4826 * mad)
    for index, row in enumerate(reps):
        before = reps[max(0, index - 1)]
        after = reps[min(len(reps) - 1, index + 1)]
        dx, dy = after["x"] - before["x"], after["y"] - before["y"]
        observation_heading = math.atan2(dy, dx) if abs(dx) + abs(dy) > 1e-9 else math.atan2(row["tangent_y"], row["tangent_x"])
        tangent_heading = math.atan2(row["tangent_y"], row["tangent_x"])
        row["observation_heading_deg"] = math.degrees(observation_heading) % 360.0
        row["gps_tangent_deg"] = math.degrees(tangent_heading) % 360.0
        row["heading_error_deg"] = math.degrees(_angle_difference(observation_heading, tangent_heading))
    rep_by_uid = {row["det_uid"]: row for row in reps}
    annotated, accepted = [], []
    for source in rows:
        row = dict(source)
        rep = rep_by_uid[row["det_uid"]]
        row["residual_m"] = rep["residual_m"]
        row["observation_heading_deg"] = rep["observation_heading_deg"]
        row["gps_tangent_deg"] = rep["gps_tangent_deg"]
        row["heading_error_deg"] = rep["heading_error_deg"]
        row["accepted"] = rep["residual_m"] <= threshold and not _bool(row.get("pre_reject"), False)
        source_factor = 0.75 if row.get("interpolated") else _quality_weight(row)
        residual_factor = max(0.0, 1.0 - row["residual_m"] / max(threshold * 2.0, 1e-6))
        displacement_factor = max(0.0, 1.0 - _float(row.get("displacement_m"), 0.0) / 5.0)
        row["qa_score"] = round(source_factor * residual_factor * displacement_factor, 6)
        if not row["accepted"] and "gross_lateral_outlier" not in row.get("correction_type", ""):
            row["correction_type"] = _append_correction(row.get("correction_type", "none"), "residual_reject")
        annotated.append(row)
        if row["accepted"]:
            accepted.append(row)
    return annotated, accepted


def _angle_difference(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _line_length_3d(coords: Sequence[Sequence[float]]) -> float:
    values = np.asarray(coords, dtype=float)
    return float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum()) if len(values) >= 2 else 0.0


def _spline_chain_coordinates(chain: Sequence[dict], config: LinearizationConfig) -> list[list[float]]:
    ordered = sorted(chain, key=lambda row: float(row["s_m"]))
    stations = np.asarray([float(row["s_m"]) for row in ordered])
    unique_stations = np.unique(stations)
    if len(unique_stations) < 2:
        return [[row["x"], row["y"], _float(row.get("altitude_m"), 0.0)] for row in ordered]
    xyz = np.asarray([
        [
            np.median([row["x"] for row in ordered if abs(float(row["s_m"]) - station) < 1e-6]),
            np.median([row["y"] for row in ordered if abs(float(row["s_m"]) - station) < 1e-6]),
            np.median([_float(row.get("altitude_m"), 0.0) for row in ordered if abs(float(row["s_m"]) - station) < 1e-6]),
        ]
        for station in unique_stations
    ], dtype=float)
    spacing = max(0.1, config.line_sample_spacing_m)
    sample_s = np.arange(unique_stations[0], unique_stations[-1], spacing)
    sample_s = np.r_[sample_s, unique_stations[-1]]
    if len(unique_stations) >= 4:
        degree = min(3, len(unique_stations) - 1)
        smooth = config.spline_smoothing * len(unique_stations)
        sample_x = UnivariateSpline(unique_stations, xyz[:, 0], k=degree, s=smooth)(sample_s)
        sample_y = UnivariateSpline(unique_stations, xyz[:, 1], k=degree, s=smooth)(sample_s)
    else:
        sample_x = np.interp(sample_s, unique_stations, xyz[:, 0])
        sample_y = np.interp(sample_s, unique_stations, xyz[:, 1])
    sample_z = np.interp(sample_s, unique_stations, xyz[:, 2])
    return np.column_stack((sample_x, sample_y, sample_z)).tolist()


def _connect_lines(rows: Sequence[dict], config: LinearizationConfig) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["det_uid"]].append(row)
    nodes = [_representative(group) for group in groups.values()]
    nodes = [row for row in nodes if row]
    by_side = defaultdict(list)
    for node in nodes:
        by_side[_int(node.get("side_sign"), 0)].append(node)
    lines = []
    for side, side_nodes in by_side.items():
        side_nodes.sort(key=lambda row: (row.get("s_m", 0.0), row["frame_index"], row["detection_id"]))
        chains: list[list[dict]] = []
        current: list[dict] = []
        for node in side_nodes:
            if not current:
                current = [node]
                continue
            previous = current[-1]
            frame_gap = _int(node.get("frame_index")) - _int(previous.get("frame_index"))
            dx, dy = node["x"] - previous["x"], node["y"] - previous["y"]
            gap = math.hypot(dx, dy)
            link_heading = math.atan2(dy, dx)
            tangent_heading = math.atan2(previous.get("tangent_y", dy), previous.get("tangent_x", dx))
            valid = (
                node.get("s_m", 0.0) > previous.get("s_m", 0.0)
                and 0 < frame_gap <= config.max_frame_gap
                and gap <= config.max_gap_connect_m
                and _angle_difference(link_heading, tangent_heading) <= math.radians(config.max_heading_deg)
                and abs(node.get("lateral_m", 0.0) - previous.get("lateral_m", 0.0)) <= config.max_lateral_step_m
            )
            if valid:
                current.append(node)
            else:
                chains.append(current)
                current = [node]
        if current:
            chains.append(current)
        for chain in chains:
            if len(chain) < config.min_line_support:
                continue
            raw_coordinates = np.asarray([[row["x"], row["y"]] for row in chain], dtype=float)
            gaps = np.linalg.norm(np.diff(raw_coordinates, axis=0), axis=1)
            coordinates = _spline_chain_coordinates(chain, config)
            length_2d = float(np.linalg.norm(np.diff(np.asarray(coordinates)[:, :2], axis=0), axis=1).sum())
            lines.append({
                "line_id": len(lines) + 1,
                "side": side,
                "start_det": chain[0]["det_uid"],
                "end_det": chain[-1]["det_uid"],
                "n_det": len(chain),
                "n_vertices": len(coordinates),
                "gap_count": int(np.count_nonzero(gaps > config.line_sample_spacing_m * 2.0)),
                "max_gap_m": float(np.max(gaps)) if gaps.size else 0.0,
                "mean_conf": round(float(np.mean([_float(row.get("confidence"), 0.0) for row in chain])), 6),
                "length_m_2d": length_2d,
                "length_m_3d": _line_length_3d(coordinates),
                "qa_score": round(float(np.mean([row.get("qa_score", 0.0) for row in chain])), 6),
                "wkt": "LINESTRING Z (" + ", ".join(f"{x:.4f} {y:.4f} {z:.4f}" for x, y, z in coordinates) + ")",
                "coordinates": coordinates,
            })
    return lines


def _write_line_shp(path: Path, lines: Sequence[dict], crs: str) -> None:
    writer = shapefile.Writer(str(path), shapeType=shapefile.POLYLINEZ, encoding="utf-8")
    writer.field("line_id", "N", 10, 0)
    writer.field("side", "N", 3, 0)
    writer.field("start_det", "C", 32)
    writer.field("end_det", "C", 32)
    writer.field("n_det", "N", 10, 0)
    writer.field("n_vertex", "N", 10, 0)
    writer.field("gap_count", "N", 10, 0)
    writer.field("max_gap_m", "F", 12, 3)
    writer.field("mean_conf", "F", 8, 4)
    writer.field("len_2d_m", "F", 16, 3)
    writer.field("len_3d_m", "F", 16, 3)
    writer.field("qa_score", "F", 8, 4)
    for row in lines:
        writer.linez([row["coordinates"]])
        writer.record(
            row["line_id"], row["side"], row["start_det"], row["end_det"], row["n_det"],
            row["n_vertices"], row["gap_count"], row["max_gap_m"], row["mean_conf"],
            row["length_m_2d"], row["length_m_3d"], row["qa_score"],
        )
    writer.close()
    _write_prj(path, crs)


def _qa_metrics(raw: Sequence[dict], corrected: Sequence[dict], accepted: Sequence[dict], lines: Sequence[dict], trajectory_source: str) -> list[dict]:
    raw_detections = {row["det_uid"] for row in raw}
    corrected_detections = {row["det_uid"] for row in corrected}
    accepted_detections = {row["det_uid"] for row in accepted}
    displacements = [_float(row.get("displacement_m"), 0.0) for row in corrected]
    residuals = [_float(row.get("residual_m"), 0.0) for row in corrected]
    accepted_residuals = [_float(row.get("residual_m"), 0.0) for row in accepted]
    heading_errors = [_float(row.get("heading_error_deg")) for row in accepted]
    heading_errors = [value for value in heading_errors if value is not None]
    mapping_depths = [
        _float(row.get("depth_mm"), 0.0) for row in raw
        if row.get("mapping_candidate") and _float(row.get("depth_mm"), 0.0) > 0
    ]
    sides = [_int(row.get("side_sign"), 0) for row in corrected if _int(row.get("side_sign"), 0)]
    side_consistency = max(sides.count(1), sides.count(-1)) / len(sides) if sides else 0.0
    values = {
        "trajectory_source": trajectory_source,
        "raw_point_count": len(raw),
        "raw_detection_count": len(raw_detections),
        "corrected_point_count": len(corrected),
        "corrected_detection_count": len(corrected_detections),
        "accepted_point_count": len(accepted),
        "accepted_detection_count": len(accepted_detections),
        "completeness": len(accepted_detections) / len(raw_detections) if raw_detections else 0.0,
        "side_consistency": side_consistency,
        "mean_displacement_m": float(np.mean(displacements)) if displacements else 0.0,
        "p95_displacement_m": float(np.percentile(displacements, 95)) if displacements else 0.0,
        "max_displacement_m": max(displacements, default=0.0),
        "residual_rmse_m": math.sqrt(float(np.mean(np.square(residuals)))) if residuals else 0.0,
        "accepted_residual_rmse_m": math.sqrt(float(np.mean(np.square(accepted_residuals)))) if accepted_residuals else 0.0,
        "residual_mad_m": _local_median_and_mad(residuals)[1],
        "depth_compliance_8m": (
            sum(value <= 8_000 for value in mapping_depths) / len(mapping_depths)
            if mapping_depths else 0.0
        ),
        "mean_heading_error_deg": float(np.mean(heading_errors)) if heading_errors else 0.0,
        "p95_heading_error_deg": float(np.percentile(heading_errors, 95)) if heading_errors else 0.0,
        "gross_lateral_outlier_count": sum(
            "gross_lateral_outlier" in str(row.get("correction_type", "")) for row in corrected
        ),
        "interpolated_point_ratio": (
            sum(_bool(row.get("interpolated"), False) for row in corrected) / len(corrected)
            if corrected else 0.0
        ),
        "line_count": len(lines),
        "line_gap_count": sum(row.get("gap_count", 0) for row in lines),
        "maximum_line_gap_m": max((row.get("max_gap_m", 0.0) for row in lines), default=0.0),
        "total_length_m_2d": sum(row["length_m_2d"] for row in lines),
        "total_length_m_3d": sum(row["length_m_3d"] for row in lines),
    }
    return [{"metric": key, "value": value} for key, value in values.items()]


def linearize_fence_points(
    points_csv: Path,
    output_dir: Path | None = None,
    trajectory_csv: Path | None = None,
    config: LinearizationConfig | None = None,
) -> dict:
    """Correct YOLO fence points and export audited point/line products."""
    config = config or LinearizationConfig()
    points_csv = points_csv.expanduser().resolve()
    output_dir = (output_dir or points_csv.parent / "linearized").expanduser().resolve()
    trajectory_csv = trajectory_csv.expanduser().resolve() if trajectory_csv else None
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = _normalise_points(points_csv, config)
    _export_stage(output_dir, "debug_00_raw", raw, CRS_WGS84, "longitude_deg", "latitude_deg", "altitude_m")
    valid = [row for row in raw if row["source_valid"] and row.get("mapping_candidate", True)]
    if not valid:
        raise ValueError("No valid world points remain after world/depth quality filtering.")

    transformer = Transformer.from_crs(CRS_WGS84, config.work_crs, always_xy=True)
    projected = []
    for source in valid:
        row = dict(source)
        row["x"], row["y"] = transformer.transform(row["longitude_deg"], row["latitude_deg"])
        projected.append(row)
    _export_stage(output_dir, "debug_01_projected", projected, config.work_crs, "x", "y", "altitude_m")

    trajectory = _trajectory_from_csv(trajectory_csv, transformer, config)
    trajectory_source = "gps" if trajectory is not None else "detection_fallback"
    if trajectory is None:
        trajectory = _fallback_trajectory(projected)
    lengths = _polyline_lengths(trajectory)
    trajectory_index = _TrajectoryIndex(trajectory)
    local = []
    for source in projected:
        row = dict(source)
        row.update(_project_to_trajectory(np.array([row["x"], row["y"]]), trajectory, lengths, trajectory_index))
        local.append(row)
    _export_stage(output_dir, "debug_02_local", local, config.work_crs, "x", "y", "altitude_m")

    side_corrected = _side_correct(local, trajectory, lengths, trajectory_index, config)
    _export_stage(output_dir, "debug_03_side_corrected", side_corrected, config.work_crs, "x", "y", "altitude_m")

    modern_observations = any(
        row.get("point_usage") == "mapping" or row.get("role") == "representative"
        for row in raw
    )
    interpolated = (
        [dict(row, interpolated=False) for row in side_corrected]
        if modern_observations
        else _interpolate_roles(raw, side_corrected)
    )
    for row in interpolated:
        row.update(_project_to_trajectory(np.array([row["x"], row["y"]]), trajectory, lengths, trajectory_index))
    _export_stage(output_dir, "debug_04_interpolated", interpolated, config.work_crs, "x", "y", "altitude_m")

    corrected, accepted = _filter_detections(interpolated, config)
    _export_stage(output_dir, "debug_05_filtered", accepted, config.work_crs, "x", "y", "altitude_m")
    _write_csv(output_dir / "points_corrected.csv", corrected)
    _write_point_shp(output_dir / "points_corrected", corrected, config.work_crs, "x", "y", "altitude_m")

    lines = _connect_lines(accepted, config)
    line_tables = [{key: value for key, value in row.items() if key != "coordinates"} for row in lines]
    _write_csv(
        output_dir / "fence_lines.csv",
        line_tables,
        ("line_id", "side", "start_det", "end_det", "n_det", "n_vertices", "gap_count", "max_gap_m", "mean_conf", "length_m_2d", "length_m_3d", "qa_score", "wkt"),
    )
    _write_line_shp(output_dir / "fence_lines", lines, config.work_crs)
    metrics = _qa_metrics(raw, corrected, accepted, lines, trajectory_source)
    _write_csv(output_dir / "qa_metrics.csv", metrics, ("metric", "value"))

    summary = {
        "points_csv": str(points_csv),
        "trajectory_csv": str(trajectory_csv) if trajectory_csv else None,
        "output_dir": str(output_dir),
        "work_crs": config.work_crs,
        "trajectory_source": trajectory_source,
        "raw_points": len(raw),
        "corrected_points": len(corrected),
        "accepted_points": len(accepted),
        "lines": len(lines),
        "total_length_m_2d": sum(row["length_m_2d"] for row in lines),
        "total_length_m_3d": sum(row["length_m_3d"] for row in lines),
        "config": asdict(config),
    }
    (output_dir / "linearization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correct fence points and generate EPSG:5179 fence lines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--points", type=Path, required=True, help="YOLO points.csv")
    parser.add_argument("--trajectory", type=Path, help="timestamps.csv or GPS CSV")
    parser.add_argument("--output-dir", type=Path, help="Output directory; defaults to a linearized folder beside points.csv")
    parser.add_argument("--depth-sample-min", type=int, default=20, help="Minimum valid Depth samples for a mapping observation")
    parser.add_argument("--max-observation-depth-mm", type=int, default=8_000, help="Maximum accepted observation Depth in millimeters")
    parser.add_argument("--side-vote-window", type=int, default=11, help="Fallback detection-count window for side-sign voting")
    parser.add_argument("--side-window-m", type=float, default=20.0, help="Chainage radius of the local Huber offset model, in meters")
    parser.add_argument("--side-frame-window", type=int, default=100, help="Maximum frame distance included in the local offset model")
    parser.add_argument("--huber-delta-m", type=float, default=0.75, help="Huber loss transition residual, in meters")
    parser.add_argument("--max-correction-m", type=float, default=3.0, help="Maximum correction displacement before rejecting a point")
    parser.add_argument("--max-gap-m", type=float, default=6.0, help="Maximum spatial gap between connected representative points")
    parser.add_argument("--max-frame-gap", type=int, default=60, help="Maximum frame interval between connected detections")
    parser.add_argument("--max-heading-deg", type=float, default=35.0, help="Maximum link-to-trajectory-tangent angular difference")
    parser.add_argument("--min-line-support", type=int, default=5, help="Minimum detections required for an exported line")
    parser.add_argument("--line-sample-spacing-m", type=float, default=0.5, help="Final spline vertex spacing, in meters")
    parser.add_argument("--spline-smoothing", type=float, default=0.15, help="Smoothing factor per source station")
    return parse_args_with_yaml(parser)


def main() -> None:
    args = parse_args()
    config = LinearizationConfig(
        depth_sample_min=args.depth_sample_min,
        max_observation_depth_mm=args.max_observation_depth_mm,
        side_vote_window=args.side_vote_window,
        side_window_m=args.side_window_m,
        side_frame_window=args.side_frame_window,
        huber_delta_m=args.huber_delta_m,
        max_correction_m=args.max_correction_m,
        max_gap_connect_m=args.max_gap_m,
        max_frame_gap=args.max_frame_gap,
        max_heading_deg=args.max_heading_deg,
        min_line_support=args.min_line_support,
        line_sample_spacing_m=args.line_sample_spacing_m,
        spline_smoothing=args.spline_smoothing,
    )
    summary = linearize_fence_points(args.points, args.output_dir, args.trajectory, config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
