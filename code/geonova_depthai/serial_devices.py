"""Safe, deterministic serial sensor discovery without camera dependencies."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from serial.tools import list_ports as serial_list_ports
except ImportError:  # pragma: no cover - pyserial normally provides this module.
    serial_list_ports = None


AUTO_SERIAL_DEVICE_VALUES = {"", "auto", "default"}
SERIAL_IDENTITY_TOKENS = {
    "gps": ("u-blox", "ublox", "gnss receiver", "gnss_receiver", "gnss", "gps"),
    "external_imu": ("ebimu", "e2box", "10c4:ea60", "cp2102", "silicon labs", "silicon_labs", "imu"),
}
SERIAL_REJECT_TOKENS = {
    "gps": ("samsung", "android", "galaxy", "04e8:6860", "smartphone", "modem"),
    "external_imu": ("samsung", "android", "04e8:6860", "u-blox", "ublox", "gnss", "gps"),
}


def _device_key(device):
    if device in (None, ""):
        return ""
    return os.path.realpath(os.fspath(device))


def _by_id_entries(by_id_dir):
    root = Path(by_id_dir)
    try:
        return sorted(
            (entry for entry in root.iterdir() if entry.is_symlink() and entry.exists()),
            key=lambda entry: entry.name.casefold(),
        )
    except OSError:
        return []


def _port_infos():
    if serial_list_ports is None:
        return []
    try:
        return sorted(serial_list_ports.comports(), key=lambda item: str(item.device).casefold())
    except Exception:
        return []


def _port_text(port):
    return " ".join(
        str(getattr(port, field, "") or "")
        for field in ("device", "name", "description", "hwid", "manufacturer", "product", "serial_number")
    ).casefold()


def _identity_score(role, text):
    normalized = str(text or "").replace("-", " ").casefold()
    if any(token.replace("-", " ") in normalized for token in SERIAL_REJECT_TOKENS[role]):
        return None
    scores = [
        100 - index
        for index, token in enumerate(SERIAL_IDENTITY_TOKENS[role])
        if token.replace("-", " ") in normalized
    ]
    return max(scores) if scores else 0


def _identity_text(device, by_id_entries, port_infos):
    key = _device_key(device)
    aliases = [entry.name for entry in by_id_entries if _device_key(entry) == key]
    matching_ports = [
        _port_text(port)
        for port in port_infos
        if _device_key(getattr(port, "device", "")) == key
    ]
    # Only the basename is an identity. Parent directories are deployment
    # details and can contain misleading words.
    return " ".join([Path(str(device)).name, *aliases, *matching_ports])


def _one_best_candidate(candidates, role, source):
    if not candidates:
        return None
    candidates.sort()
    best_score = candidates[0][0]
    best = [candidate for candidate in candidates if candidate[0] == best_score]
    if len({_device_key(candidate[2]) for candidate in best}) == 1:
        return best[0][2]
    print(f"{role} serial {source} discovery is ambiguous: {len(best)} matching identities")
    return None


def resolve_serial_device(
    requested,
    role,
    by_id_dir="/dev/serial/by-id",
    port_infos=None,
    exclude_devices=None,
):
    """Resolve a role to a stable sensor identity, or fail closed with ``None``.

    An unidentified numbered ttyACM node is never treated as GNSS because an
    Android phone can expose the same node type. Explicit non-ttyACM paths and
    Windows COM ports are preserved unless their identity is rejected.
    """
    if role not in SERIAL_IDENTITY_TOKENS:
        raise ValueError(f"Unknown serial device role: {role}")

    requested_text = str(requested or "").strip()
    entries = _by_id_entries(by_id_dir)
    ports = list(_port_infos() if port_infos is None else port_infos)
    excluded = {_device_key(device) for device in (exclude_devices or ()) if device}

    is_auto = requested_text.casefold() in AUTO_SERIAL_DEVICE_VALUES
    if not is_auto:
        requested_key = _device_key(requested_text)
        score = _identity_score(role, _identity_text(requested_text, entries, ports))
        numbered_acm = Path(requested_text).name.startswith("ttyACM")
        rejected = score is None or requested_key in excluded
        unverified_gnss_acm = role == "gps" and numbered_acm and not score
        if not rejected and not unverified_gnss_acm:
            return requested_text
        reason = "unsafe identity" if score is None or unverified_gnss_acm else "already assigned"
        print(f"Ignoring {role} serial device {requested_text}: {reason}; trying stable auto discovery")

    ranked = []
    for entry in entries:
        target_key = _device_key(entry)
        if target_key in excluded:
            continue
        score = _identity_score(role, _identity_text(entry, entries, ports))
        if score is None:
            continue
        if score:
            ranked.append((-score, entry.name.casefold(), str(entry)))
    resolved = _one_best_candidate(ranked, role, "auto")
    if resolved:
        return resolved
    if ranked:
        return None

    # Linux numbered tty paths are enumeration-order dependent. If stable by-id
    # discovery failed, leave the sensor unavailable instead of guessing. Other
    # platforms (notably Windows) can safely use identity-checked COM metadata.
    if os.name == "posix":
        return None

    port_candidates = []
    for port in ports:
        device = str(getattr(port, "device", "") or "")
        if not device or _device_key(device) in excluded:
            continue
        score = _identity_score(role, _port_text(port))
        if score:
            port_candidates.append((-score, device.casefold(), device))
    return _one_best_candidate(port_candidates, role, "fallback")


def resolve_serial_devices(args, by_id_dir="/dev/serial/by-id", port_infos=None):
    """Resolve enabled recorder roles in-place and prevent cross-role reuse."""
    used = set()
    if getattr(args, "enable_gps", False):
        args.gps_device = resolve_serial_device(
            getattr(args, "gps_device", "auto"),
            "gps",
            by_id_dir=by_id_dir,
            port_infos=port_infos,
        )
        if args.gps_device:
            used.add(args.gps_device)
            print(f"Resolved gps serial device: {args.gps_device}")
        else:
            print("GPS serial device unavailable: no safe GNSS identity found")
    if getattr(args, "enable_external_imu", False):
        args.external_imu_device = resolve_serial_device(
            getattr(args, "external_imu_device", "auto"),
            "external_imu",
            by_id_dir=by_id_dir,
            port_infos=port_infos,
            exclude_devices=used,
        )
        if args.external_imu_device:
            print(f"Resolved external_imu serial device: {args.external_imu_device}")
        else:
            print("External IMU serial device unavailable: no safe IMU identity found")
    return args
