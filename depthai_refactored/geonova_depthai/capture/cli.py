import argparse
import os
from types import SimpleNamespace

from .defaults import DEFAULTS


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def nonnegative_float(value):
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0.")
    return number


def jpeg_quality(value):
    number = int(value)
    if number < 1 or number > 100:
        raise argparse.ArgumentTypeError("JPEG quality must be 1..100")
    return number


def png_compression_level(value):
    number = int(value)
    if number < 0 or number > 9:
        raise argparse.ArgumentTypeError("PNG compression must be 0..9")
    return number


def build_parser() -> argparse.ArgumentParser:
    d = DEFAULTS
    parser = argparse.ArgumentParser(
        description=(
            "Fast per-stream DepthAI RGB-D/GPS/IMU recorder. Images are compressed "
            "on disk; synchronization is built afterwards by build_synced_dataset.py."
        )
    )
    parser.add_argument("--output-dir", default=d["output_dir"])
    parser.add_argument("--fps", type=float, default=d["fps"])
    parser.add_argument("--sync-threshold-ms", type=nonnegative_float, default=d["sync_threshold_ms"], help="Post-process pairing window; saved in metadata")
    parser.add_argument("--queue-size", type=int, default=d["queue_size"])
    parser.add_argument("--writer-threads", type=int, default=d["writer_threads"], help="Background image encoder/writer workers")
    parser.add_argument("--max-runtime-s", type=nonnegative_float, default=d["max_runtime_s"], help="0 means record until Ctrl-C")

    parser.add_argument("--rgb-format", choices=["jpg", "png"], default=d["rgb_format"])
    parser.add_argument("--rgb-jpeg-quality", "--jpg-quality", dest="rgb_jpeg_quality", type=jpeg_quality, default=d["rgb_jpeg_quality"])
    parser.add_argument("--rgb-png-compression", type=png_compression_level, default=d["rgb_png_compression"])
    parser.add_argument("--depth-png-compression", type=png_compression_level, default=d["depth_png_compression"])
    parser.add_argument("--confidence-png-compression", type=png_compression_level, default=d["confidence_png_compression"])
    parser.add_argument("--save-confidence-map", dest="save_confidence_map", action="store_true", default=d["save_confidence_map"], help="Optional debugging output; OFF by default in raw-event capture")
    parser.add_argument("--no-save-confidence-map", dest="save_confidence_map", action="store_false")
    parser.add_argument("--rotate-180", dest="rotate_180", action="store_true", default=d["rotate_180"])
    parser.add_argument("--no-rotate-180", dest="rotate_180", action="store_false")
    parser.add_argument("--flip", type=str2bool, nargs="?", const=True, default=d["flip"])
    # Production RGB-D capture always uses factory-undistorted RGB. Allowing a
    # distorted RGB stream here produces a different pixel geometry from depth.
    parser.set_defaults(rgb_undistort=True)

    parser.add_argument(
        "--depth-alignment-mode",
        choices=["auto", "image-align", "stereo"],
        default=d["depth_alignment_mode"],
        help=(
            "auto selects the supported DepthAI v3 path: StereoDepth.inputAlignTo "
            "on RVC2/RVC3 and ImageAlign on RVC4"
        ),
    )
    parser.add_argument("--depth-preset", default=d["depth_preset"])
    parser.add_argument("--lr-check", type=str2bool, nargs="?", const=True, default=d["lr_check"])
    parser.add_argument("--subpixel", type=str2bool, nargs="?", const=True, default=d["subpixel"])
    parser.add_argument("--no-subpixel", dest="subpixel", action="store_false")
    parser.add_argument("--subpixel-fractional-bits", type=int, choices=[3, 4, 5], default=d["subpixel_fractional_bits"])
    parser.add_argument("--stereo-median-filter", choices=["off", "3x3", "5x5", "7x7"], default=d["stereo_median_filter"])
    parser.add_argument("--imu-rate", type=int, default=d["imu_rate"])
    parser.add_argument("--imu-batch", type=int, default=d["imu_batch"])

    parser.add_argument("--rgb-transport", choices=["auto", "raw", "mjpeg"], default=d["rgb_transport"])
    parser.add_argument("--rgb-transport-quality", type=jpeg_quality, default=d["rgb_transport_quality"])
    parser.add_argument("--confidence-transport", choices=["auto", "raw", "mjpeg"], default=d["confidence_transport"])
    parser.add_argument("--confidence-transport-quality", type=jpeg_quality, default=d["confidence_transport_quality"])
    parser.add_argument("--confidence-match-threshold-ms", type=nonnegative_float, default=d["confidence_match_threshold_ms"])
    parser.add_argument("--allow-usb2", action="store_true", default=d["allow_usb2"])
    parser.add_argument("--usb3-retries", type=int, default=d["usb3_retries"])

    parser.add_argument("--gps-device", default=d["gps_device"])
    parser.add_argument("--gps-baudrate", type=int, default=d["gps_baudrate"])
    parser.add_argument("--gps-max-hz", type=nonnegative_float, default=d["gps_max_hz"])
    parser.add_argument("--no-gps", dest="enable_gps", action="store_false", default=d["enable_gps"])
    parser.add_argument("--external-imu-device", default=d["external_imu_device"])
    parser.add_argument("--external-imu-baudrate", type=int, default=d["external_imu_baudrate"])
    parser.add_argument("--external-imu-format", choices=["ebimu", "raw"], default=d["external_imu_format"])
    parser.add_argument("--external-imu-max-hz", type=nonnegative_float, default=d["external_imu_max_hz"])
    parser.add_argument("--serial-max-hz", type=nonnegative_float, default=d["serial_max_hz"])
    parser.add_argument("--no-external-imu", dest="enable_external_imu", action="store_false", default=d["enable_external_imu"])

    parser.add_argument("--rtk-ntrip-host", default=os.getenv("NTRIP_HOST", d["rtk_ntrip_host"]))
    parser.add_argument("--rtk-ntrip-port", type=int, default=int(os.getenv("NTRIP_PORT", d["rtk_ntrip_port"])))
    parser.add_argument("--rtk-ntrip-mountpoint", default=os.getenv("NTRIP_MOUNTPOINT", d["rtk_ntrip_mountpoint"]))
    parser.add_argument("--rtk-ntrip-username", default=os.getenv("NTRIP_USERNAME", d["rtk_ntrip_username"]))
    parser.add_argument("--rtk-ntrip-password", default=os.getenv("NTRIP_PASSWORD", d["rtk_ntrip_password"]))
    parser.add_argument("--rtk-ntrip-gga", default=d["rtk_ntrip_gga"])
    parser.add_argument("--rtk-ntrip-gga-interval", type=nonnegative_float, default=d["rtk_ntrip_gga_interval"])
    parser.add_argument("--rtk-ntrip-reconnect-delay", type=nonnegative_float, default=d["rtk_ntrip_reconnect_delay"])
    parser.add_argument("--rtk-initial-latitude-deg", type=float, default=d["rtk_initial_latitude_deg"])
    parser.add_argument("--rtk-initial-longitude-deg", type=float, default=d["rtk_initial_longitude_deg"])
    parser.add_argument("--rtk-initial-altitude-m", type=float, default=d["rtk_initial_altitude_m"])

    return parser


def apply_legacy_compat_defaults(args):
    """Fill fields expected by legacy pipeline helpers."""
    values = vars(args)
    for key, value in DEFAULTS.items():
        values.setdefault(key, value)
    values["sync_mode"] = "host"
    values.setdefault("sync_attempts", 0)
    values.setdefault("usb_speed", "UNKNOWN")
    values.setdefault("rgb_transport_effective", values.get("rgb_transport", "auto"))
    values.setdefault("confidence_transport_effective", values.get("confidence_transport", "auto"))
    return args


def parse_args(argv=None):
    parser = build_parser()
    return apply_legacy_compat_defaults(parser.parse_args(argv))
