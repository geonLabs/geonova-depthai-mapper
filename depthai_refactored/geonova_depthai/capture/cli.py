import argparse
import os
from types import SimpleNamespace

from .defaults import DEFAULTS
from ..config_cli import SafeDefaultsHelpFormatter, parse_args_with_yaml


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
        ),
        formatter_class=SafeDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default=d["output_dir"], help="Parent directory for timestamped raw datasets")
    parser.add_argument("--fps", type=float, default=d["fps"], help="Requested RGB-D frame rate in frames per second")
    parser.add_argument("--depth-fps", type=nonnegative_float, default=d["depth_fps"], help="Requested stereo mono/depth frame rate; 0 follows --fps")
    parser.add_argument("--rgb-width", type=int, default=d["rgb_width"], help="RGB output width; 0 selects automatically from the connected color sensor")
    parser.add_argument("--rgb-height", type=int, default=d["rgb_height"], help="RGB output height; 0 selects automatically from the connected color sensor")
    parser.add_argument("--sync-threshold-ms", type=nonnegative_float, default=d["sync_threshold_ms"], help="Post-process pairing window; saved in metadata")
    parser.add_argument("--queue-size", type=int, default=d["queue_size"], help="Maximum pending writer tasks before capture backpressure")
    parser.add_argument("--writer-threads", type=int, default=d["writer_threads"], help="Background image encoder/writer workers")
    parser.add_argument("--max-runtime-s", type=nonnegative_float, default=d["max_runtime_s"], help="0 means record until Ctrl-C")
    parser.add_argument("--controller-bridge-enabled", type=str2bool, nargs="?", const=True, default=d["controller_bridge_enabled"], help="Publish live sensor status and camera preview for Jetson Controller")
    parser.add_argument("--no-controller-bridge", dest="controller_bridge_enabled", action="store_false", help="Disable the Jetson Controller live bridge")
    parser.add_argument("--controller-bridge-dir", default=d["controller_bridge_dir"], help="Directory for controller status.json and camera-preview.jpg")
    parser.add_argument("--controller-status-interval-s", type=nonnegative_float, default=d["controller_status_interval_s"], help="Seconds between controller status updates")
    parser.add_argument("--controller-sensor-stale-after-s", type=nonnegative_float, default=d["controller_sensor_stale_after_s"], help="Seconds without samples before a sensor is inactive")
    parser.add_argument("--controller-preview-fps", type=nonnegative_float, default=d["controller_preview_fps"], help="Maximum controller camera preview frame rate; 0 disables preview")
    parser.add_argument("--controller-preview-max-width", type=int, default=d["controller_preview_max_width"], help="Maximum controller preview width in pixels")
    parser.add_argument("--controller-preview-jpeg-quality", type=jpeg_quality, default=d["controller_preview_jpeg_quality"], help="Controller preview JPEG quality")

    parser.add_argument("--rgb-format", choices=["jpg", "png"], default=d["rgb_format"], help="RGB image file format")
    parser.add_argument("--rgb-jpeg-quality", "--jpg-quality", dest="rgb_jpeg_quality", type=jpeg_quality, default=d["rgb_jpeg_quality"], help="On-disk JPEG quality from 1 to 100")
    parser.add_argument("--rgb-png-compression", type=png_compression_level, default=d["rgb_png_compression"], help="RGB PNG compression level from 0 to 9")
    parser.add_argument("--depth-png-compression", type=png_compression_level, default=d["depth_png_compression"], help="16-bit Depth PNG compression level from 0 to 9")
    parser.add_argument("--confidence-png-compression", type=png_compression_level, default=d["confidence_png_compression"], help="Confidence-map PNG compression level from 0 to 9")
    parser.add_argument("--save-confidence-map", dest="save_confidence_map", action="store_true", default=d["save_confidence_map"], help="Optional debugging output; OFF by default in raw-event capture")
    parser.add_argument("--no-save-confidence-map", dest="save_confidence_map", action="store_false", help="Disable confidence-map recording")
    parser.add_argument("--rotate-180", dest="rotate_180", action="store_true", default=d["rotate_180"], help="Rotate all saved image geometry by 180 degrees")
    parser.add_argument("--no-rotate-180", dest="rotate_180", action="store_false", help="Keep the camera's native image orientation")
    parser.add_argument("--flip", type=str2bool, nargs="?", const=True, default=d["flip"], help="Vertically flip saved RGB-D image geometry")
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
    parser.add_argument("--depth-preset", default=d["depth_preset"], help="DepthAI StereoDepth preset name")
    parser.add_argument("--lr-check", type=str2bool, nargs="?", const=True, default=d["lr_check"], help="Enable stereo left-right consistency checking")
    parser.add_argument("--subpixel", type=str2bool, nargs="?", const=True, default=d["subpixel"], help="Enable subpixel disparity for long-range precision")
    parser.add_argument("--no-subpixel", dest="subpixel", action="store_false", help="Disable subpixel disparity")
    parser.add_argument("--subpixel-fractional-bits", type=int, choices=[3, 4, 5], default=d["subpixel_fractional_bits"], help="Fractional disparity precision bits")
    parser.add_argument("--stereo-median-filter", choices=["off", "3x3", "5x5", "7x7"], default=d["stereo_median_filter"], help="Stereo disparity median-filter kernel")
    parser.add_argument("--imu-rate", type=int, default=d["imu_rate"], help="OAK IMU sampling rate in Hz")
    parser.add_argument("--imu-batch", type=int, default=d["imu_batch"], help="Maximum IMU reports batched per device message")

    parser.add_argument("--rgb-transport", choices=["auto", "raw", "mjpeg"], default=d["rgb_transport"], help="Device-to-host RGB transport encoding")
    parser.add_argument("--rgb-transport-quality", type=jpeg_quality, default=d["rgb_transport_quality"], help="MJPEG transport quality")
    parser.add_argument("--confidence-transport", choices=["auto", "raw", "mjpeg"], default=d["confidence_transport"], help="Device-to-host confidence transport encoding")
    parser.add_argument("--confidence-transport-quality", type=jpeg_quality, default=d["confidence_transport_quality"], help="Confidence MJPEG transport quality")
    parser.add_argument("--confidence-match-threshold-ms", type=nonnegative_float, default=d["confidence_match_threshold_ms"], help="Maximum confidence-to-Depth timestamp difference in milliseconds")
    parser.add_argument("--allow-usb2", action="store_true", default=d["allow_usb2"], help="Allow reduced-bandwidth USB2 operation")
    parser.add_argument("--usb3-retries", type=int, default=d["usb3_retries"], help="USB3 connection retries before failure")

    parser.add_argument("--gps-device", default=d["gps_device"], help="GPS serial device path or Windows COM port")
    parser.add_argument("--gps-baudrate", type=int, default=d["gps_baudrate"], help="GPS serial baud rate")
    parser.add_argument("--gps-max-hz", type=nonnegative_float, default=d["gps_max_hz"], help="Maximum GPS rows written per second; 0 keeps all")
    parser.add_argument("--no-gps", dest="enable_gps", action="store_false", default=d["enable_gps"], help="Disable GPS and NTRIP acquisition")
    parser.add_argument("--external-imu-device", default=d["external_imu_device"], help="External IMU serial device path or COM port")
    parser.add_argument("--external-imu-baudrate", type=int, default=d["external_imu_baudrate"], help="External IMU serial baud rate")
    parser.add_argument("--external-imu-format", choices=["ebimu", "raw"], default=d["external_imu_format"], help="External IMU line parser")
    parser.add_argument("--external-imu-max-hz", type=nonnegative_float, default=d["external_imu_max_hz"], help="Maximum external-IMU rows written per second")
    parser.add_argument("--serial-max-hz", type=nonnegative_float, default=d["serial_max_hz"], help="Legacy shared serial write-rate limit")
    parser.add_argument("--no-external-imu", dest="enable_external_imu", action="store_false", default=d["enable_external_imu"], help="Disable external IMU acquisition")

    parser.add_argument("--rtk-ntrip-host", default=os.getenv("NTRIP_HOST", d["rtk_ntrip_host"]), help="NTRIP caster hostname")
    parser.add_argument("--rtk-ntrip-port", type=int, default=int(os.getenv("NTRIP_PORT", d["rtk_ntrip_port"])), help="NTRIP caster TCP port")
    parser.add_argument("--rtk-ntrip-mountpoint", default=os.getenv("NTRIP_MOUNTPOINT", d["rtk_ntrip_mountpoint"]), help="NTRIP correction mountpoint")
    parser.add_argument("--rtk-ntrip-auto-mountpoint", type=str2bool, nargs="?", const=True, default=d["rtk_ntrip_auto_mountpoint"], help="Choose the nearest NTRIP mountpoint from the caster source table after GPS position is known")
    parser.add_argument("--rtk-ntrip-mountpoint-format", default=d["rtk_ntrip_mountpoint_format"], help="Mountpoint suffix/format to auto-select, such as RTCM31")
    parser.add_argument("--rtk-ntrip-mountpoint-candidates", default=d["rtk_ntrip_mountpoint_candidates"], help="Comma-separated fallback mountpoints to try when auto-selection has no position/source table")
    parser.add_argument("--rtk-ntrip-username", default=os.getenv("NTRIP_USERNAME", d["rtk_ntrip_username"]), help="NTRIP username; prefer NTRIP_USERNAME environment variable")
    parser.add_argument("--rtk-ntrip-password", default=os.getenv("NTRIP_PASSWORD", d["rtk_ntrip_password"]), help="NTRIP password; prefer NTRIP_PASSWORD environment variable")
    parser.add_argument("--rtk-ntrip-gga", default=d["rtk_ntrip_gga"], help="Fixed GGA sentence; empty uses the latest receiver position")
    parser.add_argument("--rtk-ntrip-gga-interval", type=nonnegative_float, default=d["rtk_ntrip_gga_interval"], help="Seconds between GGA updates sent to the caster")
    parser.add_argument("--rtk-ntrip-reconnect-delay", type=nonnegative_float, default=d["rtk_ntrip_reconnect_delay"], help="Seconds before reconnecting after NTRIP failure")
    parser.add_argument("--rtk-ntrip-position-wait-s", type=nonnegative_float, default=d["rtk_ntrip_position_wait_s"], help="Seconds to wait for an initial GPS position before choosing a mountpoint")
    parser.add_argument("--rtk-ntrip-connect-timeout-s", type=nonnegative_float, default=d["rtk_ntrip_connect_timeout_s"], help="Seconds before giving up on an NTRIP TCP/header connection")
    parser.add_argument("--rtk-ntrip-data-timeout-s", type=nonnegative_float, default=d["rtk_ntrip_data_timeout_s"], help="Seconds without RTCM data before switching to the next mountpoint; 0 disables")
    parser.add_argument("--rtk-ntrip-sourcetable-timeout-s", type=nonnegative_float, default=d["rtk_ntrip_sourcetable_timeout_s"], help="Seconds before giving up on the NTRIP source table request")
    parser.add_argument("--rtk-ntrip-max-mountpoints", type=int, default=d["rtk_ntrip_max_mountpoints"], help="Maximum auto-selected mountpoints to try per reconnect cycle; 0 keeps all")
    parser.add_argument("--rtk-initial-latitude-deg", type=float, default=d["rtk_initial_latitude_deg"], help="Fallback latitude used before the receiver reports a position")
    parser.add_argument("--rtk-initial-longitude-deg", type=float, default=d["rtk_initial_longitude_deg"], help="Fallback longitude used before the receiver reports a position")
    parser.add_argument("--rtk-initial-altitude-m", type=float, default=d["rtk_initial_altitude_m"], help="Fallback ellipsoidal altitude in meters")

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
    return apply_legacy_compat_defaults(parse_args_with_yaml(parser, argv))
