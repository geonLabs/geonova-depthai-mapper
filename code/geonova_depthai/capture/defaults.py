"""Default settings for field capture.

The defaults intentionally match the user's latest field command where possible:

    python synced_image_recorder.py \
      --output-dir ../data --rgb-format jpg --rotate-180 \
      --sync-threshold-ms 50 --fps 15 --gps-max-hz 120 --external-imu-max-hz 120

The new recorder stores unsynchronised per-stream event manifests first, then a
post-process step builds the synchronized timestamps.csv used for analysis.
Images are still compressed on disk; RGB defaults to JPG quality 100.
"""

WIDTH = 1280
HEIGHT = 720

DEFAULTS = {
    "output_dir": "../data",
    "fps": 15.0,
    "depth_fps": 0.0,
    "rgb_width": 0,
    "rgb_height": 0,
    "sync_threshold_ms": 50.0,
    "queue_size": 16,
    "writer_threads": 4,
    "max_runtime_s": 0.0,
    "controller_bridge_enabled": True,
    "controller_bridge_dir": "/var/lib/jetson-sensors",
    "controller_status_interval_s": 1.0,
    "controller_sensor_stale_after_s": 3.0,
    "controller_preview_fps": 4.0,
    "controller_preview_max_width": 1280,
    "controller_preview_jpeg_quality": 78,
    "rgb_format": "jpg",
    "rgb_jpeg_quality": 100,
    "rgb_png_compression": 1,
    "depth_png_compression": 0,
    "confidence_png_compression": 0,
    "save_confidence_map": False,
    "rotate_180": True,
    "flip": False,
    "rgb_undistort": True,
    # DepthAI v3 uses different RGB-D alignment paths by platform:
    # RVC2/RVC3 -> StereoDepth.inputAlignTo, RVC4 -> ImageAlign.
    "depth_alignment_mode": "auto",
    "sync_mode": "host",
    "sync_attempts": 0,
    "depth_preset": "DEFAULT",
    "lr_check": True,
    "subpixel": True,
    "subpixel_fractional_bits": 3,
    "stereo_median_filter": "7x7",
    "imu_rate": 200,
    "imu_batch": 10,
    "rgb_transport": "auto",
    "rgb_transport_quality": 100,
    "confidence_transport": "auto",
    "confidence_transport_quality": 100,
    "confidence_match_threshold_ms": 50.0,
    "allow_usb2": False,
    "usb3_retries": 2,
    "enable_gps": True,
    "gps_device": "/dev/ttyACM0",
    "gps_baudrate": 921600,
    "gps_max_hz": 120.0,
    "enable_external_imu": True,
    "external_imu_device": "/dev/ttyUSB0",
    "external_imu_baudrate": 921600,
    "external_imu_format": "ebimu",
    "external_imu_max_hz": 120.0,
    "serial_max_hz": 120.0,
    # Public connection defaults. Credentials must come from environment
    # variables or an ignored *.local.yaml file.
    "rtk_ntrip_host": "www.gnssdata.or.kr",
    "rtk_ntrip_port": 2101,
    "rtk_ntrip_mountpoint": "YANJ-RTCM31",
    "rtk_ntrip_auto_mountpoint": True,
    "rtk_ntrip_mountpoint_format": "RTCM31",
    "rtk_ntrip_mountpoint_candidates": (
        "GANS-RTCM31,GUMC-RTCM31,DBON-RTCM31,PAJU-RTCM31,YONS-RTCM31,"
        "SOUL-RTCM31,INCH-RTCM31,ICOR-RTCM31,SONP-RTCM31,OJBU-RTCM31,"
        "PJMS-RTCM31,YANJ-RTCM31,DOND-RTCM31,NAMY-RTCM31,GANH-RTCM31,"
        "SWGS-RTCM31,SUWN-RTCM31,POCN-RTCM31,YANP-RTCM31,YEOJ-RTCM31"
    ),
    "rtk_ntrip_username": "",
    "rtk_ntrip_password": "",
    "rtk_ntrip_gga": "",
    "rtk_ntrip_gga_interval": 10.0,
    "rtk_ntrip_reconnect_delay": 5.0,
    "rtk_ntrip_position_wait_s": 10.0,
    "rtk_ntrip_connect_timeout_s": 10.0,
    "rtk_ntrip_data_timeout_s": 15.0,
    "rtk_ntrip_sourcetable_timeout_s": 5.0,
    "rtk_ntrip_max_mountpoints": 12,
    "rtk_initial_latitude_deg": None,
    "rtk_initial_longitude_deg": None,
    "rtk_initial_altitude_m": 0.0,
}
