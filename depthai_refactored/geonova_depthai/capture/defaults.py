"""Default settings for field capture.

The defaults intentionally match the user's latest field command where possible:

    python synced_image_recorder.py \
      --output-dir image_records --rgb-format jpg --rotate-180 \
      --sync-threshold-ms 50 --fps 15 --gps-max-hz 120 --external-imu-max-hz 120

The new recorder stores unsynchronised per-stream event manifests first, then a
post-process step builds the synchronized timestamps.csv used for analysis.
Images are still compressed on disk; RGB defaults to JPG quality 100.
"""

WIDTH = 1280
HEIGHT = 720

DEFAULTS = {
    "output_dir": "image_records",
    "fps": 15.0,
    "sync_threshold_ms": 50.0,
    "queue_size": 16,
    "writer_threads": 4,
    "max_runtime_s": 0.0,
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
    # Field defaults copied from the original recorder as requested. Environment
    # variables and CLI arguments in cli.py can still override every value.
    "rtk_ntrip_host": "www.gnssdata.or.kr",
    "rtk_ntrip_port": 2101,
    "rtk_ntrip_mountpoint": "YANJ-RTCM31",
    "rtk_ntrip_username": "pjmsm0319@gmail.com",
    "rtk_ntrip_password": "gnss",
    "rtk_ntrip_gga": "",
    "rtk_ntrip_gga_interval": 10.0,
    "rtk_ntrip_reconnect_delay": 5.0,
    "rtk_initial_latitude_deg": None,
    "rtk_initial_longitude_deg": None,
    "rtk_initial_altitude_m": 0.0,
}
