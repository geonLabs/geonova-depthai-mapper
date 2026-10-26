"""Pure resolution policy tests; also runnable with stdlib unittest on Windows."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from geonova_depthai.camera_resolution import (
    build_resolution_metadata, describe_camera, estimate_stream_mb_s,
    resolution_limits, select_rgb_size, select_stereo_size,
)


def config(width, height, fps=120, kind="COLOR", fov=0):
    return NS(width=width, height=height, minFps=1, maxFps=fps, type=kind, fov=fov)


def feature(name="OV9782", socket="CAM_A", width=1280, height=800, configs=None):
    return NS(sensorName=name, socket=socket, width=width, height=height,
              supportedTypes=["COLOR", "MONO"], configs=configs or [])


def geo_cameras():
    rgb = feature(configs=[config(1280, 800), config(1280, 720)])
    mono = [config(1280, 800, kind="MONO"), config(1280, 720, kind="MONO"),
            config(640, 400, kind="MONO")]
    return rgb, feature("OV9282", "CAM_B", configs=mono), feature("OV9282", "CAM_C", configs=mono)


def lr_camera(socket="CAM_A"):
    return feature("AR0234", socket, 1920, 1200,
                   [config(1920, 1200), config(1920, 1080), config(1280, 800)])


class CameraResolutionTests(unittest.TestCase):
    def size(self, result):
        return result["width"], result["height"]

    def test_geo_x1_rgb_stereo_and_stream_rate(self):
        rgb, left, right = geo_cameras()
        result = select_rgb_size(rgb, fps=15, usb_speed="SUPER")
        self.assertEqual(self.size(result), (1280, 800))
        self.assertEqual(result["source"], "sensor_max")
        self.assertEqual(self.size(select_stereo_size(left, right)), (1280, 800))
        self.assertAlmostEqual(result["estimated_stream_mb_s"], 53.76)
        self.assertEqual(result["warnings"], [])

    def test_oak_d_lr_15fps_and_color_only_stereo(self):
        result = select_rgb_size(lr_camera(), fps=15, usb_speed="SUPER")
        self.assertEqual(self.size(result), (1920, 1200))
        self.assertEqual(result["source"], "sensor_max")
        self.assertAlmostEqual(result["estimated_stream_mb_s"], 120.96)
        stereo = select_stereo_size(lr_camera("CAM_B"), lr_camera("CAM_C"))
        self.assertEqual(self.size(stereo), (1280, 800))
        self.assertTrue(any("rvc2_stereo_width" in str(c) for c in stereo["rejected_candidates"]))

    def test_oak_d_lr_30fps_scales_to_exact_budget_size(self):
        result = select_rgb_size(lr_camera(), fps=30, usb_speed="SUPER")
        self.assertEqual(self.size(result), (1504, 944))
        self.assertEqual(result["source"], "sensor_max_scaled_to_limit")
        self.assertAlmostEqual(result["estimated_stream_mb_s"], 149.07648)
        self.assertTrue(any("stream_budget_mb_s" in str(c) for c in result["rejected_candidates"]))

    def test_12mp_scaled_size_wins_over_cropped_1080p(self):
        camera = feature("12MP", width=4056, height=3040, configs=[
            config(4056, 3040, 30), config(3840, 2160, 30),
            config(1920, 1080, 60), config(1352, 1012, 60),
        ])
        result = select_rgb_size(camera, fps=15, usb_speed="SUPER")
        self.assertEqual(self.size(result), (1744, 1320))
        self.assertEqual(result["source"], "sensor_max_scaled_to_limit")
        self.assertLessEqual(abs(result["width"] / result["height"] / (4056 / 3040) - 1), .01)
        self.assertLessEqual(result["width"] * result["height"], 2304000)
        self.assertTrue(any("rgb_max_pixels" in str(c) for c in result["rejected_candidates"]))

    def test_usb_high_monitor_mjpeg_has_no_depth_cost(self):
        result = select_rgb_size(lr_camera(), fps=15, usb_speed="HIGH",
                                 depth_enabled=False, rgb_transport="mjpeg")
        self.assertEqual(self.size(result), (1920, 1200))
        self.assertAlmostEqual(result["estimated_stream_mb_s"], 12.96)
        self.assertEqual(result["limits"]["stream_budget_mb_s"], 30)
        geo = select_rgb_size(geo_cameras()[0], usb_speed="HIGH", depth_enabled=False, rgb_transport="mjpeg")
        self.assertEqual(self.size(geo), (1280, 800))
        self.assertAlmostEqual(geo["estimated_stream_mb_s"], 5.76)

    def test_high_raw_capture_is_scaled_to_30mb(self):
        result = select_rgb_size(geo_cameras()[0], fps=15, usb_speed="HIGH")
        self.assertLessEqual(result["estimated_stream_mb_s"], 30)
        self.assertEqual(result["source"], "sensor_max_scaled_to_limit")

    def test_fps_exclusion_and_type_preference(self):
        camera = feature(configs=[config(1280, 800, 10), config(1280, 720, 60),
                                  config(1280, 800, 120, "MONO")])
        result = select_rgb_size(camera, fps=30, usb_speed="SUPER")
        self.assertEqual(self.size(result), (1280, 720))
        self.assertEqual(result["source"], "sensor_config")
        self.assertTrue(any("max_fps" in str(c) for c in result["rejected_candidates"]))

    def test_scaled_mode_cannot_reintroduce_slow_sensor_max(self):
        camera = feature(width=4056, height=3040, configs=[config(4056, 3040, 10), config(1920, 1080, 60)])
        self.assertEqual(self.size(select_rgb_size(camera, fps=30, usb_speed="SUPER", depth_enabled=False)), (1920, 1080))

    def test_no_config_meets_fps_is_rejected(self):
        camera = feature(configs=[config(1280, 800, 10), config(1280, 720, 10)])
        with self.assertRaisesRegex(ValueError, "rgb.*CAM_A.*OV9782.*30 FPS.*10 FPS.*lower fps"):
            select_rgb_size(camera, fps=30, usb_speed="SUPER")

    def test_each_fps_compatible_mode_supplies_scaled_rgbd_candidates(self):
        for maximum_fps, fps in ((10, 30), (30, 30), (10, 60), (30, 60)):
            with self.subTest(maximum_fps=maximum_fps, fps=fps):
                camera = feature("12MP", width=4056, height=3040, configs=[
                    config(4056, 3040, maximum_fps, fov=100),
                    config(3840, 2160, 30, fov=80), config(1920, 1080, 60, fov=60),
                ])
                result = select_rgb_size(camera, fps=fps, usb_speed="SUPER", depth_enabled=True)
                expected = {(10, 30): (1584, 896), (30, 30): (1376, 1032),
                            (10, 60): (1120, 632), (30, 60): (1120, 632)}
                self.assertEqual(self.size(result), expected[(maximum_fps, fps)])
                self.assertLessEqual(result["estimated_stream_mb_s"], 150)
                self.assertEqual(result["policy"], "auto")
                scaled = [c for c in result["candidates"] if c.get("sensor_mode", {}).get("width") == 1920]
                self.assertEqual(len(scaled), 1)
                self.assertEqual(scaled[0]["fov"], 60)
                self.assertEqual(scaled[0]["rejected_reasons"], [])
                self.assertLessEqual(abs(scaled[0]["width"] / scaled[0]["height"] / (1920 / 1080) - 1), .01)
                if fps == 60:
                    self.assertEqual(result["sensor_mode"]["width"], 1920)

    def test_unusable_limits_remain_errors_instead_of_overrunning_budget(self):
        for options in ({"rgb_max_pixels": 1}, {"stream_budget_mb_s": .00001},
                        {"stream_budget_mb_s": 10, "save_confidence_map": True, "confidence_size": (1280, 800)}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "No camera resolution fits"):
                select_rgb_size(feature(), usb_speed="SUPER", **options)

    def test_stereo_fps_error_identifies_either_sensor(self):
        _, left, right = geo_cameras()
        for slow in (left, right):
            original = slow.configs
            slow.configs = [config(1280, 800, 10, "MONO")]
            message = "stereo.*{0}.*OV9282.*30 FPS.*10 FPS.*lower depth_fps".format(slow.socket)
            with self.assertRaisesRegex(ValueError, message):
                select_stereo_size(left, right, depth_fps=30)
            slow.configs = original

    def test_explicit_modes_warn_about_fps_without_rejection(self):
        camera = feature(configs=[config(1280, 800, 10)])
        rgb = select_rgb_size(camera, fps=30, width=1280, height=800, usb_speed="SUPER")
        self.assertEqual(self.size(rgb), (1280, 800))
        self.assertIn("rgb socket=CAM_A sensor=OV9782: requested 30 FPS, supported maximum 10 FPS; lower fps",
                      rgb["warnings"])
        camera.configs = [config(1280, 800, 10, "MONO")]
        stereo = select_stereo_size(camera, camera, depth_fps=30, width=1280, height=800)
        self.assertEqual(self.size(stereo), (1280, 800))
        self.assertIn("lower depth_fps", str(stereo["warnings"]))

    def test_explicit_scaled_sizes_check_all_capable_modes_for_fps_warnings(self):
        cases = [
            ([config(1280, 800, 15)], True),
            ([config(1280, 800, 15), config(320, 800, 60)], True),
            ([config(1280, 800, 15), config(1280, 240, 60)], True),
            ([config(1280, 800, 15), config(1920, 1200, 30)], False),
            ([config(640, 400, 15), config(1280, 800, 30)], False),
            ([config(1280, 800, 15), config(640, 400, 30)], False),
            ([config(1280, 800, 15), config(1920, 1200, 0)], False),
            ([config(1280, 800, 15), config(640, 400, 0)], False),
            ([config(1280, 800, 0)], False),
            ([config(320, 240, 60)], True),
            ([], False),
        ]
        for modes, warn in cases:
            for role in ("rgb", "stereo"):
                with self.subTest(modes=modes, role=role):
                    camera = feature(configs=[config(c.width, c.height, c.maxFps,
                                                    "COLOR" if role == "rgb" else "MONO") for c in modes])
                    if role == "rgb":
                        result = select_rgb_size(camera, fps=30, width=640, height=400, usb_speed="SUPER")
                    else:
                        result = select_stereo_size(camera, camera, depth_fps=30, width=640, height=400)
                    self.assertEqual(self.size(result), (640, 400))
                    self.assertEqual(result["source"], "configured")
                    self.assertEqual(bool(result["warnings"]), warn)
                    if warn:
                        self.assertIn("requested 30 FPS", str(result["warnings"]))
                        self.assertIn(role + " socket=CAM_A sensor=OV9782", str(result["warnings"]))

    def test_depth_alignment_grid_excludes_native_unaligned_modes(self):
        for width, height in ((1352, 1012), (1344, 1013)):
            camera = feature(width=width, height=height, configs=[config(width, height)])
            result = select_rgb_size(camera, usb_speed="SUPER")
            self.assertIn("alignment_grid", str(result["rejected_candidates"]))
            self.assertEqual(result["width"] % 16, 0)
            self.assertEqual(result["height"] % 8, 0)
            self.assertEqual(self.size(select_rgb_size(camera, usb_speed="SUPER", depth_enabled=False)),
                             (width, height))

    def test_zero_max_fps_is_unreported(self):
        result = select_rgb_size(feature(configs=[config(1280, 800, 0)]), fps=60, usb_speed="SUPER", depth_enabled=False)
        self.assertEqual(self.size(result), (1280, 800))
        self.assertEqual(result["warnings"], [])
        camera = feature(configs=[config(1280, 800, 0, "MONO")])
        self.assertEqual(self.size(select_stereo_size(camera, camera, depth_fps=60)), (1280, 800))

    def test_no_configs_uses_sensor_max(self):
        result = select_rgb_size(feature(), usb_speed="SUPER")
        self.assertEqual(self.size(result), (1280, 800))
        self.assertEqual(result["source"], "sensor_max")

    def test_unknown_sensor_safe_fallback(self):
        result = select_rgb_size(None, usb_speed="SUPER")
        self.assertEqual(self.size(result), (1280, 720))
        self.assertEqual(result["source"], "fallback_unknown_sensor")

    def test_unknown_stereo_sensors_use_safe_fallback(self):
        unknown = feature(width=0, height=0)
        for cameras in ((None, None), (unknown, unknown), (None, unknown), (unknown, None)):
            with self.subTest(cameras=cameras):
                result = select_stereo_size(*cameras)
                self.assertEqual(self.size(result), (1280, 800))
                self.assertEqual(result["source"], "fallback_unknown_sensor")
                self.assertEqual(result["policy"], "auto")
                self.assertEqual(result["warnings"], [])

    def test_unknown_stereo_sensor_does_not_upscale_known_smaller_sensor(self):
        known = feature(width=640, height=400)
        for cameras in ((None, known), (known, None)):
            with self.subTest(cameras=cameras):
                result = select_stereo_size(*cameras)
                self.assertEqual(self.size(result), (640, 400))
                self.assertEqual(result["source"], "fallback_unknown_sensor")

    def test_unknown_stereo_sensor_uses_known_fps_compatible_modes(self):
        for maximum in ((1280, 800), (0, 0)):
            known = feature(width=maximum[0], height=maximum[1], configs=[
                config(1280, 800, 15, "MONO"), config(640, 400, 60, "MONO"),
                config(320, 200, 60, "MONO"),
            ])
            for cameras in ((None, known), (known, None)):
                with self.subTest(maximum=maximum, cameras=cameras):
                    result = select_stereo_size(*cameras, depth_fps=30)
                    self.assertEqual(self.size(result), (640, 400))
                    self.assertEqual(result["source"], "fallback_unknown_sensor")
                    self.assertEqual(result["warnings"], [])
                    self.assertEqual([self.size(c) for c in result["rejected_candidates"]], [(1280, 800)])
                    self.assertIn("max_fps", str(result["rejected_candidates"]))

    def test_unknown_stereo_sensor_keeps_platform_width_limit(self):
        for maximum in ((1920, 1200), (0, 0)):
            known = feature(width=maximum[0], height=maximum[1], configs=[config(1920, 1200, 60, "MONO")])
            for cameras in ((None, known), (known, None)):
                for platform, size in (("RVC2", (1280, 800)), ("RVC4", (1920, 1200))):
                    with self.subTest(maximum=maximum, cameras=cameras, platform=platform):
                        result = select_stereo_size(*cameras, platform=platform, depth_fps=30)
                        self.assertEqual(self.size(result), size)
                        self.assertEqual(result["source"], "fallback_unknown_sensor")
                        if platform == "RVC2":
                            self.assertIn("rvc2_stereo_width", str(result["rejected_candidates"]))

    def test_unknown_stereo_sensor_does_not_hide_known_fps_error(self):
        for maximum in ((1280, 800), (0, 0)):
            slow = feature("OV9282", "CAM_B", *maximum, configs=[config(640, 400, 10, "MONO")])
            for unknown in (None, feature(width=0, height=0)):
                for cameras in ((unknown, slow), (slow, unknown)):
                    with self.subTest(maximum=maximum, cameras=cameras), self.assertRaisesRegex(
                            ValueError, "stereo.*CAM_B.*OV9282.*30 FPS.*10 FPS.*lower depth_fps"):
                        select_stereo_size(*cameras, depth_fps=30)

    def test_configured_rgb_keeps_size_and_reports_violations(self):
        result = select_rgb_size(feature(), width=1920, height=1200, rgb_max_pixels=1000000,
                                 stream_budget_mb_s=30, usb_speed="SUPER")
        self.assertEqual(self.size(result), (1920, 1200))
        self.assertEqual(result["source"], "configured")
        self.assertEqual(result["policy"], "configured")
        self.assertIn("rgb_max_pixels", str(result["warnings"]))
        self.assertIn("stream_budget_mb_s", str(result["warnings"]))
        self.assertIn("upscaling", str(result["warnings"]))

    def test_stereo_override_and_rvc2_limit(self):
        _, left, right = geo_cameras()
        result = select_stereo_size(left, right, width=640, height=400)
        self.assertEqual(self.size(result), (640, 400))
        self.assertEqual(result["source"], "configured")
        with self.assertRaisesRegex(ValueError, "RVC2"):
            select_stereo_size(left, right, width=1920, height=1200)
        self.assertEqual(self.size(select_stereo_size(lr_camera(), lr_camera(), platform="RVC4")), (1920, 1200))

    def test_partial_or_negative_sizes_are_rejected(self):
        for w, h in ((0, 800), (1280, 0), (-1, 800)):
            with self.subTest(width=w, height=h):
                with self.assertRaises(ValueError):
                    select_rgb_size(feature(), width=w, height=h)
                with self.assertRaises(ValueError):
                    select_stereo_size(feature(), feature(), width=w, height=h)
        for kwargs in ({"rgb_max_pixels": -1}, {"stream_budget_mb_s": -1}):
            with self.assertRaises(ValueError):
                select_rgb_size(feature(), **kwargs)

    def test_stereo_config_intersection_and_mono_preference(self):
        left = feature(configs=[config(1280, 800, 10, "MONO"), config(640, 400, 60, "MONO"), config(1280, 800, 120)])
        right = feature(configs=[config(1280, 720, 60, "MONO"), config(640, 400, 60, "MONO")])
        result = select_stereo_size(left, right, depth_fps=30)
        self.assertEqual(self.size(result), (640, 400))
        self.assertTrue(any("max_fps" in str(c) for c in result["rejected_candidates"]))

    def test_stereo_no_intersection_uses_largest_eligible_modes(self):
        left = feature(width=1280, height=800, configs=[config(1280, 800)])
        right = feature(width=960, height=600, configs=[config(640, 400)])
        result = select_stereo_size(left, right)
        self.assertEqual(self.size(result), (640, 400))
        self.assertEqual(result["source"], "fps_common_fallback")
        self.assertIn("No common FPS-compatible stereo config", str(result["warnings"]))

    def test_stereo_disjoint_fps_modes_do_not_reintroduce_slow_maximum(self):
        left = feature(socket="CAM_B", configs=[config(1280, 800, 15, "MONO"),
                                               config(640, 400, 60, "MONO")])
        right = feature(socket="CAM_C", configs=[config(1280, 800, 15, "MONO"),
                                                config(1280, 720, 60, "MONO")])
        for cameras in ((left, right), (right, left)):
            with self.subTest(left=cameras[0].socket):
                result = select_stereo_size(*cameras, depth_fps=30)
                self.assertEqual(self.size(result), (640, 400))
                self.assertEqual(result["source"], "fps_common_fallback")
                self.assertIn("No common FPS-compatible stereo config", str(result["warnings"]))
                self.assertIn("640x400", str(result["warnings"]))
                self.assertIn("30 FPS", str(result["warnings"]))
                self.assertEqual({c["socket"] for c in result["rejected_candidates"]}, {"CAM_B", "CAM_C"})

    def test_stereo_fallback_takes_component_minimum_of_largest_fps_modes(self):
        left = feature(width=1920, height=1200, configs=[config(1600, 800, 60, "MONO"),
                                                       config(1024, 768, 60, "MONO")])
        right = feature(width=1920, height=1200, configs=[config(1280, 1000, 60, "MONO"),
                                                        config(640, 480, 60, "MONO")])
        result = select_stereo_size(left, right, depth_fps=30)
        self.assertEqual(self.size(result), (1280, 800))
        self.assertEqual(result["source"], "fps_common_fallback")
        self.assertTrue(result["warnings"])

    def test_stereo_fallback_uses_sensor_maximum_when_configs_are_missing(self):
        left = feature(width=1280, height=800)
        for modes in (None, [config(960, 600, 0, "MONO")]):
            right = feature(width=960, height=600, configs=modes)
            for cameras in ((left, right), (right, left)):
                with self.subTest(modes=modes, left=cameras[0].width):
                    result = select_stereo_size(*cameras, depth_fps=30)
                    self.assertEqual(self.size(result), (960, 600))
                    self.assertEqual(result["source"], "fps_common_fallback")
                    self.assertTrue(result["warnings"])

    def test_stereo_fallback_keeps_rvc2_width_limit_and_source(self):
        left = feature(width=1920, height=1200, configs=[config(1920, 1200, 60, "MONO")])
        right = feature(width=1920, height=1200, configs=[config(1600, 1000, 60, "MONO")])
        for platform, size, source in (("RVC2", (1280, 800), "fps_common_fallback_scaled_to_platform_limit"),
                                       ("RVC4", (1600, 1000), "fps_common_fallback")):
            with self.subTest(platform=platform):
                result = select_stereo_size(left, right, platform=platform, depth_fps=30)
                self.assertEqual(self.size(result), size)
                self.assertEqual(result["source"], source)
                self.assertTrue(result["warnings"])

    def test_stereo_fallback_without_usable_modes_or_sensor_maximum_ignores_unknown_sensor(self):
        for unknown in (None, feature(width=0, height=0), feature(width=1280, height=0)):
            for cameras in ((unknown, feature(width=640, height=400)),
                            (feature(width=640, height=400), unknown)):
                with self.subTest(cameras=cameras):
                    result = select_stereo_size(*cameras, depth_fps=30)
                    self.assertEqual(self.size(result), (640, 400))
                    self.assertEqual(result["source"], "fallback_unknown_sensor")
                    self.assertEqual(result["warnings"], [])

    def test_estimate_includes_depth_confidence_and_separate_cadences(self):
        self.assertAlmostEqual(estimate_stream_mb_s(1280, 800, 15), 53.76)
        self.assertAlmostEqual(estimate_stream_mb_s(1280, 800, 15, 30, save_confidence_map=True), 115.2)
        self.assertAlmostEqual(estimate_stream_mb_s(1280, 800, 15, 30, rgb_transport="mjpeg",
                                                   save_confidence_map=True, confidence_transport="mjpeg"), 74.88)

    def test_confidence_uses_stereo_size_when_rgb_is_smaller(self):
        options = dict(fps=15, depth_fps=15, rgb_transport="mjpeg", save_confidence_map=True,
                       confidence_transport="raw", confidence_size=(1280, 800))
        self.assertAlmostEqual(estimate_stream_mb_s(640, 400, **options), 24.48)
        result = select_rgb_size(geo_cameras()[0], usb_speed="HIGH", **options)
        self.assertEqual(self.size(result), (800, 504))
        self.assertLess(result["width"] * result["height"], 1280 * 800)
        self.assertAlmostEqual(result["estimated_stream_mb_s"], estimate_stream_mb_s(*self.size(result), **options))
        self.assertLessEqual(result["estimated_stream_mb_s"], 30)
        self.assertGreater(estimate_stream_mb_s(result["width"] + 32, result["height"] + 16, **options), 30)

    def test_confidence_only_mjpeg_checks_frame_and_shared_encoder_limits(self):
        options = dict(fps=15, rgb_transport="raw", save_confidence_map=True,
                       confidence_transport="mjpeg", rgb_max_pixels=20000000, stream_budget_mb_s=10000)
        for size in ((1352, 1012), (16416, 800), (1280, 8200), (4000, 4000)):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "confidence_mjpeg|mjpeg_throughput"):
                select_rgb_size(feature(), confidence_size=size, depth_fps=30, **options)
        result = select_rgb_size(feature(width=4056, height=3040), depth_fps=300,
                                 confidence_size=(1280, 800), rgb_transport="mjpeg",
                                 **{k: v for k, v in options.items() if k != "rgb_transport"})
        self.assertLessEqual(result["width"] * result["height"] * 15 + 1280 * 800 * 300, 450000000)
        self.assertIn("mjpeg_throughput", str(result["rejected_candidates"]))

    def test_policy_rejects_nonfinite_or_invalid_frame_rates(self):
        for value in (0, -1, float("inf"), float("-inf"), float("nan")):
            with self.subTest(fps=value):
                with self.assertRaisesRegex(ValueError, "fps"):
                    select_rgb_size(feature(), fps=value)
                with self.assertRaisesRegex(ValueError, "fps"):
                    estimate_stream_mb_s(1280, 800, value)
        for value in (-1, float("inf"), float("nan")):
            with self.subTest(depth_fps=value):
                with self.assertRaisesRegex(ValueError, "depth_fps"):
                    select_rgb_size(feature(), depth_fps=value)
                with self.assertRaisesRegex(ValueError, "depth_fps"):
                    select_stereo_size(feature(), feature(), depth_fps=value)
                with self.assertRaisesRegex(ValueError, "depth_fps"):
                    estimate_stream_mb_s(1280, 800, 15, depth_fps=value)
        self.assertEqual(estimate_stream_mb_s(1280, 800, 15, depth_fps=0), 53.76)

    def test_policy_rejects_fractional_nonfinite_sizes_and_pixel_limits(self):
        for value in (-.5, 1280.9, float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    select_rgb_size(feature(), width=value, height=800)
                with self.assertRaises(ValueError):
                    select_stereo_size(feature(), feature(), width=1280, height=value)
                with self.assertRaises(ValueError):
                    resolution_limits(rgb_max_pixels=value)

    def test_limit_sources_and_unknown_usb(self):
        for speed in ("HIGH", "UNKNOWN", "FULL", "LOW"):
            self.assertEqual(resolution_limits(usb_speed=speed)["stream_budget_mb_s"], 30)
        self.assertEqual(resolution_limits(usb_speed="SUPER_PLUS")["stream_budget_mb_s"], 150)
        limits = resolution_limits(1000, 1)
        self.assertEqual(limits["rgb_max_pixels_source"], "configured")
        self.assertEqual(limits["stream_budget_mb_s_source"], "configured")

    def test_mjpeg_alignment_and_encoder_limits(self):
        camera = feature(width=4056, height=3040)
        result = select_rgb_size(camera, fps=60, usb_speed="SUPER", rgb_max_pixels=20000000,
                                 stream_budget_mb_s=10000, rgb_transport="mjpeg", depth_enabled=False)
        self.assertEqual(result["width"] % 32, 0)
        self.assertLessEqual(result["width"] * result["height"] * 60, 450000000)
        self.assertIn("mjpeg_throughput", str(result["rejected_candidates"]))
        huge = select_rgb_size(feature(width=20000, height=10000), fps=1, depth_enabled=False,
                               rgb_transport="mjpeg", rgb_max_pixels=500000000, stream_budget_mb_s=10000)
        self.assertLessEqual(huge["width"], 16384)
        self.assertLessEqual(huge["height"], 8192)
        self.assertIn("mjpeg_frame_size", str(huge["rejected_candidates"]))

    def test_describe_deduplicates_and_sorts_configs(self):
        camera = feature(configs=[config(640, 400), config(1280, 800), config(1280, 800)])
        description = describe_camera(camera)
        self.assertEqual(len(description["configs"]), 2)
        self.assertEqual(description["configs"][0]["width"], 1280)
        json.dumps(description)

    def test_fov_accepts_depthai_rect_size_method(self):
        rect = NS(size=lambda: NS(width=1280, height=800))
        camera = feature(configs=[config(1280, 800, fov=rect)])
        self.assertEqual(describe_camera(camera)["configs"][0]["fov"], 1024000)

    def test_equal_area_ties_use_aspect_then_fov(self):
        camera = feature(width=1600, height=1000, configs=[config(1280, 800, fov=2),
                        config(1280, 800, fov=4), config(1600, 640, fov=100)])
        result = select_rgb_size(camera, usb_speed="SUPER")
        self.assertEqual(self.size(result), (1280, 800))
        self.assertEqual(result["fov"], 4)

    def test_metadata_block_preserves_camera_output_and_policy(self):
        rgb, left, right = geo_cameras()
        decision = select_rgb_size(rgb, usb_speed="SUPER")
        cameras = []
        for camera, role in ((rgb, "rgb"), (left, "stereo_left"), (right, "stereo_right"),
                             (feature(socket="CAM_D"), "unused")):
            description = describe_camera(camera)
            description["role"] = role
            cameras.append(description)
        args = NS(rgb_width=1280, rgb_height=800, rgb_camera_width=1280, rgb_camera_height=800,
                  rgb_resolution_decision=decision, stereo_resolution_decision=select_stereo_size(left, right),
                  usb_speed="SUPER", depthai_platform="RVC2", connected_camera_metadata=cameras)
        metadata = json.loads(json.dumps(build_resolution_metadata(args)))
        self.assertEqual(metadata["aligned_depth_output"], {"width": 1280, "height": 800})
        self.assertEqual(metadata["policy"], {"rgb": "auto", "stereo": "auto"})
        self.assertEqual(metadata["limits"]["rgb_max_pixels_source"], "default")
        self.assertFalse(metadata["rgb"]["upscaled"])
        self.assertEqual([c["role"] for c in metadata["cameras"]], ["rgb", "stereo_left", "stereo_right", "unused"])
        args.rgb_width, args.rgb_height = 1920, 1200
        self.assertTrue(build_resolution_metadata(args)["rgb"]["upscaled"])
        args.monitor_only = True
        self.assertIsNone(build_resolution_metadata(args)["aligned_depth_output"])

    def test_cli_defaults_overrides_yaml_and_validation(self):
        from geonova_depthai.capture.cli import parse_args
        args = parse_args(["--monitor-only"])
        self.assertEqual((args.rgb_width, args.rgb_height), (0, 0))
        self.assertEqual((args.depth_input_width, args.depth_input_height), (0, 0))
        self.assertEqual(args.rgb_max_pixels, 0)
        self.assertEqual(args.camera_stream_budget_mb_s, 0)
        args = parse_args(["--monitor-only", "--rgb-width", "1920", "--rgb-height", "1200",
                           "--depth-input-width", "640", "--depth-input-height", "400",
                           "--rgb-max-pixels", "1000000", "--camera-stream-budget-mb-s", "42.5"])
        self.assertEqual((args.rgb_width, args.rgb_height), (1920, 1200))
        self.assertEqual(args.camera_stream_budget_mb_s, 42.5)
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        for path in (root / "config.yaml", root / "code/configs/capture.yaml"):
            args = parse_args(["--config", str(path)])
            self.assertEqual((args.rgb_width, args.rgb_height), (0, 0))
        for flag in ("--rgb-max-pixels", "--camera-stream-budget-mb-s", "--depth-input-width", "--depth-input-height"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args([flag, "-1"])
        for flag in ("--rgb-width", "--depth-input-width"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args([flag, "1280"])

    def test_cli_and_yaml_reject_invalid_integer_budget_and_fps_values(self):
        from geonova_depthai.capture.cli import parse_args
        integer_keys = ("rgb_width", "rgb_height", "depth_input_width", "depth_input_height", "rgb_max_pixels")
        cases = [(key, value) for key in integer_keys for value in ("-0.5", "1280.9", ".inf", ".nan")]
        cases += [("camera_stream_budget_mb_s", v) for v in (".inf", ".nan", "inf", "nan")]
        cases += [("fps", v) for v in ("0", "-1", ".inf", ".nan", "inf", "nan")]
        cases += [("depth_fps", v) for v in ("-1", ".inf", ".nan", "inf", "nan")]
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="resolution-validation-", dir=str(root)) as directory:
            path = Path(directory) / "invalid.yaml"
            for key, value in cases:
                with self.subTest(key=key, value=value):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cli:
                        parse_args(["--monitor-only", "--" + key.replace("_", "-"), value])
                    self.assertEqual(cli.exception.code, 2)
                    path.write_text("{0}: {1}\n".format(key, value), encoding="utf-8")
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as yaml:
                        parse_args(["--monitor-only", "--config", str(path)])
                    self.assertEqual(yaml.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
