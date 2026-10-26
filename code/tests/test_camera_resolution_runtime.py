"""Resolution selection across pipeline construction and reconnects; no device I/O."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace as NS
from unittest.mock import patch

import depthai as dai

from geonova_depthai import runtime
from geonova_depthai.camera_resolution import estimate_stream_mb_s
from geonova_depthai.capture.cli import parse_args


def camera(socket, width=1280, height=800, kind="MONO", name="OV9282"):
    return NS(socket=socket, width=width, height=height, sensorName=name, supportedTypes=[kind],
              configs=[NS(width=width, height=height, minFps=1, maxFps=120, type=kind)])


class FakeDevice:
    def __init__(self, speed=None, platform=None, rgb_size=(1280, 800)):
        self.speed = speed or dai.UsbSpeed.SUPER
        self.platform = platform or dai.Platform.RVC2
        self.cameras = [camera(dai.CameraBoardSocket.CAM_A, *rgb_size, kind="COLOR", name="OV9782"),
                        camera(dai.CameraBoardSocket.CAM_B), camera(dai.CameraBoardSocket.CAM_C)]

    def getConnectedCameraFeatures(self):
        return self.cameras

    def getUsbSpeed(self):
        return self.speed

    def getPlatform(self):
        return self.platform


class Port:
    def __init__(self):
        self.links = []

    def link(self, other):
        self.links.append(other)


class FakeNode:
    def __init__(self):
        for name in ("left", "right", "input", "inputAlignTo", "depth", "confidenceMap", "outputAligned"):
            setattr(self, name, Port())

    def build(self, socket):
        self.socket = socket
        return self

    def setOutputSize(self, width, height):
        self.output_size = (width, height)


class FakePipeline:
    def __init__(self, device):
        self.device = device
        self.nodes = []

    def getDefaultDevice(self):
        return self.device

    def create(self, kind):
        node = FakeNode()
        self.nodes.append((kind, node))
        return node


class CameraResolutionRuntimeTests(unittest.TestCase):
    def pipeline(self, args, device, monitor=False):
        pipeline = FakePipeline(device)
        requested = []

        def request(node, fps, size, **kwargs):
            requested.append((node.socket, size))
            return Port()

        def resolve_rgb(*arguments, **kwargs):
            if not monitor:
                self.assertEqual(args.stereo_resolution_decision["policy"], "auto")
            return original_resolve(*arguments, **kwargs)

        original_resolve = runtime.resolve_rgb_output_size
        with ExitStack() as stack:
            for name in ("require_depthai_v3", "configure_oak_imu", "configure_stereo_exposure",
                         "configure_stereo_depth"):
                stack.enter_context(patch.object(runtime, name, return_value=None))
            stack.enter_context(patch.object(runtime, "request_camera_output", side_effect=request))
            stack.enter_context(patch.object(runtime, "resolve_rgb_output_size", side_effect=resolve_rgb))
            stack.enter_context(patch.object(runtime, "resize_camera_output", side_effect=lambda p, o, s, t: o))
            stack.enter_context(patch.object(runtime, "create_mjpeg_output", side_effect=lambda p, o, f, q: o))
            outputs = (runtime.configure_monitor_pipeline(pipeline, args) if monitor
                       else runtime.configure_pipeline(pipeline, args))
        return pipeline, outputs, requested

    def test_monitor_reconnect_reselects_with_original_auto_request_and_new_usb_speed(self):
        args = parse_args(["--monitor-only", "--rgb-max-pixels", "20000000"])
        device = FakeDevice(rgb_size=(4056, 3040))
        first = runtime.resolve_rgb_output_size(device, args)
        self.assertEqual(first, (2976, 2240))
        decision = args.rgb_resolution_decision
        self.assertEqual(runtime.resolve_rgb_output_size(device, args), first)
        self.assertEqual(args.rgb_resolution_decision, decision)
        device.speed = dai.UsbSpeed.HIGH
        second = runtime.resolve_rgb_output_size(device, args)
        self.assertEqual(second, (2656, 2008))
        self.assertNotEqual(second, first)
        self.assertEqual(args.rgb_transport_effective, "mjpeg")
        self.assertEqual(args.rgb_resolution_decision["policy"], "auto")
        self.assertLessEqual(args.rgb_resolution_decision["estimated_stream_mb_s"], 30)
        self.assertEqual((args.rgb_width_requested, args.rgb_height_requested), (0, 0))
        decision = args.rgb_resolution_decision
        self.assertEqual(runtime.resolve_rgb_output_size(device, args), second)
        self.assertEqual(args.rgb_resolution_decision, decision)

    def test_stereo_repeated_selection_and_sensor_change_keep_auto_policy(self):
        args = parse_args([])
        device = FakeDevice()
        sockets = (dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C)
        first = runtime.resolve_stereo_input_size(device, args, dai.Platform.RVC2, *sockets)
        decision = args.stereo_resolution_decision
        self.assertEqual(runtime.resolve_stereo_input_size(device, args, dai.Platform.RVC2, *sockets), first)
        self.assertEqual(args.stereo_resolution_decision, decision)
        device.cameras[1:] = [camera(socket, 640, 400) for socket in sockets]
        self.assertEqual(runtime.resolve_stereo_input_size(device, args, dai.Platform.RVC2, *sockets), (640, 400))
        self.assertEqual(args.stereo_resolution_decision["policy"], "auto")
        self.assertEqual((args.depth_input_width_requested, args.depth_input_height_requested), (0, 0))

    def test_stereo_missing_camera_features_uses_safe_fallback(self):
        for error in (None, RuntimeError("camera features unavailable")):
            with self.subTest(error=error):
                args = parse_args([])
                device = FakeDevice()
                sockets = (dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C)
                with patch.object(device, "getConnectedCameraFeatures", return_value=[], side_effect=error):
                    self.assertEqual(runtime.connected_camera_features(device), [])
                    self.assertEqual(runtime.resolve_stereo_input_size(device, args, dai.Platform.RVC2, *sockets),
                                     (1280, 800))
                self.assertEqual((args.depth_input_width, args.depth_input_height), (1280, 800))
                self.assertEqual(args.depth_input_resolution_source, "fallback_unknown_sensor")
                self.assertEqual(args.stereo_resolution_decision["source"], "fallback_unknown_sensor")
                self.assertEqual(args.stereo_resolution_decision["policy"], "auto")
                self.assertEqual((args.depth_input_width_requested, args.depth_input_height_requested), (0, 0))
                for sensor in (args.left_sensor, args.right_sensor):
                    self.assertEqual((sensor["width"], sensor["height"]), (0, 0))

    def test_explicit_requests_keep_configured_policy_on_repeated_calls(self):
        args = parse_args(["--rgb-width", "640", "--rgb-height", "400", "--depth-input-width", "640",
                           "--depth-input-height", "400"])
        device = FakeDevice()
        for speed in (dai.UsbSpeed.SUPER, dai.UsbSpeed.HIGH):
            device.speed = speed
            self.assertEqual(runtime.resolve_rgb_output_size(device, args), (640, 400))
            self.assertEqual(runtime.resolve_stereo_input_size(device, args, dai.Platform.RVC2,
                             dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C), (640, 400))
            self.assertEqual(args.rgb_resolution_decision["policy"], "configured")
            self.assertEqual(args.stereo_resolution_decision["policy"], "configured")

    def test_rvc2_confidence_uses_stereo_size_in_pipeline_budget_and_metadata(self):
        args = parse_args(["--fps", "15", "--save-confidence-map", "--confidence-transport", "raw"])
        device = FakeDevice(speed=dai.UsbSpeed.HIGH)
        pipeline, outputs, requested = self.pipeline(args, device)
        rgb = (args.rgb_width, args.rgb_height)
        self.assertLess(rgb[0] * rgb[1], 1280 * 800)
        self.assertEqual((args.depth_input_width, args.depth_input_height), (1280, 800))
        expected = estimate_stream_mb_s(*rgb, fps=15, rgb_transport="mjpeg", save_confidence_map=True,
                                        confidence_transport="raw", confidence_size=(1280, 800))
        self.assertAlmostEqual(args.camera_resolution["estimated_stream_mb_s"], expected)
        self.assertLessEqual(expected, 30)
        stereo = next(node for kind, node in pipeline.nodes if kind == dai.node.StereoDepth)
        self.assertIs(outputs["confidence"], stereo.confidenceMap)
        self.assertEqual(requested[1:], [(dai.CameraBoardSocket.CAM_B, (1280, 800)),
                                        (dai.CameraBoardSocket.CAM_C, (1280, 800))])

    def test_rvc4_image_align_confidence_uses_rgb_size(self):
        args = parse_args(["--fps", "15", "--save-confidence-map", "--confidence-transport", "raw"])
        pipeline, outputs, _ = self.pipeline(args, FakeDevice(speed=dai.UsbSpeed.HIGH, platform=dai.Platform.RVC4))
        rgb = (args.rgb_width, args.rgb_height)
        expected = estimate_stream_mb_s(*rgb, fps=15, rgb_transport="mjpeg", save_confidence_map=True)
        self.assertAlmostEqual(args.camera_resolution["estimated_stream_mb_s"], expected)
        aligns = [node for kind, node in pipeline.nodes if kind == dai.node.ImageAlign]
        self.assertEqual([node.output_size for node in aligns], [rgb, rgb])
        self.assertIs(outputs["confidence"], aligns[1].outputAligned)

    def test_raw_transport_resolves_unknown_or_missing_usb_before_direct_selection(self):
        for unknown in (True, False):
            args = parse_args(["--rgb-transport", "raw"])
            if not unknown:
                del args.usb_speed
            self.assertEqual(runtime.resolve_rgb_output_size(FakeDevice(), args), (1280, 800))
            self.assertEqual(args.usb_speed, "SUPER")
            self.assertEqual(args.rgb_transport_effective, "raw")
            self.assertEqual(args.rgb_resolution_decision["limits"]["stream_budget_mb_s"], 150)

    def test_raw_transport_resolves_usb_in_both_pipeline_entrypoints(self):
        for monitor in (False, True):
            with self.subTest(monitor=monitor):
                args = parse_args(["--rgb-transport", "raw"] + (["--monitor-only"] if monitor else []))
                self.pipeline(args, FakeDevice(), monitor=monitor)
                self.assertEqual(args.usb_speed, "SUPER")
                self.assertEqual(args.rgb_transport_effective, "raw")
                self.assertEqual(args.camera_resolution["limits"]["stream_budget_mb_s"], 150)

    def test_runtime_rejects_nonfinite_frame_rates(self):
        for name, values in (("fps", (0, -1, float("inf"), float("nan"))),
                             ("depth_fps", (-1, float("inf"), float("nan")))):
            for value in values:
                args = parse_args([])
                setattr(args, name, value)
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    runtime.resolve_depth_fps(args)


if __name__ == "__main__":
    unittest.main()
