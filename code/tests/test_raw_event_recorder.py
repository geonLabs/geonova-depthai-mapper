from pathlib import Path

import pytest

from geonova_depthai.capture import raw_event_recorder
from geonova_depthai.capture.cli import parse_args


class FakeReader:
    def __init__(self, events):
        self.events = events

    def drain(self):
        self.events.append("reader.drain")
        return [{"sample": 1}]


class FakeControllerBridge:
    def __init__(self, events):
        self.events = events

    def close(self, serial_readers, error=None):
        self.events.append(("bridge.close", error))


class FailingImagePool:
    def __init__(self, events):
        self.events = events

    def close(self):
        self.events.append("image_pool.close")
        raise RuntimeError("writer failed")


class FakeDataset:
    root = Path("/tmp/raw-dataset")

    def __init__(self, events):
        self.events = events

    def write_serial_samples(self, name, samples):
        self.events.append(("dataset.write_serial_samples", name, samples))

    def close(self):
        self.events.append("dataset.close")


class FakeDevice:
    def __init__(self, events):
        self.events = events

    def close(self):
        self.events.append("device.close")


def test_cleanup_closes_dataset_and_device_after_writer_failure(monkeypatch) -> None:
    events = []
    readers = {"gps": FakeReader(events)}
    monkeypatch.setattr(
        raw_event_recorder.runtime,
        "stop_serial_readers",
        lambda serial_readers: events.append("readers.stop"),
    )

    with pytest.raises(RuntimeError, match="writer failed"):
        raw_event_recorder._close_recording_resources(
            FakeControllerBridge(events),
            readers,
            FailingImagePool(events),
            FakeDataset(events),
            FakeDevice(events),
        )

    assert "dataset.close" in events
    assert "device.close" in events
    assert events.index("dataset.close") < events.index("device.close")


def test_main_forwards_argv_to_capture_parser(monkeypatch) -> None:
    argv = ["--max-runtime-s", "1"]
    parsed_args = object()
    calls = []

    monkeypatch.setattr(
        raw_event_recorder.signal,
        "signal",
        lambda signum, handler: calls.append(("signal", signum, handler)),
    )
    monkeypatch.setattr(
        raw_event_recorder,
        "parse_args",
        lambda value: calls.append(("parse_args", value)) or parsed_args,
    )
    monkeypatch.setattr(
        raw_event_recorder,
        "record_raw_events",
        lambda value: calls.append(("record_raw_events", value)),
    )

    raw_event_recorder.main(argv)

    assert ("parse_args", argv) in calls
    assert ("record_raw_events", parsed_args) in calls


def test_main_does_not_clear_stop_requested_after_installing_handlers(monkeypatch) -> None:
    observed = []
    monkeypatch.setattr(raw_event_recorder, "_stop_requested", True)

    def install_handler(signum, handler):
        observed.append(("install", signum, raw_event_recorder._stop_requested))
        if len(observed) == 1:
            handler(signum, None)

    monkeypatch.setattr(raw_event_recorder.signal, "signal", install_handler)
    monkeypatch.setattr(raw_event_recorder, "parse_args", lambda argv: object())
    monkeypatch.setattr(
        raw_event_recorder,
        "record_raw_events",
        lambda args: observed.append(("record", raw_event_recorder._stop_requested)),
    )

    raw_event_recorder.main([])

    assert observed[0] == ("install", raw_event_recorder.signal.SIGINT, False)
    assert observed[-1] == ("record", True)


def test_monitor_only_defaults_use_full_preview_geometry_with_bounded_bandwidth() -> None:
    args = parse_args(
        [
            "--monitor-only",
            "--fps",
            "30",
            "--rgb-width",
            "1920",
            "--rgb-height",
            "1080",
            "--imu-rate",
            "400",
            "--no-controller-bridge",
        ]
    )

    assert args.monitor_only is True
    assert args.controller_bridge_enabled is True
    assert args.allow_usb2 is True
    assert args.save_confidence_map is False
    assert args.fps == 15.0
    assert (args.rgb_width, args.rgb_height) == (1920, 1200)
    assert args.controller_preview_max_width == 1920
    assert args.controller_preview_fps == 15.0
    assert args.imu_rate == 100


def test_monitor_only_never_creates_dataset_and_releases_resources(monkeypatch) -> None:
    events = []

    class EmptyQueue:
        def tryGet(self):
            return None

    class Output:
        def createOutputQueue(self, **kwargs):  # noqa: ARG002
            return EmptyQueue()

    class Pipeline:
        def __init__(self, device):
            self.device = device

        def __enter__(self):
            events.append("pipeline.enter")
            return self

        def __exit__(self, *exc):
            events.append("pipeline.exit")

        def getDefaultDevice(self):
            return self.device

        def start(self):
            events.append("pipeline.start")

    class Bridge:
        def __init__(self, args):  # noqa: ARG002
            events.append("bridge.create")

        def publish(self, readers, force=False):  # noqa: ARG002
            events.append(("bridge.publish", force))

        def mark_device_connected(self):
            events.append("bridge.connected")

        def close(self, readers, error=None):  # noqa: ARG002
            events.append(("bridge.close", error))

        def offer_rgb(self, message):  # pragma: no cover - empty test queue
            raise AssertionError(message)

        def configure_imu(self, enabled, error=None):
            events.append(("bridge.imu", enabled, error))

        def observe_imu(self, message):  # pragma: no cover - empty test queue
            raise AssertionError

    device = FakeDevice(events)
    args = parse_args(
        [
            "--monitor-only",
            "--max-runtime-s",
            "0.001",
            "--no-gps",
        ]
    )
    monkeypatch.setattr(raw_event_recorder, "ControllerBridge", Bridge)
    monkeypatch.setattr(raw_event_recorder.dai, "Pipeline", Pipeline)
    monkeypatch.setattr(raw_event_recorder.runtime, "create_serial_readers", lambda args: {})
    monkeypatch.setattr(
        raw_event_recorder.runtime,
        "start_serial_readers",
        lambda readers: events.append("readers.start"),
    )
    monkeypatch.setattr(
        raw_event_recorder.runtime,
        "stop_serial_readers",
        lambda readers: events.append("readers.stop"),
    )
    monkeypatch.setattr(raw_event_recorder.runtime, "connect_depthai_device", lambda args: device)
    monkeypatch.setattr(raw_event_recorder.runtime, "resolve_transport_options", lambda args, device: None)
    def configure_monitor(pipeline, args):
        args.imu_enabled, args.imu_error = False, "No OAK IMU"
        return {"rgb": Output(), "imu": None}

    monkeypatch.setattr(raw_event_recorder.runtime, "configure_monitor_pipeline", configure_monitor)
    monkeypatch.setattr(
        raw_event_recorder,
        "RawEventDataset",
        lambda *args, **kwargs: pytest.fail("monitor-only created a dataset"),
    )
    monkeypatch.setattr(raw_event_recorder, "_stop_requested", False)

    raw_event_recorder.record_raw_events(args)

    assert ("bridge.imu", False, "No OAK IMU") in events
    assert "pipeline.start" in events
    assert "pipeline.exit" in events
    assert "readers.stop" in events
    assert "device.close" in events
    assert events.index("pipeline.exit") < events.index("device.close")
    assert any(event[0] == "bridge.close" for event in events if isinstance(event, tuple))


def test_monitor_camera_failure_keeps_serial_bridge_alive_until_runtime_limit(monkeypatch) -> None:
    events = []

    class Reader:
        name = "gps"
        device = "/dev/serial/by-id/test-gnss"
        baudrate = 115200
        max_hz = 10.0

        def drain(self):
            events.append("reader.drain")
            return []

        def status_text(self):
            return "gps=test"

    class Bridge:
        def __init__(self, args):  # noqa: ARG002
            pass

        def publish(self, readers, force=False):  # noqa: ARG002
            events.append(("bridge.publish", force))

        def mark_device_disconnected(self, error):
            events.append(("bridge.camera_error", str(error)))

        def close(self, readers, error=None):  # noqa: ARG002
            events.append(("bridge.close", error))

    reader = Reader()
    args = parse_args(["--monitor-only", "--max-runtime-s", "0.02"])
    monkeypatch.setattr(raw_event_recorder, "ControllerBridge", Bridge)
    monkeypatch.setattr(
        raw_event_recorder.runtime,
        "create_serial_readers",
        lambda args: {"gps": reader},
    )
    monkeypatch.setattr(raw_event_recorder.runtime, "start_serial_readers", lambda readers: None)
    monkeypatch.setattr(
        raw_event_recorder.runtime,
        "stop_serial_readers",
        lambda readers: events.append("reader.stop"),
    )
    monkeypatch.setattr(
        raw_event_recorder.runtime,
        "connect_depthai_device",
        lambda args: (_ for _ in ()).throw(RuntimeError("camera missing")),
    )
    monkeypatch.setattr(raw_event_recorder, "_stop_requested", False)

    raw_event_recorder.record_raw_events(args)

    assert ("bridge.camera_error", "camera missing") in events
    assert "reader.drain" in events
    assert "reader.stop" in events
    assert ("bridge.close", None) in events


def test_capture_without_imu_queue_still_closes_dataset(tmp_path, monkeypatch):
    events = []

    class EmptyQueue:
        def tryGet(self):
            return None

    class Output:
        def createOutputQueue(self, **kwargs):
            return EmptyQueue()

    class Pipeline:
        def __init__(self, device):
            self.device = device

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def getDefaultDevice(self):
            return self.device

        def start(self):
            events.append("pipeline.start")

    class ImagePool:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            events.append("image_pool.close")

    def configure(pipeline, args):
        args.imu_enabled, args.imu_error, args.imu_type, args.imu_sensors = False, "No OAK IMU", "NONE", []
        args.imu_metadata = raw_event_recorder.runtime.read_imu_metadata(None, args)
        return {"rgb": Output(), "depth": Output(), "imu": None}

    device = FakeDevice(events)
    monkeypatch.setattr(raw_event_recorder.dai, "Pipeline", Pipeline)
    monkeypatch.setattr(raw_event_recorder, "ImageWritePool", ImagePool)
    monkeypatch.setattr(raw_event_recorder.runtime, "create_serial_readers", lambda args: {})
    monkeypatch.setattr(raw_event_recorder.runtime, "connect_depthai_device", lambda args: device)
    monkeypatch.setattr(raw_event_recorder.runtime, "resolve_transport_options", lambda args, device: None)
    monkeypatch.setattr(raw_event_recorder.runtime, "configure_pipeline", configure)
    monkeypatch.setattr(raw_event_recorder.runtime, "read_camera_model_metadata", lambda *args: {})
    monkeypatch.setattr(raw_event_recorder.runtime, "read_stereo_depth_metadata", lambda *args: {})
    monkeypatch.setattr(raw_event_recorder, "_stop_requested", False)
    args = parse_args(["--no-gps", "--no-controller-bridge", "--output-dir", str(tmp_path), "--max-runtime-s", "0.001"])
    raw_event_recorder.record_raw_events(args)
    assert "pipeline.start" in events
    assert "image_pool.close" in events
    assert "device.close" in events
    import json
    root = next(tmp_path.glob("*_raw"))
    assert json.loads((root / "metadata.json").read_text())["imu"]["type"] == "NONE"
    assert json.loads((root / "capture_manifest.json").read_text())["state"] == "complete"


def test_frames_before_exposure_settles_are_not_saved(tmp_path, monkeypatch):
    import json

    class Queue:
        def __init__(self, count):
            self.left = count

        def tryGet(self):
            if not self.left:
                return None
            self.left -= 1
            return object()

    queues = {"rgb": Queue(3), "depth": Queue(2)}

    class Output:
        def __init__(self, stream):
            self.stream = stream

        def createOutputQueue(self, **kwargs):
            return queues[self.stream]

    class Pipeline:
        def __init__(self, device):
            self.device = device

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def getDefaultDevice(self):
            return self.device

        def start(self):
            pass

    submitted = []

    class ImagePool:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, job):
            submitted.append(job["stream"])

        def close(self):
            pass

    def configure(pipeline, args):
        args.imu_enabled, args.imu_error, args.imu_type, args.imu_sensors = False, "No OAK IMU", "NONE", []
        args.imu_metadata = raw_event_recorder.runtime.read_imu_metadata(None, args)
        return {"rgb": Output("rgb"), "depth": Output("depth"), "imu": None}

    monkeypatch.setattr(raw_event_recorder.dai, "Pipeline", Pipeline)
    monkeypatch.setattr(raw_event_recorder, "ImageWritePool", ImagePool)
    monkeypatch.setattr(raw_event_recorder.runtime, "create_serial_readers", lambda args: {})
    monkeypatch.setattr(raw_event_recorder.runtime, "connect_depthai_device", lambda args: FakeDevice([]))
    monkeypatch.setattr(raw_event_recorder.runtime, "resolve_transport_options", lambda args, device: None)
    monkeypatch.setattr(raw_event_recorder.runtime, "configure_pipeline", configure)
    monkeypatch.setattr(raw_event_recorder.runtime, "read_camera_model_metadata", lambda *args: {})
    monkeypatch.setattr(raw_event_recorder.runtime, "read_stereo_depth_metadata", lambda *args: {})
    monkeypatch.setattr(raw_event_recorder, "_stop_requested", False)
    args = parse_args([
        "--no-gps", "--no-controller-bridge", "--output-dir", str(tmp_path),
        "--max-runtime-s", "0.001", "--exposure-settle-s", "0.05",
    ])

    raw_event_recorder.record_raw_events(args)

    assert submitted == []
    root = next(tmp_path.glob("*_raw"))
    assert json.loads((root / "metadata.json").read_text())["exposure_settle"] == {
        "seconds": 0.05,
        "dropped_frames": {"rgb": 3, "depth": 2},
    }


def test_exposure_settle_off_by_default_drops_nothing():
    events = []
    dropped = raw_event_recorder._drop_frames_until_exposure_settles(
        0.0, {"rgb": object()}, None, None, None, {}, FakeDataset(events)
    )

    assert dropped == {"rgb": 0}
    assert events == []
