from pathlib import Path

import pytest

from geonova_depthai.capture import raw_event_recorder


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
