from __future__ import annotations

import queue
import socket
import threading
import time

import pytest

from geonova_depthai import runtime


def _rtcm_frame(payload: bytes) -> bytes:
    header = bytes((0xD3, (len(payload) >> 8) & 0x03, len(payload) & 0xFF))
    body = header + payload
    return body + runtime.rtcm3_crc24q(body).to_bytes(3, "big")


class FakeStreamSocket:
    def __init__(self, name, events, recv_data=None, close_event=None):
        self.name = name
        self.events = events
        self.recv_data = recv_data
        self.close_event = close_event
        self.closed = False

    def settimeout(self, timeout):  # noqa: ARG002
        pass

    def sendall(self, data):
        self.events.append(("send", self.name, data))

    def recv(self, size):  # noqa: ARG002
        if self.closed:
            return b""
        if self.recv_data is None:
            raise socket.timeout()
        threading.Event().wait(0.001)
        return self.recv_data

    def close(self):
        if not self.closed:
            self.closed = True
            self.events.append(("close", self.name))
            if self.close_event is not None:
                self.close_event.set()


class ScriptedSocket(FakeStreamSocket):
    def __init__(self, chunks):
        super().__init__("scripted", [])
        self.chunks = list(chunks)

    def recv(self, size):
        if not self.chunks:
            return b""
        chunk = self.chunks.pop(0)
        if len(chunk) <= size:
            return chunk
        self.chunks.insert(0, chunk[size:])
        return chunk[:size]


def _client_config(**overrides):
    config = {
        "host": "caster.test",
        "port": 2101,
        "auto_mountpoint": True,
        "mountpoint_format": "RTCM31",
        "gga_interval": 0.0,
        "connect_timeout": 0.5,
        "data_timeout": 1.0,
        "reselect_interval": 0.01,
        "switch_min_improvement_m": 1000.0,
    }
    config.update(overrides)
    return config


def test_periodic_handover_keeps_old_stream_until_new_rtcm_is_ready(monkeypatch) -> None:
    old_frame = _rtcm_frame(b"old")
    new_frame = _rtcm_frame(b"new")
    events = []
    writes = []
    probe_started = threading.Event()
    old_continued = threading.Event()
    release_probe = threading.Event()
    switched = threading.Event()
    stop_event = threading.Event()
    old_writes_during_probe = 0

    old_socket = FakeStreamSocket("old", events, recv_data=old_frame)
    new_socket = FakeStreamSocket("new", events)

    def serial_write(data):
        nonlocal old_writes_during_probe
        writes.append(data)
        events.append(("write", data))
        if probe_started.is_set() and data == old_frame:
            old_writes_during_probe += 1
            if old_writes_during_probe >= 2:
                old_continued.set()
        if data == new_frame:
            switched.set()
            stop_event.set()

    client = runtime.NtripCorrectionClient(
        _client_config(),
        serial_write,
        stop_event,
        latest_nmea={"latitude_deg": 37.11, "longitude_deg": 127.0},
    )
    current = {"mountpoint": "OLD", "latitude": 37.0, "longitude": 127.0}
    candidate = {
        "mountpoint": "NEW",
        "latitude": 37.1,
        "longitude": 127.0,
        "distance_improvement_m": 5000.0,
    }

    def open_stream(
        entry,
        require_frame=False,
        cancel_event=None,
        handover_state=None,
    ):  # noqa: ARG001
        if entry["mountpoint"] == "OLD":
            return {
                "socket": old_socket,
                "entry": dict(entry),
                "framer": runtime.Rtcm3Framer(),
                "frames": [old_frame],
                "last_gga_time": 0.0,
            }
        probe_started.set()
        assert release_probe.wait(1.0)
        return {
            "socket": new_socket,
            "entry": dict(entry),
            "framer": runtime.Rtcm3Framer(),
            "frames": [new_frame],
            "last_gga_time": 0.0,
        }

    monkeypatch.setattr(client, "_open_mountpoint_stream", open_stream)
    monkeypatch.setattr(client, "_periodic_switch_candidates", lambda entry: [candidate])

    thread = threading.Thread(target=client._stream_mountpoint, args=(current,), daemon=True)
    thread.start()

    assert probe_started.wait(1.0)
    assert old_continued.wait(1.0)
    assert client.connected is True
    assert client.current_mountpoint == "OLD"
    assert old_socket.closed is False

    release_probe.set()
    assert switched.wait(1.0)
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert client.current_mountpoint == "NEW"
    assert old_socket.closed is True
    assert new_socket.closed is True
    assert client.bytes_received == sum(len(data) for data in writes)
    assert events.index(("write", new_frame)) < events.index(("close", "old"))


def test_candidate_connection_waits_for_complete_valid_rtcm(monkeypatch) -> None:
    first = _rtcm_frame(b"candidate-1")
    second = _rtcm_frame(b"candidate-2")
    sock = ScriptedSocket(
        [b"HTTP/1.1 200 OK\r\n\r\n" + first[:4], first[4:] + second]
    )
    client = runtime.NtripCorrectionClient(
        _client_config(),
        lambda data: None,
        threading.Event(),
    )
    monkeypatch.setattr(runtime.socket, "create_connection", lambda *args, **kwargs: sock)

    stream = client._open_mountpoint_stream(
        {"mountpoint": "NEW"},
        require_frame=True,
    )

    assert stream["frames"] == [first, second]
    assert sock.closed is False
    client._close_socket(stream["socket"])


def test_ntrip_v1_icy_status_line_starts_rtcm_stream(monkeypatch) -> None:
    first = _rtcm_frame(b"legacy-one")
    second = _rtcm_frame(b"legacy-two")
    sock = ScriptedSocket([b"ICY 200 OK\r\n" + first + second])
    client = runtime.NtripCorrectionClient(
        _client_config(),
        lambda data: None,
        threading.Event(),
    )
    monkeypatch.setattr(runtime.socket, "create_connection", lambda *args, **kwargs: sock)

    stream = client._open_mountpoint_stream(
        {"mountpoint": "LEGACY"},
        require_frame=True,
    )

    assert stream["frames"] == [first, second]
    assert sock.closed is False
    client._close_socket(stream["socket"])


def test_ntrip_v1_rejected_status_is_reported(monkeypatch) -> None:
    sock = ScriptedSocket([b"ICY 401 Unauthorized\n"])
    client = runtime.NtripCorrectionClient(
        _client_config(),
        lambda data: None,
        threading.Event(),
    )
    monkeypatch.setattr(runtime.socket, "create_connection", lambda *args, **kwargs: sock)

    with pytest.raises(RuntimeError, match="401 Unauthorized"):
        client._open_mountpoint_stream({"mountpoint": "LEGACY"})

    assert sock.closed is True


def test_empty_caster_response_reports_missing_credentials(monkeypatch) -> None:
    sock = ScriptedSocket([])
    client = runtime.NtripCorrectionClient(
        _client_config(),
        lambda data: None,
        threading.Event(),
    )
    monkeypatch.setattr(runtime.socket, "create_connection", lambda *args, **kwargs: sock)

    with pytest.raises(RuntimeError, match="verify NTRIP username and password"):
        client._open_mountpoint_stream({"mountpoint": "AUTH_REQUIRED"})

    assert sock.closed is True


def test_rejected_candidate_connection_is_closed(monkeypatch) -> None:
    sock = ScriptedSocket([b"HTTP/1.1 401 Unauthorized\r\n\r\n"])
    client = runtime.NtripCorrectionClient(
        _client_config(),
        lambda data: None,
        threading.Event(),
    )
    monkeypatch.setattr(runtime.socket, "create_connection", lambda *args, **kwargs: sock)

    with pytest.raises(RuntimeError, match="401 Unauthorized"):
        client._open_mountpoint_stream({"mountpoint": "NEW"}, require_frame=True)

    assert sock.closed is True


def test_single_rtcm_frame_then_eof_is_not_handover_ready(monkeypatch) -> None:
    frame = _rtcm_frame(b"only-frame")
    sock = ScriptedSocket([b"HTTP/1.1 200 OK\r\n\r\n" + frame])
    client = runtime.NtripCorrectionClient(
        _client_config(),
        lambda data: None,
        threading.Event(),
    )
    monkeypatch.setattr(runtime.socket, "create_connection", lambda *args, **kwargs: sock)

    with pytest.raises(RuntimeError, match="closed before valid RTCM data"):
        client._open_mountpoint_stream({"mountpoint": "NEW"}, require_frame=True)

    assert sock.closed is True


def test_non_rtcm3_stream_keeps_raw_payload_compatibility(monkeypatch) -> None:
    sock = ScriptedSocket([b"HTTP/1.1 200 OK\r\n\r\nraw-one", b"raw-two"])
    client = runtime.NtripCorrectionClient(
        _client_config(auto_mountpoint=False, mountpoint_format="RAW"),
        lambda data: None,
        threading.Event(),
    )
    monkeypatch.setattr(runtime.socket, "create_connection", lambda *args, **kwargs: sock)

    stream = client._open_mountpoint_stream(
        {"mountpoint": "RAW", "format": "RAW"},
        require_frame=True,
    )

    assert stream["framer"] is None
    assert stream["frames"] == [b"raw-one", b"raw-two"]
    client._close_socket(stream["socket"])


def test_failed_handover_does_not_change_active_connection(monkeypatch) -> None:
    stop_event = threading.Event()
    client = runtime.NtripCorrectionClient(
        _client_config(),
        lambda data: None,
        stop_event,
        latest_nmea={"latitude_deg": 37.11, "longitude_deg": 127.0},
    )
    client.connected = True
    client.current_mountpoint = "OLD"
    current = {"mountpoint": "OLD", "latitude": 37.0, "longitude": 127.0}
    candidate = {
        "mountpoint": "NEW",
        "latitude": 37.1,
        "longitude": 127.0,
        "distance_improvement_m": 5000.0,
    }
    results = queue.Queue(maxsize=1)

    monkeypatch.setattr(client, "_periodic_switch_candidates", lambda entry: [candidate])

    def reject(
        entry,
        require_frame=False,
        cancel_event=None,
        handover_state=None,
    ):  # noqa: ARG001
        raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr(client, "_open_mountpoint_stream", reject)

    client._handover_worker(current, results, threading.Event())
    result = results.get_nowait()

    assert result["status"] == "failed"
    assert "401 Unauthorized" in result["reason"]
    assert client.connected is True
    assert client.current_mountpoint == "OLD"
    assert client.bytes_received == 0


def test_handover_is_abandoned_if_fix_is_lost_during_validation(monkeypatch) -> None:
    old_frame = _rtcm_frame(b"old")
    new_frame = _rtcm_frame(b"new")
    events = []
    writes = []
    probe_started = threading.Event()
    release_probe = threading.Event()
    candidate_closed = threading.Event()
    stop_event = threading.Event()
    latest = {
        "latitude_deg": 37.11,
        "longitude_deg": 127.0,
        "position_valid": True,
        "position_monotonic": time.monotonic(),
    }
    old_socket = FakeStreamSocket("old", events, recv_data=old_frame)
    new_socket = FakeStreamSocket("new", events, close_event=candidate_closed)
    client = runtime.NtripCorrectionClient(
        _client_config(),
        writes.append,
        stop_event,
        latest_nmea=latest,
    )
    current = {"mountpoint": "OLD", "latitude": 37.0, "longitude": 127.0}
    candidate = {"mountpoint": "NEW", "latitude": 37.1, "longitude": 127.0}

    def open_stream(entry, require_frame=False, cancel_event=None, handover_state=None):  # noqa: ARG001
        if entry["mountpoint"] == "OLD":
            return {
                "socket": old_socket,
                "entry": dict(entry),
                "framer": runtime.Rtcm3Framer(),
                "frames": [old_frame],
                "last_gga_time": 0.0,
            }
        probe_started.set()
        assert release_probe.wait(1.0)
        return {
            "socket": new_socket,
            "entry": dict(entry),
            "framer": runtime.Rtcm3Framer(),
            "frames": [new_frame],
            "last_gga_time": 0.0,
        }

    monkeypatch.setattr(client, "_open_mountpoint_stream", open_stream)
    monkeypatch.setattr(client, "_periodic_switch_candidates", lambda entry: [candidate])

    thread = threading.Thread(target=client._stream_mountpoint, args=(current,), daemon=True)
    thread.start()
    assert probe_started.wait(1.0)

    latest["position_valid"] = False
    latest["gga"] = "$GPGGA,120000.00,,,,,0,00,99.99,,,,,,*00"
    release_probe.set()

    assert candidate_closed.wait(1.0)
    stop_event.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert client.current_mountpoint == "OLD"
    assert new_frame not in writes


def test_cancelling_handover_closes_registered_probe_socket(monkeypatch) -> None:
    events = []
    probe_started = threading.Event()
    probe_closed = threading.Event()
    stop_event = threading.Event()
    client = runtime.NtripCorrectionClient(
        _client_config(),
        lambda data: None,
        stop_event,
        latest_nmea={"latitude_deg": 37.11, "longitude_deg": 127.0},
    )
    current = {"mountpoint": "OLD", "latitude": 37.0, "longitude": 127.0}
    candidate = {"mountpoint": "NEW", "latitude": 37.1, "longitude": 127.0}
    probe_socket = FakeStreamSocket("probe", events, close_event=probe_closed)

    monkeypatch.setattr(client, "_periodic_switch_candidates", lambda entry: [candidate])

    def blocked_open(
        entry,
        require_frame=False,
        cancel_event=None,
        handover_state=None,
    ):  # noqa: ARG001
        assert client._register_handover_socket(handover_state, probe_socket)
        probe_started.set()
        while not cancel_event.wait(0.01):
            pass
        raise RuntimeError("cancelled")

    monkeypatch.setattr(client, "_open_mountpoint_stream", blocked_open)

    handover = client._start_handover(current)
    assert probe_started.wait(1.0)
    client._cancel_handover(handover)

    assert probe_closed.wait(1.0)
    handover["thread"].join(timeout=1.0)
    assert not handover["thread"].is_alive()


def test_failure_after_handover_reranks_instead_of_using_stale_candidates(
    monkeypatch,
    capsys,
) -> None:
    stop_event = threading.Event()
    client = runtime.NtripCorrectionClient(
        _client_config(reconnect_delay=0.0),
        lambda data: None,
        stop_event,
    )
    ranking_calls = 0
    attempted = []

    def candidates():
        nonlocal ranking_calls
        ranking_calls += 1
        if ranking_calls == 1:
            return [{"mountpoint": "A"}, {"mountpoint": "STALE-C"}]
        stop_event.set()
        return []

    def stream(entry):
        attempted.append(entry["mountpoint"])
        if entry["mountpoint"] == "STALE-C":
            raise AssertionError("stale candidate list must not be resumed")
        client.connected = True
        client.current_mountpoint = "B"
        raise RuntimeError("stream lost")

    monkeypatch.setattr(client, "_mountpoint_candidates", candidates)
    monkeypatch.setattr(client, "_stream_mountpoint", stream)

    client._run()

    assert ranking_calls == 2
    assert attempted == ["A"]
    assert "B: stream lost" in capsys.readouterr().out


def test_periodic_selection_is_disabled_without_auto_mountpoint(monkeypatch) -> None:
    client = runtime.NtripCorrectionClient(
        _client_config(auto_mountpoint=False),
        lambda data: None,
        threading.Event(),
        latest_nmea={"latitude_deg": 37.1, "longitude_deg": 127.0},
    )

    def unexpected_ranking(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("candidate ranking must stay disabled")

    monkeypatch.setattr(client, "_mountpoint_candidates", unexpected_ranking)

    assert client._periodic_switch_candidates({"mountpoint": "OLD"}) == []


def test_zero_reselect_interval_starts_no_handover(monkeypatch) -> None:
    frame = _rtcm_frame(b"initial")
    stop_event = threading.Event()
    sock = FakeStreamSocket("old", [])

    def serial_write(data):  # noqa: ARG001
        stop_event.set()

    client = runtime.NtripCorrectionClient(
        _client_config(reselect_interval=0.0),
        serial_write,
        stop_event,
    )

    monkeypatch.setattr(
        client,
        "_open_mountpoint_stream",
        lambda *args, **kwargs: {
            "socket": sock,
            "entry": {"mountpoint": "OLD"},
            "framer": runtime.Rtcm3Framer(),
            "frames": [frame],
            "last_gga_time": 0.0,
        },
    )

    def unexpected_handover(entry):  # noqa: ARG001
        raise AssertionError("interval=0 must not start a periodic handover")

    monkeypatch.setattr(client, "_start_handover", unexpected_handover)

    client._stream_mountpoint({"mountpoint": "OLD"})

    assert sock.closed is True


def test_invalid_gga_does_not_reuse_the_previous_position(monkeypatch) -> None:
    latest = {
        "latitude_deg": 37.5,
        "longitude_deg": 127.0,
        "gga": "$GPGGA,120000.00,,,,,0,00,99.99,,,,,,*00",
        "position_monotonic": time.monotonic(),
    }
    client = runtime.NtripCorrectionClient(
        _client_config(),
        lambda data: None,
        threading.Event(),
        latest_nmea=latest,
    )

    def unexpected_ranking(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("invalid GNSS position must not trigger candidate ranking")

    monkeypatch.setattr(client, "_mountpoint_candidates", unexpected_ranking)

    assert client._rover_position() is None
    assert client._periodic_switch_candidates({"mountpoint": "OLD"}) == []


def test_stale_live_position_is_not_used_for_handover() -> None:
    latest = {
        "gga": runtime.build_gga_sentence(37.5, 127.0),
        "position_monotonic": time.monotonic() - 31.0,
    }
    client = runtime.NtripCorrectionClient(
        _client_config(position_max_age=30.0),
        lambda data: None,
        threading.Event(),
        latest_nmea=latest,
    )

    assert client._rover_position() is None


def test_latest_valid_cached_position_wins_over_an_older_gga() -> None:
    latest = {
        "gga": runtime.build_gga_sentence(37.0, 127.0),
        "latitude_deg": 37.5,
        "longitude_deg": 127.5,
        "position_monotonic": time.monotonic(),
        "position_valid": True,
    }
    client = runtime.NtripCorrectionClient(
        _client_config(position_max_age=30.0),
        lambda data: None,
        threading.Event(),
        latest_nmea=latest,
    )

    assert client._rover_position() == (37.5, 127.5)
