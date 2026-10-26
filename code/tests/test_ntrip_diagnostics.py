from __future__ import annotations

import base64
import math
import socket
import threading
from types import SimpleNamespace

import pytest

from geonova_depthai import runtime
from geonova_depthai.capture import cli
from geonova_depthai.controller_bridge import ControllerBridge
from .test_ntrip_switching import ScriptedSocket, _client_config, _rtcm_frame


def gga(quality=1, latitude="3730.0000", longitude="12700.0000"):
    body = f"GNGGA,120000.00,{latitude},N,{longitude},E,{quality},18,0.7,42.5,M,20.0,M,1.2,0123"
    return f"${body}*{runtime.nmea_checksum(body)}\r\n"


def client(latest=None, **overrides):
    config = _client_config(selection="caster", mountpoint="RTK-RTCM32", position_wait_s=0.0)
    config.update(overrides)
    return runtime.NtripCorrectionClient(config, lambda data: None, threading.Event(), latest_nmea=latest)


@pytest.mark.parametrize("quality", range(9))
def test_gga_quality_and_checksum_gate(quality):
    sentence = gga(quality)
    assert bool(runtime.valid_ntrip_gga(sentence)) == (1 <= quality <= 5)
    assert runtime.valid_ntrip_gga(sentence[:-4] + "FF\r\n") is None


@pytest.mark.parametrize("sentence", [
    "$GNGGA,120000,3730,N,12700,E,1,18,0.7,42,M,,M,,",
    gga(latitude=""), gga(longitude=""), gga(latitude="3760.0000"),
    gga(latitude="9130.0000"), gga(longitude="18100.0000"),
    gga(latitude="-3730.0000"), gga(longitude="+12700.0000"),
    gga().replace("GNGGA", "GNRMC"), gga() + "X\r\nInjected: true",
])
def test_malformed_gga_is_never_transmitted(sentence):
    assert runtime.valid_ntrip_gga(sentence) is None
    assert client(gga=sentence)._current_gga() is None


def test_receiver_age_fallback_and_transition_logs(monkeypatch, capsys):
    now = [100.0]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])
    latest = {"gga": gga(), "gga_monotonic": 95.0}
    fixed = gga(2)
    caster = client(latest)
    legacy = client(latest, selection="client", gga=fixed, latitude=36.0, longitude=128.0)
    assert caster._current_gga() == gga()
    now[0] += 0.01
    assert caster._current_gga() is None
    assert caster._current_gga() is None
    assert legacy._current_gga() == fixed
    legacy.config["gga"] = "bad"
    assert runtime.parse_nmea_line(legacy._current_gga())["latitude_deg"] == 36.0
    latest["gga_monotonic"] = now[0]
    assert caster._current_gga() == gga()
    output = capsys.readouterr().out
    assert output.count("GGA unavailable") == 1
    assert output.count("GGA available") == 1


def test_receiver_without_timestamp_and_invalid_fix_use_only_valid_fallback():
    caster = client({"gga": gga()}, latitude=37.5, longitude=127.0)
    assert caster._current_gga() is None
    caster.config["gga"] = gga(4)
    assert caster._current_gga() == gga(4)
    caster.latest_nmea.update(gga=gga(0), gga_monotonic=runtime.time.monotonic())
    assert caster._current_gga() == gga(4)


def test_caster_never_queries_or_ranks_and_reconnects_same_mountpoint(monkeypatch, capsys):
    caster = client(mountpoint_candidates="OTHER,THIRD", reconnect_delay=7.0)
    attempted, waited = [], []

    def unexpected(*args, **kwargs):
        raise AssertionError("caster selection must not load or rank stations")

    monkeypatch.setattr(runtime, "fetch_ntrip_source_table", unexpected)
    monkeypatch.setattr(runtime, "fallback_rtk_mountpoint_entries", unexpected)
    monkeypatch.setattr(runtime, "sort_ntrip_mountpoint_entries", unexpected)
    assert caster._periodic_switch_candidates({"mountpoint": "OTHER"}) == []

    def stream(entry):
        attempted.append(entry["mountpoint"])
        raise TimeoutError("no RTCM data")

    def wait(delay):
        waited.append(delay)
        if len(waited) == 2:
            caster.stop_event.set()

    monkeypatch.setattr(caster, "_stream_mountpoint", stream)
    monkeypatch.setattr(caster.stop_event, "wait", wait)
    caster._run()
    assert attempted == ["RTK-RTCM32", "RTK-RTCM32"]
    assert waited == [7.0, 7.0]
    assert "reconnecting after error" in capsys.readouterr().out


@pytest.mark.parametrize("has_fix", [False, True])
def test_caster_gga_header_and_post_response_update(monkeypatch, has_fix):
    latest = {"gga": gga(), "gga_monotonic": runtime.time.monotonic()} if has_fix else {}
    caster = client(latest)
    sock = ScriptedSocket([b"HTTP/1.1 200 OK\r\n\r\n" + _rtcm_frame(b"one") + _rtcm_frame(b"two")])
    monkeypatch.setattr(runtime.socket, "create_connection", lambda *args, **kwargs: sock)
    stream = caster._open_mountpoint_stream({"mountpoint": "RTK-RTCM32"}, require_frame=True)
    sends = [event[2] for event in sock.events if event[0] == "send"]
    assert (b"Ntrip-GGA: " + gga().strip().encode() in sends[0]) == has_fix
    assert sends[1:] == ([gga().encode()] if has_fix else [])
    timestamp = caster.diagnostics_snapshot()["ntripLastGgaSentAtEpochMillis"]
    assert isinstance(timestamp, int) if has_fix else timestamp is None
    stream["socket"].close()


def test_caster_waits_only_for_receiver_gga_with_a_bounded_wait(monkeypatch):
    now = [10.0]
    latest = {}
    caster = client(latest, position_wait_s=0.3, gga=gga(), latitude=37.5, longitude=127.0)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])
    waits = []

    def wait(delay):
        waits.append(delay)
        now[0] += delay

    monkeypatch.setattr(caster.stop_event, "wait", wait)
    caster._wait_for_receiver_gga()
    assert sum(waits) == pytest.approx(0.3)
    waits.clear()
    latest.update(gga=gga(), gga_monotonic=now[0])
    caster._wait_for_receiver_gga()
    assert waits == []


def test_gga_is_sent_during_candidate_validation_and_when_first_fix_arrives(monkeypatch):
    now = [100.0]
    latest = {}
    caster = client(latest, gga_interval=0.2, connect_timeout=5.0, data_timeout=5.0)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])

    class ValidationSocket(ScriptedSocket):
        def recv(self, size):
            now[0] += 0.3
            if self.chunks:
                return super().recv(size)
            if now[0] < 101.5:
                latest.update(gga=gga(), gga_monotonic=now[0])
                raise socket.timeout()
            return _rtcm_frame(b"one") + _rtcm_frame(b"two")

    sock = ValidationSocket([b"HTTP/1.1 200 OK\r\n\r\n"])
    monkeypatch.setattr(runtime.socket, "create_connection", lambda *args, **kwargs: sock)
    stream = caster._open_mountpoint_stream({"mountpoint": "RTK-RTCM32"}, require_frame=True)
    sends = [event[2] for event in sock.events if event[0] == "send"]
    assert b"Ntrip-GGA:" not in sends[0]
    assert len(sends[1:]) >= 3
    assert all(sentence == gga().encode() for sentence in sends[1:])
    stream["socket"].close()


def test_gga_interval_staleness_and_immediate_recovery(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])
    latest = {"gga": gga(), "gga_monotonic": now[0]}
    caster = client(latest, gga_interval=10.0)
    sock = ScriptedSocket([])
    last = caster._send_gga(sock)
    now[0] += 1.0
    assert caster._send_gga(sock, last) == last
    now[0] += 5.0
    last = caster._send_gga(sock, last)
    assert last is None
    latest["gga_monotonic"] = now[0]
    assert caster._send_gga(sock, last) == now[0]
    assert len(sock.events) == 2


def test_caster_stream_disables_periodic_handover(monkeypatch):
    caster = client(reselect_interval=0.00001)
    frames = _rtcm_frame(b"one") + _rtcm_frame(b"two")
    sock = ScriptedSocket([b"HTTP/1.1 200 OK\r\n\r\n" + frames, frames])
    writes = []

    def write(data):
        writes.append(data)
        if len(writes) == 2:
            caster.stop_event.set()

    caster.serial_write = write
    monkeypatch.setattr(runtime.socket, "create_connection", lambda *args, **kwargs: sock)
    monkeypatch.setattr(caster, "_start_handover", lambda *args: pytest.fail("unexpected handover"))
    caster._stream_mountpoint({"mountpoint": "RTK-RTCM32"})
    assert b"".join(writes) == frames * 2
    assert sock.closed


def chunked(data, split=7):
    parts = [data[index:index + split] for index in range(0, len(data), split)]
    return b"".join(f"{len(part):x};test=1\r\n".encode() + part + b"\r\n" for part in parts) + b"0\r\nX-Trailer: ignored\r\n\r\n"


def test_chunk_decoder_all_byte_boundaries_and_crc():
    frames = _rtcm_frame(b"first") + _rtcm_frame(b"second")
    encoded = chunked(frames, 2)
    for split in range(1, len(encoded)):
        decoder, framer = runtime.NtripChunkDecoder(), runtime.Rtcm3Framer()
        received = []
        for offset in range(0, len(encoded), split):
            received.extend(framer.feed(decoder.feed(encoded[offset:offset + split])))
        assert received == [_rtcm_frame(b"first"), _rtcm_frame(b"second")]
        assert decoder.finished


@pytest.mark.parametrize("encoded", [b"oops\r\n", b"-1\r\n", b"1\r\naXX", b"X" * 4097])
def test_chunk_decoder_rejects_invalid_framing(encoded):
    with pytest.raises(RuntimeError, match="chunk"):
        runtime.NtripChunkDecoder().feed(encoded)


def test_chunked_stream_decodes_before_candidate_and_active_framing(monkeypatch):
    first = _rtcm_frame(b"first") + _rtcm_frame(b"second")
    third = _rtcm_frame(b"third")
    # Validation consumes only the first two frames; the third arrives in the active loop.
    initial = chunked(first, 3).split(b"0\r\nX-Trailer:", 1)[0]
    sock = ScriptedSocket([b"HTTP/1.1 200 OK\r\nTransfer-Encoding: Chunked\r\n\r\n", initial, chunked(third, 2)])
    caster = client()
    writes = []

    def write(data):
        writes.append(data)
        if third in data:
            caster.stop_event.set()

    caster.serial_write = write
    monkeypatch.setattr(runtime.socket, "create_connection", lambda *args, **kwargs: sock)
    caster._stream_mountpoint({"mountpoint": "RTK-RTCM32"})
    assert b"".join(writes) == first + third


def base_frame(message=1005, station=123, latitude=37.5, longitude=127.0):
    lat, lon = math.radians(latitude), math.radians(longitude)
    normal = 6378137.0 / math.sqrt(1.0 - 6.69437999014e-3 * math.sin(lat) ** 2)
    xyz = ((normal + 42.0) * math.cos(lat) * math.cos(lon),
           (normal + 42.0) * math.cos(lat) * math.sin(lon),
           (normal * (1.0 - 6.69437999014e-3) + 42.0) * math.sin(lat))
    # Pack the fields in wire order rather than reusing the decoder's offsets.
    fields = [(message, 12), (station, 12), (0, 6), (1, 1), (1, 1), (1, 1), (0, 1),
              (round(xyz[0] * 10000), 38), (0, 1), (0, 1),
              (round(xyz[1] * 10000), 38), (0, 2), (round(xyz[2] * 10000), 38)]
    if message == 1006:
        fields.append((1200, 16))
    bits, length = 0, 0
    for value, width in fields:
        bits = (bits << width) | (value & ((1 << width) - 1))
        length += width
    return _rtcm_frame(bits.to_bytes(length // 8, "big"))


def msm_frame(message=1077, signal_ids=(2, 10, 23)):
    # Pack in wire order, independently of the decoder's offsets. Nonzero fields
    # around the masks expose offset errors; both ends of the satellite mask are set.
    fields = [(message, 12), (0xABC, 12), (0x12345678, 30), (1, 1), (5, 3),
              (0x55, 7), (2, 2), (1, 2), (1, 1), (5, 3)]
    fields.extend((int(satellite in (1, 9, 64)), 1) for satellite in range(1, 65))
    fields.extend((int(signal in signal_ids), 1) for signal in range(1, 33))
    # Three satellites, all cells present. Zero observation fields complete MSM4-7.
    fields.extend((1, 1) for _ in range(3 * len(signal_ids)))
    variant = message % 10
    satellite_width = 36 if variant in (5, 7) else 18
    cell_width = {4: 48, 5: 63, 6: 65, 7: 80}.get(variant, 0)
    fields.extend((0, satellite_width) for _ in range(3))
    fields.extend((0, cell_width) for _ in range(3 * len(signal_ids)))
    wire = "".join(format(value, f"0{width}b") for value, width in fields if width)
    wire += "0" * (-len(wire) % 8)
    return _rtcm_frame(int(wire, 2).to_bytes(len(wire) // 8, "big"))


@pytest.mark.parametrize("message,signal_ids,expected", [
    (1077, (2, 10, 23), ["GPS 1C", "GPS 2W", "GPS 5Q"]),
    (1097, (2, 15, 23), ["GAL 1C", "GAL 7Q", "GAL 5Q"]),
    (1127, (2, 23), ["BDS 2I", "BDS 5P"]),
    (1084, (2, 11, 13), ["GLO 1C", "GLO 3I", "GLO 3X"]),
    (1115, (2, 9, 23, 32), ["QZS 1C", "QZS 6S", "QZS 5Q", "QZS 1X"]),
    (1106, (1, 2, 32), ["SBS ?1", "SBS ?2", "SBS ?32"]),
    (1137, (1, 22, 32), ["IRN ?1", "IRN ?22", "IRN ?32"]),
    (1077, (1, 5, 32), ["GPS ?1", "GPS ?5", "GPS 1X"]),
])
def test_msm_signal_masks_and_rinex_codes(message, signal_ids, expected):
    frame = msm_frame(message, signal_ids)
    wire = "".join(format(byte, "08b") for byte in frame[3:-3])
    # 12+12+30+1+3+7+2+2+1+3 = 73, then 64 satellite bits and 32 signal bits.
    assert wire[73:137] == "".join("1" if satellite in (1, 9, 64) else "0" for satellite in range(1, 65))
    assert wire[137:169] == "".join("1" if signal in signal_ids else "0" for signal in range(1, 33))
    assert runtime.rtcm3_msm_signals(frame) == expected


@pytest.mark.parametrize("variant", range(4, 8))
def test_all_msm4_to_7_signal_masks(variant):
    assert runtime.rtcm3_msm_signals(msm_frame(1070 + variant, (2, 23))) == ["GPS 1C", "GPS 5Q"]


@pytest.mark.parametrize("message", [1004, 1012, 1071, 1072, 1073, 1078, 1147])
def test_legacy_and_other_messages_do_not_add_msm_signals(message):
    caster = client()
    frame = msm_frame(message)
    assert runtime.rtcm3_msm_signals(frame) == []
    caster._forward_rtcm_frames([frame])
    state = caster.diagnostics_snapshot()
    assert state["rtcmSignals"] == []
    assert state["rtcmMessageTypes"] == [message]


def test_msm_signals_union_expiry_status_and_transition_logs(monkeypatch, capsys):
    now = [100.0]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])
    caster = client()
    frames = [msm_frame(), msm_frame(1097, (2, 15, 23)), msm_frame(1127, (2, 23))]
    writes = []
    caster.serial_write = writes.append
    caster._forward_rtcm_frames(frames)
    caster._forward_rtcm_frames(frames)
    expected = sorted(["GPS 1C", "GPS 2W", "GPS 5Q", "GAL 1C", "GAL 5Q", "GAL 7Q", "BDS 2I", "BDS 5P"])
    assert caster.diagnostics_snapshot()["rtcmSignals"] == expected
    assert writes == [b"".join(frames)] * 2
    bridge = ControllerBridge(SimpleNamespace(controller_bridge_enabled=False, enable_gps=True))
    reader = SimpleNamespace(rtk_client=caster, latest_sample=lambda: {})
    assert bridge._gnss_state(reader, 1000, 1000)["rtcmSignals"] == expected
    assert bridge._gnss_state(None, 1000, 1000)["rtcmSignals"] == []
    assert bridge._gnss_state(SimpleNamespace(rtk_client=object()), 1000, 1000)["rtcmSignals"] == []
    snapshot = caster.diagnostics_snapshot()
    snapshot["rtcmSignals"].clear()
    assert caster.diagnostics_snapshot()["rtcmSignals"] == expected
    logs = [line for line in capsys.readouterr().out.splitlines() if "MSM signals:" in line]
    assert logs == ["RTK NTRIP MSM signals: GPS 1C,2W,5Q; GAL 1C,5Q,7Q; BDS 2I,5P"]

    now[0] = 150.0
    caster._forward_rtcm_frames([msm_frame(signal_ids=(2,))])
    now[0] = 160.0
    assert caster.diagnostics_snapshot()["rtcmSignals"] == expected
    assert capsys.readouterr().out == ""
    now[0] = 160.01
    assert caster.diagnostics_snapshot()["rtcmSignals"] == ["GPS 1C"]
    assert capsys.readouterr().out.count("MSM signals:") == 1
    now[0] = 210.01
    assert caster.diagnostics_snapshot()["rtcmSignals"] == []
    assert "RTK NTRIP MSM signals: -" in capsys.readouterr().out
    caster.diagnostics_snapshot()
    assert capsys.readouterr().out == ""


def test_msm_signals_ignore_bad_crc_and_short_headers(capsys):
    caster = client()
    damaged = bytearray(msm_frame())
    damaged[-1] ^= 1
    caster._forward_rtcm_frames([bytes(damaged)])
    assert caster.diagnostics_snapshot()["rtcmMessageTypes"] == []
    assert caster.diagnostics_snapshot()["rtcmSignals"] == []
    # Preserve valid CRCs while truncating every possible MSM header byte.
    payload = msm_frame()[3:-3]
    for length in range(22):
        short = _rtcm_frame(payload[:length])
        assert runtime.rtcm3_msm_signals(short) == []
        caster._forward_rtcm_frames([short])
    assert caster.diagnostics_snapshot()["rtcmSignals"] == []
    assert "MSM signals:" not in capsys.readouterr().out
    caster._forward_rtcm_frames([msm_frame()])
    assert caster.diagnostics_snapshot()["rtcmSignals"] == ["GPS 1C", "GPS 2W", "GPS 5Q"]


def test_msm_signals_sorted_and_bounded():
    caster = client()
    frames = [msm_frame(group * 10 + 7, range(1, 33)) for group in range(107, 114)]
    caster._forward_rtcm_frames(frames)
    expected = sorted({signal for frame in frames for signal in runtime.rtcm3_msm_signals(frame)})[:48]
    assert len(expected) == 48
    assert caster.diagnostics_snapshot()["rtcmSignals"] == expected


@pytest.mark.parametrize("message,latitude,longitude", [(1005, 37.5, 127.0), (1006, -35.0, -58.0), (1005, 90.0, 0.0)])
def test_rtcm_base_coordinates_signed_ecef(message, latitude, longitude):
    station, lat, lon = runtime.rtcm3_base_position(base_frame(message, 4095, latitude, longitude))
    assert station == 4095
    assert lat == pytest.approx(latitude, abs=1e-7)
    assert lon == pytest.approx(longitude, abs=1e-7)


def test_rtcm_diagnostics_crc_window_station_changes_and_distance(monkeypatch, capsys):
    now = [100.0]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])
    latest = {"latitude_deg": 37.51, "longitude_deg": 127.0, "position_monotonic": now[0], "position_valid": True}
    caster = client(latest)
    valid = base_frame()
    bad = bytearray(base_frame(station=777))
    bad[-1] ^= 1
    caster._forward_rtcm_frames([bytes(bad), valid, valid])
    state = caster.diagnostics_snapshot()
    assert state["rtcmMessageTypes"] == [1005]
    assert state["rtcmBaseStationId"] == 123
    assert state["rtcmBaseDistanceM"] == pytest.approx(1111.9, abs=0.1)
    assert isinstance(state["rtcmLastAtEpochMillis"], int)
    assert capsys.readouterr().out.count("base station:") == 1
    caster._forward_rtcm_frames([base_frame(1006, station=124)])
    assert caster.diagnostics_snapshot()["rtcmBaseStationId"] == 124
    assert "id=0124" in capsys.readouterr().out
    for number in range(1070, 1110):
        caster._forward_rtcm_frames([_rtcm_frame((number << 4).to_bytes(2, "big"))])
    types = caster.diagnostics_snapshot()["rtcmMessageTypes"]
    assert types == sorted(types) and len(types) == 32
    types.clear()
    assert len(caster.diagnostics_snapshot()["rtcmMessageTypes"]) == 32
    now[0] = 160.01
    state = caster.diagnostics_snapshot()
    assert state["rtcmMessageTypes"] == []
    assert state["rtcmBaseDistanceM"] is None
    assert state["rtcmBaseStationId"] == 124


def test_diagnostics_rejects_truncated_and_undefined_base_positions():
    assert runtime.rtcm3_base_position(_rtcm_frame(b"\x3e\xd0")) is None
    assert runtime.rtcm3_base_position(_rtcm_frame(b"\x3e\xd0" + b"\0" * 17)) is None
    caster = client(latitude=37.5, longitude=127.0)
    caster._forward_rtcm_frames([base_frame()])
    assert caster.diagnostics_snapshot()["rtcmBaseDistanceM"] is None


def test_status_contract_and_error_redaction():
    latest = {"gga": gga(), "gga_monotonic": runtime.time.monotonic()}
    caster = client(latest, username="test-account", password="test-password")
    caster._forward_rtcm_frames([base_frame()])
    caster._send_gga(ScriptedSocket([]))
    token = base64.b64encode(b"test-account:test-password").decode()
    caster.error = f"test-account test-password {token}\nAuthorization: Basic other-token"
    diagnostics = caster.diagnostics_snapshot()
    assert "test-account" not in diagnostics["ntripError"]
    assert "test-password" not in diagnostics["ntripError"]
    assert token not in diagnostics["ntripError"]
    assert "other-token" not in diagnostics["ntripError"]
    bridge = ControllerBridge(SimpleNamespace(controller_bridge_enabled=False, enable_gps=True))
    reader = SimpleNamespace(rtk_client=caster, started=True, error=None, latest_sample=lambda: {})
    state = bridge._gnss_state(reader, 1000, 1000)
    for key, value in diagnostics.items():
        assert state[key] == value
    assert state["ntripSelection"] == "caster"
    assert state["rtcmMessageTypes"] == [1005]
    assert state["rtcmBaseStationId"] == 123
    assert state["rtcmBaseDistanceM"] == 0.0
    assert isinstance(state["rtcmLastAtEpochMillis"], int)
    assert isinstance(state["ntripLastGgaSentAtEpochMillis"], int)
    empty = bridge._gnss_state(None, 1000, 1000)
    assert {key: empty[key] for key in diagnostics} == {
        key: ([] if key in ("rtcmMessageTypes", "rtcmSignals") else None) for key in diagnostics
    }


def test_bad_checksum_does_not_change_nmea_state_fields():
    parser = runtime.NmeaParserState()
    good = parser(gga(4))
    assert good["rtk_status"] == "fixed"
    assert good["satellites"] == "18"
    assert good["differential_age_s"] == "1.2"
    assert good["reference_station_id"] == "0123"
    previous = dict(parser.latest)
    assert parser(gga(0)[:-4] + "FF\r\n") == {}
    assert parser.latest == previous
    assert parser(gga(5))["rtk_status"] == "float"


def test_serial_reopen_clears_gga_and_bad_checksum_is_recorded(monkeypatch):
    reader = runtime.SerialRateLimitedReader("gps", "TEST", 115200, 0, parser=runtime.NmeaParserState(), rtk_config=_client_config())
    monkeypatch.setattr(runtime.NtripCorrectionClient, "start", lambda self: None)

    class Port:
        def __init__(self, lines):
            self.lines = iter(lines)

        def readline(self):
            try:
                return next(self.lines).encode()
            except StopIteration:
                reader.connection_stop_event.set()
                return b""

        def write(self, data):
            pass

    reader.connection_stop_event = threading.Event()
    reader._read_connection(Port([gga(4)]), reader.connection_stop_event)
    assert runtime.valid_ntrip_gga(reader.latest_nmea["gga"])
    assert reader.latest_nmea["gga_monotonic"] is not None
    reader.connection_stop_event = threading.Event()
    bad = gga(0)[:-4] + "FF\r\n"
    reader._read_connection(Port([bad]), reader.connection_stop_event)
    assert reader.latest_nmea == {}
    assert reader.sample_count == 2
    assert reader.drain()[-1]["raw"] == bad.strip()
    assert reader.recent[-1]["raw"] == bad.strip()


@pytest.mark.parametrize("ntrip_enabled", [False, True])
def test_bad_gga_is_recorded_without_changing_last_fix_or_sending(monkeypatch, ntrip_enabled):
    # A constant receive time also checks that bad GGA bypasses the GPS row limit.
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(runtime.NtripCorrectionClient, "start", lambda self: None)
    config = _client_config(selection="caster", mountpoint="RTK-RTCM32") if ntrip_enabled else None
    reader = runtime.SerialRateLimitedReader("gps", "TEST", 115200, 120, parser=runtime.NmeaParserState(), rtk_config=config)
    bridge = ControllerBridge(SimpleNamespace(controller_bridge_enabled=False, enable_gps=True))
    stop = threading.Event()
    bad = gga(0, latitude="0100.0000", longitude="00200.0000")[:-4] + "FF\r\n"
    before = {}
    sock = ScriptedSocket([])

    class Port:
        def __init__(self):
            self.index = 0

        def readline(self):
            self.index += 1
            if self.index == 1:
                return gga(4).encode()
            if self.index == 2:
                before["sample"] = reader.latest_sample()
                before["parser"] = dict(reader.parser.latest)
                before["nmea"] = dict(reader.latest_nmea)
                before["state"] = bridge._gnss_state(reader, 1000, runtime.time.monotonic_ns())
                if ntrip_enabled:
                    assert reader.rtk_client._send_gga(sock) == 100.0
                return bad.encode()
            stop.set()
            return b""

    reader._read_connection(Port(), stop)
    recorded = reader.drain()
    assert [sample["raw"] for sample in recorded] == [gga(4).strip(), bad.strip()]
    assert list(reader.recent) == recorded
    assert "fix_quality" not in recorded[-1]
    assert "latitude_deg" not in recorded[-1]
    assert "nmea_type" not in recorded[-1]
    assert reader.sample_count == 2 and reader.dropped_count == 0
    assert reader.parser.latest == before["parser"]
    assert reader.latest_sample() == before["sample"]
    state = bridge._gnss_state(reader, 1000, runtime.time.monotonic_ns())
    for key in ("fixQuality", "fixType", "rtkStatus", "latitude", "longitude", "altitudeM",
                "satellites", "hdop", "differentialAgeS", "referenceStationId"):
        assert state[key] == before["state"][key]
    assert state["fixQuality"] == 4 and state["latitude"] == 37.5
    if ntrip_enabled:
        assert reader.latest_nmea == {key: value for key, value in before["nmea"].items() if key not in ("gga", "gga_monotonic")}
        assert reader.rtk_client._send_gga(sock) is None
        assert [event[2] for event in sock.events] == [gga(4).encode()]


@pytest.mark.parametrize("selection", ["wrong", "", None])
def test_invalid_selection_rejected_at_startup(selection):
    with pytest.raises(ValueError, match="selection"):
        runtime.build_rtk_config(SimpleNamespace(rtk_ntrip_selection=selection))


@pytest.mark.parametrize("mountpoint", ["", "ONE,TWO"])
def test_caster_requires_one_mountpoint(mountpoint):
    with pytest.raises(ValueError, match="exactly one"):
        client(mountpoint=mountpoint)


def test_cli_yaml_new_settings_and_invalid_selection(tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text("rtk_ntrip_selection: caster\nrtk_ntrip_required_messages: [1077, 1097]\nrtk_ntrip_gga_max_age_s: 3.0\n", encoding="utf-8")
    args = cli.parse_args(["--config", str(path)])
    assert args.rtk_ntrip_selection == "caster"
    assert args.rtk_ntrip_required_messages == [1077, 1097]
    assert args.rtk_ntrip_gga_max_age_s == 3.0
    args = cli.parse_args(["--rtk-ntrip-selection", "client", "--rtk-ntrip-required-messages", "1077"])
    assert args.rtk_ntrip_selection == "client"
    assert args.rtk_ntrip_required_messages == [1077]
    assert cli.parse_args([]).rtk_ntrip_gga_max_age_s == 5.0
    path.write_text("rtk_ntrip_selection: typo\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        cli.parse_args(["--config", str(path)])


@pytest.mark.parametrize("relay,explicit", [(False, False), (True, False), (True, True)])
def test_mobile_relay_yaml_and_explicit_cli_priority(tmp_path, monkeypatch, relay, explicit):
    path = tmp_path / "profile.yaml"
    path.write_text("rtk_ntrip_host: rts1.ngii.go.kr\nrtk_ntrip_port: 2101\n", encoding="utf-8")
    monkeypatch.setenv("JETSON_PIPELINE_MOBILE_RTK_RELAY", "1" if relay else "0")
    monkeypatch.setenv("NTRIP_HOST", "192.168.49.61")
    monkeypatch.setenv("NTRIP_PORT", "39535")
    argv = ["--config", str(path)]
    if explicit:
        argv += ["--rtk-ntrip-host=explicit.test", "--rtk-ntrip-port", "9876"]
    args = cli.parse_args(argv)
    if explicit:
        assert (args.rtk_ntrip_host, args.rtk_ntrip_port) == ("explicit.test", 9876)
    elif relay:
        assert (args.rtk_ntrip_host, args.rtk_ntrip_port) == ("192.168.49.61", 39535)
        assert runtime.build_rtk_config(args)["caster_host"] == "rts1.ngii.go.kr"
    else:
        assert (args.rtk_ntrip_host, args.rtk_ntrip_port) == ("rts1.ngii.go.kr", 2101)


def test_mobile_relay_uses_default_upstream_when_yaml_has_no_host(monkeypatch):
    monkeypatch.setenv("JETSON_PIPELINE_MOBILE_RTK_RELAY", "1")
    monkeypatch.setenv("NTRIP_HOST", "relay.test")
    monkeypatch.setenv("NTRIP_PORT", "3333")
    args = cli.parse_args([])
    assert (args.rtk_ntrip_host, args.rtk_ntrip_port) == ("relay.test", 3333)
    assert args.rtk_ntrip_caster_host == "www.gnssdata.or.kr"


def test_mobile_relay_expands_tilde_config_path(tmp_path, monkeypatch):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / "profile.yaml"
    path.write_text("rtk_ntrip_host: rts1.ngii.go.kr\nrtk_ntrip_port: 2101\n", encoding="utf-8")
    monkeypatch.setenv("JETSON_PIPELINE_MOBILE_RTK_RELAY", "1")
    monkeypatch.setenv("NTRIP_HOST", "relay.test")
    monkeypatch.setenv("NTRIP_PORT", "3333")
    args = cli.parse_args(["--config", "~/profile.yaml"])
    assert (args.rtk_ntrip_host, args.rtk_ntrip_port) == ("relay.test", 3333)
    assert (args.rtk_ntrip_caster_host, args.rtk_ntrip_caster_port) == ("rts1.ngii.go.kr", 2101)


@pytest.mark.parametrize("options,expected", [
    (["--rtk-ntrip-h", "explicit.test", "--rtk-ntrip-por", "9876"], ("explicit.test", 9876)),
    (["--rtk-ntrip-h=explicit.test", "--rtk-ntrip-por=9876"], ("explicit.test", 9876)),
    (["--rtk-ntrip-h", "explicit.test"], ("explicit.test", 3333)),
    (["--rtk-ntrip-por", "9876"], ("relay.test", 9876)),
])
def test_mobile_relay_preserves_abbreviated_cli_options(tmp_path, monkeypatch, options, expected):
    path = tmp_path / "profile.yaml"
    path.write_text("rtk_ntrip_host: rts1.ngii.go.kr\nrtk_ntrip_port: 2101\n", encoding="utf-8")
    monkeypatch.setenv("JETSON_PIPELINE_MOBILE_RTK_RELAY", "1")
    monkeypatch.setenv("NTRIP_HOST", "relay.test")
    monkeypatch.setenv("NTRIP_PORT", "3333")
    args = cli.parse_args(["--config", str(path)] + options)
    assert (args.rtk_ntrip_host, args.rtk_ntrip_port) == expected
    if "relay.test" in expected or 3333 in expected:
        assert (args.rtk_ntrip_caster_host, args.rtk_ntrip_caster_port) == ("rts1.ngii.go.kr", 2101)


@pytest.mark.parametrize("username,password", [(123456, 987654), (0, 987654), (123456, 0)],
                         ids=["integers", "zero-username", "zero-password"])
def test_numeric_yaml_credentials_are_strings_and_reconnect(tmp_path, monkeypatch, capsys, username, password):
    path = tmp_path / "profile.yaml"
    path.write_text("rtk_ntrip_host: caster.test\nrtk_ntrip_selection: caster\n"
                    "rtk_ntrip_mountpoint: RTK-RTCM32\nrtk_ntrip_position_wait_s: 0\n"
                    f"rtk_ntrip_username: {username}\nrtk_ntrip_password: {password}\n", encoding="utf-8")
    args = cli.parse_args(["--config", str(path)])
    assert isinstance(args.rtk_ntrip_username, str)
    assert isinstance(args.rtk_ntrip_password, str)
    config = runtime.build_rtk_config(args)
    assert config["username"] == str(username) and config["password"] == str(password)
    # Also exercise the configuration boundary independently of argparse.
    args.rtk_ntrip_username, args.rtk_ntrip_password = username, password
    config = runtime.build_rtk_config(args)
    assert isinstance(config["username"], str) and isinstance(config["password"], str)
    caster = runtime.NtripCorrectionClient(config, lambda data: None, threading.Event())
    attempts, waits = [], []
    token = base64.b64encode(f"{username}:{password}".encode()).decode()

    def connect(*args, **kwargs):
        attempts.append(True)
        raise OSError(f"connection failed {username}:{password} Basic {token}")

    def wait(delay):
        waits.append(delay)
        if len(waits) == 2:
            caster.stop_event.set()

    monkeypatch.setattr(runtime.socket, "create_connection", connect)
    monkeypatch.setattr(caster.stop_event, "wait", wait)
    caster._run()
    assert len(attempts) == 2 and len(waits) == 2
    output = capsys.readouterr().out + caster.diagnostics_snapshot()["ntripError"]
    assert not any(secret in output for secret in (str(username), str(password), token))


@pytest.mark.parametrize("username,password", [(123456, 987654), (0, 987654), (None, 987654), (False, [987654])],
                         ids=["integers", "zero", "none", "other-types"])
def test_redaction_handles_non_string_credentials(username, password):
    user = "" if username is None else str(username)
    secret = "" if password is None else str(password)
    token = base64.b64encode(f"{user}:{secret}".encode()).decode()
    result = runtime.redact_ntrip_error(f"failed {user}:{secret} Basic {token}", {"username": username, "password": password})
    assert not any(value in result for value in (user, secret, token) if value)


def test_redaction_never_raises_for_unprintable_or_invalid_inputs():
    class Unprintable:
        def __str__(self):
            raise ValueError("cannot format")

    for error, config in ((Unprintable(), {}), ("failure", None), ("failure", []),
                          ("failure", {"username": Unprintable()})):
        assert runtime.redact_ntrip_error(error, config) == "NTRIP error (details redacted)"


@pytest.mark.parametrize("transition", ["base", "messages", "signals", "gga-unavailable", "gga-available",
                                      "expired-messages", "expired-signals"])
def test_blocked_diagnostic_stdout_does_not_hold_client_lock(monkeypatch, transition):
    now = [100.0]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])
    caster = client()
    frames = [base_frame(), msm_frame()]
    if transition.startswith("expired-"):
        caster._forward_rtcm_frames(frames)
        now[0] = 160.01
    elif transition == "gga-available":
        caster._current_gga()
        caster.latest_nmea.update(gga=gga(), gga_monotonic=now[0])
    prefixes = {"base": "RTK NTRIP base station:", "messages": "RTK NTRIP RTCM message types:",
                "signals": "RTK NTRIP MSM signals:", "gga-unavailable": "RTK NTRIP GGA unavailable:",
                "gga-available": "RTK NTRIP GGA available:",
                "expired-messages": "RTK NTRIP RTCM message types:", "expired-signals": "RTK NTRIP MSM signals:"}
    entered, release, observed = threading.Event(), threading.Event(), threading.Event()
    errors, snapshots = [], []

    def blocked_print(*args, **kwargs):
        if str(args[0]).startswith(prefixes[transition]) and not entered.is_set():
            entered.set()
            release.wait(5.0)

    def produce():
        try:
            if transition.startswith("expired-"):
                caster.diagnostics_snapshot()
            elif transition.startswith("gga-"):
                caster._current_gga()
            else:
                caster._forward_rtcm_frames(frames)
        except Exception as error:
            errors.append(error)

    def observe():
        try:
            # Streaming updates and bridge queries must both work while stdout blocks.
            caster._forward_rtcm_frames(frames)
            bridge = ControllerBridge(SimpleNamespace(controller_bridge_enabled=False, enable_gps=True))
            reader = SimpleNamespace(rtk_client=caster, latest_sample=lambda: {})
            snapshots.append(bridge._gnss_state(reader, 1000, 1000))
        except Exception as error:
            errors.append(error)
        finally:
            observed.set()

    monkeypatch.setattr(runtime, "print", blocked_print, raising=False)
    producer = threading.Thread(target=produce, daemon=True)
    observer = threading.Thread(target=observe, daemon=True)
    producer.start()
    try:
        assert entered.wait(1.0)
        observer.start()
        assert observed.wait(1.0)
        assert snapshots[0]["rtcmMessageTypes"] == [1005, 1077]
        assert snapshots[0]["rtcmSignals"] == ["GPS 1C", "GPS 2W", "GPS 5Q"]
        assert snapshots[0]["rtcmBaseStationId"] == 123
    finally:
        release.set()
        producer.join(timeout=1.0)
        if observer.ident is not None:
            observer.join(timeout=1.0)
    assert errors == []
    assert not producer.is_alive() and not observer.is_alive()


def test_diagnostics_snapshots_are_safe_during_stream_updates():
    caster = client()
    finished = threading.Event()
    errors = []

    def update():
        try:
            for number in range(1070, 1100):
                caster._forward_rtcm_frames([msm_frame(number, (2, 23))])
                caster.error = "temporary failure" if number % 2 else None
        except Exception as error:
            errors.append(error)
        finally:
            finished.set()

    worker = threading.Thread(target=update)
    worker.start()
    while not finished.is_set():
        state = caster.diagnostics_snapshot()
        assert state["rtcmMessageTypes"] == sorted(set(state["rtcmMessageTypes"]))
        assert len(state["rtcmMessageTypes"]) <= 32
        assert state["rtcmSignals"] == sorted(set(state["rtcmSignals"]))
        assert len(state["rtcmSignals"]) <= 48
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert errors == []


def test_caster_start_logs_ignored_auto_selection_once(monkeypatch, capsys):
    caster = client()
    monkeypatch.setattr(caster, "_run", lambda: None)
    caster.start()
    caster.stop()
    assert capsys.readouterr().out.count("auto_mountpoint ignored") == 1


def test_error_logs_do_not_echo_credentials_or_response_headers(monkeypatch, capsys):
    caster = client(username="test-account", password="test-password", reconnect_delay=0.0)

    def stream(entry):
        caster.stop_event.set()
        raise RuntimeError("test-account:test-password\nAuthorization: Basic injected-token")

    monkeypatch.setattr(caster, "_stream_mountpoint", stream)
    caster._run()
    assert "test-account" not in caster.error
    assert "test-password" not in caster.error
    assert "injected-token" not in caster.error
    sock = ScriptedSocket([b"HTTP/1.1 401 test-account:test-password\r\nAuthorization: Basic injected-token\r\n\r\n"])
    caster.stop_event.clear()
    monkeypatch.setattr(runtime.socket, "create_connection", lambda *args, **kwargs: sock)
    with pytest.raises(RuntimeError, match="401 Unauthorized") as error:
        caster._open_mountpoint_stream({"mountpoint": "RTK-RTCM32"})
    assert str(error.value) == "NTRIP caster rejected request: 401 Unauthorized"
    assert "test-account" not in capsys.readouterr().out


def test_valid_checksum_does_not_allow_embedded_line_breaks():
    body = gga().strip()[1:].split("*", 1)[0] + "\r\nInjected: value"
    assert runtime.valid_ntrip_gga(f"${body}*{runtime.nmea_checksum(body)}") is None


def test_relay_enabled_log_includes_configured_upstream(monkeypatch, capsys):
    reader = runtime.SerialRateLimitedReader("gps", "TEST", 115200, 0, rtk_config=_client_config(
        selection="caster", host="192.168.49.61", port=39535, mountpoint="RTK-RTCM32",
        caster_host="rts1.ngii.go.kr", caster_port=2101))
    monkeypatch.setattr(runtime.NtripCorrectionClient, "start", lambda self: None)
    stop = threading.Event()
    stop.set()
    reader._read_connection(SimpleNamespace(write=lambda data: None), stop)
    assert "RTK NTRIP enabled: 192.168.49.61:39535/RTK-RTCM32 (mobile relay to rts1.ngii.go.kr:2101)" in capsys.readouterr().out
