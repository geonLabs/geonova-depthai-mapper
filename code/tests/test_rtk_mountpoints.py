from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from geonova_depthai import runtime


SOURCE_TABLE = """HTTP/1.1 200 OK\r
Content-Type: text/plain\r
\r
STR;YANJ-RTCM31;YANJ-RTCM31;RTCM 3.1;RTCM(1);2;GPS+GLONASS;Single Base;KOR;37.41;126.55;0;0;;;B;N;0;;
STR;GANS-RTCM31;GANS-RTCM31;RTCM 3.1;RTCM(1);2;GPS+GLONASS;SMG;KOR;37.50;126.90;0;0;;;B;N;0;;
STR;SUWN-RTCM32;SUWN-RTCM32;RTCM 3.2;RTCM(1);2;GPS+GLONASS;Single Base;KOR;37.28;127.05;0;0;;;B;N;0;;
STR;PAJU-RTCM31;PAJU-RTCM31;RTCM 3.1;RTCM(1);2;GPS+GLONASS;Single Base;KOR;37.75;126.74;0;0;;;B;N;0;;
ENDSOURCETABLE
"""


class SourceTableSocket:
    def __init__(self, response):
        self.response = response.encode("utf-8")
        self.request = b""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def settimeout(self, timeout):  # noqa: ARG002
        return None

    def sendall(self, data):
        self.request += data

    def recv(self, size):
        if not self.response:
            return b""
        chunk, self.response = self.response[:size], self.response[size:]
        return chunk


def test_source_table_request_omits_stream_credentials_for_public_catalogue(monkeypatch) -> None:
    sock = SourceTableSocket(SOURCE_TABLE)
    monkeypatch.setattr(runtime.socket, "create_connection", lambda *args, **kwargs: sock)

    text = runtime.fetch_ntrip_source_table({
        "host": "www.gnssdata.or.kr",
        "port": 2101,
        "username": "field-user",
        "password": "field-password",
    })

    assert "STR;YANJ-RTCM31" in text
    assert b"Authorization:" not in sock.request


def test_private_source_table_retries_with_stream_credentials(monkeypatch) -> None:
    unauthorized = SourceTableSocket("HTTP/1.1 401 Unauthorized\r\n\r\n")
    authorized = SourceTableSocket(SOURCE_TABLE)
    sockets = iter((unauthorized, authorized))
    monkeypatch.setattr(
        runtime.socket,
        "create_connection",
        lambda *args, **kwargs: next(sockets),
    )

    text = runtime.fetch_ntrip_source_table({
        "host": "private-caster.test",
        "port": 2101,
        "username": "field-user",
        "password": "field-password",
    })

    token = base64.b64encode(b"field-user:field-password")
    assert "STR;YANJ-RTCM31" in text
    assert b"Authorization:" not in unauthorized.request
    assert b"Authorization: Basic " + token in authorized.request


@pytest.mark.parametrize("kind", ["username", "password", "basic", "authorization"])
@pytest.mark.parametrize("boundary", [130, 153, 159])
def test_source_table_redacts_credentials_before_status_truncation(monkeypatch, kind, boundary):
    config = {"host": "caster.test", "port": 2101,
              "username": "boundary-account-must-remain-private",
              "password": "boundary-password-must-remain-private"}
    token = base64.b64encode((config["username"] + ":" + config["password"]).encode()).decode()
    secret = config.get(kind, token)
    prefix = "HTTP/1.1 401 Unauthorized "
    if kind == "authorization":
        prefix += "Authorization: Basic "
    response = prefix + "x" * (boundary - len(prefix)) + secret + "\r\n\r\n"
    sockets = []

    def connect(*args, **kwargs):
        sock = SourceTableSocket(response)
        sockets.append(sock)
        return sock

    monkeypatch.setattr(runtime.socket, "create_connection", connect)
    with pytest.raises(RuntimeError, match="401 Unauthorized") as error:
        runtime.fetch_ntrip_source_table(config)
    assert len(sockets) == 2
    fragments = [value[:7] for value in (config["username"], config["password"], token)]
    assert not any(fragment in str(error.value) for fragment in fragments)
    safe_status = (prefix + "x" * (boundary - len(prefix)) + "[redacted]")[:160]
    if kind == "authorization":
        safe_status = "HTTP/1.1 401 Unauthorized [redacted]"
    assert str(error.value) == (
        "NTRIP source table did not contain STR mountpoint entries "
        f"(responses: {safe_status}; {safe_status})"
    )


@pytest.mark.parametrize("required", [[1005], [1007], [1005, 1007], [1014, 1015, 1016]])
def test_rts1_source_table_accepts_slash_grouped_message_numbers(required):
    # Captured RTS-1 STR lines from pipeline-rtk.md, including commas in periods.
    table = "\n".join([
        "STR;VRS-RTCM31;VRS-RTCM31;RTCM 3.1;1004(1),1005/1007(5),1012(1),PBS(10);2;GPS+GLONASS;NGII;KOR;0;0;1;1;Trimble Pivot Platform;none;B;N;0;;",
        "STR;RTK-RTCM32;RTK-RTCM32;RTCM 3.2;1004(1),1005/1007(5),1014(1, 1 msgs),1015(1, all msgs),1016(1, all msgs);2;GPS+GLO+GAL+BDS+QZS;NGII;KOR;0;0;1;1;Trimble Pivot Platform;none;B;N;0;;",
        "STR;VRS-RTCM34;VRS-RTCM34;RTCM 3.4;1004(1),1005/1007(5),1014(1, 1 msgs),1015(1, all msgs),1016(1, all msgs);2;GPS+GLO+GAL+BDS+QZS;NGII;KOR;0;0;1;1;Trimble Pivot Platform;none;B;N;0;;",
    ])
    entries = runtime.parse_ntrip_source_table(table, "RTCM32", required)
    assert [entry["mountpoint"] for entry in entries] == ["RTK-RTCM32"]
    expected = [] if 1014 in required else ["VRS-RTCM31"]
    assert [entry["mountpoint"] for entry in runtime.parse_ntrip_source_table(table, "RTCM31", required)] == expected
    assert runtime.parse_ntrip_source_table(table, "RTCM32", [1]) == []
    assert runtime.parse_ntrip_source_table(table, "RTCM32", [5]) == []


def test_rts3_source_table_keeps_all_comma_separated_message_numbers():
    # Expand the message list represented by an ellipsis in the brief's STR example.
    table = "STR;Seoul-Gangseo-Gaehwa;GP168;RTCM32;1006(1),1008(1),1013(1),1032(1),1033(1),1077(1),1097(1),1127(1);2;GNSS;NGII-NET;KOR;37.6;126.8;0;1;GVRS;none;B;N;4800;source"
    required = [1006, 1008, 1013, 1032, 1033, 1077, 1097, 1127]
    assert [entry["mountpoint"] for entry in runtime.parse_ntrip_source_table(table, "RTCM32", required)] == ["Seoul-Gangseo-Gaehwa"]
    for missing in (1005, 1007, 1):
        assert runtime.parse_ntrip_source_table(table, "RTCM32", [missing]) == []


def test_parse_ntrip_source_table_filters_requested_mountpoint_format() -> None:
    entries = runtime.parse_ntrip_source_table(SOURCE_TABLE, "RTCM31")

    assert [entry["mountpoint"] for entry in entries] == [
        "YANJ-RTCM31",
        "GANS-RTCM31",
        "PAJU-RTCM31",
    ]
    assert entries[1]["latitude"] == 37.50
    assert entries[1]["longitude"] == 126.90


def test_sort_ntrip_mountpoint_entries_uses_nearest_rover_position() -> None:
    entries = runtime.parse_ntrip_source_table(SOURCE_TABLE, "RTCM31")
    ranked = runtime.sort_ntrip_mountpoint_entries(
        entries,
        latitude_deg=37.62045,
        longitude_deg=126.82182,
    )

    assert ranked[0]["mountpoint"] == "GANS-RTCM31"
    assert ranked[0]["distance_m"] < ranked[-1]["distance_m"]
    assert ranked[-1]["mountpoint"] == "YANJ-RTCM31"


def test_closer_mountpoints_require_a_real_configured_improvement() -> None:
    entries = [
        {"mountpoint": "CURRENT", "latitude": 37.0, "longitude": 127.0},
        {"mountpoint": "EQUAL", "latitude": 37.0, "longitude": 127.0},
        {"mountpoint": "NEAR", "latitude": 37.09, "longitude": 127.0},
    ]
    current = {"mountpoint": "CURRENT", "latitude": 37.0, "longitude": 127.0}

    closer = runtime.closer_ntrip_mountpoint_entries(
        entries,
        current,
        latitude_deg=37.1,
        longitude_deg=127.0,
    )

    assert [entry["mountpoint"] for entry in closer] == ["NEAR"]
    improvement = closer[0]["distance_improvement_m"]
    assert runtime.closer_ntrip_mountpoint_entries(
        entries,
        current,
        latitude_deg=37.1,
        longitude_deg=127.0,
        min_improvement_m=improvement,
    )
    assert not runtime.closer_ntrip_mountpoint_entries(
        entries,
        current,
        latitude_deg=37.1,
        longitude_deg=127.0,
        min_improvement_m=improvement + 0.01,
    )


def test_closer_mountpoints_skip_an_active_station_without_coordinates() -> None:
    entries = [{"mountpoint": "NEAR", "latitude": 37.1, "longitude": 127.0}]

    assert not runtime.closer_ntrip_mountpoint_entries(
        entries,
        {"mountpoint": "UNKNOWN"},
        latitude_deg=37.1,
        longitude_deg=127.0,
    )


def _rtcm_frame(payload: bytes) -> bytes:
    header = bytes((0xD3, (len(payload) >> 8) & 0x03, len(payload) & 0xFF))
    body = header + payload
    return body + runtime.rtcm3_crc24q(body).to_bytes(3, "big")


def test_rtcm3_framer_waits_for_a_complete_crc_valid_frame() -> None:
    assert runtime.rtcm3_crc24q(b"123456789") == 0xCDE703
    frame = _rtcm_frame(b"split-frame")
    framer = runtime.Rtcm3Framer()

    assert framer.feed(b"junk" + frame[:2]) == []
    assert framer.feed(frame[2:-1]) == []
    assert framer.feed(frame[-1:]) == [frame]


def test_rtcm3_framer_recovers_after_bad_crc_and_emits_multiple_frames() -> None:
    bad = bytearray(_rtcm_frame(b"bad"))
    bad[-1] ^= 0xFF
    first = _rtcm_frame(b"first")
    second = _rtcm_frame(b"second")
    framer = runtime.Rtcm3Framer()

    assert framer.feed(bytes(bad) + first + second) == [first, second]


def test_build_rtk_config_allows_auto_mountpoint_without_primary_mountpoint() -> None:
    args = SimpleNamespace(
        rtk_ntrip_host="www.gnssdata.or.kr",
        rtk_ntrip_port=2101,
        rtk_ntrip_mountpoint="",
        rtk_ntrip_auto_mountpoint=True,
        rtk_ntrip_mountpoint_format="RTCM31",
        rtk_ntrip_mountpoint_candidates="GANS-RTCM31,YANJ-RTCM31",
        rtk_ntrip_username="",
        rtk_ntrip_password="",
        rtk_initial_latitude_deg=None,
        rtk_initial_longitude_deg=None,
        rtk_initial_altitude_m=0.0,
        rtk_ntrip_gga="",
        rtk_ntrip_gga_interval=10.0,
        rtk_ntrip_reconnect_delay=5.0,
        rtk_ntrip_position_wait_s=10.0,
        rtk_ntrip_connect_timeout_s=10.0,
        rtk_ntrip_data_timeout_s=15.0,
        rtk_ntrip_sourcetable_timeout_s=5.0,
        rtk_ntrip_max_mountpoints=12,
        rtk_ntrip_reselect_interval_s=300.0,
        rtk_ntrip_switch_min_improvement_m=1000.0,
        rtk_ntrip_position_max_age_s=30.0,
    )

    config = runtime.build_rtk_config(args)

    assert config["mountpoint"] == ""
    assert config["auto_mountpoint"] is True
    assert config["mountpoint_candidates"] == "GANS-RTCM31,YANJ-RTCM31"
    assert config["reselect_interval"] == 300.0
    assert config["switch_min_improvement_m"] == 1000.0
    assert config["position_max_age"] == 30.0


def test_source_table_format_field_accepts_suffixless_msm7_stations() -> None:
    table = "\n".join([
        "STR;Seoul-Gangseo-Gaehwa;GP168;RTCM32;1006(1),1008(1),1033(1),1077(1),1097(1),1127(1);2;GNSS;NGII-NET;KOR;37.6;126.8;0;1;GVRS;none;B;N;4800;source",
        "STR;OTHER;OTHER;RTCM 3.2;1006(1),1077(1);2;GNSS;NGII-NET;KOR;37.5;126.9;0;1;;;;;;",
        "STR;WRONG;WRONG;RTCM 3.1;1077(1);2;GNSS;NGII-NET;KOR;37.5;126.9;0;1;;;;;;",
        "STR;MISSING;MISSING;RTCM32;10770(1),1097(1);2;GNSS;NGII-NET;KOR;37.5;126.9;0;1;;;;;;",
    ])
    assert [entry["mountpoint"] for entry in runtime.parse_ntrip_source_table(table, "RTCM32", [1077])] == ["Seoul-Gangseo-Gaehwa", "OTHER"]
    entries = runtime.parse_ntrip_source_table(table, "RTCM 3.2", [1077, 1097])
    assert [entry["mountpoint"] for entry in entries] == ["Seoul-Gangseo-Gaehwa"]
    assert runtime.ntrip_mountpoint_matches_format("NAME-RTCM32", "RTCM32", "OTHER")
    assert not runtime.ntrip_mountpoint_matches_format("NAMERTCM32", "RTCM32", "OTHER")


def test_required_messages_do_not_leak_unverified_fallback_candidates(monkeypatch) -> None:
    import threading

    client = runtime.NtripCorrectionClient({
        "host": "rts3.ngii.go.kr", "auto_mountpoint": True, "mountpoint_format": "RTCM32",
        "required_messages": [1077], "mountpoint": "YANJ-RTCM31",
        "mountpoint_candidates": "GANS-RTCM31", "position_wait_s": 0.0,
    }, lambda data: None, threading.Event())
    available = "STR;Seoul-Gangseo-Gaehwa;GP168;RTCM32;1006(1),1077(1);2;GNSS;NGII-NET;KOR;37.6;126.8;0;1;GVRS;none;B;N;4800;source"
    responses = iter((SOURCE_TABLE, available))
    monkeypatch.setattr(runtime, "fetch_ntrip_source_table", lambda config: next(responses))
    assert client._mountpoint_candidates() == []
    assert [entry["mountpoint"] for entry in client._mountpoint_candidates()] == ["Seoul-Gangseo-Gaehwa"]


def test_source_table_chunked_response_is_decoded(monkeypatch) -> None:
    body = SOURCE_TABLE.split("\r\n\r\n", 1)[1]
    response = f"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n{len(body.encode()):x}\r\n{body}\r\n0\r\n\r\n"
    sock = SourceTableSocket(response)
    monkeypatch.setattr(runtime.socket, "create_connection", lambda *args, **kwargs: sock)
    entries = runtime.parse_ntrip_source_table(runtime.fetch_ntrip_source_table({"host": "caster.test", "port": 2101}))
    assert len(entries) == 3


def test_gnssdata_relay_retains_builtin_fallback_but_rts3_does_not(monkeypatch) -> None:
    import threading

    monkeypatch.setattr(runtime, "fetch_ntrip_source_table", lambda config: "ENDSOURCETABLE")
    config = {"host": "relay.test", "caster_host": "www.gnssdata.or.kr", "auto_mountpoint": True, "position_wait_s": 0.0}
    client = runtime.NtripCorrectionClient(config, lambda data: None, threading.Event())
    assert "GANS-RTCM31" in [entry["mountpoint"] for entry in client._mountpoint_candidates()]
    config["caster_host"] = "rts3.ngii.go.kr"
    assert client._mountpoint_candidates() == []
