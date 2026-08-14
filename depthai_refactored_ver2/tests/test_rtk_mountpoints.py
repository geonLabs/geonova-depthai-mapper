from __future__ import annotations

from types import SimpleNamespace

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
    )

    config = runtime.build_rtk_config(args)

    assert config["mountpoint"] == ""
    assert config["auto_mountpoint"] is True
    assert config["mountpoint_candidates"] == "GANS-RTCM31,YANJ-RTCM31"
