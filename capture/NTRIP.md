# NTRIP 기준국 선택과 RTCM 진단

기본 선택 방식은 `rtk_ntrip_selection: client`입니다. 아래 예시를 사용할 YAML에 병합합니다. 계정은 `NTRIP_USERNAME`, `NTRIP_PASSWORD` 환경변수로 공급합니다.

## ① gnssdata 단일 기준국, 클라이언트 선택 (현재 기본)

```yaml
rtk_ntrip_host: www.gnssdata.or.kr
rtk_ntrip_port: 2101
rtk_ntrip_selection: client
rtk_ntrip_mountpoint: YANJ-RTCM31
rtk_ntrip_auto_mountpoint: true
rtk_ntrip_mountpoint_format: RTCM31
rtk_ntrip_required_messages: []
rtk_ntrip_gga_interval: 10
rtk_ntrip_gga_max_age_s: 5.0
```

수신기 위치와 소스테이블로 가까운 단일 기준국을 선택하고, 기본 300초마다 더 가까운 후보를 검토합니다. 기존 후보 목록과 gnssdata 기준국의 내장 좌표는 유지됩니다.

## ② RTS-1 RTK-RTCM32, 캐스터 선택

```yaml
rtk_ntrip_host: rts1.ngii.go.kr
rtk_ntrip_port: 2101
rtk_ntrip_selection: caster
rtk_ntrip_mountpoint: RTK-RTCM32
rtk_ntrip_auto_mountpoint: true
rtk_ntrip_mountpoint_format: RTCM32
rtk_ntrip_required_messages: []
rtk_ntrip_gga_interval: 5
rtk_ntrip_gga_max_age_s: 5.0
rtk_ntrip_position_wait_s: 10
rtk_ntrip_reconnect_delay: 5
```

`caster`에서는 지정한 마운트포인트 하나만 사용합니다. `auto_mountpoint: true`여도 후보 조회·거리 정렬·주기적 재선정·handover를 하지 않으며, 실패하면 같은 마운트포인트로 재접속합니다. 캐스터가 GGA 위치를 받아 기준국을 고릅니다.

접속 전에 `position_wait_s` 동안 신선한 수신기 GGA를 기다립니다. 없으면 접속을 진행하고 첫 유효 GGA가 생기는 즉시 보냅니다. 접속 때 GGA가 있으면 `Ntrip-GGA` 헤더와 200 응답 직후에 보내고, 이후 설정 간격으로 보냅니다. 합성 GGA는 사용하지 않습니다.

## ③ RTS-3 MSM7 단일 기준국, 클라이언트 선택

```yaml
rtk_ntrip_host: rts3.ngii.go.kr
rtk_ntrip_port: 2101
rtk_ntrip_selection: client
rtk_ntrip_mountpoint: ""
rtk_ntrip_mountpoint_candidates: ""
rtk_ntrip_auto_mountpoint: true
rtk_ntrip_mountpoint_format: RTCM32
rtk_ntrip_required_messages: [1077]
rtk_ntrip_gga_interval: 10
rtk_ntrip_gga_max_age_s: 5.0
```

이름의 `-RTCM32` 접미사 또는 STR 형식 필드(`RTCM 3.2`, `RTCM32`)가 일치하는 항목을 고릅니다. `required_messages`는 STR 상세 필드에 지정한 모든 번호가 있는지 확인합니다. 괄호 안의 주기 설명을 제거한 뒤 쉼표와 슬래시로 구분하므로 `1005/1007(5)`는 1005와 1007 모두에 일치하고, 주기 숫자는 메시지로 취급하지 않습니다. 필수 메시지를 설정하면 확인되지 않은 설정 후보나 내장 후보로 우회하지 않으므로, 조건에 맞는 소스테이블이 없으면 재시도합니다.

## GGA·중계·상태

모든 모드에서 송신용 수신기 GGA는 체크섬·좌표·품질 1~5를 검사하고 host monotonic 기준 `gga_max_age_s` 이내인 것만 사용합니다. 직렬 포트를 다시 열면 저장된 GGA를 비웁니다. `rtk_ntrip_gga`는 유효한 정적 시험 문장일 때만 쓰며 수신기 나이 제한의 예외입니다. `client`만 마지막 수단으로 초기 좌표에서 GGA를 합성합니다. GGA 송신 중단과 재개는 상태가 바뀔 때 한 번씩 로그에 남깁니다.

체크섬이 틀린 GGA도 원문은 `gps.csv`로 전달되는 샘플 큐와 recent 기록에 남깁니다. 파싱 필드는 비우고, 송신용 `gga`·`gga_monotonic`은 지웁니다. 상태 조회는 해당 행을 건너뛰어 마지막 정상 fix와 그 수신 시각을 유지하므로, 잘못된 행이 정상 fix를 덮거나 신선도를 갱신하지 않습니다.

`JETSON_PIPELINE_MOBILE_RTK_RELAY=1`이면 주소 우선순위는 명시적 CLI > `NTRIP_HOST`/`NTRIP_PORT` > YAML > 기본값입니다. 로그에는 실제 중계 주소와 YAML/기본 캐스터 주소를 함께 남깁니다. 예: `RTK NTRIP enabled: 192.168.49.61:39535/RTK-RTCM32 (mobile relay to rts1.ngii.go.kr:2101)`.

`--config "~/profile.yaml"`도 같은 경로 정규화를 적용합니다. argparse가 허용하는 `--rtk-ntrip-h`, `--rtk-ntrip-por` 등의 축약 옵션도 명시적 CLI로 취급합니다. YAML의 숫자형 계정은 CLI와 `build_rtk_config`에서 문자열로 정규화하며, 0도 보존합니다. 오류 마스킹은 입력 형식이 잘못되어도 예외를 내지 않고 일반 오류 문구를 반환합니다. 소스테이블 응답의 첫 줄은 계정과 Basic 토큰을 먼저 마스킹한 다음 160자로 제한합니다.

CRC가 맞고 직렬로 전달한 RTCM3에서 최근 60초의 메시지 번호(정렬, 최대 32개), 1005/1006 기준국 ID·좌표, 신선한 수신기 위치와의 지표면 거리를 기록합니다. 기준국 ID와 메시지 집합이 바뀔 때 로그를 남기며, chunked HTTP 응답은 프레임 검사 전에 디코딩합니다.

MSM4~7(1074~1077, 1084~1087, 1094~1097, 1104~1107, 1114~1117, 1124~1127, 1134~1137)의 신호 마스크도 읽습니다. 접두사는 각각 GPS·GLO·GAL·SBS·QZS·BDS·IRN입니다. signal ID를 RTKLIB `msm_sig` 표의 RINEX 관측 코드로 바꾸고, 미정의 ID와 SBS·IRN은 `?<ID>`로 표시합니다. 최근 60초의 합집합을 `"GPS 1C"`, `"GPS 5Q"`, `"GAL 5Q"` 형태로 정렬해 최대 48개 내보냅니다. 같은 신호가 다시 오면 만료 시각을 갱신하고, 수신이 멎어도 상태 조회 때 오래된 항목을 지웁니다. 1004·1012는 메시지 번호에만 포함하며 위성 수는 계산하지 않습니다.

MSM 헤더의 비트 번호는 RTCM 프레이밍을 제외한 메시지 번호의 첫 비트를 0으로 셉니다. 위성 마스크 앞 필드의 합은 `12+12+30+1+3+7+2+2+1+3=73`비트입니다. 위성 마스크는 `[73,137)`, 신호 마스크는 `[137,169)`이며, 신호 마스크의 MSB가 signal ID 1입니다. 헤더가 169비트보다 짧으면 신호를 읽지 않습니다. 테스트는 필드를 전송 순서대로 만들고 비트 구간과 ID 1·32 경계를 확인합니다.

신호 집합이 바뀔 때만 한 줄을 남깁니다: `RTK NTRIP MSM signals: GPS 1C,2W,5Q; GAL 1C,5Q,7Q; BDS 2I,5P`. 모두 만료되면 `RTK NTRIP MSM signals: -`입니다.

기준국·메시지·신호·GGA 가용성의 변경 상태와 로그 문자열은 클라이언트 잠금 안에서 결정하고, 출력은 잠금을 놓은 뒤 실행합니다. `_refresh_message_types`와 `_refresh_signals`는 직접 출력하지 않고 변경 로그 문자열 또는 `None`을 반환합니다. stdout이 막혀도 다른 스레드가 클라이언트 상태를 갱신하고 조회할 수 있습니다.

Controller bridge는 기존 GNSS 키와 함께 `ntripSelection`, `ntripError`, `rtcmLastAtEpochMillis`, `rtcmMessageTypes`, `rtcmSignals`, `rtcmBaseStationId`, `rtcmBaseDistanceM`, `ntripLastGgaSentAtEpochMillis`를 내보냅니다. 시간은 epoch ms, 거리는 m 소수 1자리이고, 값이 없으면 null(메시지·신호 목록은 빈 목록)입니다. `rtcmSignals`는 정렬된 문자열 목록이며 진단 클라이언트가 없어도 빈 목록입니다. 기준국 거리에는 초기 설정 좌표를 사용하지 않습니다.

ZED-F9P HPG L1L5 수신기에서 1004(GPS L1/L2)·MAC(1014~1016)만 오는 스트림은 GPS 관측의 공통 대역이 L1뿐입니다. 실제 메시지 번호와 수신기의 사용 가능한 신호를 확인하고, L5 관측이 포함된 MSM 스트림인지 현장에서 검증합니다.

## 옛 릴리스로 이식할 때

기존 함수 경계를 유지했으며 변경된 함수는 다음과 같습니다.

- NMEA·소스테이블: `parse_nmea_line`, `ntrip_mountpoint_matches_format`, `parse_ntrip_source_table`, `_fetch_ntrip_source_table_response`, `_ntrip_response_status`, `fetch_ntrip_source_table`, `build_rtk_config`.
- `NtripCorrectionClient`: `__init__`, `start`, `_fallback_gga`, `_current_gga`, `_live_position_is_fresh`, `_rover_position`, `_load_source_table_entries`, `_mountpoint_candidates`, `_periodic_switch_candidates`, `_request`, `_uses_rtcm3_framing`, `_open_mountpoint_stream`, `_forward_rtcm_frames`, `_handover_worker`, `_stream_mountpoint`, `_run`.
- `SerialRateLimitedReader`: `__init__`, `_read_connection`, `latest_sample`(기록된 잘못된 GGA를 상태에서 제외).
- 설정·상태: `capture.cli.build_parser`, `capture.cli.parse_args`, `ControllerBridge._gnss_state` 및 `DEFAULTS`의 새 설정 세 개.

새 헬퍼도 함께 이식합니다: `nmea_checksum_is_valid`, `valid_ntrip_gga`, `NtripChunkDecoder.__init__/feed`, `redact_ntrip_error`, `rtcm3_base_position`, `rtcm3_msm_signals`와 매핑 상수 `RTCM_MSM_SIGNAL_CODES`. 클라이언트에는 `error` 속성, `_nmea_snapshot`, `_refresh_message_types`, `_refresh_signals`, `diagnostics_snapshot`, `_diagnostic_rover_position`, `_record_rtcm`, `_receiver_gga`, `_send_gga`, `_wait_for_receiver_gga`가 추가됐습니다.

이번 MSM 추가분은 `NtripCorrectionClient.__init__`의 `_signal_seen`·`_signals` 초기화, `_record_rtcm`의 신호 수신 시각 갱신, `_refresh_signals`의 60초 만료·정렬·로그, `diagnostics_snapshot` 및 `ControllerBridge._gnss_state`의 `rtcmSignals` 내보내기를 함께 옮깁니다. 신호 진단은 기존 CRC 검증 경로와 클라이언트 잠금 안에서 갱신합니다. GGA 수정분은 `_read_connection`의 원문 기록·GGA 행 제한 예외와 `latest_sample`의 상태 조회 처리를 함께 옮깁니다.

검토 수정 6건은 `parse_ntrip_source_table`, `redact_ntrip_error`, `_ntrip_response_status(text, config)`와 호출부 `fetch_ntrip_source_table`, `build_rtk_config`, `capture.cli.build_parser`·`parse_args`, `NtripCorrectionClient._record_rtcm`·`_refresh_message_types`·`_refresh_signals`·`diagnostics_snapshot`·`_current_gga`를 함께 옮깁니다. `_refresh_*` 호출부는 반환된 문자열을 잠금 밖에서 출력해야 합니다. 중계 경로의 YAML 재조회에는 `expanduser().resolve()`를 적용하고, CLI 명시 여부는 기본값을 sentinel로 바꾼 argparse 재파싱 결과로 확인합니다.

직렬 리더와 클라이언트가 같은 `latest_nmea_lock`을 공유하고 `gga_monotonic`을 함께 저장하도록 옮깁니다. `error`는 잠금을 사용하는 속성이므로 잠금 생성 순서도 유지합니다. 스트림의 `decoder`는 후보 검증에서 활성 스트림·handover까지 전달하고, `last_gga_time=None`은 아직 보낼 GGA가 없다는 뜻으로 유지합니다. 진단용 프레이머는 새 연결과 handover 때 초기화합니다. 옛 CLI의 YAML 처리 위치에서도 모바일 중계 환경변수를 적용한 뒤 명시적 CLI가 최우선인지 확인합니다.
