# 수집·후처리 분리 및 서버 이전 안내

## 결정

기존 저장소를 유지하고 실행 구성만 분리한다. 앱에 등록할 수집 진입점은 루트 `main.py`다. 서버는 `postprocess/main.py`로 실행한다. 독립 Git 작업 트리가 필요하면 export 도구로 각각 내보낸다. 별도 GitHub 저장소 생성, 운영 장치 배포, 서버 전송 서비스 구축은 이번 코드 분리의 실행 범위에 포함하지 않는다.

검토 기준: mapper `main` 커밋 `de0fa99511c4fea063fbb5c2a3aa06671ddbd239`, Controller 계약 파일 blob `de468580c5e8fd95423cb6eb19be1718e14014ad`. 정확한 앱 센서 형식은 Controller `backend/jetson_control/sensors.py`의 `SensorBridgeStore`와 대조했다. 기준 문서는 [EXTERNAL_PIPELINE_CONTRACT.md](https://github.com/dbparkJ/JetsonControllerApp/blob/main/docs/EXTERNAL_PIPELINE_CONTRACT.md)다.

## 코드 소유권

| 이전 | 새 구현 | 실행 환경 |
|---|---|---|
| `code/geonova_depthai/capture/` | `capture/src/geonova_depthai/capture/` | Jetson |
| `runtime.py`, `serial_devices.py`, `controller_bridge.py` | `capture/src/geonova_depthai/` | Jetson |
| `code/geonova_depthai/config_cli.py` | `shared/src/geonova_common/config_cli.py` | 양쪽, 하드웨어 의존 없음 |
| `code/geonova_depthai/postprocess/sync_builder.py` | `postprocess/src/geonova_postprocess/sync_builder.py` | 서버 |
| `yolo_seg_shp.py`, `fence_linearization.py` | `postprocess/src/geonova_postprocess/` | 서버 |
| `debug_ui.py`의 데이터·좌표 계산 | `postprocess/src/geonova_postprocess/dataset.py` | 서버 배치/뷰어 공용 |
| `debug_ui.py`의 HTTP 화면 | `postprocess/src/geonova_postprocess/debug_ui.py` | 선택적 서버 진단 |
| `code/inference/infer_class_change.py` | `postprocess/src/geonova_postprocess/infer_class_change.py` | 서버 |
| LiDAR 오프라인 변환 | 기존 `tools/lidar/`, 서버 export에 포함 | 서버, 별도 requirements |
| 학습·라벨 전처리 | 기존 `model/` | 학습 환경, 수집에 포함하지 않음 |

`code/`의 어댑터는 기존 import와 진단 명령을 연결한다. 동일 알고리즘을 두 벌 유지하지 않는다. 새 기능은 위 새 구현 위치에서 수정한다. `code/setup_env.py`와 기존 requirements는 혼합 개발환경용으로 남고, 루트 설치기는 수집 전용으로 변경됐다. 예전 `./install.sh --dev/--platform/--recreate`는 새 설치기 옵션이 아니다.

## Controller 계약 적용

- 루트 `main.py`와 한 개의 `config.yaml`을 유지한다. `src`는 통합 저장소에서 `capture/src`, 독립 export에서 `src`에 있다. 계약상 예시 폴더와 달라도 실제 진입점에서 이 경로를 자체 해결한다.
- `.venv`는 Controller API venv와 별도로 해당 Jetson에서 생성한다. Python 3.8~3.12, DepthAI 3.1.0, aarch64 wheel/장치 SDK 호환성을 장치에서 검증한다. `--system-site-packages`는 선택 사항이다.
- `--config`가 최우선, 없으면 `JETSON_PIPELINE_CONFIG`, 그다음 저장소 `config.yaml`이다.
- 결과 경로는 `JETSON_PIPELINE_RESULTS_DIR`, 센서 브리지는 `JETSON_PIPELINE_SENSOR_BRIDGE_DIR`가 우선한다. 독립 실행 상대 출력은 YAML 위치 기준이다. release 내부 bytecode 생성을 비활성화한다.
- `JETSON_PIPELINE_ID`와 `JETSON_PIPELINE_RELEASE`를 수집 manifest에 기록한다. stdout/stderr는 실행 로그이며 이 코드가 로그 경로에 별도 daemon을 만들지 않는다.
- 카메라 JPEG와 schemaVersion 1 센서 상태를 기존 원자적 파일 게시 방식으로 유지한다. GNSS의 실제 샘플 나이를 유지하며 검지 결과를 센서 필드로 위장하지 않는다.
- Controller의 센서 handoff/lease를 이용한다. 수집기가 부팅 모니터와 동시에 센서를 열도록 별도 systemd 서비스를 추가하지 않는다. 센서 해제 시그널과 파일 정리를 유지한다.
- 모바일 RTK relay 활성 플래그만으로 연결 endpoint/protocol을 추정하지 않는다. 현재 NTRIP 경로를 유지하고 실제 Controller 버전/네트워크 설정으로 검증한다.

## 수집본 전달 계약 v1

기존 `raw_events_v1` 이미지/CSV 구조와 카메라 내부표정, 이미지 변환, 깊이 단위(mm)를 유지한다. 폴더명은 기존 로컬 timestamp 형식을 유지하고 manifest의 상태 시각은 명시적 UTC다. run_id는 수집 폴더명이다.

새 파일 `capture_manifest.json`:

```json
{
  "schema": "geonova.raw-run/v1",
  "state": "complete",
  "run_id": "2026-09-12-14-00-00_raw",
  "updated_at_utc": "2026-09-12T05:10:00+00:00",
  "pipeline_id": "geonova-capture",
  "pipeline_release": "/path/to/immutable/release",
  "raw_format": "raw_events_v1",
  "counts": {"rgb": 100, "depth": 100},
  "files": [{"path": "rgb/frame.jpg", "size": 12345}],
  "error_type": null
}
```

예시는 축약형이다. 실제 complete manifest에는 필수 CSV/metadata와 기록된 모든 데이터 파일 목록을 포함한다. 수집 시작 상태는 `recording`; 쓰기 큐, CSV, 센서 종료에 성공하면 `complete`; 수집 또는 정리 실패는 `failed`다. 강제 전원 차단/SIGKILL은 complete를 게시하지 못하므로 서버가 거부한다. 정리 중 오류가 나도 다른 자원의 종료는 시도한다.

이미지 파일은 존재/크기, CSV·JSON은 SHA-256까지 검증한다. 이미지 전체 체크섬은 수집 종료 시 대용량 재읽기를 피하기 위해 계산하지 않는다. 이 manifest는 전송 완료를 대신하지 않으며, 서버가 검증에 통과한 뒤에만 작업을 시작해야 한다. 원본 삭제/보관 정책은 별도로 결정한다.

새 서버 `sync` CLI는:

1. complete 상태와 스키마, 파일 전송 상태, 경로를 검증한다.
2. 원본과 겹치지 않는 새/빈 출력 폴더만 허용한다.
3. 이미지를 실제 복사하고 동기 CSV와 QA를 생성한다.
4. 0프레임이면 실패 처리한다. 성공/실패를 `processing_manifest.json`에 기록한다.
5. 재처리는 새 job 디렉터리에서 수행한다. 실패한 결과를 완료본으로 취급하지 않는다.

기존 저수준 `build_synced_dataset()` Python 함수의 in-place 기본값은 기존 테스트/개발 호출 호환을 위해 유지한다. 운영 서버와 이전 어댑터 CLI는 새 `main()` 검증 경로를 사용한다. 과거 manifest 없는 원본은 `--allow-legacy`로 명시 승인하며, 이미 failed/recording 표시가 있는 원본은 이 옵션으로 통과하지 못한다.

## 독립 배포 순서

통합 저장소에서 검증한 커밋을 체크아웃하고 다음을 실행한다.

```bash
python3 tools/export_component.py capture --destination /tmp/geonova-capture --init-git
python3 tools/export_component.py postprocess --destination /tmp/geonova-postprocess --init-git
```

export는 대상 폴더 덮어쓰기를 거부한다. `source_revision.json`에 기준 커밋과 미커밋 변경 여부를 기록한다. 운영본은 `source_worktree_dirty: false`인지 확인한다. 새 로컬 Git 커밋을 생성하지만 GitHub 저장소를 자동 생성/업로드하지 않는다.

수집본은 `/home/<pipeline-user>/geonova-capture` 같은 최종 경로로 옮긴 후 Jetson에서 설치한다. `.git`, 실행 가능한 `.venv/bin/python`, `main.py`, 하나의 YAML, 일반 `results/` 디렉터리를 확인하고 앱 폴더 등록 → 작업 설정 → 시작 전 확인 → 실행 순서로 연결한다. 외부 수집 계정에 USB·시리얼·결과·브리지 접근 권한이 필요하다.

서버본은 서버의 최종 경로로 옮긴 뒤 Python 3.11~3.12로 설치한다. 기본 numerical 환경은 GPU가 필요 없다. YOLO용 torch/torchvision은 CUDA/드라이버를 확인해 별도로 설치하고 `--with-yolo`를 실행한다. 모델은 파일 경로를 명시한다. weights의 라이선스·클래스·입출력 형상·checksum은 실제 모델 소유자가 확인한다. `environment.freeze.txt`와 `environment.json`을 배포 기록으로 보관한다. 코드 변경이 수집 정확도 향상을 의미하지는 않는다.

## 서버 이전 시 다음 연결 작업

자동 업로드·인증·작업 큐·DB·API는 아직 구현하지 않는다. 실제 서버 저장 경로와 운영 방식을 확인한 뒤, `raw 수신 → validate_capture → sync → YOLO → QA → 결과 등록`에 연결한다. 원본 복사가 끝난 수신 디렉터리만 worker에 넘기고, 실행 ID별 작업 공간과 재시도 기록을 둔다. 수집 코드가 서버 API 응답을 기다리도록 엮지 않는다.

## 검증 및 실기 인수

자동 검증은 독립 export 실행/의존성 차단, 표준 라이브러리만으로 서버 동기화, 원본 보존, 전송 누락/손상/위험 경로 거부, 미완료 수집 거부, SIGTERM 정상 종료, 정리 실패 상태, 기존 좌표/마스크/브리지/센서 파서 회귀를 포함한다. 센서 없는 테스트에서 작성한 이미지 payload는 전송 검증용 합성 데이터이며 실제 카메라 영상 품질을 검증한 것은 아니다.

실기에서는 다음을 확인한다.

- 앱 등록과 재등록, 시작/중지/재시작, 부팅 센서 모니터 인계.
- OAK RGB·Depth·IMU 및 GNSS/EBIMU 수집량, 저장 속도, 프리뷰, 신규 GPS 샘플 시각.
- 모바일 데이터와 NTRIP RTCM 지속 수신, RTK fix, 장치 해제 후 포트 반환.
- 디스크 부족·케이블 분리·종료 시 failed/complete 상태와 결과 파일.
- 수집 폴더 전체 전송 후 동기화 QA와 기존 결과 비교.
- 동일 모델/설정/데이터의 YOLO·대표점·SHP·선형화 결과와 공간 오차.

이 환경에는 실제 Jetson·OAK·GNSS·EBIMU가 연결되지 않았고 서버 모델 추론도 실측하지 않았다. 실기 검증 전 자동으로 main 병합하거나 운영 장치 설정을 바꾸지 않는다.

## 이번 작업의 확인 결과

- 자동 회귀: 108개 통과(합성 데이터/모의 장치 포함).
- 독립 서버 설치: Python 3.12 x86_64에서 고정 수치 패키지 설치·pip check·CLI import 통과.
- 독립 수집 설치: Python 3.12 x86_64에서 고정 수집 패키지 import 확인. DepthAI 3.1.0의 wheel 내부 cp311 태그/다운로드 multi-Python 태그 불일치는 native import 성공 후 단독 경고로만 기록하며 다른 의존성 실패는 허용하지 않음. Jetson aarch64 설치는 실기 확인 필요.
- 현재 Controller의 discover_pipeline_folder로 수집 export 구조 검사, SensorBridgeStore로 게시/종료 상태 읽기 확인.
- YOLO GPU 추론, 모바일 RTK, USB/시리얼 실기 수집은 미검증.
