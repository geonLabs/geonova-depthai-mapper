# Geonova Capture — Jetson 수집 전용

OAK RGB·Depth와 온보드 IMU, GNSS/NTRIP 원본을 수집하고 GEO& 앱에 센서 상태와 JPEG 프리뷰를 게시합니다. YOLO·SHP·동기화·선형화는 후처리 서버가 실행합니다.

## 실행

통합 저장소에서는 저장소 루트, 독립 export에서는 현재 폴더에서 실행합니다.

```bash
# Jetson 자체에서 시스템 Python 3.8~3.12로 생성. 다른 PC의 .venv를 복사하지 않습니다.
./install.sh
.venv/bin/python main.py --config config.yaml
```

`--system-site-packages`는 시스템 패키지가 필요한 장치에서만 명시적으로 사용합니다. 기본값은 독립 venv입니다. 수집에 CUDA/PyTorch/TensorRT는 필요하지 않습니다. DepthAI 3.1.0과 장치 USB/시리얼 권한은 필요합니다.

앱에는 `.git`, `.venv/bin/python`, `main.py`, `config.yaml`이 있는 저장소 루트를 등록합니다. 결과 폴더는 `results/`입니다. `config.yml`을 함께 만들거나 등록 폴더·venv·YAML·main.py·results를 심볼릭 링크로 구성하지 않습니다. 일반 venv의 Python 링크는 허용됩니다.

Controller가 전달한 `JETSON_PIPELINE_RESULTS_DIR`, `JETSON_PIPELINE_SENSOR_BRIDGE_DIR`가 우선합니다. `--config`가 없으면 `JETSON_PIPELINE_CONFIG`, 그다음 루트 `config.yaml`을 사용합니다. 독립 실행의 상대 출력 경로는 YAML 디렉터리를 기준으로 해석합니다. stdout/stderr는 Controller 로그로 전달되며 SIGTERM/SIGINT에서 쓰기 큐와 센서를 정리합니다.

## 서버로 전달

`results/<timestamp>_raw/` 전체를 전송합니다. 기존 파일명과 이미지/CSV 형식을 유지합니다. 폴더의 `capture_manifest.json`이 `state: complete`인 경우만 새 서버 CLI가 처리합니다. `recording` 또는 `failed`는 자동 처리하지 않습니다. 수집 중 폴더를 전송하기 시작했다면 전송 종료 후 서버 검증까지 수행해야 합니다.

원본에는 `rgb/`, `depth_mm/`, 선택적 `confidence/`, 이벤트 CSV, 센서 CSV, 카메라 보정정보가 있는 `metadata.json`이 포함됩니다. 파일 크기와 CSV/JSON 체크섬으로 불완전한 전송을 검사합니다. 이미지 전체 해시는 계산하지 않으므로 같은 크기의 이미지 손상까지 검출하는 기능은 아닙니다.

## 깊이(StereoDepth) 설정과 밀도 점검

`depth_preset`이 confidence·LR-check 임계값을 정하고, `lr_check`·`subpixel`·`subpixel_fractional_bits`·`stereo_median_filter`는 프리셋 뒤에 항상 설정값으로 덮어씁니다. 확장 시차는 끕니다. DepthAI 3.1.0 RVC2에서 `FAST_ACCURACY`는 confidence 55·LR 5, `FAST_DENSITY`는 15·10이며 두 프리셋 모두 후처리 필터를 켜지 않습니다. v3에서는 confidence 값이 낮을수록 더 많은 픽셀을 남깁니다.

선택 항목은 기본값일 때 아무것도 바꾸지 않습니다.

- `stereo_confidence_threshold`, `stereo_lr_check_threshold`: -1이면 프리셋 값
- `stereo_speckle_filter`, `stereo_spatial_filter`: 장치에서 작은 오정합 제거와 가로 방향 구멍 메움. false면 프리셋 선택을 유지합니다. 켜면 장치 SHAVE를 쓰므로 30초 시험 녹화로 depth 15 FPS를 확인한 뒤 사용합니다. FPS가 떨어지면 `stereo_postprocessing_shaves`·`stereo_postprocessing_memory_slices`를 함께 3으로 둡니다.
- `stereo_ae_max_exposure_us`: 두 스테레오 카메라의 자동노출 상한(µs). 기본 자동노출은 약 프레임 시간까지 길어져 주행 중 노면이 흐려질 수 있습니다.
- `stereo_ae_region`: 노출을 잴 영역. 저장 영상 기준 `x0,y0,x1,y1`(0~1)이며 `flip`·`rotate_180`을 반영해 센서 좌표로 바꿉니다.

실제로 적용된 값은 `metadata.json`의 `stereo_config`에 기록됩니다. 녹화 폴더의 깊이 밀도와 depth FPS는 다음으로 확인합니다. 폴더는 읽기만 합니다.

```bash
# 독립 export에서는 src/geonova_depthai/capture/depth_density.py
.venv/bin/python capture/src/geonova_depthai/capture/depth_density.py results/<timestamp>_raw --band 0.55,0.70
```

`--band`는 노면 띠의 세로 범위(영상 높이 비율)입니다. 행별 유효 비율과 중앙 깊이를 출력하고, 화면 아래쪽에서 1.2 m보다 가까운 행은 차체로 표시합니다.

## OAK 온보드 IMU와 USB

연결 시 OAK IMU 칩을 감지합니다. 지원 칩(BNO086 등 BNO08x 계열 또는 BMI270)의 raw 가속도와 자이로는 항상 `imu_events.csv`에 기록합니다. BNO08x에서는 game rotation vector도 `rv_*` 열에 기록합니다. `metadata.json`의 `imu` 블록에는 칩 이름(`type`), 센서 목록(`sensors`), 속도(`rate_hz`), IMU-to-camera extrinsics(`imu_to_camera`)가 기록됩니다.

`--allow-usb2`를 켜면 USB 3 연결 시작에 실패했을 때 `usb3_retries` 횟수만큼 USB 3 재시도한 뒤 USB 2 연결을 한 번 시도합니다. 기본 `usb3_retries`는 2입니다.

NTRIP 계정은 `NTRIP_USERNAME`, `NTRIP_PASSWORD` 환경변수로 주입합니다. 앱의 모바일 RTK relay 플래그 자체를 새로운 TCP endpoint로 해석하지 않습니다. 기존 네트워크/NTRIP 구성과 Controller 센서 인계 설정을 사용하고 실기에서 확인합니다.

`environment.freeze.txt`와 `environment.json`은 설치 후 생성되는 장치 환경 기록입니다. 앱은 패키지 설치·모델 다운로드를 수행하지 않습니다. 변경한 소스를 적용하려면 앱에서 재등록해 새 실행 스냅샷을 만듭니다.

DepthAI 3.1.0의 multi-Python wheel 내부 태그가 cp311 하나로 기록되는 경우가 있습니다. 설치기는 모든 실제 import가 성공한 뒤 이 버전의 단독 태그 경고만 환경 기록에 남깁니다. 다른 pip 의존성 오류나 SDK import 실패는 설치 실패입니다.
