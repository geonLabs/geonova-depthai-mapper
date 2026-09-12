# Geonova Capture — Jetson 수집 전용

OAK RGB·Depth·IMU, GNSS/NTRIP, EBIMU 원본을 수집하고 GEO& 앱에 센서 상태와 JPEG 프리뷰를 게시합니다. YOLO·SHP·동기화·선형화는 후처리 서버가 실행합니다.

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

NTRIP 계정은 `NTRIP_USERNAME`, `NTRIP_PASSWORD` 환경변수로 주입합니다. 앱의 모바일 RTK relay 플래그 자체를 새로운 TCP endpoint로 해석하지 않습니다. 기존 네트워크/NTRIP 구성과 Controller 센서 인계 설정을 사용하고 실기에서 확인합니다.

`environment.freeze.txt`와 `environment.json`은 설치 후 생성되는 장치 환경 기록입니다. 앱은 패키지 설치·모델 다운로드를 수행하지 않습니다. 변경한 소스를 적용하려면 앱에서 재등록해 새 실행 스냅샷을 만듭니다.

DepthAI 3.1.0의 multi-Python wheel 내부 태그가 cp311 하나로 기록되는 경우가 있습니다. 설치기는 모든 실제 import가 성공한 뒤 이 버전의 단독 태그 경고만 환경 기록에 남깁니다. 다른 pip 의존성 오류나 SDK import 실패는 설치 실패입니다.
