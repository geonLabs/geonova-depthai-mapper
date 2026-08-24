# Geonova DepthAI Mapper (RGB-D / RTK Fence Linearization)

OAK RGB-D 카메라, GPS/RTK, IMU를 동기 수집해 YOLO segmentation 결과를
지도 좌표계에서 방호울타리 선형(polyline)로 변환하는 파이프라인입니다.

> 참고: 이 저장소는 LiDAR를 사용하지 않으며, 깊이 정보는 OAK의 stereo depth를
> 기반으로 처리합니다.

## 프로젝트 개요

- 실시간 현장 수집: RGB-D, GPS/RTK, IMU 동기화
- 오프라인/온라인 처리: 동기화 데이터셋 생성, 마스크 후처리, 지리좌표 변환, 선형화
- 배포 플로우: Jetson + Controller로 부팅 시 자동 실행(필요 시 시스템 서비스 등록)
- 학습/추론 준비: YOLO 세그먼테이션 라벨 변환 및 재학습 스크립트 제공

## 빠른 시작

### 1) 환경 설치

Jetson / Ubuntu PC:

```bash
chmod +x install.sh
./install.sh --dev
```

Windows PowerShell:

```powershell
.\install.ps1 --dev
```

설치 후 가상환경은 저장소 루트의 `.venv`에 생성됩니다.

- Jetson: L4T를 자동 감지하고 JetPack Python과 NVIDIA CUDA PyTorch를 보존합니다.
  NVIDIA PyTorch가 이미 설치되어 있으면 그대로 재사용하며, 호환되는 torchvision을
  자동 구성합니다. 첫 설치에서는 torchvision 빌드에 시간이 걸릴 수 있습니다.
- Ubuntu/Windows PC: `uv`가 Python 3.11을 준비하고, `nvcc` 또는 `nvidia-smi`로
  CUDA를 감지해 공식 PyTorch CUDA/CPU wheel을 설치합니다.
- 기존 환경을 완전히 다시 만들 때만 `./install.sh --recreate --dev` 또는
  `.\install.ps1 --recreate --dev`를 사용합니다.

활성화:

```bash
. .venv/bin/activate
```

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2) 데이터 수집/처리

```bash
cd code
../.venv/bin/python synced_image_recorder.py --config configs/capture.yaml
../.venv/bin/python build_synced_dataset.py --config configs/sync.yaml
../.venv/bin/python tests/test_yolo_seg_shp.py --config configs/yolo.yaml
```

데이터셋을 만들지 않고 Controller 센서 상태와 카메라 프리뷰만 계속 게시하려면
모니터 모드를 사용합니다. 이 모드는 센서에 맞는 최대 RGB 캡처 크기를
선택하되 최대 5 FPS로 제한하고, USB2 급 연결에서는 MJPEG 전송으로
대역폭을 제한합니다. Controller JPEG 프리뷰는 최대 1920 px 폭입니다.

```bash
../.venv/bin/python synced_image_recorder.py --config configs/capture.yaml --monitor-only
```

Linux의 GNSS와 외부 IMU 기본 포트는 `auto`이며 `/dev/serial/by-id` 장치 ID로
결정됩니다. Android 휴대폰처럼 GNSS가 아닌 `ttyACM` 장치는 자동 선택하지 않습니다.

### 3) 디버그 뷰어

```bash
../.venv/bin/python -m geonova_depthai.debug_ui --config configs/debug_ui.yaml
```

### 4) 매핑 실행 (Jetson Controller 진입점)

```bash
../.venv/bin/python ../main.py --config ../config.yaml
```

Controller 실행 시 `JETSON_PIPELINE_RESULTS_DIR`과 `JETSON_PIPELINE_SENSOR_BRIDGE_DIR`
환경변수가 우선되어 쓰기 경로 충돌을 줄입니다.
Controller의 DepthAI preset은 모든 원시 수집을 단일 `/data/collections` 작업 폴더에
저장하며, 각 데이터셋은 `yyyy-mm-dd-hh-mm-ss_raw` 이름으로 직접 생성됩니다.
같은 초 이름이 이미 있으면 기존 데이터를 덮어쓰지 않고 다음 빈 초 이름을 원자적으로
예약합니다.

## 저장소 구조

```text
code/
  geonova_depthai/    핵심 캡처/후처리/변환/매핑 라이브러리
  inference/          클래스 추론 유틸
  scripts/            uv/venv 기반 설치 스크립트
  configs/            수집·동기화·캘리브레이션·선형화 설정
  tests/              동작 검증 테스트
model/
  data.yaml           학습 데이터셋 메타
  safe_gard_train.py  Ultralytics 학습 진입점
  convert_label.py     COCO polygon → YOLO segmentation 변환
  n_model/            저장소 포함 소형 모델 (`best.pt`)
  x_model/            현장 대형 모델(로컬 전용)
data/
  README.md           데이터 루트 정책
results/              런타임 산출물(권장: Git 제외, 기본값 경로)
install.sh / install.ps1  Jetson·Ubuntu·Windows 공통 환경 설치 진입점
main.py / config.yaml
```

## 기본 실행 파일

```bash
cd code
../.venv/bin/python synced_image_recorder.py --help
../.venv/bin/python build_synced_dataset.py --help
../.venv/bin/python tests/test_config_cli.py
```

설정 파일 기본값은 `code/configs/*.yaml`이며, 현장 고정값은
`configs/*.local.yaml` 또는 환경별 override로 분리해 Git 추적에서 제외합니다.

## Jetson Controller 포인트

- `main.py` + `config.yaml`이 고정 진입점입니다.
- 센서 상태/프리뷰는 controller bridge로 게시되어 운영 UI와 연동됩니다.
- 쓰기 경로는 `JETSON_PIPELINE_RESULTS_DIR`을 우선 적용해 읽기 전용 릴리스 경로 문제를 피합니다.

## 모델·학습

```bash
../.venv/bin/python model/convert_label.py --help
../.venv/bin/python model/safe_gard_train.py --config model/safe_gard_train_config.yaml
```

- `model/safe_gard_train_config.yaml`: 단일 GPU 로컬 학습 예시
- `model/safe_gard_train_config.4gpu.yaml`: 멀티 GPU 예시
- GitHub 대형 가중치(>100 MiB)는 로컬 보관 후 배포 시 별도 채널 사용

## 민감 정보 처리

- 노출되면 안 되는 NTRIP 계정은 공유 YAML에 두지 말고 환경변수 또는 `*.local.yaml`로 분리
- 환경변수: `NTRIP_USERNAME`, `NTRIP_PASSWORD`
- 크레덴셜이 포함되지 않도록 `.gitignore`에 맞춰 운영
- 이동 중에는 기본 5분마다 더 가까운 기준국을 확인하고, 새 RTCM3 스트림을
  검증한 뒤 기존 연결을 닫는 make-before-break 방식으로 전환

## 로컬 환경과 산출물

- `.venv`는 Git에 올리지 않는 로컬 실행 환경이며, `install.sh` 또는 `install.ps1`로
  언제든 동일하게 재생성합니다.
- `__pycache__`, `.pytest_cache`, `results`, 수집 데이터와 학습 산출물은
  `.gitignore`로 제외합니다.
