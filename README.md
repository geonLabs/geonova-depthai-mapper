# Geonova DepthAI Mapper — Canonical Multi-Fusion Repository

이 저장소는 Geonova 현장 센서 파이프라인의 **단일 기준 저장소**입니다.
OAK RGB-D 카메라, GPS/RTK, OAK IMU와 외부 IMU를 동기 수집하고 YOLO
segmentation 결과를 지도 좌표계의 방호울타리 선형으로 변환합니다. 기존
`dbparkJ/geo_multifusion_sensors`에 있던 고유 LiDAR/ROS 도구도
`tools/lidar/`로 통합했습니다.

핵심 Jetson 런타임은 OAK stereo depth를 사용합니다. LiDAR 기능은 ROS bag 변환과
캘리브레이션 검증을 위한 **오프라인 도구**이며 Controller 자동 실행 경로와
의존성을 분리합니다.

## 프로젝트 구성

### DepthAI 현장 런타임

- RGB-D, GPS/RTK, OAK IMU, 외부 EBIMU 동기 수집
- 데이터셋 저장 없이 상태와 프리뷰만 게시하는 `--monitor-only`
- Jetson Controller 센서 bridge와 부팅 자동 실행
- USB2 대역폭 제한, OAK 재연결, 안전한 시리얼 장치 자동 탐색
- 이동 중 NTRIP 기준국 재선택과 make-before-break RTCM 전환
- YOLO segmentation, Depth 관측점 생성, 지리좌표 변환, SHP 선형화
- 단일 GPU와 멀티 GPU 학습 진입점

### LiDAR / ROS 오프라인 도구

- ROS1 bag의 PointCloud2를 binary PCD로 변환
- 각 LiDAR 프레임에 가장 가까운 카메라 이미지 추출
- 캘리브레이션 JSON을 이용한 LiDAR–카메라 overlay 생성
- ROS 설치가 필요 없는 `rosbags` 변환기와 ROS1 전용 투영 도구 분리

통합 배경과 경로 대응은 [`docs/CONSOLIDATION.md`](docs/CONSOLIDATION.md),
LiDAR 사용법은 [`tools/lidar/README.md`](tools/lidar/README.md)를 참고합니다.

## 빠른 시작

### 1. DepthAI 환경 설치

Jetson 또는 Ubuntu PC:

```bash
chmod +x install.sh
./install.sh --dev
```

Windows PowerShell:

```powershell
.\install.ps1 --dev
```

기본 가상환경은 저장소 루트의 `.venv`입니다. 다른 위치가 필요하면 설치기가
사용자 지정값을 유지합니다.

```bash
./install.sh --venv /data/venvs/geonova-depthai --dev
```

```powershell
.\install.ps1 --venv D:\venvs\geonova-depthai --dev
```

- Jetson: L4T와 JetPack Python ABI를 감지하고 NVIDIA CUDA PyTorch를 재사용하거나
  호환 wheel을 구성합니다.
- Ubuntu/Windows: `uv`가 Python 3.11을 준비하고 `nvcc` 또는 `nvidia-smi` 결과에
  맞는 공식 PyTorch CPU/CUDA wheel을 설치합니다.
- 기존 환경을 완전히 다시 만들 때만 `--recreate`를 사용합니다.

기본 환경 활성화:

```bash
. .venv/bin/activate
```

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. 데이터 수집과 동기화

```bash
cd code
../.venv/bin/python synced_image_recorder.py --config configs/capture.yaml
../.venv/bin/python build_synced_dataset.py --config configs/sync.yaml
../.venv/bin/python tests/test_yolo_seg_shp.py --config configs/yolo.yaml
```

Controller 센서 상태와 카메라 프리뷰만 계속 게시하고 데이터셋은 만들지 않으려면
모니터 모드를 사용합니다.

```bash
../.venv/bin/python synced_image_recorder.py \
  --config configs/capture.yaml \
  --monitor-only
```

모니터 모드는 지원되는 최대 RGB 크기를 선택하되 최대 5 FPS로 제한하고, USB2
연결에서는 MJPEG 전송으로 대역폭을 줄입니다. 카메라가 끊기면 GPS와 외부 IMU
상태를 유지한 채 OAK 연결을 다시 시도합니다.

Linux의 GNSS와 외부 IMU 기본 포트는 `auto`이며 `/dev/serial/by-id` 장치 식별자로
결정합니다. Android 휴대폰처럼 GNSS가 아닌 번호형 `ttyACM` 장치는 자동 선택하지
않습니다.

### 3. 디버그 뷰어

```bash
../.venv/bin/python -m geonova_depthai.debug_ui \
  --config configs/debug_ui.yaml
```

### 4. Jetson Controller 진입점

```bash
../.venv/bin/python ../main.py --config ../config.yaml
```

Controller가 실행할 때 `JETSON_PIPELINE_RESULTS_DIR`과
`JETSON_PIPELINE_SENSOR_BRIDGE_DIR` 환경변수가 YAML이나 이전 CLI 경로보다
우선합니다. 원시 수집은 단일 작업 폴더 아래 `yyyy-mm-dd-hh-mm-ss_raw` 이름으로
생성하며, 같은 초 이름이 이미 있어도 기존 폴더를 다시 열거나 덮어쓰지 않습니다.

### 5. ROS bag → PCD/이미지

LiDAR 도구는 DepthAI 환경과 분리하는 것을 권장합니다.

```bash
cd tools/lidar
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp config.yaml config.local.yaml
python convert_lidar_bag_to_pcd.py config.local.yaml
```

`project_lidar_overlay.py`는 ROS1 Noetic의 `rosbag`, `sensor_msgs`, `cv_bridge`가
필요합니다.

```bash
source /opt/ros/noetic/setup.bash
python project_lidar_overlay.py \
  --bag /data/sample.bag \
  --calib /data/calib.json \
  --output overlay-output/sample.jpg
```

## 저장소 구조

```text
code/
  geonova_depthai/       DepthAI·GPS·NTRIP·IMU 핵심 라이브러리
  inference/             클래스 추론 유틸
  configs/               수집·동기화·캘리브레이션·선형화 설정
  scripts/               uv/venv 설치 스크립트
  tests/                 센서·매핑·회귀 테스트

model/
  data.yaml              학습 데이터셋 메타
  safe_gard_train.py     Ultralytics 학습 진입점
  convert_label.py       COCO polygon → YOLO segmentation 변환
  n_model/               저장소에 포함 가능한 소형 모델
  x_model/               현장 대형 모델 로컬 위치

tools/lidar/
  convert_lidar_bag_to_pcd.py
  project_lidar_overlay.py
  config.yaml
  requirements.txt
  tests/

docs/
  CONSOLIDATION.md       저장소 통합 결정과 개발 규칙

data/                    현장 데이터 정책 문서만 추적
results/                 런타임 산출물, Git 제외
main.py / config.yaml    Jetson Controller 고정 진입점
install.sh / install.ps1 공통 환경 설치 진입점
```

## 기본 검증 명령

```bash
cd code
../.venv/bin/python synced_image_recorder.py --help
../.venv/bin/python build_synced_dataset.py --help
../.venv/bin/python tests/test_config_cli.py
../.venv/bin/python -m pytest -q tests
```

LiDAR 순수 회귀 테스트:

```bash
cd tools/lidar
. .venv/bin/activate
python -m pip install pytest
python -m pytest -q tests
```

GitHub Actions는 Python 구문 검사, LiDAR 보조 함수 테스트, 공유 Controller 설정의
NTRIP 자격증명 키 검사를 수행합니다.

## 모델과 학습

```bash
.venv/bin/python model/convert_label.py --help
.venv/bin/python model/safe_gard_train.py \
  --config model/safe_gard_train_config.yaml
```

- `model/safe_gard_train_config.yaml`: 단일 GPU 예시
- `model/safe_gard_train_config.4gpu.yaml`: 멀티 GPU 예시
- GitHub 일반 파일 한도를 넘는 가중치는 로컬 또는 별도 배포 채널에 보관

## 설정과 민감 정보

공유 YAML에는 NTRIP 사용자명과 비밀번호를 넣지 않습니다.

```bash
export NTRIP_USERNAME='...'
export NTRIP_PASSWORD='...'
```

자동 실행 장비에서는 Jetson Controller가 관리하는 보호 환경 설정으로 두 값을
주입합니다. 장비별 포트, 경로, 자격증명은 `*.local.yaml` 또는 환경변수로 관리하며
Git에서 제외합니다.

이동 수집 중에는 기본 5분마다 현재 위치에서 더 가까운 기준국을 확인합니다. 새
연결에서 CRC가 유효한 RTCM3 프레임을 확인한 뒤 기존 스트림을 닫으므로 후보 연결이
실패해도 현재 보정 스트림을 유지합니다.

## 단일 원본 정책

`geonLabs/geonova-depthai-mapper`가 앞으로의 유일한 개발 원본입니다.
`dbparkJ/geo_multifusion_sensors/safe_gard_test/code`에 같은 코드를 다시 반영하지
않습니다. 신규 센서 공통 기능은 `code/geonova_depthai`, ROS 전용 기능은
`tools/lidar`에서 개발합니다.

- `.venv`, `__pycache__`, 테스트 캐시와 런타임 결과는 Git 제외
- `.bag`, `.pcd`, 수집 데이터와 대형 학습 산출물은 로컬 보관
- 코드 변경은 기준 저장소의 브랜치와 PR에서 검증 후 반영
