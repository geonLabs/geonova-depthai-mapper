# Geonova DepthAI Mapper

OAK RGB-D, GPS/RTK, 내장·외부 IMU를 동기 수집하고 YOLO segmentation 결과를
지도 좌표의 방호울타리 선형으로 변환하는 프로젝트입니다. 저장소 구조는 현장
작업 폴더와 동일하게 `code/`, `model/`, `data/`로 나눕니다.

```text
code/                         수집·동기화·캘리브레이션·매핑 애플리케이션
model/                        학습·라벨 변환 코드와 배포 가능한 소형 모델
  n_model/best.pt             Git에 포함된 guardrail YOLO26n-seg 모델
  x_model/                    대형 현장 모델을 로컬에 두는 위치
data/                         원시/동기 데이터셋(README 외에는 Git 제외)
install.sh, install.ps1       루트 .venv 설치 진입점
```

## 설치

Linux/macOS에서는 저장소 루트에서 실행합니다.

```bash
chmod +x install.sh
./install.sh --dev
```

Windows PowerShell에서는 다음을 실행합니다.

```powershell
.\install.ps1 --dev
```

설치기는 Python 3.11 가상환경을 저장소 루트의 `.venv`에 만들고 CUDA Toolkit을 감지해
PyTorch 빌드를 선택합니다.

## 수집과 매핑

```bash
cd code
../.venv/bin/python synced_image_recorder.py --config configs/capture.yaml
../.venv/bin/python build_synced_dataset.py --config configs/sync.yaml
../.venv/bin/python tests/test_yolo_seg_shp.py --config configs/yolo.yaml
../.venv/bin/python -m geonova_depthai.debug_ui --config configs/debug_ui.yaml
```

기본 설정은 저장소 루트의 `data/`와 `model/n_model/best.pt`를 사용합니다.
현장별 경로와 파라미터는 `configs/*.local.yaml`에 두면 Git에 포함되지 않습니다.
전체 수집·검증·선형화 절차는 [`code/README.md`](code/README.md)를 참고하세요.

## Jetson Controller 자동실행

저장소 루트는 Controller의 폴더 작업 규칙을 따릅니다. `./install.sh`로 실제
`.venv/bin/python`을 만든 뒤 앱에서 이 폴더를 등록하면 `main.py`와
`config.yaml`이 고정 진입점으로 사용되고, 부팅 시 pipeline service가 자동
활성화됩니다.

```bash
./.venv/bin/python main.py --config config.yaml
```

Controller 실행에서는 `JETSON_PIPELINE_RESULTS_DIR`가 YAML이나 기존 CLI 값보다
우선하므로 읽기 전용 release에 기록하지 않습니다. 일반 폴더 등록의 센서 상태와
프리뷰는 `results/controller-bridge/`에 저장합니다. Controller 앱의 공용 live
preview까지 연결하는 장비 preset은 JetsonControllerApp의
`install-depthai-pipeline.sh`를 사용합니다.

## 학습과 로컬 모델

```bash
.venv/bin/python model/convert_label.py --help
.venv/bin/python model/safe_gard_train.py \
  --config model/safe_gard_train_config.yaml
```

`model/safe_gard_train_config.yaml`은 로컬 단일 GPU 설정이고,
`model/safe_gard_train_config.4gpu.yaml`은 4-GPU YOLO26x 예시입니다.
`model/x_model/best.pt`처럼 GitHub 일반 파일 제한(100 MiB)을 넘는 가중치는
로컬에 보존하며 커밋하지 않습니다. 배포가 필요하면 Git LFS 또는 Release
asset을 별도로 사용하세요.

## 자격증명

NTRIP 계정은 코드나 공유 YAML에 넣지 않고 `NTRIP_USERNAME`,
`NTRIP_PASSWORD` 환경변수 또는 Git에서 제외되는 `*.local.yaml`로 제공합니다.
