# Geonova DepthAI Mapper

Jetson의 **원본 수집**과 서버의 **데이터 후처리**를 분리한 저장소입니다. 수집 실행은 [JetsonControllerApp 외부 파이프라인 계약](https://github.com/dbparkJ/JetsonControllerApp/blob/main/docs/EXTERNAL_PIPELINE_CONTRACT.md)을 따릅니다.

| 구성 | 위치 | 역할 |
|---|---|---|
| Jetson 수집 | `main.py`, `config.yaml`, `capture/` | OAK RGB-D·IMU, GNSS/NTRIP, EBIMU, 앱 프리뷰·센서 상태 |
| 서버 후처리 | `postprocess/` | 이벤트 동기화, YOLO, Depth 대표점, 좌표 변환, SHP, 선형화, 확인 화면 |
| 공유 계약 | `shared/` | YAML 파서, 수집 완료 표시, 파일 전달 검증 |
| 기존 명령 호환 | `code/` | 이전 import/CLI 어댑터, 하드웨어 진단 도구와 기존 개발용 설치기 |
| 별도 도구 | `tools/lidar/`, `model/` | LiDAR/ROS 오프라인 변환, 모델 학습 |

## Jetson 수집

저장소 루트에서 Jetson 시스템 Python 3.8~3.12로 실행합니다.

```bash
./install.sh
.venv/bin/python main.py --config config.yaml
```

설치기는 수집 전용 `.venv`를 만듭니다. YOLO/PyTorch/지도 후처리 패키지를 설치하지 않습니다. 시스템 패키지가 꼭 필요한 장치에서만 `./install.sh --system-site-packages`를 사용합니다. Windows 수집/진단 환경은 `python` 3.8~3.12로 `./install.ps1`을 실행합니다. Controller 등록은 Jetson에서 생성한 Linux venv가 필요합니다.

앱에서 이 저장소 루트를 등록하면 기존 센서 인계와 systemd 실행 경로를 사용합니다. 출력은 `JETSON_PIPELINE_RESULTS_DIR`가 우선하고, 프리뷰·GNSS는 기존 센서 브리지에 게시합니다. 등록한 뒤 시작해야 하며, 소스 변경 후에는 재등록해야 새 스냅샷이 적용됩니다.

자세한 장치 설정과 전달 규칙: [capture/README.md](capture/README.md).

## 서버 후처리

서버 Python 3.11~3.12로 별도 환경을 만듭니다.

```bash
PYTHON=python3.11 ./postprocess/install.sh
postprocess/.venv/bin/python postprocess/main.py sync \
  --dataset /srv/geonova/raw/RUN_ID \
  --output-dir /srv/geonova/processed/RUN_ID/synced
```

`RUN_ID`는 실제 수집 폴더명입니다. 수집 폴더 전체를 먼저 서버로 전달합니다. `capture_manifest.json`의 완료 상태, 파일 크기, CSV/JSON 해시를 확인한 다음 원본과 별도 폴더에 동기화 결과를 만듭니다. 이미지도 복사해 결과 폴더를 다른 서버로 다시 옮길 수 있습니다.

YOLO·선형화·SHP 명령과 GPU 환경 준비: [postprocess/README.md](postprocess/README.md). 동기화만 실행할 때는 DepthAI, 시리얼 포트, GPU가 필요하지 않습니다.

## 수집과 후처리를 독립 폴더로 내보내기

```bash
python3 tools/export_component.py capture \
  --destination /tmp/geonova-capture --init-git
python3 tools/export_component.py postprocess \
  --destination /tmp/geonova-postprocess --init-git
```

각 export는 필요한 소스와 공유 계약만 포함합니다. 모델·데이터·venv는 포함하지 않습니다. 수집 export에는 Git 작업 트리, `main.py`, `config.yaml`, 수집 의존성 파일이 준비됩니다. 폴더를 Jetson의 실제 운영 경로로 옮긴 후 그 장치에서 `./install.sh`를 실행하고 앱에 등록합니다. 새 GitHub 저장소는 자동 생성하지 않습니다.

실제 경로 대응·서버 이전 순서·실기 확인 항목: [docs/CAPTURE_POSTPROCESS_SPLIT_KO.md](docs/CAPTURE_POSTPROCESS_SPLIT_KO.md).

## 진단·검증

이전 상세 알고리즘 문서와 진단 도구는 [code/README.md](code/README.md), LiDAR 도구는 [tools/lidar/README.md](tools/lidar/README.md)에 있습니다. 모델 학습은 [model/README.md](model/README.md)에서 별도로 다룹니다.

```bash
# 개발/CI 환경에서만 두 구성의 테스트 의존성을 함께 설치합니다.
python -m pip install -r postprocess/requirements.lock depthai==3.1.0 pyserial==3.5 pytest
PYTHONPATH=code python -m pytest -q tests code/tests
```

실제 센서 수집과 모델 추론 정확도 검증은 Jetson·카메라·GNSS·EBIMU 및 현장 데이터로 별도 수행합니다.
