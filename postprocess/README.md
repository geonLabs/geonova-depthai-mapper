# Geonova Postprocess — 서버 후처리 전용

센서를 열지 않고 전달받은 원본으로 동기화, YOLO segmentation, Depth 대표점, 지도 좌표 변환, SHP, 방호울타리 선형화를 수행합니다. `src/geonova_postprocess/dataset.py`는 데이터 읽기·좌표 계산, `debug_ui.py`는 선택적 확인 화면입니다. LiDAR/ROS 변환기는 통합 저장소 또는 독립 export의 `tools/lidar/`에 있습니다. 학습 코드는 통합 저장소 `model/`에서 별도로 관리합니다.

## 설치

통합 저장소 루트 기준입니다. 서버 Python 3.11~3.12를 사용합니다.

```bash
PYTHON=python3.11 ./postprocess/install.sh
```

독립 export에서는 `PYTHON=python3.11 ./install.sh`입니다. `.venv`는 각 서버에서 새로 생성합니다. 동기화 자체는 표준 라이브러리로 실행 가능하고, 기본 설치는 좌표 계산·선형화·확인 화면을 위한 수치 패키지를 포함합니다. DepthAI, pyserial, JetPack은 설치하지 않습니다.

YOLO가 필요하면 서버의 CUDA/드라이버와 맞는 PyTorch 및 torchvision을 먼저 이 venv에 설치한 뒤 다음을 실행합니다.

```bash
./postprocess/install.sh --with-yolo
```

독립 export는 `./install.sh --with-yolo`입니다. 기존 통합 설치기의 PyTorch 2.7.1/torchvision 0.22.1 조합을 참고할 수 있지만 CUDA wheel index는 서버 환경을 확인해 선택합니다. 모델은 별도 경로로 관리하며 checksum, 라이선스, 학습 revision, 클래스, 입력 크기를 기록합니다. 기본 설치로 YOLO 추론까지 준비되지는 않습니다. `--with-yolo`는 선택한 torch/torchvision 버전과 수치 패키지를 제약으로 고정하고, Ultralytics 요구에 따라 headless OpenCV를 같은 버전의 `opencv-python`으로 교체합니다. 서버의 OpenCV 시스템 공유 라이브러리도 필요하며 실제 GPU 추론 설치는 서버에서 검증합니다.

## 단계별 실행

`RUN_ID`는 실제 원본 폴더명으로 바꿉니다. 다음 명령은 통합 저장소 루트 기준이며, 독립 export에서는 `postprocess/` 접두사를 빼면 됩니다.

```bash
postprocess/.venv/bin/python postprocess/main.py sync \
  --dataset /srv/geonova/raw/RUN_ID \
  --output-dir /srv/geonova/processed/RUN_ID/synced

postprocess/.venv/bin/python postprocess/main.py yolo \
  --dataset /srv/geonova/processed/RUN_ID/synced \
  --model /srv/geonova/models/guardrail.pt \
  --output-dir /srv/geonova/processed/RUN_ID/yolo \
  --start-frame 200 --max-frames 999999 --orientation-source ebimu --device 0

postprocess/.venv/bin/python postprocess/main.py linearize \
  --points /srv/geonova/processed/RUN_ID/yolo/points.csv \
  --trajectory /srv/geonova/processed/RUN_ID/synced/timestamps.csv \
  --output-dir /srv/geonova/processed/RUN_ID/linearized
```

YOLO는 기본적으로 선형화도 실행합니다. 세 번째 명령은 설정을 바꿔 선형화만 다시 실행할 때 사용합니다. 기존 알고리즘의 프레임 200 이후 시작·Depth 8m 조건을 유지했습니다. 새 장치/지역에 맞는 값은 실측 데이터로 검증해야 합니다.

각 명령은 `--config configs/<명령>.yaml`도 지원합니다. 예시의 `REPLACE_RUN_ID`와 모델 경로를 수정합니다. `sync`의 상대경로는 YAML 기준이며, 기존 yolo/linearize CLI의 상대경로는 실행 디렉터리 기준입니다. 운영 설정에서는 예시처럼 절대경로를 사용합니다.

새 `sync` CLI는 원본과 분리된 새/빈 출력 폴더가 필수이고 이미지도 복사합니다. 원본 파일을 덮어쓰거나 이미지 링크로 원본 서버에 종속되지 않습니다. 같은 출력 폴더 재사용은 거부하므로 재처리는 다른 job 폴더를 지정합니다. 결과 복사에 원본 이미지 크기만큼 공간이 필요합니다. 동기화 실패/0프레임은 실패 상태로 남고 재실행은 새 출력 폴더에서 합니다.

이전 수집본은 manifest가 없으므로, 수집이 완전히 종료된 `raw_events_v1` 원본에만 `--allow-legacy`를 명시합니다. 이 옵션도 `recording`/`failed` 표시가 있는 수집본을 허용하지 않습니다.

## 선택적 확인 화면

```bash
postprocess/.venv/bin/python postprocess/main.py inspect --config postprocess/configs/debug_ui.yaml
```

화면은 기본 배치 처리에서 시작하지 않습니다. 예시 설정은 127.0.0.1:8088이며, 원격 확인은 서버 접근 정책/SSH 터널을 사용합니다. 이 기존 도구는 인증을 갖춘 업로드 API나 작업 대기열이 아닙니다. 자동 전송, 서버 인증, 작업 큐, DB, 결과 API, 재시도 운영은 실제 후처리 서버 이전 단계에서 연결합니다.
