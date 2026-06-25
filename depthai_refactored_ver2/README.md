# DepthAI RGB-D·RTK 방호울타리 선형화

OAK RGB-D 카메라, GPS/RTK, 내장 IMU와 외부 EBIMU를 동기 수집하고 YOLO
segmentation 결과를 지도 좌표의 방호울타리 선형으로 만드는 오프라인
파이프라인입니다. 최종 공간 연산과 길이 계산은 미터 단위 `EPSG:5179`에서
수행합니다.

## 처리 파이프라인

```text
센서 수집
  RGB + RGB 정렬 Depth + IMU + GPS/RTK + EBIMU
        ↓
후처리 동기화
  RGB 프레임 기준 device timestamp nearest matching
        ↓
YOLO segmentation
  마스크·박스 plot 저장
        ↓
관측점 생성
  마스크 내부 8 m 이하 유효 Depth fragment → representative 1점
  마스크 상·중·하 3점 → 형상 확인용 feature
        ↓
세계좌표 변환
  카메라 내부표정 + 장착각/lever-arm + RTK/자세
        ↓
점 보정
  EPSG:5179 → RTK 궤적 접선·법선 좌표 → Huber 측방 offset 보정
        ↓
선형화
  잔차 필터 → 방향·거리·시간 제약 → chainage spline 재샘플링
        ↓
CSV + Point/Polyline SHP + QA
```

## 설치

Python 3.11을 권장합니다. 설치기는 `uv`를 준비하고 `nvcc --version` 결과에
따라 PyTorch 2.7.1의 CPU/CUDA 빌드를 자동 선택합니다.

### Windows PowerShell

```powershell
.\scripts\setup_env.ps1 --config configs\setup.yaml
.\.venv\Scripts\Activate.ps1
```

### Linux

```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh --config configs/setup.yaml
. .venv/bin/activate
```

`configs/setup.yaml`에서 `dev: true`로 바꾸면 `pytest`도 설치합니다. 자동 선택
대신 빌드를 지정하려면 YAML의 `cuda`를 바꾸거나 CLI로 한 값만 덮어씁니다.

```bash
python setup_env.py --config configs/setup.yaml --cuda cu118
```

YAML에 없는 값은 기존 코드 기본값을 그대로 사용하며, 명시한 CLI 옵션은 YAML보다
우선합니다.

## 1. 센서 연결 검증

본수집 전에 사용하는 장치만 개별 확인합니다.

```bash
python tests/test_depthai_rgbd.py --config configs/rgbd_test.yaml
python tests/test_gps_ntrip.py --config configs/gps_ntrip_test.yaml
python tests/test_ebimu.py --config configs/ebimu_test.yaml
```

카메라 내부표정과 RGB–Depth 정렬을 확인해야 할 때는 체커보드 도구를 사용합니다.

```bash
python tests/test_checkerboard_calibration.py --config configs/camera_calibration.yaml
python tests/test_rgb_depth_checkerboard.py --config configs/rgb_depth_validation.yaml
```

각 도구의 판정 임계값과 입력 방식은 해당 명령의 `--help`에 설명되어 있습니다.

## 2. 원시 이벤트 수집

```bash
python synced_image_recorder.py --config configs/capture.yaml
```

Windows PowerShell에서는 `\` 대신 백틱을 사용하거나 한 줄로 실행합니다.

기본 RGB-D 정렬은 `--depth-alignment-mode auto`입니다.

- RVC2/RVC3: 실제 RGB 출력을 `StereoDepth.inputAlignTo`에 연결
- RVC4: `ImageAlign`으로 Depth를 RGB에 정렬
- 저장 RGB: 장치 factory calibration으로 undistort
- Depth: `uint16` millimeter PNG
- confidence map: 마스크 fragment의 soft quality gate에 사용하므로 가능하면 저장

수집 폴더에는 스트림별 이벤트 CSV와 `rgb/`, `depth_mm/` 이미지가 생성됩니다.
GPS/NTRIP/EBIMU 장치 및 보정 서버 설정은 다음으로 확인합니다.

```bash
python synced_image_recorder.py --help
```

NTRIP 값은 CLI 또는 `NTRIP_HOST`, `NTRIP_PORT`, `NTRIP_MOUNTPOINT`,
`NTRIP_USERNAME`, `NTRIP_PASSWORD` 환경변수로 지정할 수 있습니다.

## 3. 데이터 동기화

원본 폴더 안에 `timestamps.csv`, `imu.csv`와 품질 보고서를 생성합니다.

```bash
python build_synced_dataset.py --config configs/sync.yaml
```

별도 폴더를 만들려면 `--output-dir`을 지정합니다. 같은 파일시스템에서는 이미지
symlink를 사용하며, 실제 파일 복사가 필요하면 `--copy-images`를 추가합니다.

```bash
python build_synced_dataset.py --config configs/sync.yaml --copy-images
```

동기 데이터셋은 최소한 다음 항목을 포함해야 합니다.

```text
timestamps.csv
imu.csv
gps.csv                  # GPS를 사용한 경우
external_imu.csv         # EBIMU를 사용한 경우
metadata.json
rgb/
depth_mm/
```

## 4. YOLO 검출과 선형화 일괄 실행

동기화된 데이터셋과 segmentation 모델을 입력합니다. 초기 카메라 안정화 구간을
제외하기 위해 `--start-frame`은 200 이상이어야 합니다.

```bash
python tests/test_yolo_seg_shp.py --config configs/yolo.yaml
```

`--orientation-source ebimu`는 보정된 EBIMU/외부표정을 사용하는 운영 모드입니다.
`gps-course-level`은 GPS course를 yaw로 쓰고 카메라를 수평으로 가정하므로
초기 확인용 근사 모드입니다. 정밀 산출 전 `metadata.json`의 카메라 장착각과
GPS–카메라 lever-arm을 실제 측정값으로 설정해야 합니다.

YOLO 실행이 끝나면 선형화가 자동 실행됩니다. YOLO 결과만 만들려면
`--no-linearize`를 추가합니다.

### YOLO 관측점 생성 알고리즘

1. segmentation polygon을 원본 RGB 크기의 binary mask로 복원합니다.
2. 마스크 경계를 erosion해 배경 Depth 혼입을 줄입니다.
3. Depth 0, 최대거리 초과, confidence 기준 초과 픽셀을 제거합니다.
4. Depth median/MAD로 fragment 내부 이상치를 제거합니다.
5. 중심 위치·Depth 잔차가 안정적인 픽셀을 `representative`로 선택합니다.
6. 마스크 PCA의 `endpoint_a/midpoint/endpoint_b`는 `feature`로만 저장합니다.
7. 최종 지도 선형에는 `point_usage=mapping`인 representative만 사용합니다.

검출된 프레임 이미지는 원본 복사본이 아니라 YOLO 마스크·박스·라벨이 표시된
`result.plot()` 이미지로 `detected_plot/`에 저장됩니다.

## 5. 방호 울타리 점 보정과 선형화만 재실행

YOLO의 `points.csv`가 있으면 모델 추론 없이 보정 파라미터만 바꿔 빠르게
재실행할 수 있습니다.

```bash
python -m geonova_depthai.fence_linearization --config configs/linearization.yaml
```

### 점 보정 알고리즘

1. WGS84 관측점을 `always_xy=True`로 `EPSG:5179`에 한 번 투영합니다.
2. RTK fixed/position/HDOP가 유효한 GPS epoch를 선택합니다.
3. GPS 궤적을 이동 median과 Savitzky–Golay로 평활화합니다.
4. 각 관측점을 가장 가까운 궤적 station의 접선·법선 좌표 `(s, lateral)`로
   변환합니다.
5. chainage·프레임 로컬 윈도우에서 품질 가중 Huber 회귀로 측방 offset을
   추정합니다.
6. 허용 이동량 안의 점은 `C(s) + b(s)n(s)` 방향으로 재투영합니다.
7. `--max-correction-m`보다 크게 움직여야 하는 점은
   `gross_lateral_outlier`로 표시하고 최종선에서 제외합니다.
8. 로컬 TLS 직교 잔차와 MAD 임계값으로 남은 이상치를 필터링합니다.

### 선 연결 알고리즘

1. 대표점을 좌·우 side 및 chainage 순서로 정렬합니다.
2. 프레임 순서, 최대 공간 gap, 최대 heading 차이, 측방 offset 변화를 동시에
   검사합니다.
3. 조건을 벗어나면 실제 단절로 간주해 새로운 segment를 시작합니다.
4. 최소 지지 검출 수를 만족하는 segment만 선으로 만듭니다.
5. segment를 chainage 기준 smoothing spline으로 적합합니다.
6. `--line-sample-spacing-m` 간격으로 재샘플링해 GPS 진행방향을 따르는
   `POLYLINEZ`를 생성합니다.
7. 2D 길이는 EPSG:5179 XY, 3D 길이는 XYZ 차분으로 각각 계산합니다.

### 주요 튜닝 옵션

| 옵션 | 기본값 | 의미 | 조정 방향 |
|---|---:|---|---|
| `--max-observation-depth-mm` | 8000 | 지도점으로 사용할 최대 Depth | 장비 거리 검증 후에만 증가 |
| `--side-window-m` | 20 | Huber offset의 chainage 윈도우 | 급곡선은 감소, 노이즈가 크면 증가 |
| `--side-frame-window` | 100 | 같은 주행 구간으로 볼 프레임 범위 | 중복 주행 혼입 시 감소 |
| `--huber-delta-m` | 0.75 | Huber loss 전환 잔차 | 관측 노이즈에 맞춰 조정 |
| `--max-correction-m` | 3 | 강제로 이동할 수 있는 최대 거리 | 큰 값은 반대편 오연결 위험 |
| `--max-gap-m` | 6 | 이웃 대표점 연결 최대 거리 | 과분절 시 소폭 증가 |
| `--max-frame-gap` | 60 | 연결 가능한 최대 프레임 간격 | 큰 값은 다른 주행 혼입 위험 |
| `--max-heading-deg` | 35 | GPS 접선 대비 연결 각도 | 곡선에서만 소폭 증가 |
| `--min-line-support` | 5 | 선 하나의 최소 검출 수 | 짧은 시설물이 사라질 때 감소 |
| `--line-sample-spacing-m` | 0.5 | 최종 spline vertex 간격 | 정밀도와 파일 크기 절충 |
| `--spline-smoothing` | 0.15 | 선형 평활 강도 | 과평활 시 감소 |

파라미터 비교 시 같은 출력 폴더를 덮어쓰지 말고 `linearized_baseline`,
`linearized_gap8`처럼 별도 폴더를 사용합니다.

## 산출물

```text
yolo_seg/
  detected_plot/                     YOLO plot 이미지
  detections.jsonl                   프레임별 검출·관측점
  points.csv                         feature/mapping 점과 센서 품질
  yolo_seg_points_pixels.*           픽셀 POINT SHP
  yolo_seg_points_wgs84.*            좌표화 성공 WGS84 POINTZ SHP
  summary.json
  linearized/
    debug_00_raw.csv/.shp            원본과 유효성 판정
    debug_01_projected.csv/.shp      EPSG:5179 투영점
    debug_02_local.csv/.shp          chainage/lateral 좌표
    debug_03_side_corrected.csv/.shp Huber offset 보정 결과
    debug_04_interpolated.csv/.shp   하위 호환 보간 단계
    debug_05_filtered.csv/.shp       최종 채택점
    points_corrected.csv/.shp        보정 이력·QA 전체점
    fence_lines.csv/.shp             최종 POLYLINEZ
    qa_metrics.csv                    완전성·잔차·방향·gap·길이
    linearization_summary.json        설정과 실행 요약
```

SHP는 DBF 필드명/타입 제한이 있으므로 상세 원인은 CSV를 기준으로 확인합니다.
`qa_metrics.csv`에서는 특히 다음을 함께 봅니다.

- `completeness`
- `accepted_residual_rmse_m`
- `mean_heading_error_deg`, `p95_heading_error_deg`
- `gross_lateral_outlier_count`
- `line_count`, `line_gap_count`, `maximum_line_gap_m`
- `total_length_m_2d`, `total_length_m_3d`

## Debug UI

```bash
python -m geonova_depthai.debug_ui --config configs/debug_ui.yaml
```

브라우저에서 `http://127.0.0.1:8088`을 열어 RGB, Depth, GPS/IMU 동기화,
YOLO 관측점과 세계좌표 품질을 확인합니다.

## CLI 도움말

모든 실행 파일은 YAML을 지원합니다. `--config`에 없는 키는 기존 기본값을
유지하고, 명시적인 CLI 옵션은 YAML 값을 덮어씁니다. 전체 기본 설정 파일은
각 명령의 `--write-default-config`로 생성할 수 있습니다.

YAML 키는 CLI의 `--max-gap-m`을 `max_gap_m`으로 적는 식으로 하이픈을
밑줄로 바꿉니다. NTRIP username/password는 기본 YAML에 기록하지 않으므로
환경변수나 별도 비공개 YAML로 제공합니다.

```bash
python synced_image_recorder.py --write-default-config capture.defaults.yaml
python tests/test_yolo_seg_shp.py --write-default-config yolo.defaults.yaml
python -m geonova_depthai.fence_linearization --write-default-config linearization.defaults.yaml
```

옵션의 의미와 단위는 `--help`로 확인합니다.

```bash
python setup_env.py --help
python synced_image_recorder.py --help
python build_synced_dataset.py --help
python tests/test_yolo_seg_shp.py --help
python -m geonova_depthai.fence_linearization --help
python -m geonova_depthai.debug_ui --help
python tests/test_depthai_rgbd.py --help
python tests/test_gps_ntrip.py --help
python tests/test_ebimu.py --help
python tests/test_checkerboard_calibration.py --help
python tests/test_rgb_depth_checkerboard.py --help
```

## 코드 구성

```text
setup_env.py                         uv/PyTorch/의존성 설치
configs/                             운영 단계별 YAML 예제
synced_image_recorder.py             센서 수집 진입점
build_synced_dataset.py              후처리 동기화 진입점
geonova_depthai/capture/             수집 CLI와 raw writer
geonova_depthai/postprocess/         이벤트 동기화
geonova_depthai/runtime.py           DepthAI·GPS·NTRIP·EBIMU runtime
geonova_depthai/debug_ui.py          데이터셋 확인과 좌표 변환
geonova_depthai/yolo_seg_shp.py      YOLO·Depth fragment·WGS84 관측점
geonova_depthai/fence_linearization.py EPSG:5179 보정·spline·SHP·QA
tests/                               센서/캘리브레이션/회귀 검증
```
