# DepthAI RGB-D·RTK 방호울타리 선형화

OAK RGB-D 카메라, GPS/RTK, 내장 IMU와 외부 EBIMU를 동기 수집하고 YOLO
segmentation 결과를 지도 좌표의 방호울타리 선형으로 만드는 오프라인
파이프라인입니다. 최종 공간 연산과 길이 계산은 미터 단위 `EPSG:5179`에서
수행합니다.

명령은 저장소의 `code/`에서 실행합니다. 기본 데이터 경로는 `../data`, 기본
소형 모델은 `../model/n_model/best.pt`입니다. 기존 2026-06-26 현장값은
`configs/profiles/local_2026_06_26/`에 상대경로로 보존되어 있습니다.

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
  confidence-qualified 마스크 Depth 중앙값이 8 m 이내인 fragment → representative 1점
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

저장소 루트의 설치 진입점 하나로 플랫폼을 자동 판별합니다. 부트스트랩용 Python
3.8 이상만 있으면 `uv`와 실제 프로젝트 환경은 설치기가 준비합니다.

### Jetson / Ubuntu PC

```bash
chmod +x install.sh
./install.sh --dev
. .venv/bin/activate
```

### Windows PowerShell

```powershell
.\install.ps1 --dev
.\.venv\Scripts\Activate.ps1
```

| 환경 | Python | PyTorch | 요구사항 파일 |
|---|---|---|---|
| Jetson | JetPack 시스템 Python | NVIDIA CUDA 빌드 재사용 또는 NVIDIA wheel 탐색 | `requirements-jetson.txt` |
| Ubuntu PC | uv Python 3.11 | 공식 CUDA/CPU wheel 자동 선택 | `requirements.txt` |
| Windows PC | uv Python 3.11 | 공식 CUDA/CPU wheel 자동 선택 | `requirements.txt` |

Jetson에서는 `/etc/nv_tegra_release`로 L4T를 감지합니다. 일반 PC용 PyTorch wheel로
NVIDIA 빌드를 덮어쓰지 않으며, PyTorch 버전에 맞는 torchvision을 자동 구성합니다.
Jetson 첫 설치의 torchvision 소스 빌드는 몇 분 걸릴 수 있습니다.

기존 환경을 보존한 채 누락 패키지만 맞추는 것이 기본입니다. 완전 재생성은 명시적으로
요청할 때만 `./install.sh --recreate --dev` 또는
`.\install.ps1 --recreate --dev`를 사용합니다.

데스크톱 PyTorch 빌드를 직접 고르려면 `--cuda cpu|cu118|cu126|cu128`, Jetson에
별도 NVIDIA wheel을 지정하려면 `--jetson-torch <wheel-or-url>`을 사용합니다.
YAML에 없는 값은 기본값을 사용하며 CLI 옵션이 `configs/setup.yaml`보다 우선합니다.

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
순수 회귀 테스트는 하드웨어 수집을 시작하지 않고 다음처럼 실행합니다.

```bash
python -m pytest -q tests
```

### Depth가 실측과 크게 다를 때

Luxonis 문서 기준 OAK-D W wide FOV 800P/75 mm baseline은 3.5~6.5 m 구간에서
대략 몇 % 수준의 depth error가 정상 범위입니다. 4 m 대상이 7 m로 보이는 정도는
post-processing 튜닝이나 임의 scale 보정으로 처리할 문제가 아니라 stereo
calibration/구성 문제로 봅니다.

1. 같은 위치에서 OAK Viewer의 depth도 같은 오차인지 확인합니다.
2. `metadata.json`의 `stereo_depth_model`에서 CAM_B/C sensor, baseline,
   left/right socket, input size가 실제 장치와 맞는지 확인합니다.
3. 좌우 카메라가 바뀌었거나 board config/HFOV/baseline이 틀렸다면 Luxonis
   manual calibration으로 CAM_B/C intrinsics/extrinsics를 다시 만들어 EEPROM에
   플래시합니다.

Luxonis 절차 요약:

```bash
git clone https://github.com/luxonis/depthai.git --branch main
cd depthai
git submodule update --init --recursive
python3 install_requirements.py
python3 calibrate.py --help
```

Charuco board는 화면 전체에 띄우고 실제 square size를 cm 단위로 재서 `-s`에
넣습니다. OAK-D-LR처럼 지원 board가 있는 compact device는 해당 board 이름을
사용하고, 렌즈/모듈 구성이 바뀐 장치는 `resources/depthai_boards/boards/`에
맞는 board config를 준비해 `-brd <board>.json`으로 실행합니다. 촬영은 정면,
가까운 거리의 상하좌우/기울임, 중간 거리, 먼 거리와 코너까지 FOV 전체를 덮도록
진행합니다. 처리 단계에서 epipolar line을 확인한 뒤 성공한 calibration을 EEPROM에
플래시하고, 다시 OAK Viewer와 `tests/test_depthai_rgbd.py`로 거리 스케일을 확인합니다.

참고 문서:

- Luxonis depth accuracy: <https://docs.luxonis.com/hardware/platform/depth/depth-accuracy/>
- Luxonis manual calibration: <https://docs.luxonis.com/hardware/platform/depth/manual-calibration/>
- Luxonis stereo depth 설정: <https://docs.luxonis.com/hardware/platform/depth/configuring-stereo-depth/>
- OAK-D-LR 제품 정보: <https://shop.luxonis.com/products/oak-d-lr>

## 2. 원시 이벤트 수집

```bash
python synced_image_recorder.py --config configs/capture.yaml
```

### Jetson Controller 앱 연동

수집 프로세스는 기본적으로 `/var/lib/jetson-sensors`에 앱 연동용 최신 상태를
게시합니다.

- `status.json`: 카메라·GNSS·IMU heartbeat, GNSS fix quality, NTRIP 상태와 위치
- `camera-preview.jpg`: 최대 4 Hz, 1280 px 폭의 최신 RGB 프리뷰

파일은 같은 디렉터리에서 원자적으로 교체되므로 Jetson Control API가 기록 중인
파일을 읽지 않습니다. `controller_sensor_stale_after_s` 동안 새 샘플이 없으면 해당
센서는 비활성으로 표시됩니다. 프리뷰 인코딩은 별도 스레드에서 동작하며 수집 큐를
막지 않습니다. 경로, 상태 주기, 프리뷰 FPS·폭·JPEG 품질은 `configs/capture.yaml`의
`controller_*` 값으로 조정할 수 있고 `controller_bridge_enabled: false`로 끌 수
있습니다.

systemd로 실행할 때 파이프라인 사용자에게 `/var/lib/jetson-sensors` 쓰기 권한과
`ReadWritePaths`를 함께 부여해야 합니다. JetsonControllerApp의
`install-depthai-pipeline.sh`가 이 설정을 자동으로 추가합니다.

Windows PowerShell에서는 `\` 대신 백틱을 사용하거나 한 줄로 실행합니다.

기본 RGB-D 정렬은 `--depth-alignment-mode auto`입니다.

- RGB 해상도: `rgb_width: 0`, `rgb_height: 0`이면 연결된 컬러 센서의
  `getConnectedCameraFeatures()` 결과를 보고 자동 선택
  (`1920x1200` → `1920x1080` → `1280x720` 후보)
- Stereo 입력: EEPROM의 `getStereoLeftCameraId()`/`getStereoRightCameraId()`를
  우선 사용하고, 연결된 좌우 센서 크기에 맞춰 자동 선택합니다. RVC2에서는 stereo
  matching 폭 한계 때문에 1920x1200 AR0234 계열도 기본적으로 1280x800 입력을
  사용해 전체 FOV를 유지합니다.
- FPS: `fps`는 RGB 출력과 RGB 기준 저장 cadence를 정합니다.
  `depth_fps: 0`이면 기존처럼 depth도 같은 fps를 쓰고, 양수이면 좌우
  mono/stereo depth 입력만 별도 fps로 요청합니다. 예: RGB는 15 Hz로 두고
  depth만 30 Hz로 올리려면 `--fps 15 --depth-fps 30`.
- RVC2/RVC3: 실제 RGB 출력을 `StereoDepth.inputAlignTo`에 연결
- RVC4: `ImageAlign`으로 Depth를 RGB에 정렬
- 저장 RGB: 장치 factory calibration으로 undistort
- Depth: `uint16` millimeter PNG
- confidence map: 마스크 fragment의 soft quality gate에 사용하므로 가능하면 저장

예를 들어 OAK-D-W의 `CAM_A IMX378 4056x3040 COLOR` 센서는 자동으로
`1920x1200` RGB 출력을 선택하고, RVC2에서는 같은 크기의 depth를
`StereoDepth.inputAlignTo`로 맞춥니다. RGB와 depth 크기는 `metadata.json`의
`image_size`, `rgb_sensor`, `depth_alignment`에 기록됩니다. 특정 크기로 강제할
때는 `--rgb-width 1920 --rgb-height 1080`처럼 두 값을 함께 지정합니다.
렌즈/왜곡 보정은 첫 RGB 프레임의 `ImgFrame.getTransformation()`에서 실제 저장
출력 intrinsics를 읽어 `camera_model.intrinsics`에 기록합니다. OAK-D-W처럼
wide lens인 경우 factory intrinsics와 undistorted 출력 intrinsics가 다르므로,
세계좌표 계산은 `factory_distortion_coefficients`가 아니라 저장 픽셀 기준
`camera_model.intrinsics`를 사용합니다. Depth PNG의 mm 값은 DepthAI stereo
calibration으로 이미 계산된 값이고, 앱에서 바뀌는 식은 픽셀+depth를 3D ray로
푸는 unprojection입니다.

OAK-D-LR은 triple AR0234 2.3 MP global-shutter color sensor와 5/10/15 cm
baseline을 가진 RVC2 장치입니다. 코드가 연결된 camera feature와 EEPROM stereo
pair를 읽어 RGB/Depth socket과 입력 크기를 자동 선택하므로 별도 카메라명
하드코딩 없이 사용할 수 있습니다. 선택된 socket, sensor, stereo 입력 크기는
`metadata.json`의 `camera_sockets`, `rgb_sensor`, `stereo_sensors`,
`stereo_depth_model`에 기록됩니다.

OAK-D-LR에서 저장되는 RGB와 depth 출력은 `1920x1200` RGB geometry에 맞춰집니다.
다만 RVC2의 StereoDepth matching 입력은 1920 폭을 직접 받을 수 없어서 AR0234
1920x1200 stereo frame을 full-FOV `1280x800`으로 내려 계산한 뒤, RGB 출력에
align된 depth를 `1920x1200`으로 저장합니다. 따라서 LR 데이터셋 metadata에는
`depth_alignment.depth_output_size=1920x1200`과
`stereo_sensors.stereo_matching_input_size=1280x800`이 함께 기록됩니다.

이전 코드로 찍은 데이터셋이 `metadata.json`에 frame transformation intrinsics를
갖고 있지 않다면, 같은 카메라를 연결한 상태에서 다음처럼 갱신할 수 있습니다.

```bash
python refresh_camera_metadata.py --dataset ../data/2026-06-25_14-30-59_raw
```

Depth fps를 RGB보다 높이면 후처리 동기화가 RGB timestamp에 가장 가까운 depth
이벤트를 고르므로 시간 오차를 줄일 수 있지만, USB 대역폭과 저장량은 늘어납니다.
현재 저장 depth는 RGB 좌표계에 정렬된 출력입니다. RVC2의
`StereoDepth.inputAlignTo` 경로에서는 `depth_fps`를 더 높게 요청해도 실제
aligned depth 저장 cadence가 RGB fps에 가까워질 수 있으므로, 저장 프레임률까지
올려야 할 때는 `fps`도 함께 올립니다.

DepthAI raw depth는 stereo pair의 calibration으로 계산된 millimeter 값입니다.
OAK-D W wide FOV 800P/75 mm baseline 기준으로 4 m 부근에서 4 m가 7 m로 보이는
수준은 정상 정확도 범위를 벗어납니다. 이 경우 임의 scale 보정보다는
CAM_B/C 렌즈가 EEPROM calibration과 일치하는지, 좌우 카메라가 바뀌지 않았는지,
board config/HFOV/baseline이 맞는지 확인하고 Luxonis manual calibration 절차로
stereo intrinsics/extrinsics를 다시 플래시해야 합니다.
정확도 확인용 RVC2 수집 설정은 `depth_preset: FAST_ACCURACY`, `lr_check: true`,
`subpixel: true`, `subpixel_fractional_bits: 5`, `stereo_median_filter: off`로
두어 post-processing 영향을 줄이고 stereo calibration 상태를 먼저 봅니다.

수집 폴더에는 스트림별 이벤트 CSV와 `rgb/`, `depth_mm/` 이미지가 생성됩니다.
GPS/NTRIP/EBIMU 장치 및 보정 서버 설정은 다음으로 확인합니다.

```bash
python synced_image_recorder.py --help
```

NTRIP 값은 CLI 또는 `NTRIP_HOST`, `NTRIP_PORT`, `NTRIP_MOUNTPOINT`,
`NTRIP_USERNAME`, `NTRIP_PASSWORD` 환경변수로 지정할 수 있습니다.
기본값은 `www.gnssdata.or.kr:2101` source table에서 `RTCM31` mountpoint를
읽어 GPS 초기 위치와 가장 가까운 기준국부터 연결합니다. source table을 받지
못하거나 GPS 위치가 아직 없으면 서울/경기권 fallback 후보
`GANS-RTCM31,GUMC-RTCM31,DBON-RTCM31,PAJU-RTCM31,...` 순서로 시도하며,
연결 또는 RTCM 데이터 수신 timeout이 나면 다음 기준국으로 넘어갑니다.

## 3. 데이터 동기화

원본 폴더 안에 `timestamps.csv`, `imu.csv`와 품질 보고서를 생성합니다.

```bash
python build_synced_dataset.py --config configs/sync.yaml
```

RGB와 aligned depth는 `rgb_depth_threshold_ms`로 따로 제한합니다. 기본값 `10ms`는
30 FPS에서 이전/다음 depth 프레임이 잘못 붙는 `±33ms` 매칭을 버리기 위한 값입니다.
IMU 매칭은 기존 `sync_threshold_ms`를 사용합니다.

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
3. Depth 0과 confidence 기준 초과 픽셀을 제거한 분포의 중앙값을 구합니다.
4. 이 중앙값이 `max_depth_mm`를 넘으면 representative와 feature를 모두
   `beyond_max_depth`로 제외합니다.
5. 거리 이내 검출에서만 최대거리 초과 픽셀과 median/MAD 이상치를 제거합니다.
6. 중심 위치·Depth 잔차가 안정적인 픽셀을 `representative`로 선택합니다.
7. 마스크 PCA의 `endpoint_a/midpoint/endpoint_b`는 `feature`로만 저장합니다.
8. 최종 지도 선형에는 `point_usage=mapping`인 representative만 사용합니다.

`max_depth_mm`는 초과 깊이를 상한값으로 잘라 쓰는 설정이 아니라 검출 제외
임계값입니다. 최대거리 필터를 적용하기 전의 confidence-qualified 중앙값으로 먼저
거리 이내인지 판정하므로, 원거리 마스크에 섞인 소수의 8 m 직하 픽셀이 지도점으로
채택되지 않습니다. 정확히 임계값인 Depth는 포함하고 이를 초과한 검출은 좌표를
만들지 않습니다.

`points.csv`의 mapping 행에는 필터 전 거리 판정을 감사할 수 있도록
`depth_measured_count`와 `depth_measured_median_mm`도 기록됩니다. 거절 행에도
임계값 이내 후보 수가 `depth_sample_count`로 남을 수 있으므로, 유효 지도점을
고를 때는 이 개수만 보지 말고 `depth_fragment_status=ok`, `depth_mm>0`,
`world_status=ok`를 함께 확인해야 합니다.

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
geonova_depthai/controller_bridge.py 앱용 센서 상태와 카메라 프리뷰 게시
geonova_depthai/postprocess/         이벤트 동기화
geonova_depthai/runtime.py           DepthAI·GPS·NTRIP·EBIMU runtime
geonova_depthai/debug_ui.py          데이터셋 확인과 좌표 변환
geonova_depthai/yolo_seg_shp.py      YOLO·Depth fragment·WGS84 관측점
geonova_depthai/fence_linearization.py EPSG:5179 보정·spline·SHP·QA
../model/n_model/best.pt            기본 guardrail segmentation 모델
tools/configure_ebimu.py            EBIMU 출력 설정·보정 도구
tests/                               센서/캘리브레이션/회귀 검증
```
