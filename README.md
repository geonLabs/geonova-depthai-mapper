# DepthAI Camera Recorder

Luxonis DepthAI v3 기반으로 RGB 이미지, depth(mm) 이미지, IMU 데이터를 동기화해서 저장하고, 저장된 데이터셋을 브라우저 UI로 디버깅하는 도구입니다.

## 구성

- `synced_image_recorder.py`: RGB 사진, 16-bit depth PNG, IMU CSV를 timestamp 기준으로 프레임 단위 저장
- `configure_ebimu.py`: EBIMU-9DOFV5 외부 IMU를 921600bps, quaternion, gyro/accel/mag/timestamp 출력으로 설정
- `dataset_debug_ui.py`: 저장된 데이터셋을 열어 RGB/depth를 같이 보고, hover/click 위치의 거리와 IMU 값을 확인하는 로컬 UI
- `diagnose_dataset_geometry.py`: 저장된 데이터셋의 depth 품질, sync, orientation 축, 랜덤 좌표 투영을 진단
- `synced_depthai_recorder.py`: RGB 비디오, depth raw, IMU를 세그먼트 단위로 저장하는 실험용/이전 버전
- `depthai_recorder.py`: 기존 H.265 세그먼트 녹화 코드
- `requirements.txt`: Python 패키지 고정 버전

## 설치

```bash
cd /home/jm/26_camera_record
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

DepthAI v3 API를 사용합니다. `requirements.txt` 기준 버전은 `depthai==3.1.0`입니다.

## 이미지 데이터셋 저장

무손실 RGB PNG와 16-bit depth PNG로 저장합니다.

```bash
python synced_image_recorder.py --output-dir image_records
```

저장 속도를 높이고 싶으면 RGB를 JPG로 저장합니다. Depth는 계속 16-bit PNG라 mm 거리값이 유지됩니다.

```bash
python synced_image_recorder.py --output-dir image_records --rgb-format jpg
```

짧게 테스트하려면:

```bash
python synced_image_recorder.py --output-dir image_records --duration 10 --rgb-format jpg
```

JPG 저장 품질과 장치 MJPEG 전송 품질의 기본값은 모두 `100`입니다. JPEG 규격 특성상 품질 100도 완전한 무손실은 아니며, 픽셀 단위 무손실 RGB가 필요하면 `--rgb-format png`를 사용하세요.

JPG 압축 강도는 `--jpg-quality` 또는 `--rgb-jpeg-quality`로 조절합니다. 범위는 `1~100`이며 값이 낮을수록 압축이 강하고 파일이 작아집니다.

```bash
python synced_image_recorder.py --output-dir image_records --rgb-format jpg --jpg-quality 90
```

녹화 시작 직후 자동 노출/화이트밸런스가 안정되기 전의 흰 프레임은 기본 3초 동안 버립니다. 필요하면 `--camera-warmup-seconds`로 조정할 수 있습니다. 컬러 카메라 출력의 세 채널이 완전히 동일한 비정상 흑백 프레임이면 녹화를 중단해 잘못된 데이터셋이 계속 쌓이지 않게 합니다.

카메라가 180도 거꾸로 장착되어 있다면 RGB와 depth를 저장 전에 같이 회전시킵니다.

```bash
python synced_image_recorder.py \
  --output-dir image_records \
  --rgb-format jpg \
  --rotate-180 \
  --save-confidence-map
```

저장 구조:

```text
image_records/YYYY-MM-DD_HH-MM-SS/
  rgb/YYYY-MM-DD-HH-MM-SS-frame0000000_rgb.png
  depth_mm/YYYY-MM-DD-HH-MM-SS-frame0000000_depth_mm.png
  confidence/YYYY-MM-DD-HH-MM-SS-frame0000000_confidence.png  # --save-confidence-map 사용 시
  timestamps.csv
  imu.csv
  gps.csv
  external_imu.csv
  metadata.json
```

`depth_mm/*.png`는 `uint16` PNG입니다. 픽셀값이 mm 단위 거리입니다.

```python
import cv2

depth = cv2.imread("depth_mm/....png", cv2.IMREAD_UNCHANGED)
distance_mm = int(depth[y, x])
```

`--save-confidence-map`을 켜면 `confidence/*.png`도 저장됩니다. confidence는 RGB/depth/IMU Sync 그룹에 넣지 않고 별도 큐에서 depth 타임스탬프와 가장 가까운 프레임을 best-effort로 붙입니다. 그래서 confidence가 늦거나 누락되어도 RGB/depth/IMU 프레임 저장은 멈추지 않습니다. depth가 42m 근처로 튀거나 물체 경계에서 흔들릴 때 confidence와 주변 depth 분산을 같이 보고 해당 픽셀을 좌표 계산에서 제외하는 용도로 사용합니다.

카메라가 USB 2.0(`UsbSpeed.HIGH`)으로 연결되면 1280x720 RGB NV12 + depth RAW16 + confidence RAW8의 비압축 전송량이 링크 용량을 넘습니다. 녹화기는 USB 3.x 재연결을 기본 2회 추가 시도하고, 계속 USB 2.0이면 불완전한 데이터셋 생성을 막기 위해 녹화를 시작하지 않습니다. OAK 카메라를 USB 3.x 포트에 직접 연결하고 USB 3.x 케이블을 사용하세요. 실행 로그의 `DepthAI USB speed`와 `transports` 항목에서 실제 연결을 확인할 수 있습니다.

USB 2.0에서 저속 시험을 꼭 해야 한다면 `--allow-usb2`를 추가할 수 있습니다. 이 경우 RGB와 confidence를 장치 MJPEG로 압축 전송하지만 1280x720 15 FPS는 보장되지 않습니다.

confidence의 MJPEG 전송은 품질 100이어도 완전한 무손실은 아닙니다. confidence 원본 픽셀값이 반드시 필요하면 USB 3.x 포트/케이블로 연결하고 `--confidence-transport raw`를 사용하세요. USB 2.0에서 raw를 강제하면 다시 프레임이 심하게 누락될 수 있습니다.

## 동기화와 정렬

녹화 파이프라인은 기본으로 RGB, depth, IMU를 각각 큐로 받은 뒤 host에서 장치 타임스탬프 기준 nearest matching을 합니다. DepthAI `Sync` 노드가 일부 환경에서 그룹 출력을 드물게 만드는 경우가 있어, 기본값은 `--sync-mode host`입니다. 기존 장치 Sync 노드 방식을 다시 테스트하려면 `--sync-mode device`를 사용할 수 있습니다.

기본 sync 허용 범위는 `50ms`입니다. RGB/depth/IMU를 더 엄격히 묶고 싶으면 `--sync-threshold-ms`를 낮출 수 있지만, 너무 낮으면 프레임이 많이 버려질 수 있습니다.

프레임의 host 시각은 큐에서 꺼낸 순간을 그대로 쓰지 않습니다. RGB device timestamp와 관측된 최소 transport latency로 실제 촬영 host monotonic/wall 시각을 복원하고, dequeue 시각과 `frame_queue_lag_ms`를 별도로 기록합니다. 기본 host queue도 4로 제한해 저장 병목이 생길 때 오래된 프레임 대신 최신 프레임을 유지합니다.

StereoDepth의 subpixel disparity는 기본으로 켜져 있고 3 fractional bits를 사용합니다. 장거리 depth 계단 현상을 줄이면서 median filter를 유지하기 위한 설정입니다. disparity/depth median은 기본 `7x7`이며 `--stereo-median-filter off|3x3|5x5|7x7`로 바꿀 수 있습니다. `--subpixel-fractional-bits 4|5`에서는 DepthAI 제한 때문에 median이 자동으로 꺼집니다.

Depth는 기본으로 RGB 카메라 기준 `StereoDepth.setDepthAlign(CAM_A)`로 align합니다. 이 모드는 RGB 저장 스트림과 depth 정렬을 동시에 안정적으로 유지합니다.

```python
StereoDepth.setDepthAlign(CAM_A)
```

새로 저장되는 데이터셋의 `metadata.json`에는 alignment 정보가 포함됩니다.

RGB 저장 영상 자체는 자동 undistort하지 않아 원래 화각을 유지합니다. 대신 장치 calibration에서 RGB intrinsics와 OpenCV distortion coefficients를 저장하고, Debug UI의 픽셀→3D ray 계산에서 `cv2.undistortPoints`로 왜곡을 제거합니다. 180도 회전/flip이 있으면 원본 픽셀로 역변환한 뒤 왜곡을 제거하고 다시 저장 좌표계로 변환합니다.

```json
{
  "depth_alignment": {
    "enabled": true,
    "mode": "stereo",
    "aligned_to": "rgb",
    "aligned_to_socket": "CAM_A",
    "method": "StereoDepth.setDepthAlign(CAM_A)",
    "depth_pixel_coordinates_match_rgb": true
  }
}
```

DepthAI `ImageAlign` 노드를 별도로 테스트해야 할 때는 다음 옵션을 사용할 수 있습니다. 단, 현재 장비/DepthAI v3 조합에서는 RGB 저장 스트림과 함께 쓸 때 출력 타입 협상이 불안정할 수 있어 기본 수집에는 `stereo` 모드를 권장합니다.

```bash
python synced_image_recorder.py --output-dir image_records --depth-alignment-mode image-align
```

## GPS와 외부 IMU serial 저장

`synced_image_recorder.py`는 기본으로 다음 serial 장치를 같이 읽습니다.

- GPS: `/dev/ttyACM0`, `921600 baud`
- 외부 IMU: `/dev/ttyUSB0`, `921600 baud`

각 serial 장치는 저장률을 기본 최대 `120Hz`로 제한합니다. 장치에서 더 높은 빈도로 데이터가 들어와도 CSV에는 초당 최대 120개 샘플만 저장됩니다. GPS/외부 IMU를 따로 제한하려면 `--gps-max-hz`, `--external-imu-max-hz`를 사용합니다.

```bash
python synced_image_recorder.py --output-dir image_records --rgb-format jpg
```

장치나 속도를 바꾸려면:

```bash
python synced_image_recorder.py \
  --output-dir image_records \
  --rgb-format jpg \
  --gps-device /dev/ttyACM0 \
  --gps-baudrate 921600 \
  --external-imu-device /dev/ttyUSB0 \
  --external-imu-baudrate 921600 \
  --gps-max-hz 120 \
  --external-imu-max-hz 120 \
  --sync-threshold-ms 50 \
  --save-confidence-map
```

GPS 또는 외부 IMU를 잠시 끌 수도 있습니다.

```bash
python synced_image_recorder.py --output-dir image_records --no-gps
python synced_image_recorder.py --output-dir image_records --no-external-imu
```

출력 파일:

- `gps.csv`: GPS serial 원문과 NMEA `GGA/RMC` 파싱 결과
- `external_imu.csv`: EBIMU 원문과 quaternion/gyro/accel/magnetometer/timestamp 파싱 결과
- `timestamps.csv`: 각 카메라 프레임에 가장 가까운 GPS/외부 IMU 샘플 인덱스와 시간차

GPS는 serial 도착 시각이 아니라 GGA의 `date_utc + gps_time_utc` 측정 epoch를 host monotonic clock으로 환산해 RGB 촬영시각과 매칭합니다. `gps.csv`에는 측정시각, 환산 monotonic 시각, serial 수신 지연이 함께 저장됩니다. 외부 IMU는 현재 serial 수신 `host_monotonic_ns()` 기준 nearest sample을 사용합니다.

### RTK 좌표 확인

기본 실행은 NTRIP caster에서 받은 RTCM 보정 데이터를 GPS 수신기로 다시 전송합니다. CSV에 저장되는 위도/경도는 GPS 수신기가 RTCM을 적용해 출력한 해이며, RTK 상태는 GGA의 `fix_quality`로 판별합니다.

- `4`, `RTK fixed`: 정수 모호성이 해결된 RTK 고정해. 정밀 좌표에는 이 샘플을 사용합니다.
- `5`, `RTK float`: RTK 보정 중이지만 아직 고정되지 않은 해입니다.
- `2`, `DGPS`: 차분 보정해이지만 RTK fixed는 아닙니다.
- `1`, `standalone`: 일반 단독 측위입니다.

`gps.csv`에는 `fix_quality_name`, `rtk_status`, `rtk_fixed`, `rtk_corrected`, `differential_age_s`, `reference_station_id`가 저장됩니다. FIX 플래그만으로는 오래된 보정해를 거르지 못하므로 정밀 처리는 `gps_fix_quality == 4`, `gps_differential_age_s <= 2.0`, `gps_hdop <= 2.0`을 모두 만족하는 행을 사용하세요. Debug UI도 이 조건을 벗어난 FIX를 `RTK FIXED / STALE`로 표시합니다. 한계값은 녹화 시 `--rtk-max-correction-age-s`, `--rtk-max-hdop`으로 바꿀 수 있습니다.

GPS만 먼저 점검하려면 다음 명령을 사용합니다.

```bash
python test_gps_rtk.py --duration 20
```

절대좌표 디버깅용 보정값을 metadata에 같이 남길 수 있습니다. 기본 장착 높이는 GPS 1.50m, 카메라 1.30m, 외부 IMU 1.15m로 설정되어 있습니다. GPS 기준 카메라 ENU 높이 오프셋은 `-0.20m`, 카메라 기준 외부 IMU 아래 방향 오프셋은 `+0.15m`입니다.

```bash
python synced_image_recorder.py \
  --output-dir image_records \
  --rgb-format jpg \
  --rotate-180 \
  --imu-from-camera-roll-deg 0 \
  --imu-from-camera-pitch-deg 0 \
  --imu-from-camera-yaw-deg 0 \
  --camera-mount-roll-deg 0 \
  --camera-mount-pitch-deg 0 \
  --camera-mount-yaw-deg 0 \
  --gps-to-camera-east-m 0 \
  --gps-to-camera-north-m 0 \
  --gps-to-camera-up-m -0.20 \
  --gps-from-camera-right-m 0.30 \
  --gps-from-camera-down-m 0 \
  --gps-from-camera-forward-m -0.20 \
  --external-imu-from-camera-right-m -0.30 \
  --external-imu-from-camera-down-m 0.15 \
  --external-imu-from-camera-forward-m 0.20 \
  --magnetic-declination-deg 0
```

`gps-from-camera-*`와 `external-imu-from-camera-*`는 저장된 카메라 이미지 좌표계 기준입니다. `+right`는 이미지 오른쪽, `+down`은 이미지 아래, `+forward`는 카메라가 보는 앞쪽입니다. 예를 들어 GPS 안테나가 카메라보다 오른쪽 30cm, 뒤쪽 20cm이면 `--gps-from-camera-right-m 0.30 --gps-from-camera-forward-m -0.20`입니다.

현재 높이 관계는 `GPS → 카메라: 아래 0.20m`, `카메라 → IMU: 아래 0.15m`로 해석합니다. 따라서 IMU 높이는 지면에서 1.15m입니다.

`camera-mount-*`는 차량 진행 방향 기준 카메라 장착 각도입니다. `yaw +`는 카메라가 차량 정면보다 오른쪽을 보는 경우, `pitch +`는 아래를 보는 경우, `roll +`는 이미지가 시계방향으로 도는 경우입니다.

Debug UI의 절대좌표 계산은 `GPS 위치 + EBIMU 자세 + depth 픽셀 + RGB intrinsics + GPS antenna/camera lever arm`을 사용합니다. 정확도를 높이려면 EBIMU-to-camera 축 정렬값, GPS 안테나-to-camera offset, 지자기 보정과 지역 magnetic declination을 맞춰야 합니다.

## EBIMU-9DOFV5 설정

EBIMU를 카메라 자세 추정용 외부 IMU로 쓸 때는 먼저 출력 포맷을 고정해 둡니다. 기본 추천값은 다음과 같습니다.

- baudrate: `921600`
- 출력 주기: `10ms`, 약 `100Hz`
- 출력 모드: ASCII
- 자세 출력: quaternion
- 출력 항목: quaternion, gyro, accel, magnetometer, timestamp
- magnetometer: active mode `sem2`

이미 EBIMU가 `921600`으로 설정되어 있다면:

```bash
python configure_ebimu.py --port /dev/ttyUSB0
```

처음 연결해서 기본 baudrate `115200`에서 `921600`으로 바꾸는 경우:

```bash
python configure_ebimu.py \
  --port /dev/ttyUSB0 \
  --change-baud-from 115200 \
  --baudrate 921600
```

실제로 전송되는 주요 명령은 다음과 같습니다.

```text
<stop>
<soc1>
<sor10>
<sof2>
<sog1>
<soa1>
<som1>
<sots1>
<sem2>
<start>
```

실행 후 출력 한 줄은 대략 다음 순서로 들어옵니다.

```text
*qz,qy,qx,qw,gx,gy,gz,ax,ay,az,mx,my,mz,timestamp_ms
```

사람이 바로 보기 쉬운 roll/pitch/yaw 출력으로 확인하고 싶으면:

```bash
python configure_ebimu.py --port /dev/ttyUSB0 --orientation-format euler
```

카메라에 장착한 뒤 보정까지 같이 수행하려면:

```bash
python configure_ebimu.py --port /dev/ttyUSB0 --guided-calibration
```

`--guided-calibration`은 다음 순서로 안내 문구를 출력합니다.

```text
1. EBIMU 스트리밍 정지
2. 출력 설정 초기화
3. Gyro 보정
4. Accelerometer 보정
5. Magnetometer 보정
6. <stop> 전송 후 종료
```

gyro 보정은 센서를 완전히 정지시킨 상태에서 진행합니다. accel 보정은 수평 정지 상태에서 진행하고, magnetometer 보정은 실제 장착 상태에서 센서를 여러 방향으로 충분히 회전시킨 뒤 Enter를 누릅니다.

처음 연결해서 기본 baudrate `115200`에서 `921600`으로 바꾸면서 보정까지 하는 경우:

```bash
python configure_ebimu.py \
  --port /dev/ttyUSB0 \
  --change-baud-from 115200 \
  --baudrate 921600 \
  --guided-calibration
```

보정을 끝낸 뒤 데이터 출력이 정상인지 확인하려면:

```bash
python configure_ebimu.py --port /dev/ttyUSB0
```

개별 보정만 실행해야 할 때는 아래 옵션을 조합해서 사용할 수 있습니다.

```bash
python configure_ebimu.py --port /dev/ttyUSB0 --calibrate-gyro
python configure_ebimu.py --port /dev/ttyUSB0 --calibrate-accel
python configure_ebimu.py --port /dev/ttyUSB0 --calibrate-mag
```

## 디버그 UI

```bash
python dataset_debug_ui.py --port 8088
```

기본으로 `0.0.0.0`에 바인딩되어 같은 네트워크나 Tailscale 등에서 외부 접속할 수 있습니다. 실행하면 접속 가능한 URL 후보가 출력됩니다.

```text
Local URL: http://127.0.0.1:8088
External/LAN URL candidates:
  http://10.144.45.251:8088
  http://100.76.81.78:8088
```

같은 PC 브라우저에서는:

```text
http://127.0.0.1:8088
```

외부 접속이 안 되면 방화벽에서 포트 `8088`을 허용해야 할 수 있습니다. 로컬에서만 열고 싶을 때는:

```bash
python dataset_debug_ui.py --host 127.0.0.1 --port 8088
```

기능:

- RGB와 depth preview를 반반으로 표시
- 폴더 단위 데이터셋 열기
- `Latest` 버튼으로 루트 폴더 아래 최신 데이터셋 열기
- RGB/depth 화면 hover 위치의 거리 실시간 표시
- 클릭 위치의 거리 고정 표시
- RGB/depth 화면 hover/click 위치의 ENU offset, 위도, 경도, 고도 표시
- 절대좌표 orientation source를 `Compare`, `EBIMU`, `GPS Course + Tilt`, `GPS Course Level`로 전환해서 같은 픽셀의 계산 결과 비교
- invalid depth는 `invalid depth`로 표시
- exact depth가 0이면 주변 9x9 유효 depth median으로 보완
- IMU accel/gyro, RGB-depth-IMU timestamp delta 표시
- 현재 프레임의 유효 depth 픽셀 수 표시
- `First Valid` 버튼으로 depth가 있는 첫 프레임 이동

단축키:

- `D` 또는 `←`: 이전 프레임
- `F` 또는 `→`: 다음 프레임
- `Home`: 첫 프레임
- `End`: 마지막 프레임

## 데이터셋 좌표 진단

좌표가 앞뒤/좌우/고도 방향으로 맞지 않을 때는 먼저 랜덤 샘플 진단을 실행합니다.

```bash
python diagnose_dataset_geometry.py \
  image_records/2026-06-18_13-28-25 \
  --output-csv image_records/2026-06-18_13-28-25/geometry_samples.csv
```

진단 결과는 다음을 보여줍니다.

- depth 유효 픽셀 비율과 42m 근처 max-like depth 비율
- RGB-depth, GPS-frame, external-IMU-frame 시간차
- `EBIMU`, `GPS Course + Tilt`, `GPS Course Level`의 전방축 heading/elevation 통계
- 랜덤 픽셀 좌표 투영 CSV

`GPS Course + Tilt`는 yaw를 GPS course로 맞추되 pitch/roll은 EBIMU 기준을 사용합니다. EBIMU-camera extrinsic이 틀어져 있으면 전방축이 위/아래로 크게 틀어질 수 있습니다. `GPS Course Level`은 GPS course를 전방 yaw로 쓰고 카메라를 수평으로 가정하므로, 주행 중 전방 객체의 좌우/앞뒤 방향을 빠르게 진단할 때 유용합니다.

GPS 속도가 `2m/s`보다 낮거나 course 샘플 품질이 약하면 Debug UI는 가까운 주행 중 GPS course 샘플을 최대 5초 범위에서 찾아 사용합니다. 화면의 좌표 상세에는 `course_source`와 `course sample delta`가 표시됩니다.

### 카메라 장착각 보정

Debug UI의 `Mount Calibration` 섹션에서 `camera_mount_rpy_deg`를 조절할 수 있습니다. 수동으로 roll/pitch/yaw를 바꾼 뒤 `Apply`를 누르면 현재 UI 세션에 반영되고, `Save`를 누르면 데이터셋의 `metadata.json`에 저장됩니다.

실제 좌표를 아는 물체가 있으면 역보정도 가능합니다.

1. RGB 또는 depth 화면에서 기준 물체를 클릭합니다.
2. 지도나 기준 측량값에서 그 물체의 실제 latitude/longitude/altitude를 입력합니다.
3. `Calibrate & Save`를 누릅니다.

이 기능은 클릭한 한 점을 기준으로 yaw/pitch를 역산하고 roll은 기존 값을 유지합니다. 한 점만으로 roll까지 안정적으로 풀 수는 없으므로, roll은 수평선이나 여러 기준점으로 별도 조정하는 편이 좋습니다.

## IMU 자세와 절대좌표

이미지 저장 스크립트는 DepthAI IMU accel/gyro와 외부 EBIMU quaternion/gyro/accel/magnetometer를 함께 저장합니다. 절대 좌표 계산에서 카메라가 보는 절대 방향까지 쓰려면 raw accel/gyro보다 EBIMU quaternion처럼 보정된 자세값을 쓰는 편이 좋습니다.

추가로 저장해야 할 값:

- `ROTATION_VECTOR` quaternion
- IMU-to-RGB extrinsics
- RGB camera intrinsics
- GPS timestamp, latitude, longitude, altitude
- GPS antenna-to-camera offset 또는 camera-frame lever arm

좌표 변환 흐름:

```text
depth pixel + RGB intrinsics -> camera frame 3D point
IMU rotation vector + IMU-to-RGB extrinsics -> camera orientation
GPS position + antenna/camera offset -> camera world position
world_point = camera_position + R_world_camera @ point_camera
```

## 참고

- DepthAI Sync 노드: https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/sync
- DepthAI StereoDepth 노드: https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/stereo_depth
- DepthAI IMU 노드: https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/imu
