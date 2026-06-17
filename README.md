# DepthAI Camera Recorder

Luxonis DepthAI v3 기반으로 RGB 이미지, depth(mm) 이미지, IMU 데이터를 동기화해서 저장하고, 저장된 데이터셋을 브라우저 UI로 디버깅하는 도구입니다.

## 구성

- `synced_image_recorder.py`: RGB 사진, 16-bit depth PNG, IMU CSV를 `dai.node.Sync` 기준으로 프레임 단위 저장
- `dataset_debug_ui.py`: 저장된 데이터셋을 열어 RGB/depth를 같이 보고, hover/click 위치의 거리와 IMU 값을 확인하는 로컬 UI
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

저장 구조:

```text
image_records/YYYY-MM-DD_HH-MM-SS/
  rgb/YYYY-MM-DD-HH-MM-SS-frame0000000_rgb.png
  depth_mm/YYYY-MM-DD-HH-MM-SS-frame0000000_depth_mm.png
  timestamps.csv
  imu.csv
  metadata.json
```

`depth_mm/*.png`는 `uint16` PNG입니다. 픽셀값이 mm 단위 거리입니다.

```python
import cv2

depth = cv2.imread("depth_mm/....png", cv2.IMREAD_UNCHANGED)
distance_mm = int(depth[y, x])
```

## 동기화와 정렬

녹화 파이프라인은 DepthAI의 `Sync` 노드로 RGB, depth, IMU 메시지를 장치 타임스탬프 기준으로 묶습니다.

Depth는 RGB 카메라 기준으로 align합니다.

```python
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setOutputSize(1280, 720)
```

새로 저장되는 데이터셋의 `metadata.json`에는 alignment 정보가 포함됩니다.

```json
{
  "depth_alignment": {
    "enabled": true,
    "aligned_to": "rgb",
    "aligned_to_socket": "CAM_A",
    "method": "StereoDepth.setDepthAlign(CAM_A)",
    "depth_pixel_coordinates_match_rgb": true
  }
}
```

## 디버그 UI

```bash
python dataset_debug_ui.py --port 8088
```

브라우저에서 엽니다.

```text
http://127.0.0.1:8088
```

기능:

- RGB와 depth preview를 반반으로 표시
- 폴더 단위 데이터셋 열기
- `Latest` 버튼으로 루트 폴더 아래 최신 데이터셋 열기
- RGB/depth 화면 hover 위치의 거리 실시간 표시
- 클릭 위치의 거리 고정 표시
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

## IMU 자세와 절대좌표

현재 이미지 저장 스크립트는 accel/gyro 원시값을 저장합니다. 절대 좌표 계산에서 카메라가 보는 절대 방향까지 쓰려면 raw accel/gyro만으로는 부족합니다.

추가로 저장해야 할 값:

- `ROTATION_VECTOR` quaternion
- IMU-to-RGB extrinsics
- RGB camera intrinsics
- GPS timestamp, latitude, longitude, altitude
- GPS antenna-to-camera offset

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
