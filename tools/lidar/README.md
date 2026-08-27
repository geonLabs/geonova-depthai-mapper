# LiDAR / ROS 오프라인 도구

이 디렉터리는 기존 `dbparkJ/geo_multifusion_sensors` 저장소의 고유 기능을
기준 저장소로 통합한 위치입니다. Jetson에서 자동 실행되는 DepthAI 수집 런타임과는
의존성을 분리했으며, ROS bag 변환과 LiDAR–카메라 검증 작업에만 사용합니다.

## 포함 도구

| 파일 | 용도 | 실행 환경 |
|---|---|---|
| `convert_lidar_bag_to_pcd.py` | ROS1 bag의 PointCloud2와 가장 가까운 카메라 프레임을 PCD/이미지로 추출 | 일반 Python 3.11, ROS 설치 불필요 |
| `project_lidar_overlay.py` | 캘리브레이션 JSON으로 LiDAR 포인트를 카메라 영상에 투영 | ROS1 Noetic Python 환경 필요 |
| `config.yaml` | bag 추출 기본 설정 예시 | 현장값은 `config.local.yaml`로 복사 |
| `requirements.txt` | bag 추출기 전용 고정 의존성 | DepthAI `.venv`와 분리 권장 |

## 1. ROS bag → PCD/이미지 변환

DepthAI와 PyTorch 환경을 오염시키지 않도록 전용 가상환경을 만듭니다.

```bash
cd tools/lidar
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp config.yaml config.local.yaml
```

`config.local.yaml`에서 bag 경로, 토픽, 출력 경로를 바꾼 뒤 실행합니다.

```bash
python convert_lidar_bag_to_pcd.py config.local.yaml
```

주요 설정은 다음과 같습니다.

- `bags.paths`: 단일 `.bag`, 디렉터리 또는 목록
- `bags.recursive`: 하위 폴더까지 bag 검색
- `topics.lidar`, `topics.camera`: PointCloud2와 Image 토픽
- `extract.timestamp_source`: `bag` 또는 message `header`
- `extract.match_tolerance_ms`: 최근접 프레임 허용 시간차
- `extract.save_all`: 전체 매칭 프레임 저장 여부
- `pcd.fields`: `null`이면 PointCloud2의 모든 필드 보존
- `output.group_by_bag`: bag별 출력 디렉터리 분리

기본 출력은 다음과 같습니다.

```text
extracted/
  <bag-name>/
    pcd/
    images/
```

## 2. LiDAR–카메라 투영 확인

이 도구는 `rosbag`, `sensor_msgs`, `cv_bridge`를 사용하므로 ROS1 Noetic 환경에서
실행해야 합니다.

```bash
source /opt/ros/noetic/setup.bash
python project_lidar_overlay.py \
  --bag /data/sample.bag \
  --calib /data/calib.json \
  --output overlay-output/sample.jpg \
  --image-topic /roof_clpe_ros/roof_cam_1/image_raw \
  --points-topic /lidar0/velodyne_points
```

캘리브레이션 JSON은 다음 항목을 사용합니다.

```text
results.T_lidar_camera
results.init_T_lidar_camera
results.init_T_lidar_camera_auto
camera.intrinsics = [fx, fy, cx, cy]
camera.distortion_coeffs
```

변환은 LiDAR→카메라 초기값을 읽고 카메라 좌표계 변환으로 역변환한 뒤 OpenCV로
투영합니다. 영상 앞쪽에 있고 이미지 경계 안에 들어오는 포인트만 표시합니다.

## 테스트

하드웨어와 bag 파일 없이 순수 변환 보조 함수와 PCD 기록 형식을 확인합니다.

```bash
python -m pip install pytest
python -m pytest -q tests
```

루트 GitHub Actions도 이 테스트와 전체 Python 구문 검사를 수행합니다.

## 이전 저장소 경로 대응

```text
dbparkJ/geo_multifusion_sensors/test/convert_lidar_bag_to_pcd.py
  → tools/lidar/convert_lidar_bag_to_pcd.py

dbparkJ/geo_multifusion_sensors/test/config.yaml
  → tools/lidar/config.yaml

dbparkJ/geo_multifusion_sensors/test/requirements.txt
  → tools/lidar/requirements.txt

dbparkJ/geo_multifusion_sensors/scripts/project_lidar_overlay.py
  → tools/lidar/project_lidar_overlay.py
```

`safe_gard_test/code`는 복사하지 않았습니다. 동일 기능의 최신 원본은 저장소 루트의
`code/`이며, 앞으로 DepthAI 수집·동기화·RTK·매핑 변경은 그 위치에서만 진행합니다.
