# DepthAI RGB-D 센서 수집/검증 코드

이 디렉터리는 OAK-D-PRO-W의 RGB·Depth·내장 IMU와 외부 GPS/NTRIP·EBIMU를
수집하고, 각 센서를 독립적으로 검증하는 코드만 남긴 버전입니다.

## 기존 코드와 달라진 점

| 항목 | 기존 `synced_image_recorder.py` | 리팩터 |
|---|---|---|
| 저장 | 수집 루프에서 동기 프레임을 즉시 저장 | 스트림별 이벤트를 먼저 저장하고 후처리에서 동기화 |
| 동기화 | RGB 기준 host nearest matching | 동일한 device timestamp 기반 후처리, 재실행 가능 |
| RGB-D 정렬 | `StereoDepth.setDepthAlign(CAM_A)` | 플랫폼 자동 분기 |
| RVC2/RVC3 | target 출력 geometry가 명확하지 않음 | 실제 RGB 출력을 `StereoDepth.inputAlignTo`에 연결 |
| RVC4 | 별도 보장 없음 | `ImageAlign(depth, RGB)` 사용 |
| RGB 왜곡 | 왜곡된 RGB 저장 | `requestOutput(..., enableUndistortion=True)` 강제 |
| confidence | 기본 저장 | 센서 진단 시에만 선택 저장 |
| NTRIP | 기존 코드에 기본값 존재 | 동일 기본값을 복원하고 환경변수/CLI override 지원 |

DepthAI v3 공식 예제와 동일하게 RVC2/RVC3는 `StereoDepth.inputAlignTo`,
RVC4는 `ImageAlign`을 사용합니다.

- [Depth Align 공식 예제](https://docs.luxonis.com/software-v3/depthai/examples/image_align/depth_align)
- [ImageAlign API](https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/image_align/)
- [StereoDepth API](https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/stereo_depth)
- [Sync API](https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/sync/)
- [Camera undistort 예제](https://docs.luxonis.com/software-v3/depthai/examples/camera/camera_undistort)

## 확인된 기존 데이터 상태

가장 최근 장시간 수집 데이터 `2026-06-22_14-44-18`에서 초기 200프레임을
제외하고 검사했습니다.

- 총 11,654개 RGB-D 행
- RGB-depth device timestamp 차이: median 5.443 ms, p95 5.444 ms
- 표본 프레임 200/500/2000/5000/10002/11655의 유효 depth: 약 59~78%
- RGB와 depth는 동일 장면이지만, 기존 저장 RGB는 왜곡이 남아 있고
  `setDepthAlign(CAM_A)`만 사용해 127° 광각 영상 가장자리의 동일 픽셀 좌표를
  보장하기 어려운 구성

수정된 RVC2 경로는 실제 OAK-D-PRO-W에서 200프레임 warm-up 후 검사했으며,
10개 평가 프레임의 median 유효 depth 84.1%, 최대 RGB-depth 차이 7.404 ms로
통과했습니다.

## 설치와 수집

```bash
cd depthai_refactored
../.venv/bin/python synced_image_recorder.py
```

수집 결과는 `image_records/<timestamp>_raw`에 저장됩니다. 동기 데이터 인덱스는
다음 명령으로 만듭니다.

```bash
../.venv/bin/python build_synced_dataset.py \
  --dataset image_records/<timestamp>_raw
```

기본 RGB-D 정렬은 `--depth-alignment-mode auto`입니다. RVC2 장비에서
`image-align`을 강제하면 잘못된 조합을 막기 위해 즉시 오류를 냅니다.
본수집에서는 raw 왜곡 RGB가 Depth와 다른 geometry가 되는 것을 막기 위해
factory RGB undistortion을 항상 사용하며 비활성화 옵션을 제공하지 않습니다.
실행 로그에 다음 줄이 표시되어야 합니다.

```text
RGB-D geometry: factory-undistorted RGB -> StereoDepth.inputAlignTo
```

## 센서 테스트

테스트 코드는 모두 `tests/`에 있습니다.

```bash
# RGB-D: 200프레임 warm-up 후 30프레임 평가
../.venv/bin/python tests/test_depthai_rgbd.py

# GPS + NTRIP/RTCM
../.venv/bin/python tests/test_gps_ntrip.py --duration-s 30

# EBIMU 출력/주기
../.venv/bin/python tests/test_ebimu.py --duration-s 10
```

RGB-D 테스트 결과는 기본적으로 `test_output/rgbd/`에 저장됩니다. 원본 RGB,
16-bit depth(mm), depth 컬러맵, RGB-depth overlay, JSON 판정 결과를 함께 봅니다.

## Debug UI

저장 데이터의 RGB·Depth·동기화·GPS·EBIMU와 YOLO-seg 결과를 한 화면에서
확인할 수 있습니다.

```bash
../.venv/bin/python debug_ui.py --host 127.0.0.1 --port 8088
```

브라우저에서 `http://127.0.0.1:8088`을 열고 데이터셋을 선택합니다. YOLO 검출
프레임에서는 오른쪽 패널에서 각 점의 픽셀·Depth·좌표 품질을 확인할 수 있습니다.

## YOLO-seg 3점 및 SHP 테스트

`best.pt`는 `guardrail` segmentation 모델입니다. 검출 마스크의 주축을 구한 뒤
한쪽 끝점, 중앙점, 반대쪽 끝점의 3점을 마스크 내부에 찍습니다. 따라서 난간이
비스듬하거나 휘어 보이는 경우에도 단순 사각형 중심보다 검출 형상을 잘 따릅니다.
초기 불안정 데이터가 섞이지 않도록 `--start-frame`은 200 미만을 허용하지 않습니다.

```bash
../.venv/bin/python tests/test_yolo_seg_shp.py \
  --dataset image_records/2026-06-23_14-20-20_raw \
  --model best.pt \
  --start-frame 200 \
  --max-frames 100 \
  --batch 4 \
  --conf 0.25
```

`--batch-size`(축약 `--batch`)는 한 번의 YOLO 추론에 묶을 이미지 수이며 기본값은
4입니다. GPU/메모리에 여유가 있으면 높이고 메모리 부족이 발생하면 낮춥니다.
`--confidence`(축약 `--conf`)는 검출 confidence 임계값이며 기본값은 0.25입니다.

기본 출력은 데이터셋의 `yolo_seg/`에 생성됩니다.

```text
yolo_seg/
  detected_rgb/                     검출된 프레임의 수정하지 않은 원본 RGB만 저장
  detections.jsonl                  Debug UI용 프레임별 결과
  points.csv                        모든 점과 Depth/좌표 상태
  yolo_seg_points_pixels.shp        항상 생성되는 픽셀 좌표 SHP
  yolo_seg_points_wgs84.shp         좌표 계산에 성공한 WGS84 POINTZ SHP
  summary.json
```

검출이 없는 프레임은 결과 파일에 기록하거나 복사하지 않습니다. 검출 프레임도
마스크·선·점을 그린 overlay는 만들지 않고 원본 RGB 파일을 그대로 복사합니다.

WGS84 좌표의 기본 자세 소스는 `gps-course-level`입니다. GPS 진행방향을 yaw로
사용하고 카메라가 수평이라고 가정하므로 테스트용 근사 좌표입니다. 정밀 SHP에는
RTK fixed 상태와 실제 카메라 장착각/레버암을 metadata에 넣고 `--orientation-source
ebimu` 또는 보정된 자세를 사용해야 합니다. 유효 Depth가 없는 점은 픽셀 SHP와
CSV에는 남지만 WGS84 SHP에서는 제외됩니다.

## 체커보드 카메라 캘리브레이션 테스트

지정 보드는 가로 14칸 × 세로 10칸, 한 칸 30 mm입니다. OpenCV에 넘기는
내부 코너 수는 가로 13 × 세로 9입니다.

```bash
../.venv/bin/python tests/test_checkerboard_calibration.py
```

보드를 여러 거리·위치·각도로 움직이고 `LOCKED - press Space` 초록 표시가
나오면 Space를 눌러 20장을 모읍니다. 검출은 1.5초간 고정되므로 한 프레임
실패로 즉시 사라지지 않습니다. `Q`로 조기 종료할 수 있으며 최소 12장이
필요합니다. 수락한 원본은 즉시 `test_output/checkerboard/captures/`에 저장됩니다.
결과는 `test_output/checkerboard/calibration.json`에 저장됩니다. RMS reprojection
error 1.0 px 이하뿐 아니라 추정 초점거리가 OAK factory calibration과 25% 안에서
일치해야 합격합니다. 평면 보드가 화면 일부 위치에만 모이면 RMS가 낮아도
초점거리와 거리 스케일을 잘못 추정할 수 있기 때문입니다.

자동 수집 또는 기존 이미지 폴더도 사용할 수 있습니다.

```bash
../.venv/bin/python tests/test_checkerboard_calibration.py --auto-capture
../.venv/bin/python tests/test_checkerboard_calibration.py --images ./checkerboard_images
```

### RGB–Depth 캘리브레이션 검증

RGB 캘리브레이션이 끝난 다음 같은 체커보드로 RGB와 Depth 사이의 정렬·거리·
평면 방향까지 검사합니다.

```bash
../.venv/bin/python tests/test_rgb_depth_checkerboard.py
```

이 테스트도 200개의 동기 프레임을 워밍업으로 버린 뒤 시작합니다. 체커보드를
약 0.7~2 m 거리에서 화면의 중앙/가장자리와 서로 다른 기울기로 보여주고,
`LOCKED - press Space`가 나오면 Space를 눌러 8개 자세를 저장합니다.
RGB와 Depth의 외곽선 이동도 측정하므로 체커보드를 벽에 붙이지 말고 뒤쪽
배경과 거리가 생기도록 들고 촬영해야 합니다.

판정 항목은 다음과 같습니다.

- 체커보드 내부의 유효 depth 비율
- RGB 코너와 `solvePnP`로 예측한 체커보드 거리 대비 실제 depth 오차
- RGB 체커보드 평면 법선과 depth 포인트로 피팅한 평면 법선의 각도
- depth 체커보드 평면의 RMSE
- RGB가 예측한 보드 외곽선과 실제 depth 경계의 픽셀 이동

결과와 각 프레임 overlay는 `test_output/rgb_depth_checkerboard/`에 저장됩니다.
기본 합격 기준은 유효률 50% 이상, median 거리 오차 80 mm 이하, 최악 view의
p95 오차 250 mm 이하, 평면 법선 5° 이하, depth 평면 RMSE 30 mm 이하,
외곽선 median 12 px/p95 30 px 이하입니다.
저장된 캡처는 카메라 없이 다시 분석할 수 있습니다.

```bash
../.venv/bin/python tests/test_rgb_depth_checkerboard.py \
  --captures test_output/rgb_depth_checkerboard/captures
```

이전 버전으로 저장해 RGB에는 광각 왜곡이 남고 Depth만 정렬 geometry인 경우,
위 재분석 명령이 OAK factory CAM_A 값으로 RGB를 자동 undistort합니다. 원본은
건드리지 않고 `test_output/rgb_depth_checkerboard/corrected_captures/`에 보정본을
만듭니다. 향후 라이브 RGB–Depth 테스트는 처음부터 undistorted RGB를 수집합니다.

주의: 14×30 mm = 420 mm, 10×30 mm = 300 mm이므로 정확한 30 mm 정사각형 보드는
420×300 mm입니다. A3는 420×297 mm라 짧은 변이 3 mm 부족합니다. A3에 맞춤
축소 인쇄하면 칸이 정확히 30×30 mm가 아니게 되므로, 100% 배율로 인쇄하고
3 mm가 더 긴 용지/여백을 쓰거나 인쇄 후 실제 칸 크기를 재서
`--square-size-mm`에 입력해야 합니다.

## NTRIP 기본값

리팩터의 비어 있던 값은 기존 코드와 동일하게 복원했습니다.

- caster: `www.gnssdata.or.kr:2101`
- mountpoint: `YANJ-RTCM31`
- username/password: 기존 수집 코드 값
- GGA interval: 10초, reconnect delay: 5초

CLI 인자 또는 `NTRIP_HOST`, `NTRIP_PORT`, `NTRIP_MOUNTPOINT`,
`NTRIP_USERNAME`, `NTRIP_PASSWORD` 환경변수로 덮어쓸 수 있습니다.

## 남긴 코드

```text
synced_image_recorder.py
build_synced_dataset.py
geonova_depthai/capture/       수집 CLI, writer, defaults
geonova_depthai/postprocess/   timestamp 동기화
geonova_depthai/runtime.py          수집에 필요한 카메라/serial runtime
geonova_depthai/debug_ui.py         저장 데이터 Debug UI 및 좌표 투영
geonova_depthai/yolo_seg_shp.py     YOLO-seg 3점/Depth/WGS84/SHP 처리
debug_ui.py                         Debug UI 실행 진입점
tools/configure_ebimu.py            EBIMU 설정 도구
tests/                         RGB-D, GPS/NTRIP, EBIMU, checkerboard 테스트
```
