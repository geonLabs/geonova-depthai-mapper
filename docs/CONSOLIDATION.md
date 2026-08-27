# 저장소 통합 결정 기록

## 결론

2026년 8월 27일부터 다음 저장소를 단일 기준 저장소로 사용합니다.

```text
Canonical: geonLabs/geonova-depthai-mapper
Legacy:    dbparkJ/geo_multifusion_sensors
```

`geo_multifusion_sensors`에 있던 DepthAI 코드 사본은 더 이상 독립 원본으로 취급하지
않습니다. 수집, 동기화, GPS/RTK, 외부 IMU, Controller bridge, YOLO/SHP 및
방호울타리 선형화 변경은 모두 기준 저장소에서만 진행합니다.

## 기준 저장소를 선택한 이유

기준 저장소의 `code/`에는 레거시 `safe_gard_test/code/`보다 다음 운영 기능이 더
발전된 상태로 들어 있습니다.

- 데이터셋을 만들지 않는 `--monitor-only` 센서 모니터링
- OAK 카메라 장애 시 GPS/IMU 상태를 유지한 재연결
- `/dev/serial/by-id` 기반 GNSS/EBIMU 안전 자동 탐색
- Android `ttyACM` 장치를 GNSS로 오인하지 않는 fail-closed 처리
- 이동 중 NTRIP 기준국 재선택과 make-before-break RTCM 전환
- 동일 초 재시작 시 기존 수집 폴더를 덮어쓰지 않는 원자적 디렉터리 예약
- 위 동작을 검증하는 더 넓은 회귀 테스트

따라서 레거시 저장소의 중복 `safe_gard_test`를 다시 복사하면 최신 기능이 후퇴하고
두 원본이 재발생합니다.

## 병합한 고유 기능

레거시 저장소에서만 존재하던 LiDAR/ROS 기능은 다음 위치로 이동했습니다.

| 레거시 경로 | 기준 저장소 경로 |
|---|---|
| `test/convert_lidar_bag_to_pcd.py` | `tools/lidar/convert_lidar_bag_to_pcd.py` |
| `test/config.yaml` | `tools/lidar/config.yaml` |
| `test/requirements.txt` | `tools/lidar/requirements.txt` |
| `scripts/project_lidar_overlay.py` | `tools/lidar/project_lidar_overlay.py` |

bag 변환기는 원본 Git blob을 그대로 보존했습니다. 투영 도구는 ROS1 모듈이 없는
환경에서 이해하기 쉬운 오류를 내고, 입력과 출력 실패를 검증하도록 보강했습니다.

## 추가 정리

- Linux와 Windows 설치기가 사용자가 명시한 `--venv` 경로를 더 이상 덮어쓰지 않음
- 공유 `config.yaml`에서 NTRIP 사용자명과 비밀번호 제거
- `.bag`, `.pcd`, LiDAR 산출물을 Git에서 제외
- LiDAR 순수 함수 및 PCD writer 회귀 테스트 추가
- PR과 `main` push에서 실행되는 경량 GitHub Actions 추가

## 앞으로의 디렉터리 책임

```text
code/
  Jetson/OAK RGB-D/GNSS/IMU 수집과 매핑의 유일한 원본

model/
  YOLO 학습, 라벨 변환, 배포 가능한 모델 메타데이터

tools/lidar/
  ROS bag 변환과 LiDAR–카메라 오프라인 검증

data/, results/
  Git에 올리지 않는 현장 데이터와 런타임 결과
```

`tools/lidar`는 Jetson Controller 자동 실행 진입점에 포함하지 않습니다. ROS와
LiDAR 의존성은 DepthAI/PyTorch 가상환경과 분리합니다.

## 개발 규칙

1. DepthAI 핵심 코드를 다른 저장소로 복제하지 않습니다.
2. 센서 공통 기능이 필요하면 `code/geonova_depthai`에서 모듈화합니다.
3. ROS 전용 의존성은 `tools/lidar` 안에 격리합니다.
4. 공유 YAML에는 계정, 비밀번호, 토큰을 기록하지 않습니다.
5. 변경은 기준 저장소의 브랜치와 PR에서 테스트한 뒤 반영합니다.
6. 레거시 저장소는 과거 이력 확인용으로만 유지하고 신규 기능을 추가하지 않습니다.

## 로컬 작업공간 이전

레거시 저장소를 사용하던 장비에서는 새 저장소를 별도 clone한 뒤 데이터 경로만
연결합니다. `.venv`, 수집 데이터, 모델 대형 가중치를 저장소 간에 복사하지 않습니다.

```bash
git clone https://github.com/geonLabs/geonova-depthai-mapper.git
cd geonova-depthai-mapper
chmod +x install.sh
./install.sh --dev
```

LiDAR 도구는 별도 환경을 사용합니다.

```bash
cd tools/lidar
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

NTRIP 계정은 `NTRIP_USERNAME`, `NTRIP_PASSWORD` 환경변수나 Jetson Controller의
보호된 환경 설정으로 다시 주입합니다. 과거 Git 커밋에 노출된 인증정보는 별도로
폐기하고 재발급해야 합니다.
