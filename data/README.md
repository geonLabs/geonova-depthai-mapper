# Local data workspace

원시 RGB-D/GPS/IMU 수집물, 동기 데이터셋, YOLO 결과와 SHP 산출물을 두는
로컬 전용 폴더입니다. 대용량 데이터는 Git에 커밋하지 않습니다.

`code/configs/capture.yaml`은 기본적으로 이 폴더 아래에 `<timestamp>_raw`
데이터셋을 생성합니다.
