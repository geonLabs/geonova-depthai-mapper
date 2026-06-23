# DepthAI Camera Record

현재 사용하는 수집기, 센서 테스트, 체커보드 캘리브레이션, Debug UI,
YOLO-seg/SHP 코드는 모두 [`depthai_refactored/`](depthai_refactored/)에 있습니다.
루트에 있던 이전 수집기와 임시 진단 스크립트는 중복 실행을 막기 위해 제거했습니다.

```bash
cd depthai_refactored
../.venv/bin/python synced_image_recorder.py
../.venv/bin/python debug_ui.py
```

설치, 센서 테스트, 200프레임 이후 RGB–Depth 검증, YOLO-seg 실행 방법은
[`depthai_refactored/README.md`](depthai_refactored/README.md)를 참고하세요.
