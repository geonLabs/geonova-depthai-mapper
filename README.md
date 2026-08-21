# DepthAI Camera Record

현재 사용하는 수집기, 센서 테스트, 체커보드 캘리브레이션, Debug UI,
YOLO-seg/SHP 코드는 모두 [`depthai_refactored/`](depthai_refactored/)에 있습니다.
이전 버전은 최신 코드로 통합해 중복 실행 경로를 제거했습니다.

새 Linux/macOS 환경에서는 저장소 루트에서 다음 한 명령으로 Python 3.11,
가상환경, PyTorch 및 실행 의존성을 설치합니다.

```bash
chmod +x install.sh
./install.sh
```

Windows PowerShell에서는 다음을 실행합니다.

```powershell
.\install.ps1
```

설치기는 CUDA Toolkit을 감지해 PyTorch 빌드를 선택하며, CUDA가 없으면 CPU
빌드를 사용합니다. 테스트 의존성도 필요하면 `./install.sh --dev` 또는
`.\install.ps1 --dev`를 사용합니다.

```bash
cd depthai_refactored
.venv/bin/python synced_image_recorder.py --config configs/capture.yaml
.venv/bin/python -m geonova_depthai.debug_ui --config configs/debug_ui.yaml
```

설치, 센서 테스트, 200프레임 이후 RGB–Depth 검증, YOLO-seg 실행 방법은
[`depthai_refactored/README.md`](depthai_refactored/README.md)를 참고하세요.
