# 2026-06-26 field profile

기존 로컬 작업본의 현장 튜닝값을 최신 설정 스키마 위에 보존한 프로필입니다.
절대경로는 저장소 루트의 `data/`와 `model/x_model/best.pt`를 사용하는 상대경로로
바꿨습니다. YAML에 없는 신규 옵션은 코드 기본값을 사용합니다.

```bash
cd code
.venv/bin/python synced_image_recorder.py \
  --config configs/profiles/local_2026_06_26/capture.yaml
.venv/bin/python tests/test_yolo_seg_shp.py \
  --config configs/profiles/local_2026_06_26/yolo.yaml
```
