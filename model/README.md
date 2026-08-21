# Model workspace

- `convert_label.py`: COCO polygon JSON을 YOLO segmentation 데이터셋으로 변환
- `safe_gard_train.py`: YAML 기반 Ultralytics 학습 진입점
- `safe_gard_train_config.yaml`: 로컬 단일 GPU YOLO26n 설정
- `safe_gard_train_config.4gpu.yaml`: 4-GPU YOLO26x 설정 예시
- `n_model/best.pt`: 저장소에 포함되는 소형 guardrail 모델
- `x_model/best.pt`: 현장용 대형 모델의 로컬 위치(Git 제외)

학습 데이터와 결과는 각각 `../data/`와 `runs/`에 저장하며 Git에 포함하지
않습니다. Ultralytics는 설정에 적힌 `yolo26n-seg.pt` 또는 `yolo26x-seg.pt`가
없으면 공식 seed 모델을 내려받으며, 내려받은 파일도 Git에서 제외됩니다.
