#!/usr/bin/env python3
"""Run segmentation inference and override display class names."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def choose_default_model(repository_root: Path = REPOSITORY_ROOT) -> Path:
    """Prefer the local field model and fall back to the tracked nano model."""
    field_model = repository_root / "model" / "x_model" / "best.pt"
    if field_model.is_file():
        return field_model
    return repository_root / "model" / "n_model" / "best.pt"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO segmentation 결과의 표시 클래스 이름을 바꿔 저장합니다."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=choose_default_model(),
        help="segmentation 모델 경로; 로컬 x_model이 있으면 우선 사용",
    )
    parser.add_argument(
        "--source",
        default="raw_images",
        help="입력 이미지, 폴더, glob 또는 Ultralytics 지원 source",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/design_fence_plot"),
        help="plot 이미지를 저장할 폴더",
    )
    parser.add_argument(
        "--class-name",
        default="design fence",
        help="모든 검출 클래스에 표시할 이름",
    )
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    import cv2
    from ultralytics import YOLO

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"모델 파일이 없습니다: {model_path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predict_args = {"source": args.source, "save": False}
    if args.confidence is not None:
        predict_args["conf"] = args.confidence
    if args.iou is not None:
        predict_args["iou"] = args.iou

    results = YOLO(str(model_path)).predict(**predict_args)
    for result in results:
        result.names = {class_id: args.class_name for class_id in result.names}
        plotted = result.plot(conf=True, labels=True, boxes=True, masks=True)
        output_path = output_dir / Path(result.path).name
        if not cv2.imwrite(str(output_path), plotted):
            raise RuntimeError(f"이미지를 저장하지 못했습니다: {output_path}")

    print(f"저장 완료: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
