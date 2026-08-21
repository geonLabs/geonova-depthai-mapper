#!/usr/bin/env python3

import argparse
import json
import math
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
DEFAULT_LABEL_SOURCE = REPOSITORY_ROOT / "data" / "training_source" / "labels"
DEFAULT_IMAGE_SOURCE = REPOSITORY_ROOT / "data" / "training_source" / "images"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "safety_guard"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


@dataclass
class Sample:
    json_path: Path
    relative_path: Path
    image_path: Path
    lines: list[str]
    skipped_polygons: int
    split: str = ""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "COCO 형식 JSON에서 지정한 category_id의 폴리곤을 변환하고, "
            "YOLO segmentation 학습용 이미지/라벨/data.yaml을 생성합니다."
        )
    )
    parser.add_argument(
        "--label-source",
        "--source",
        dest="label_source",
        type=Path,
        default=DEFAULT_LABEL_SOURCE,
        help=f"원본 JSON 폴더 (기본값: {DEFAULT_LABEL_SOURCE})",
    )
    parser.add_argument(
        "--image-source",
        type=Path,
        default=DEFAULT_IMAGE_SOURCE,
        help=f"원본 이미지 폴더 (기본값: {DEFAULT_IMAGE_SOURCE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"YOLO 데이터셋 출력 폴더 (기본값: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--category-id",
        type=int,
        default=13,
        help="추출할 원본 category_id (기본값: 13)",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="출력할 YOLO class id (단일 클래스이므로 기본값: 0)",
    )
    parser.add_argument(
        "--class-name",
        default="guardrail",
        help="data.yaml에 기록할 클래스 이름 (기본값: guardrail)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="검증 데이터 비율 (기본값: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="train/val 분할 난수 시드 (기본값: 42)",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="대상 객체가 없는 이미지를 데이터셋에서 제외합니다.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="출력 폴더에 파일이 있어도 해당 경로의 생성 파일을 덮어씁니다.",
    )
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def iter_polygons(segmentation):
    if not isinstance(segmentation, list) or not segmentation:
        return

    if all(isinstance(value, (int, float)) for value in segmentation):
        yield segmentation
        return

    for polygon in segmentation:
        if isinstance(polygon, list):
            yield polygon


def normalize_polygon(polygon, width, height):
    if len(polygon) < 6 or len(polygon) % 2:
        return None

    normalized = []
    unique_points = set()

    for index in range(0, len(polygon), 2):
        x = float(polygon[index])
        y = float(polygon[index + 1])
        if not math.isfinite(x) or not math.isfinite(y):
            return None

        x_normalized = min(max(x / width, 0.0), 1.0)
        y_normalized = min(max(y / height, 0.0), 1.0)
        normalized.extend((x_normalized, y_normalized))
        unique_points.add((round(x_normalized, 8), round(y_normalized, 8)))

    if len(unique_points) < 3:
        return None

    return normalized


def make_yolo_lines(document, category_id, class_id):
    images = document.get("images") or []
    if len(images) != 1:
        raise ValueError(f"JSON 하나당 이미지 정보가 1개여야 합니다: {len(images)}개")

    image = images[0]
    image_id = image.get("id")
    width = float(image.get("width", 0))
    height = float(image.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"잘못된 이미지 크기: width={width}, height={height}")

    lines = []
    skipped_polygons = 0

    for annotation in document.get("annotations") or []:
        if annotation.get("category_id") != category_id:
            continue
        if annotation.get("image_id") != image_id:
            skipped_polygons += 1
            continue

        for polygon in iter_polygons(annotation.get("segmentation")):
            normalized = normalize_polygon(polygon, width, height)
            if normalized is None:
                skipped_polygons += 1
                continue

            coordinates = " ".join(f"{value:.6f}" for value in normalized)
            lines.append(f"{class_id} {coordinates}")

    return lines, skipped_polygons, image


def find_existing_file(candidate):
    if candidate.is_file():
        return candidate

    parent = candidate.parent
    if not parent.is_dir():
        return None

    target_name = candidate.name.casefold()
    for child in parent.iterdir():
        if child.is_file() and child.name.casefold() == target_name:
            return child

    return None


def iter_metadata_image_paths(file_name):
    if not file_name:
        return

    normalized = str(file_name).strip().replace("\\", "/")
    if not normalized:
        return

    parts = [
        part
        for part in PurePosixPath(normalized).parts
        if part not in {"", "/", "."}
    ]
    if parts and parts[0].endswith(":"):
        parts = parts[1:]
    if not parts:
        return

    yield Path(*parts)
    if len(parts) > 1:
        yield Path(parts[-1])


def iter_unique_paths(paths):
    seen = set()
    for path in paths:
        key = path.as_posix().casefold()
        if key in seen:
            continue
        seen.add(key)
        yield path


def find_image(image_source, relative_json_path, image_info):
    relative_stem = relative_json_path.with_suffix("")

    for extension in IMAGE_EXTENSIONS:
        candidate = (image_source / relative_stem).with_suffix(extension)
        match = find_existing_file(candidate)
        if match:
            return match

    file_name = image_info.get("file_name")
    metadata_candidates = []
    for metadata_path in iter_metadata_image_paths(file_name):
        metadata_candidates.append(image_source / metadata_path)
        metadata_candidates.append(
            image_source / relative_json_path.parent / metadata_path.name
        )

    for candidate in iter_unique_paths(metadata_candidates):
        match = find_existing_file(candidate)
        if match:
            return match

    raise FileNotFoundError(f"대응하는 이미지가 없습니다: {relative_json_path}")


def collect_samples(label_source, image_source, category_id, class_id, skip_empty):
    json_paths = sorted(label_source.rglob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"JSON 파일이 없습니다: {label_source}")

    samples = []
    failed_files = []

    for json_path in json_paths:
        relative_path = json_path.relative_to(label_source)
        try:
            document = load_json(json_path)
            lines, skipped_polygons, image_info = make_yolo_lines(
                document, category_id, class_id
            )
            image_path = find_image(image_source, relative_path, image_info)
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            failed_files.append((json_path, str(error)))
            continue

        if not lines and skip_empty:
            continue

        samples.append(
            Sample(
                json_path=json_path,
                relative_path=relative_path,
                image_path=image_path,
                lines=lines,
                skipped_polygons=skipped_polygons,
            )
        )

    return json_paths, samples, failed_files


def split_samples(samples, val_ratio, seed):
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.relative_path.parent].append(sample)

    for group_index, group_path in enumerate(sorted(groups, key=lambda path: str(path))):
        group = sorted(groups[group_path], key=lambda sample: str(sample.relative_path))
        random.Random(seed + group_index).shuffle(group)

        if val_ratio == 0 or len(group) < 2:
            validation_count = 0
        else:
            validation_count = max(1, round(len(group) * val_ratio))
            validation_count = min(validation_count, len(group) - 1)

        for index, sample in enumerate(group):
            sample.split = "val" if index < validation_count else "train"


def ensure_output_available(output, overwrite):
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"출력 폴더가 비어 있지 않습니다: {output}\n"
            "기존 파일을 덮어쓰려면 --overwrite 옵션을 사용하세요."
        )
    output.mkdir(parents=True, exist_ok=True)


def write_sample(sample, output):
    image_relative = sample.relative_path.with_suffix(sample.image_path.suffix.lower())
    label_relative = sample.relative_path.with_suffix(".txt")
    image_output = output / "images" / sample.split / image_relative
    label_output = output / "labels" / sample.split / label_relative

    image_output.parent.mkdir(parents=True, exist_ok=True)
    label_output.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(sample.image_path, image_output)
    content = "\n".join(sample.lines)
    if content:
        content += "\n"
    label_output.write_text(content, encoding="utf-8")


def yaml_string(value):
    return json.dumps(str(value), ensure_ascii=False)


def write_data_yaml(output, class_id, class_name):
    if class_id != 0:
        raise ValueError(
            "단일 클래스 데이터셋의 YOLO class id는 0이어야 합니다. "
            "--class-id 0을 사용하세요."
        )

    yaml_path = output / "data.yaml"
    content = (
        f"path: {yaml_string(output.resolve().as_posix())}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        f"  0: {yaml_string(class_name)}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def build_dataset(
    label_source,
    image_source,
    output,
    category_id,
    class_id,
    class_name,
    val_ratio,
    seed,
    skip_empty,
    overwrite,
):
    if not label_source.is_dir():
        raise FileNotFoundError(f"원본 JSON 폴더를 찾을 수 없습니다: {label_source}")
    if not image_source.is_dir():
        raise FileNotFoundError(f"원본 이미지 폴더를 찾을 수 없습니다: {image_source}")
    if class_id < 0:
        raise ValueError("YOLO class id는 0 이상이어야 합니다.")
    if not 0 <= val_ratio < 1:
        raise ValueError("--val-ratio는 0 이상 1 미만이어야 합니다.")
    if not class_name.strip():
        raise ValueError("--class-name은 비어 있을 수 없습니다.")

    json_paths, samples, failed_files = collect_samples(
        label_source=label_source,
        image_source=image_source,
        category_id=category_id,
        class_id=class_id,
        skip_empty=skip_empty,
    )

    if failed_files:
        return {
            "json_files": len(json_paths),
            "samples": samples,
            "failed_files": failed_files,
            "yaml_path": None,
        }
    if not samples:
        raise ValueError("출력할 데이터가 없습니다.")

    split_samples(samples, val_ratio, seed)
    ensure_output_available(output, overwrite)

    for sample in samples:
        write_sample(sample, output)

    yaml_path = write_data_yaml(output, class_id, class_name)
    return {
        "json_files": len(json_paths),
        "samples": samples,
        "failed_files": [],
        "yaml_path": yaml_path,
    }


def summarize(result):
    samples = result["samples"]
    train_samples = [sample for sample in samples if sample.split == "train"]
    val_samples = [sample for sample in samples if sample.split == "val"]

    return {
        "json_files": result["json_files"],
        "total_images": len(samples),
        "train_images": len(train_samples),
        "val_images": len(val_samples),
        "empty_images": sum(not sample.lines for sample in samples),
        "objects": sum(len(sample.lines) for sample in samples),
        "skipped_polygons": sum(sample.skipped_polygons for sample in samples),
        "failed_files": result["failed_files"],
        "yaml_path": result["yaml_path"],
    }


def main():
    args = parse_args()

    try:
        result = build_dataset(
            label_source=args.label_source,
            image_source=args.image_source,
            output=args.output,
            category_id=args.category_id,
            class_id=args.class_id,
            class_name=args.class_name,
            val_ratio=args.val_ratio,
            seed=args.seed,
            skip_empty=args.skip_empty,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        raise SystemExit(f"오류: {error}") from error

    summary = summarize(result)
    print(f"원본 JSON: {summary['json_files']}개")
    print(f"전체 이미지: {summary['total_images']}개")
    print(f"학습 이미지: {summary['train_images']}개")
    print(f"검증 이미지: {summary['val_images']}개")
    print(f"빈 라벨 이미지: {summary['empty_images']}개")
    print(f"변환 객체: {summary['objects']}개")
    print(f"건너뛴 폴리곤: {summary['skipped_polygons']}개")
    print(f"실패 파일: {len(summary['failed_files'])}개")

    for path, message in summary["failed_files"][:10]:
        print(f"  - {path}: {message}")

    if summary["failed_files"]:
        raise SystemExit(1)

    print(f"data.yaml: {summary['yaml_path'].resolve()}")


if __name__ == "__main__":
    main()
