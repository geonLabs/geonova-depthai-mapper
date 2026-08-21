import argparse
import os
from pathlib import Path


os.environ.setdefault("NCCL_P2P_DISABLE", "1")


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "safe_gard_train_config.yaml"
DEFAULT_DATA = PROJECT_ROOT / "data.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "segment"

DEFAULT_TRAIN_CONFIG = {
    "data": DEFAULT_DATA,
    "model": "yolo26n-seg.pt",
    "epochs": 100,
    "imgsz": 640,
    "batch": 8,
    "device": "0",
    "workers": 4,
    "patience": 20,
    "project": DEFAULT_PROJECT,
    "name": "guardrail_yolo26n_seg",
    "seed": 42,
    "cache": False,
    "exist_ok": False,
}
PATH_KEYS = {"data", "project"}
INT_KEYS = {"epochs", "imgsz", "batch", "workers", "patience", "seed"}
BOOL_KEYS = {"cache", "exist_ok"}


def load_yaml(path):
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "YAML 설정 파일을 읽으려면 PyYAML이 필요합니다. "
            "conda activate yolo_26 후 `pip install pyyaml`을 실행하세요."
        ) from error

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML 최상위 구조는 key: value 형태여야 합니다: {path}")
    return data


def normalize_config_keys(raw_config):
    config = {}
    for key, value in raw_config.items():
        normalized_key = str(key).replace("-", "_")
        if normalized_key in config:
            raise ValueError(f"중복 설정 키가 있습니다: {key}")
        config[normalized_key] = value

    unknown_keys = sorted(set(config) - set(DEFAULT_TRAIN_CONFIG))
    if unknown_keys:
        joined_keys = ", ".join(unknown_keys)
        raise ValueError(f"알 수 없는 설정 키입니다: {joined_keys}")
    return config


def to_bool(value, key):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"{key} 값은 true 또는 false여야 합니다: {value}")


def coerce_config_types(config, config_path):
    coerced = {}
    for key, value in config.items():
        if key in PATH_KEYS:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = config_path.parent / path
            coerced[key] = path
        elif key in INT_KEYS:
            coerced[key] = int(value)
        elif key in BOOL_KEYS:
            coerced[key] = to_bool(value, key)
        elif key == "device" and isinstance(value, list):
            coerced[key] = ",".join(str(item) for item in value)
        else:
            coerced[key] = str(value)
    return coerced


def load_train_config(config_path, required):
    config = dict(DEFAULT_TRAIN_CONFIG)
    if not config_path.is_file():
        if required:
            raise FileNotFoundError(f"설정 YAML 파일이 없습니다: {config_path}")
        return config

    raw_config = load_yaml(config_path)
    yaml_config = normalize_config_keys(raw_config)
    config.update(coerce_config_types(yaml_config, config_path))
    return config


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"학습 설정 YAML 경로 (기본값: {DEFAULT_CONFIG})",
    )
    config_args, _ = config_parser.parse_known_args()
    config_path = config_args.config or DEFAULT_CONFIG
    required_config = config_args.config is not None or DEFAULT_CONFIG.is_file()
    train_config = load_train_config(config_path, required=required_config)

    parser = argparse.ArgumentParser(
        description="YOLO nano 모델로 category_id 13 segmentation 데이터셋을 학습합니다.",
        parents=[config_parser],
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=train_config["data"],
        help=f"데이터셋 YAML 경로 (설정값: {train_config['data']})",
    )
    parser.add_argument(
        "--model",
        default=train_config["model"],
        help=f"사전 학습 segmentation 모델 (설정값: {train_config['model']})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=train_config["epochs"],
        help=f"학습 epoch 수 (설정값: {train_config['epochs']})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=train_config["imgsz"],
        help=f"학습 이미지 크기 (설정값: {train_config['imgsz']})",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=train_config["batch"],
        help=f"batch 크기 (설정값: {train_config['batch']})",
    )
    parser.add_argument(
        "--device",
        default=train_config["device"],
        help=f"학습 장치: 0, 0,1, cpu 등 (설정값: {train_config['device']})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=train_config["workers"],
        help=f"데이터 로더 worker 수 (설정값: {train_config['workers']})",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=train_config["patience"],
        help=f"조기 종료 patience (설정값: {train_config['patience']})",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=train_config["project"],
        help=f"학습 결과 상위 폴더 (설정값: {train_config['project']})",
    )
    parser.add_argument(
        "--name",
        default=train_config["name"],
        help=f"학습 실행 이름 (설정값: {train_config['name']})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=train_config["seed"],
        help=f"재현용 난수 시드 (설정값: {train_config['seed']})",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        default=train_config["cache"],
        help="이미지를 RAM 또는 디스크 캐시에 저장합니다.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        default=train_config["exist_ok"],
        help="동일한 학습 결과 폴더 이름을 허용합니다.",
    )
    return parser.parse_args()


def load_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyTorch가 설치되지 않았습니다. 먼저 setup_yolo_env.ps1을 실행하세요."
        ) from error
    return torch


def validate_args(args, torch):
    if not args.data.is_file():
        raise FileNotFoundError(
            f"data.yaml이 없습니다: {args.data}\n"
            "먼저 convert_category13_to_yolo_seg.py를 실행하세요."
        )
    if args.epochs <= 0:
        raise ValueError("--epochs는 1 이상이어야 합니다.")
    if args.imgsz <= 0:
        raise ValueError("--imgsz는 1 이상이어야 합니다.")
    if args.batch == 0 or args.batch < -1:
        raise ValueError("--batch는 -1 또는 1 이상의 값이어야 합니다.")
    if args.workers < 0:
        raise ValueError("--workers는 0 이상이어야 합니다.")

    device = args.device.strip().lower()
    uses_cuda = device not in {"cpu", "mps"} and device != ""
    if uses_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU를 사용할 수 없습니다. NVIDIA 드라이버를 확인하거나 "
            "--device cpu로 실행하세요."
        )


def print_environment(args, torch):
    print(f"Config: {args.config or DEFAULT_CONFIG}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data.resolve()}")


def train(args):
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Ultralytics가 설치되지 않았습니다. 먼저 setup_yolo_env.ps1을 실행하세요."
        ) from error

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(args.project.resolve()),
        name=args.name,
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        optimizer="auto",
        amp=True,
        cache=args.cache,
        plots=True,
        save=True,
        exist_ok=args.exist_ok,
    )
    return model


def main():
    try:
        args = parse_args()
        torch = load_torch()
        validate_args(args, torch)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"오류: {error}") from error

    print_environment(args, torch)
    try:
        model = train(args)
    except RuntimeError as error:
        raise SystemExit(f"오류: {error}") from error
    print(f"학습 완료: {model.trainer.save_dir}")


if __name__ == "__main__":
    main()
