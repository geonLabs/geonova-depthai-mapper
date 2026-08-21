from pathlib import Path

from inference.infer_class_change import choose_default_model, parse_args


def test_choose_default_model_prefers_field_model(tmp_path: Path) -> None:
    nano_model = tmp_path / "model" / "n_model" / "best.pt"
    nano_model.parent.mkdir(parents=True)
    nano_model.touch()
    assert choose_default_model(tmp_path) == nano_model

    field_model = tmp_path / "model" / "x_model" / "best.pt"
    field_model.parent.mkdir(parents=True)
    field_model.touch()
    assert choose_default_model(tmp_path) == field_model


def test_parse_args_accepts_portable_paths() -> None:
    args = parse_args(
        [
            "--model",
            "../model/n_model/best.pt",
            "--source",
            "images",
            "--output-dir",
            "runs/test",
            "--class-name",
            "guardrail",
        ]
    )
    assert args.model == Path("../model/n_model/best.pt")
    assert args.source == "images"
    assert args.output_dir == Path("runs/test")
    assert args.class_name == "guardrail"
