from pathlib import Path

from model.safe_gard_train import coerce_config_types


def test_multigpu_device_list_becomes_ultralytics_device_string(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    converted = coerce_config_types(
        {"device": [0, 1, 2, 3], "data": "data.yaml"},
        config_path,
    )
    assert converted["device"] == "0,1,2,3"
    assert converted["data"] == tmp_path / "data.yaml"
