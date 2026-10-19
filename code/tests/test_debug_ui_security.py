from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geonova_depthai.debug_ui import create_server, parse_args, print_server_urls  # noqa: E402


def _make_dataset(path: Path, rgb_file: str = "rgb/frame.png") -> Path:
    path.mkdir(parents=True)
    (path / "timestamps.csv").write_text(
        "stem,rgb_file,depth_file\nframe0,{},depth_mm/frame.png\n".format(rgb_file),
        encoding="utf-8",
    )
    (path / "imu.csv").write_text("frame_index\n", encoding="utf-8")
    return path


@contextmanager
def _running_server(dataset_root: Path):
    server = create_server("127.0.0.1", 0, dataset_root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield "http://{}:{}".format(host, port)
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def _url(base: str, path: str, params=None) -> str:
    suffix = "?" + urllib.parse.urlencode(params) if params else ""
    return base + path + suffix


def _status(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def test_server_defaults_to_loopback() -> None:
    assert parse_args([]).host == "127.0.0.1"
    server = create_server(port=0)
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()


def test_network_bind_prints_one_warning(capsys: pytest.CaptureFixture[str]) -> None:
    print_server_urls("0.0.0.0", 8088)
    output = capsys.readouterr().out
    assert output.count("WARNING:") == 1
    assert "reachable from the network" in output


def test_mutation_requires_post_json_and_read_only_get_works(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    root.mkdir()
    dataset = _make_dataset(root / "run")

    with _running_server(root) as base:
        status, _ = _status(_url(base, "/"))
        assert status == 200

        status, body = _status(_url(base, "/api/dataset", {"path": str(dataset)}))
        assert status == 200
        assert json.loads(body)["frame_count"] == 1

        status, body = _status(_url(base, "/api/latest", {"path": str(root)}))
        assert status == 200
        assert Path(json.loads(body)["path"]) == dataset.resolve()

        status, _ = _status(_url(base, "/api/update_mount_rpy"))
        assert status == 405
        status, _ = _status(_url(base, "/api/solve_mount_calibration"))
        assert status == 405

        bad_request = urllib.request.Request(
            _url(base, "/api/update_mount_rpy"),
            data=b"{}",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(bad_request, timeout=3)
        assert error.value.code == 400

        payload = json.dumps({
            "path": str(dataset),
            "roll_deg": 1.5,
            "pitch_deg": -2.0,
            "yaw_deg": 3.25,
            "save": True,
        }).encode("utf-8")
        request = urllib.request.Request(
            _url(base, "/api/update_mount_rpy"),
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            result = json.loads(response.read())
        assert result["status"] == "ok"
        assert result["saved"] is True
        assert result["camera_mount_rpy_deg"] == [1.5, -2.0, 3.25]
        assert (dataset / "metadata.json").is_file()


def test_dataset_traversal_and_outside_absolute_path_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    root.mkdir()
    dataset = _make_dataset(root / "run")
    outside = _make_dataset(tmp_path / "outside")

    with _running_server(root) as base:
        status, body = _status(_url(base, "/api/dataset", {"path": "../outside"}))
        assert status == 403
        assert str(outside.resolve()).encode("utf-8") not in body

        status, body = _status(_url(base, "/api/dataset", {"path": str(outside)}))
        assert status == 403
        assert str(outside.resolve()).encode("utf-8") not in body

        status, _ = _status(_url(base, "/api/dataset", {"path": str(dataset)}))
        assert status == 200


def test_symlink_dataset_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    root.mkdir()
    outside = _make_dataset(tmp_path / "outside")
    link = root / "linked-run"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip("Directory symlinks are unavailable: {}".format(error))

    with _running_server(root) as base:
        status, body = _status(_url(base, "/api/dataset", {"path": str(link)}))
        assert status == 403
        assert str(outside.resolve()).encode("utf-8") not in body


def test_media_frame_path_cannot_escape_dataset_root(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    root.mkdir()
    dataset = _make_dataset(root / "run", "../outside.png")

    with _running_server(root) as base:
        status, body = _status(_url(
            base,
            "/media/rgb",
            {"path": str(dataset), "index": 0},
        ))
        assert status == 403
        assert str(tmp_path.resolve()).encode("utf-8") not in body
