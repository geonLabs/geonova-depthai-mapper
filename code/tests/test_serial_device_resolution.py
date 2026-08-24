from pathlib import Path
from types import SimpleNamespace

from geonova_depthai import runtime
from geonova_depthai import serial_devices


def add_by_id(root: Path, name: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch()
    root.mkdir(parents=True, exist_ok=True)
    link = root / name
    link.symlink_to(target)
    return link


def test_gnss_auto_resolution_ignores_samsung_ttyacm0(tmp_path: Path) -> None:
    by_id = tmp_path / "dev" / "serial" / "by-id"
    samsung = add_by_id(
        by_id,
        "usb-SAMSUNG_SAMSUNG_Android_PHONE-if01",
        tmp_path / "dev" / "ttyACM0",
    )
    gnss = add_by_id(
        by_id,
        "usb-u-blox_AG_u-blox_GNSS_receiver-if00",
        tmp_path / "dev" / "ttyACM1",
    )

    resolved = runtime.resolve_serial_device(
        "auto", "gps", by_id_dir=by_id, port_infos=[]
    )

    assert resolved == str(gnss)
    assert resolved != str(samsung)


def test_legacy_samsung_ttyacm_path_is_rejected_and_gnss_is_rediscovered(
    tmp_path: Path,
) -> None:
    by_id = tmp_path / "dev" / "serial" / "by-id"
    add_by_id(
        by_id,
        "usb-SAMSUNG_SAMSUNG_Android_PHONE-if01",
        tmp_path / "dev" / "ttyACM0",
    )
    gnss = add_by_id(
        by_id,
        "usb-u-blox_AG_u-blox_GNSS_receiver-if00",
        tmp_path / "dev" / "ttyACM1",
    )

    resolved = runtime.resolve_serial_device(
        str(tmp_path / "dev" / "ttyACM0"),
        "gps",
        by_id_dir=by_id,
        port_infos=[],
    )

    assert resolved == str(gnss)


def test_external_imu_uses_stable_cp2102_identity(tmp_path: Path) -> None:
    by_id = tmp_path / "dev" / "serial" / "by-id"
    imu = add_by_id(
        by_id,
        "usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0",
        tmp_path / "dev" / "ttyUSB0",
    )

    resolved = runtime.resolve_serial_device(
        "auto", "external_imu", by_id_dir=by_id, port_infos=[]
    )

    assert resolved == str(imu)


def test_gnss_resolution_fails_closed_without_a_safe_identity(tmp_path: Path) -> None:
    by_id = tmp_path / "dev" / "serial" / "by-id"
    add_by_id(
        by_id,
        "usb-SAMSUNG_SAMSUNG_Android_PHONE-if01",
        tmp_path / "dev" / "ttyACM0",
    )

    assert (
        runtime.resolve_serial_device(
            "auto", "gps", by_id_dir=by_id, port_infos=[]
        )
        is None
    )


def test_samsung_hard_deny_wins_even_when_identity_contains_gnss(tmp_path: Path) -> None:
    by_id = tmp_path / "dev" / "serial" / "by-id"
    add_by_id(
        by_id,
        "usb-SAMSUNG_Android_GNSS_PHONE-if01",
        tmp_path / "dev" / "ttyACM0",
    )

    assert (
        runtime.resolve_serial_device(
            "auto", "gps", by_id_dir=by_id, port_infos=[]
        )
        is None
    )


def test_dangling_by_id_link_is_not_selected(tmp_path: Path) -> None:
    by_id = tmp_path / "dev" / "serial" / "by-id"
    by_id.mkdir(parents=True)
    (by_id / "usb-u-blox_GNSS_receiver-if00").symlink_to(
        tmp_path / "dev" / "missing-ttyACM1"
    )

    assert (
        runtime.resolve_serial_device(
            "auto", "gps", by_id_dir=by_id, port_infos=[]
        )
        is None
    )

def test_gnss_resolution_fails_closed_when_two_receivers_match(tmp_path: Path) -> None:
    by_id = tmp_path / "dev" / "serial" / "by-id"
    add_by_id(
        by_id,
        "usb-u-blox_GNSS_receiver_A-if00",
        tmp_path / "dev" / "ttyACM1",
    )
    add_by_id(
        by_id,
        "usb-u-blox_GNSS_receiver_B-if00",
        tmp_path / "dev" / "ttyACM2",
    )

    assert (
        runtime.resolve_serial_device(
            "auto", "gps", by_id_dir=by_id, port_infos=[]
        )
        is None
    )


def test_linux_does_not_fallback_to_numbered_tty_when_by_id_is_missing(
    tmp_path: Path,
) -> None:
    port = SimpleNamespace(
        device="/dev/ttyACM1",
        name="ttyACM1",
        description="u-blox GNSS receiver",
        hwid="USB VID:PID=1546:01A9",
        manufacturer="u-blox",
        product="GNSS receiver",
        serial_number=None,
    )

    assert (
        runtime.resolve_serial_device(
            "auto",
            "gps",
            by_id_dir=tmp_path / "missing-by-id",
            port_infos=[port],
        )
        is None
    )


def test_external_imu_resolution_fails_closed_when_cp2102_is_ambiguous(
    tmp_path: Path,
) -> None:
    by_id = tmp_path / "dev" / "serial" / "by-id"
    add_by_id(
        by_id,
        "usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_A-if00-port0",
        tmp_path / "dev" / "ttyUSB0",
    )
    add_by_id(
        by_id,
        "usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_B-if00-port0",
        tmp_path / "dev" / "ttyUSB1",
    )

    assert (
        runtime.resolve_serial_device(
            "auto", "external_imu", by_id_dir=by_id, port_infos=[]
        )
        is None
    )


def test_gnss_and_imu_cannot_resolve_aliases_of_the_same_physical_port(
    tmp_path: Path,
) -> None:
    by_id = tmp_path / "dev" / "serial" / "by-id"
    target = tmp_path / "dev" / "ttyUSB0"
    add_by_id(by_id, "usb-u-blox_GNSS_receiver-if00", target)
    add_by_id(
        by_id,
        "usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0",
        target,
    )
    args = SimpleNamespace(
        enable_gps=True,
        gps_device="auto",
        enable_external_imu=True,
        external_imu_device="auto",
    )

    runtime.resolve_serial_devices(args, by_id_dir=by_id, port_infos=[])

    assert args.gps_device is not None
    assert args.external_imu_device is None


def test_disabled_serial_roles_are_not_discovered(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        serial_devices,
        "resolve_serial_device",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    args = SimpleNamespace(
        enable_gps=False,
        gps_device="auto",
        enable_external_imu=False,
        external_imu_device="auto",
    )

    runtime.resolve_serial_devices(args, port_infos=[])

    assert calls == []


def test_serial_reader_reconnects_and_requests_exclusive_access(monkeypatch) -> None:
    calls = []
    resolve_calls = []
    reader = None

    class Port:
        def __init__(self, fail):
            self.fail = fail

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def readline(self):
            if self.fail:
                raise OSError("device unplugged")
            reader.stop_event.set()
            return b"sample\n"

    def open_port(device, baudrate, **kwargs):
        calls.append((device, baudrate, kwargs))
        return Port(fail=len(calls) == 1)

    def resolve_device():
        resolve_calls.append(True)
        if len(resolve_calls) == 1:
            return None
        return "/dev/serial/by-id/test-imu"

    monkeypatch.setattr(runtime.serial, "Serial", open_port)
    reader = runtime.SerialRateLimitedReader(
        "external_imu",
        "/dev/serial/by-id/test-imu",
        115200,
        0,
        parser=lambda line: {"line": line},
        device_resolver=resolve_device,
        reconnect_delay_s=0.1,
    )

    reader.start()
    reader.thread.join(timeout=2.0)
    reader.stop()

    assert not reader.thread.is_alive()
    assert len(resolve_calls) >= 3
    assert len(calls) >= 2
    assert calls[0][2]["exclusive"] is True
    assert reader.latest_sample()["line"] == "sample"
