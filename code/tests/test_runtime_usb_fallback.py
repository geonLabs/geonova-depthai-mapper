from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from geonova_depthai import runtime


def args(allow_usb2):
    return SimpleNamespace(usb3_retries=2, allow_usb2=allow_usb2, fps=15)


def test_usb2_allowed_retries_super_boot_failures_before_high(monkeypatch, capsys):
    device = Mock()
    device.getUsbSpeed.return_value = runtime.dai.UsbSpeed.HIGH
    factory = Mock(
        side_effect=[
            RuntimeError('X_LINK_DEVICE_NOT_FOUND'),
            RuntimeError('X_LINK_DEVICE_NOT_FOUND'),
            RuntimeError('X_LINK_DEVICE_NOT_FOUND'),
            device,
        ]
    )
    sleep = Mock()
    monkeypatch.setattr(runtime.dai, 'Device', factory)
    monkeypatch.setattr(runtime.time, 'sleep', sleep)

    assert runtime.connect_depthai_device(args(True)) is device
    assert [call.args[0] for call in factory.call_args_list] == [
        runtime.dai.UsbSpeed.SUPER,
        runtime.dai.UsbSpeed.SUPER,
        runtime.dai.UsbSpeed.SUPER,
        runtime.dai.UsbSpeed.HIGH,
    ]
    assert sleep.call_count == 3
    sleep.assert_called_with(1.0)
    assert capsys.readouterr().out.count('DepthAI USB 3.x boot failed') == 3


def test_usb2_allowed_later_super_success_skips_high(monkeypatch):
    device = Mock()
    device.getUsbSpeed.return_value = runtime.dai.UsbSpeed.SUPER
    factory = Mock(side_effect=[RuntimeError('boot failed'), device])
    sleep = Mock()
    monkeypatch.setattr(runtime.dai, 'Device', factory)
    monkeypatch.setattr(runtime.time, 'sleep', sleep)

    assert runtime.connect_depthai_device(args(True)) is device
    assert [call.args[0] for call in factory.call_args_list] == [
        runtime.dai.UsbSpeed.SUPER,
        runtime.dai.UsbSpeed.SUPER,
    ]
    sleep.assert_called_once_with(1.0)


def test_usb2_allowed_accepts_lower_speed_after_super_boot(monkeypatch):
    device = Mock()
    device.getUsbSpeed.return_value = runtime.dai.UsbSpeed.HIGH
    factory = Mock(return_value=device)
    monkeypatch.setattr(runtime.dai, 'Device', factory)

    assert runtime.connect_depthai_device(args(True)) is device
    factory.assert_called_once_with(runtime.dai.UsbSpeed.SUPER)


def test_usb2_allowed_high_boot_failure_propagates_once_after_super_attempts(monkeypatch):
    factory = Mock(
        side_effect=[
            RuntimeError('USB 3 boot failed'),
            RuntimeError('USB 3 boot failed'),
            RuntimeError('USB 3 boot failed'),
            RuntimeError('USB 2 boot failed'),
        ]
    )
    sleep = Mock()
    monkeypatch.setattr(runtime.dai, 'Device', factory)
    monkeypatch.setattr(runtime.time, 'sleep', sleep)

    with pytest.raises(RuntimeError, match='USB 2 boot failed'):
        runtime.connect_depthai_device(args(True))

    assert [call.args[0] for call in factory.call_args_list] == [
        runtime.dai.UsbSpeed.SUPER,
        runtime.dai.UsbSpeed.SUPER,
        runtime.dai.UsbSpeed.SUPER,
        runtime.dai.UsbSpeed.HIGH,
    ]
    assert sleep.call_count == 3


def test_usb3_required_preserves_boot_failure(monkeypatch):
    factory = Mock(side_effect=RuntimeError('X_LINK_DEVICE_NOT_FOUND'))
    monkeypatch.setattr(runtime.dai, 'Device', factory)
    with pytest.raises(RuntimeError, match='X_LINK_DEVICE_NOT_FOUND'):
        runtime.connect_depthai_device(args(False))
    factory.assert_called_once_with(runtime.dai.UsbSpeed.SUPER)


def test_working_usb3_is_preserved_when_usb2_is_allowed(monkeypatch):
    device = Mock()
    device.getUsbSpeed.return_value = runtime.dai.UsbSpeed.SUPER
    factory = Mock(return_value=device)
    monkeypatch.setattr(runtime.dai, 'Device', factory)
    assert runtime.connect_depthai_device(args(True)) is device
    factory.assert_called_once_with(runtime.dai.UsbSpeed.SUPER)


def test_usb3_required_retries_low_bandwidth_negotiation(monkeypatch):
    slow = Mock()
    slow.getUsbSpeed.return_value = runtime.dai.UsbSpeed.HIGH
    fast = Mock()
    fast.getUsbSpeed.return_value = runtime.dai.UsbSpeed.SUPER
    factory = Mock(side_effect=[slow, fast])
    monkeypatch.setattr(runtime.dai, 'Device', factory)
    monkeypatch.setattr(runtime.time, 'sleep', lambda seconds: None)
    assert runtime.connect_depthai_device(args(False)) is fast
    slow.close.assert_called_once()


def test_usb3_required_closes_final_low_bandwidth_device(monkeypatch):
    slow_devices = [Mock() for _ in range(3)]
    for device in slow_devices:
        device.getUsbSpeed.return_value = runtime.dai.UsbSpeed.HIGH
    factory = Mock(side_effect=slow_devices)
    monkeypatch.setattr(runtime.dai, 'Device', factory)
    monkeypatch.setattr(runtime.time, 'sleep', lambda seconds: None)

    with pytest.raises(RuntimeError, match='not USB 3.x'):
        runtime.connect_depthai_device(args(False))

    for device in slow_devices:
        device.close.assert_called_once()
