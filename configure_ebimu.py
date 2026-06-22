#!/usr/bin/env python3
"""Configure EBIMU-9DOFV5 serial output for camera/GPS recording."""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Iterable

import serial


BAUD_COMMANDS = {
    9600: "1",
    19200: "2",
    38400: "3",
    57600: "4",
    115200: "5",
    230400: "6",
    460800: "7",
    921600: "8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Configure EBIMU-9DOFV5 for synced camera/GPS recording."
    )
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial device path.")
    parser.add_argument(
        "--baudrate",
        type=int,
        default=921600,
        choices=sorted(BAUD_COMMANDS),
        help="Baudrate used after optional baudrate change.",
    )
    parser.add_argument(
        "--change-baud-from",
        type=int,
        choices=sorted(BAUD_COMMANDS),
        help=(
            "Open the device at this baudrate first, send the EBIMU baudrate "
            "change command, then reopen at --baudrate."
        ),
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=100.0,
        help="Maximum output rate target. The script chooses an EBIMU ms period that does not exceed it.",
    )
    parser.add_argument(
        "--orientation-format",
        choices=("quaternion", "euler"),
        default="quaternion",
        help="Use quaternion for production logging or euler for quick human-readable checks.",
    )
    parser.add_argument(
        "--verify-lines",
        type=int,
        default=5,
        help="Print this many output lines after starting. Use 0 to skip.",
    )
    parser.add_argument(
        "--verify-timeout",
        type=float,
        default=3.0,
        help="Seconds to wait for streaming output after <start>.",
    )
    parser.add_argument(
        "--calibrate-gyro",
        action="store_true",
        help="Run gyro calibration. Keep the sensor completely still.",
    )
    parser.add_argument(
        "--calibrate-accel",
        action="store_true",
        help="Run simple accelerometer calibration. Keep the sensor level and still.",
    )
    parser.add_argument(
        "--calibrate-mag",
        action="store_true",
        help="Run free magnetometer calibration. You will be prompted to rotate the mounted sensor.",
    )
    parser.add_argument(
        "--guided-calibration",
        action="store_true",
        help="Guide initialization and gyro/accel/magnetometer calibration, then stop streaming and exit.",
    )
    parser.add_argument(
        "--no-start",
        action="store_true",
        help="Configure the device but leave streaming stopped.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without opening the serial device.",
    )
    return parser.parse_args()


def output_period_ms(max_rate_hz: float) -> int:
    if max_rate_hz <= 0:
        raise ValueError("--rate-hz must be positive")
    return max(1, min(1000, math.ceil(1000.0 / max_rate_hz)))


def decode_bytes(data: bytes) -> str:
    return data.decode("ascii", errors="replace")


def drain(ser: serial.Serial, seconds: float = 0.1) -> bytes:
    deadline = time.monotonic() + seconds
    chunks: list[bytes] = []
    while time.monotonic() < deadline:
        waiting = ser.in_waiting
        if waiting:
            chunks.append(ser.read(waiting))
        else:
            time.sleep(0.01)
    return b"".join(chunks)


def read_until_response(ser: serial.Serial, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    chunks: list[bytes] = []
    while time.monotonic() < deadline:
        waiting = ser.in_waiting
        data = ser.read(waiting or 1)
        if data:
            chunks.append(data)
            merged = b"".join(chunks)
            if b"<ok>" in merged or b"<er>" in merged:
                return merged
        else:
            time.sleep(0.01)
    return b"".join(chunks)


def send_command(ser: serial.Serial, command: str, timeout: float = 1.5) -> bytes:
    print(f"send {command}")
    ser.write(command.encode("ascii"))
    ser.flush()
    response = read_until_response(ser, timeout)
    text = decode_bytes(response).strip()
    if text:
        print(f"  response: {text}")
    else:
        print("  response: <timeout/no response>")
    if b"<er>" in response:
        raise RuntimeError(f"EBIMU returned error for {command}")
    return response


def open_serial(port: str, baudrate: int) -> serial.Serial:
    return serial.Serial(
        port=port,
        baudrate=baudrate,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.1,
        write_timeout=1.0,
        rtscts=False,
        dsrdtr=False,
        xonxoff=False,
    )


def recommended_commands(args: argparse.Namespace) -> list[str]:
    period_ms = output_period_ms(args.rate_hz)
    orientation_command = "<sof2>" if args.orientation_format == "quaternion" else "<sof1>"
    commands = [
        "<stop>",
        "<soc1>",
        f"<sor{period_ms}>",
        orientation_command,
        "<sog1>",
        "<soa1>",
        "<som1>",
        "<sots1>",
        "<sem2>",
    ]
    if not args.no_start and not args.guided_calibration:
        commands.append("<start>")
    return commands


def print_plan(args: argparse.Namespace, commands: Iterable[str]) -> None:
    period_ms = output_period_ms(args.rate_hz)
    actual_rate = 1000.0 / period_ms
    print("EBIMU recommended setup")
    print(f"  port: {args.port}")
    print(f"  baudrate: {args.baudrate}")
    print(f"  orientation: {args.orientation_format}")
    print(f"  output period: {period_ms} ms ({actual_rate:.2f} Hz)")
    print("  commands:")
    for command in commands:
        print(f"    {command}")


def maybe_change_baudrate(args: argparse.Namespace) -> None:
    if args.change_baud_from is None or args.change_baud_from == args.baudrate:
        return

    baud_code = BAUD_COMMANDS[args.baudrate]
    print(
        f"Changing EBIMU baudrate: {args.change_baud_from} -> {args.baudrate} "
        f"with <sb{baud_code}>"
    )
    with open_serial(args.port, args.change_baud_from) as ser:
        drain(ser, 0.2)
        send_command(ser, "<stop>", timeout=1.0)
        send_command(ser, f"<sb{baud_code}>", timeout=2.0)
    time.sleep(0.8)


def run_calibrations(ser: serial.Serial, args: argparse.Namespace) -> None:
    if args.calibrate_gyro:
        print("Gyro calibration: keep the sensor completely still.")
        send_command(ser, "<cg>", timeout=4.0)

    if args.calibrate_accel:
        print("Accel calibration: keep the sensor level and still.")
        send_command(ser, "<cas>", timeout=3.0)

    if args.calibrate_mag:
        print("Mag calibration: rotate the mounted sensor through many directions.")
        print("Press Enter after rotating it enough; the script will send the final '>' byte.")
        ser.write(b"<cmf>")
        ser.flush()
        time.sleep(0.5)
        pending = drain(ser, 0.2)
        if pending:
            print(f"  response: {decode_bytes(pending).strip()}")
        input()
        ser.write(b">")
        ser.flush()
        response = read_until_response(ser, timeout=4.0)
        text = decode_bytes(response).strip()
        print(f"  response: {text or '<timeout/no response>'}")
        if b"<er>" in response:
            raise RuntimeError("EBIMU returned error for magnetometer calibration")


def wait_enter(message: str) -> None:
    try:
        input(message)
    except EOFError as exc:
        raise RuntimeError("Guided calibration requires an interactive terminal") from exc


def print_guided_intro(args: argparse.Namespace) -> None:
    period_ms = output_period_ms(args.rate_hz)
    actual_rate = 1000.0 / period_ms
    print()
    print("=== EBIMU guided initialization/calibration ===")
    print("이 모드는 EBIMU 출력 설정을 먼저 고정하고, 보정을 순서대로 진행합니다.")
    print("보정이 끝나면 <stop> 명령을 보내 스트리밍을 멈춘 상태로 종료합니다.")
    print()
    print("사용 전 확인:")
    print(f"  - 포트: {args.port}")
    print(f"  - baudrate: {args.baudrate}")
    print(f"  - 자세 출력: {args.orientation_format}")
    print(f"  - 출력 주기: {period_ms} ms ({actual_rate:.2f} Hz)")
    print("  - 가능하면 IMU를 카메라에 실제 장착한 상태로 보정하세요.")
    print("  - 지자기 보정은 금속, 자석, 큰 전류가 흐르는 장비에서 멀리 떨어져 진행하세요.")
    print()


def run_guided_calibration(ser: serial.Serial, args: argparse.Namespace) -> None:
    print_guided_intro(args)

    wait_enter("준비되면 Enter를 누르세요. EBIMU 스트리밍을 정지하고 출력 설정을 초기화합니다.")
    drain(ser, 0.2)
    for command in recommended_commands(args):
        send_command(ser, command)

    print()
    print("[1/3] Gyro calibration")
    print("센서를 완전히 정지시켜 주세요. 손으로 들고 있지 말고 바닥이나 고정된 지그 위에 두는 것이 좋습니다.")
    wait_enter("센서가 멈춰 있으면 Enter를 누르세요. <cg> 명령을 보냅니다.")
    send_command(ser, "<cg>", timeout=4.0)
    print("Gyro calibration done.")

    print()
    print("[2/3] Accelerometer calibration")
    print("센서를 수평으로 두고 움직이지 않게 유지하세요.")
    wait_enter("수평 정지 상태가 되면 Enter를 누르세요. <cas> 명령을 보냅니다.")
    send_command(ser, "<cas>", timeout=3.0)
    print("Accelerometer calibration done.")

    print()
    print("[3/3] Magnetometer calibration")
    print("이 단계는 실제 장착 상태에서 하는 것이 가장 좋습니다.")
    print("시작 후 센서를 roll/pitch/yaw 방향으로 천천히 골고루 회전시키세요.")
    wait_enter("회전시킬 준비가 되면 Enter를 누르세요. <cmf> 명령을 보내 보정을 시작합니다.")
    print("Magnetometer calibration running. 충분히 회전시킨 뒤 Enter를 누르면 종료 명령 '>'를 보냅니다.")
    ser.write(b"<cmf>")
    ser.flush()
    time.sleep(0.5)
    pending = drain(ser, 0.2)
    if pending:
        print(f"  response: {decode_bytes(pending).strip()}")
    wait_enter("충분히 회전했다면 Enter를 누르세요.")
    ser.write(b">")
    ser.flush()
    response = read_until_response(ser, timeout=4.0)
    text = decode_bytes(response).strip()
    print(f"  response: {text or '<timeout/no response>'}")
    if b"<er>" in response:
        raise RuntimeError("EBIMU returned error for magnetometer calibration")
    print("Magnetometer calibration done.")

    print()
    print("보정이 완료되었습니다. 스트리밍을 멈추고 종료합니다.")
    send_command(ser, "<stop>", timeout=2.0)
    print("Done. 다음 녹화 전에는 `python configure_ebimu.py --port /dev/ttyUSB0`로 출력 확인을 할 수 있습니다.")


def verify_output_lines(ser: serial.Serial, count: int, timeout: float) -> None:
    if count <= 0:
        return
    print(f"Reading up to {count} output line(s) for {timeout:.1f}s:")
    deadline = time.monotonic() + timeout
    buffer = bytearray()
    lines: list[bytes] = []

    while time.monotonic() < deadline and len(lines) < count:
        waiting = ser.in_waiting
        chunk = ser.read(waiting or 1)
        if not chunk:
            time.sleep(0.01)
            continue

        buffer.extend(chunk)
        while b"\n" in buffer and len(lines) < count:
            raw_line, _, rest = buffer.partition(b"\n")
            buffer = bytearray(rest)
            raw_line = raw_line.strip(b"\r")
            if raw_line.strip():
                lines.append(bytes(raw_line))

    for line in lines:
        print(f"  {decode_bytes(line).strip()}")

    if len(lines) < count:
        print(f"  only received {len(lines)} complete line(s)")
        if buffer:
            preview = decode_bytes(bytes(buffer[:240])).replace("\r", "\\r").replace("\n", "\\n")
            print(f"  partial/raw preview: {preview}")


def main() -> int:
    args = parse_args()
    commands = recommended_commands(args)
    print_plan(args, commands)

    if args.dry_run:
        return 0

    maybe_change_baudrate(args)

    with open_serial(args.port, args.baudrate) as ser:
        drain(ser, 0.3)

        if args.guided_calibration:
            run_guided_calibration(ser, args)
            return 0

        commands_without_start = [cmd for cmd in commands if cmd != "<start>"]
        for command in commands_without_start:
            send_command(ser, command)

        run_calibrations(ser, args)

        if not args.no_start:
            send_command(ser, "<start>")
            verify_output_lines(ser, args.verify_lines, args.verify_timeout)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        raise SystemExit(130)
