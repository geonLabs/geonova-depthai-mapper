import signal
import time
import threading

import depthai as dai

from geonova_depthai import runtime
from geonova_depthai.controller_bridge import ControllerBridge
from .cli import apply_monitor_only_defaults, parse_args
from .raw_writer import ImageWritePool, RawEventDataset

_stop_requested = False


class ThreadSafeClockMapper:
    def __init__(self):
        self.mapper = runtime.DeviceHostClockMapper()
        self.lock = threading.Lock()

    def stamp(self, device_ts_ns):
        with self.lock:
            return self.mapper.stamp(device_ts_ns)


def _request_stop(signum, frame):  # noqa: ARG001
    global _stop_requested
    _stop_requested = True


def _drain_device_queue(q, handler, max_per_loop=256):
    if q is None:
        return 0
    count = 0
    while count < max_per_loop:
        msg = q.tryGet()
        if msg is None:
            break
        handler(msg)
        count += 1
    return count


def _close_recording_resources(
    controller_bridge,
    serial_readers,
    image_pool,
    dataset,
    device,
    error=None,
):
    """Close every recorder resource even when an earlier cleanup step fails."""
    serial_readers = serial_readers or {}
    try:
        if controller_bridge is not None:
            controller_bridge.close(serial_readers, error=error)
    finally:
        try:
            runtime.stop_serial_readers(serial_readers)
        finally:
            try:
                if image_pool is not None:
                    print("Finishing pending image writes...")
                    image_pool.close()
            finally:
                try:
                    if dataset is not None:
                        try:
                            for name, reader in serial_readers.items():
                                dataset.write_serial_samples(name, reader.drain())
                        finally:
                            dataset.close()
                        print(f"Raw dataset closed: {dataset.root}")
                        print(f"Next: python build_synced_dataset.py --dataset {dataset.root}")
                finally:
                    if device is not None:
                        try:
                            device.close()
                        except Exception:
                            pass


def _monitor_runtime_reached(args, started):
    return bool(args.max_runtime_s and time.monotonic() - started >= args.max_runtime_s)


def _monitor_controller_bridge(
    args,
    pipeline,
    outputs,
    controller_bridge,
    serial_readers,
    started,
):
    """Publish live sensor state without creating or writing a dataset."""
    rgb_q = outputs["rgb"].createOutputQueue(maxSize=max(2, args.queue_size), blocking=False)
    imu_q = outputs["imu"].createOutputQueue(maxSize=max(4, args.queue_size * 2), blocking=False)
    pipeline.start()

    last_status = time.monotonic()
    last_rgb = last_status
    last_imu = last_status
    device_stale_after_s = max(
        5.0,
        float(getattr(args, "controller_sensor_stale_after_s", 3.0)) * 2.0,
    )
    rgb_count = 0
    imu_count = 0
    print("Monitor-only mode active; no dataset will be created. Press Ctrl-C to stop.")

    def observe_rgb(message):
        nonlocal last_rgb, rgb_count
        rgb_count += 1
        last_rgb = time.monotonic()
        controller_bridge.offer_rgb(message)

    def observe_imu(message):  # noqa: ARG001
        nonlocal last_imu, imu_count
        imu_count += 1
        last_imu = time.monotonic()
        controller_bridge.observe_imu()

    while not _stop_requested:
        drained = 0
        drained += _drain_device_queue(rgb_q, observe_rgb)
        drained += _drain_device_queue(imu_q, observe_imu)
        for reader in serial_readers.values():
            # latest_sample() remains available to ControllerBridge.  Discard the
            # persistence queue so a long-running monitor cannot grow memory.
            drained += len(reader.drain())
        controller_bridge.publish(serial_readers)

        now = time.monotonic()
        if _monitor_runtime_reached(args, started):
            break
        if now - last_rgb > device_stale_after_s:
            raise RuntimeError(
                f"OAK RGB stream produced no frames for {device_stale_after_s:.1f}s"
            )
        if now - last_imu > device_stale_after_s:
            raise RuntimeError(
                f"OAK IMU stream produced no messages for {device_stale_after_s:.1f}s"
            )
        if now - last_status >= 2.0:
            print(
                f"monitor rgb={rgb_count} imu_msgs={imu_count} | "
                f"{runtime.gps_status_text(serial_readers)}"
            )
            last_status = now
        if drained == 0:
            time.sleep(0.01)


def _wait_for_monitor_camera_retry(args, controller_bridge, serial_readers, started, delay_s=2.0):
    deadline = time.monotonic() + delay_s
    while (
        not _stop_requested
        and not _monitor_runtime_reached(args, started)
        and time.monotonic() < deadline
    ):
        for reader in serial_readers.values():
            reader.drain()
        controller_bridge.publish(serial_readers)
        time.sleep(0.05)


def _run_monitor_only(args, controller_bridge, serial_readers):
    """Keep serial telemetry alive while reconnecting a failed OAK camera."""
    started = time.monotonic()
    while not _stop_requested and not _monitor_runtime_reached(args, started):
        device = None
        try:
            device = runtime.connect_depthai_device(args)
            with dai.Pipeline(device) as pipeline:
                device = pipeline.getDefaultDevice()
                controller_bridge.mark_device_connected()
                runtime.resolve_transport_options(args, device)
                outputs = runtime.configure_monitor_pipeline(pipeline, args)
                _monitor_controller_bridge(
                    args,
                    pipeline,
                    outputs,
                    controller_bridge,
                    serial_readers,
                    started,
                )
        except Exception as error:
            if _stop_requested:
                break
            controller_bridge.mark_device_disconnected(error)
            controller_bridge.publish(serial_readers, force=True)
            print(f"Monitor camera unavailable; retrying: {error}")
        finally:
            if device is not None:
                try:
                    device.close()
                except Exception:
                    pass

        if _stop_requested or _monitor_runtime_reached(args, started):
            break
        _wait_for_monitor_camera_retry(args, controller_bridge, serial_readers, started)


def record_raw_events(args):
    apply_monitor_only_defaults(args)
    # Enforce the same pixel geometry before camera metadata is read.
    args.rgb_undistort = True
    args.rgb_undistort_effective = True
    controller_bridge = None
    serial_readers = {}
    dataset = None
    image_pool = None
    device = None
    recording_error = None
    try:
        controller_bridge = ControllerBridge(args)
        serial_readers = runtime.create_serial_readers(args)
        runtime.start_serial_readers(serial_readers)
        controller_bridge.publish(serial_readers, force=True)

        if getattr(args, "monitor_only", False):
            _run_monitor_only(args, controller_bridge, serial_readers)
            return

        device = runtime.connect_depthai_device(args)

        # DepthAI v3 starts pipelines from the Pipeline object, not Device.startPipeline().
        # Keep the context alive for the entire recording loop so output queues remain valid.
        with dai.Pipeline(device) as pipeline:
            device = pipeline.getDefaultDevice()
            controller_bridge.mark_device_connected()
            runtime.resolve_transport_options(args, device)

            outputs = runtime.configure_pipeline(pipeline, args)
            camera_model = runtime.read_camera_model_metadata(device, args)
            camera_model["image_stream_undistorted"] = True
            stereo_depth_model = runtime.read_stereo_depth_metadata(device, args)

            dataset = RawEventDataset(
                args.output_dir,
                args,
                camera_model=camera_model,
                stereo_depth_model=stereo_depth_model,
            )
            image_pool = ImageWritePool(dataset, worker_count=args.writer_threads)
            mapper = ThreadSafeClockMapper()

            rgb_q = outputs["rgb"].createOutputQueue(maxSize=args.queue_size * 2, blocking=False)
            depth_q = outputs["depth"].createOutputQueue(maxSize=args.queue_size * 2, blocking=False)
            imu_q = outputs["imu"].createOutputQueue(maxSize=args.queue_size * 4, blocking=False)
            confidence_q = None
            if args.save_confidence_map and outputs.get("confidence") is not None:
                confidence_q = outputs["confidence"].createOutputQueue(maxSize=args.queue_size * 2, blocking=False)

            pipeline.start()

            started = time.monotonic()
            last_status = started
            print(f"Recording raw event dataset to: {dataset.root}")
            print("Press Ctrl-C to stop. Run build_synced_dataset.py on this folder afterwards.")

            def submit_image(stream, msg):
                if stream == "rgb":
                    controller_bridge.offer_rgb(msg)
                stamp = mapper.stamp(runtime.get_device_ts_ns(msg))
                event_index = dataset.next_index(stream)
                image_pool.submit({
                    "stream": stream,
                    "message": msg,
                    "stamp": stamp,
                    "event_index": event_index,
                })

            while not _stop_requested:
                drained = 0
                drained += _drain_device_queue(rgb_q, lambda msg: submit_image("rgb", msg))
                drained += _drain_device_queue(depth_q, lambda msg: submit_image("depth", msg))
                def write_imu(msg):
                    controller_bridge.observe_imu()
                    dataset.write_imu_message(
                        msg,
                        mapper.stamp(runtime.get_device_ts_ns(msg)),
                    )

                drained += _drain_device_queue(imu_q, write_imu)
                if confidence_q is not None:
                    drained += _drain_device_queue(confidence_q, lambda msg: submit_image("confidence", msg))

                for name, reader in serial_readers.items():
                    dataset.write_serial_samples(name, reader.drain())
                controller_bridge.publish(serial_readers)

                now = time.monotonic()
                if args.max_runtime_s and now - started >= args.max_runtime_s:
                    break
                if now - last_status >= 2.0:
                    gps_status = runtime.gps_status_text(serial_readers)
                    print(
                        "raw events "
                        f"rgb={dataset.counters['rgb']} depth={dataset.counters['depth']} "
                        f"imu_msgs={dataset.counters['imu']} gps={dataset.counters['gps']} "
                        f"ext_imu={dataset.counters['external_imu']} | {gps_status}"
                    )
                    last_status = now
                if drained == 0:
                    time.sleep(0.002)
    except Exception as error:
        recording_error = error
        raise
    finally:
        _close_recording_resources(
            controller_bridge,
            serial_readers,
            image_pool,
            dataset,
            device,
            error=recording_error,
        )


def main(argv=None):
    global _stop_requested
    _stop_requested = False
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    record_raw_events(parse_args(argv))


if __name__ == "__main__":
    main()
