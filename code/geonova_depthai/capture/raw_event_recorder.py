import signal
import time
import threading

import depthai as dai

from geonova_depthai import runtime
from geonova_depthai.controller_bridge import ControllerBridge
from .cli import parse_args
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
    try:
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


def record_raw_events(args):
    # Enforce the same pixel geometry before camera metadata is read.
    args.rgb_undistort = True
    args.rgb_undistort_effective = True
    controller_bridge = ControllerBridge(args)
    serial_readers = runtime.create_serial_readers(args)
    runtime.start_serial_readers(serial_readers)
    controller_bridge.publish(serial_readers, force=True)

    dataset = None
    image_pool = None
    device = None
    recording_error = None
    try:
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
