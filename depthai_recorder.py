#!/usr/bin/env python3


import sys
sys.path.append("/usr/lib/python3/dist-packages")
import depthai as dai
import gi
import argparse
import sys
import time
import os
import threading
import queue
from datetime import datetime

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

# Queue to handle conversion tasks
conversion_queue = queue.Queue()

# Stop event for signaling the end of recording and conversion
stop_event = threading.Event()

def record_h265_segmented(output_dir, segment_duration=10, flip=False):
    """Records H.265 video using DepthAI, saves it to segmented files in output_dir, and adds them to a queue."""
    # Create DepthAI pipeline
    pipeline = dai.Pipeline()
    camRgb = pipeline.create(dai.node.ColorCamera)
    videoEnc = pipeline.create(dai.node.VideoEncoder)
    xout = pipeline.create(dai.node.XLinkOut)
    controlIn = pipeline.create(dai.node.XLinkIn)

    xout.setStreamName('h265')
    controlIn.setStreamName('control')

    # Camera properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)

    # Encoder properties
    videoEnc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.H265_MAIN)

    # Linking
    camRgb.video.link(videoEnc.input)
    videoEnc.bitstream.link(xout.input)
    controlIn.out.link(camRgb.inputControl)

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:

        # Control queue for real-time control
        controlQueue = device.getInputQueue("control")

        # Default camera control settings
        ctrl = dai.CameraControl()
        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
        ctrl.setAutoExposureEnable()

        # Set auto exposure region based on flip
        if flip:
            startX, startY, width, height = 300, 0, 1320, 600
        else:
            startX, startY, width, height = 0, 1080, 3840, 1080

        ctrl.setAutoExposureRegion(startX, startY, width, height)
        controlQueue.send(ctrl)

        # Output queue to get encoded data
        q = device.getOutputQueue(name="h265", maxSize=60, blocking=True)

        print("Recording... Press Ctrl+C to stop.")
        try:
            while not stop_event.is_set():

                # Generate a new file name with the current date and time
                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_file = os.path.join(output_dir, f"{current_time}.h265")

                # Record for the segment duration
                with open(output_file, 'wb') as videoFile:
                    start_time = time.time()
                    while time.time() - start_time < segment_duration and not stop_event.is_set():
                        h265Packet = q.tryGet()  # Non-blocking call to avoid being stuck
                        if h265Packet is not None:
                            h265Packet.getData().tofile(videoFile)  # Appends the packet data to the opened file

                print(f"Segment saved: {output_file}")

                # Add file to conversion queue
                conversion_queue.put(output_file)

        except KeyboardInterrupt:
            # Keyboard interrupt (Ctrl + C) detected
            print("Recording stopped.")
            stop_event.set()
        finally:
            # Ensure the last segment is saved and added to the queue
            if not videoFile.closed:
                videoFile.close()
                conversion_queue.put(output_file)

def convert_h265_to_mp4(input_file, output_file, flip):
    """Converts H.265 file to MP4 using GStreamer."""
    # Create an empty pipeline
    pipeline = Gst.Pipeline.new("convert_pipeline")
    
    # Create elements
    source = Gst.ElementFactory.make("filesrc", "source")
    h265parse = Gst.ElementFactory.make("h265parse", "h265parse")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "decoder")
    converter = Gst.ElementFactory.make("nvvidconv", "converter")
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
    parser = Gst.ElementFactory.make("h264parse", "parser")
    muxer = Gst.ElementFactory.make("qtmux", "muxer")
    sink = Gst.ElementFactory.make("filesink", "sink")
    
    # queue 추가
    queue1 = Gst.ElementFactory.make("queue", "queue1")
    queue2 = Gst.ElementFactory.make("queue", "queue2")
    
    if not pipeline or not source or not h265parse or not decoder or not converter or not encoder or not parser or not muxer or not sink:
        print("Not all elements could be created.", file=sys.stderr)
        sys.exit(1)
    
    # Set properties
    source.set_property("location", input_file)
    sink.set_property("location", output_file)
    encoder.set_property("bitrate", 25000000)
    
    if flip:
        converter.set_property("flip-method", 2)  # 2 means vertical flip
    
    # Build the pipeline
    pipeline.add(source)
    pipeline.add(h265parse)
    pipeline.add(queue1)  # h265parse 다음에 queue 추가
    pipeline.add(decoder)
    pipeline.add(converter)
    pipeline.add(queue2)  # converter 다음에 queue 추가
    pipeline.add(encoder)
    pipeline.add(parser)
    pipeline.add(muxer)
    pipeline.add(sink)
    
    # Link elements
    source.link(h265parse)
    h265parse.link(queue1)
    queue1.link(decoder)
    decoder.link(converter)
    converter.link(queue2)
    queue2.link(encoder)
    encoder.link(parser)
    parser.link(muxer)
    muxer.link(sink)
    
    # Start playing the pipeline
    pipeline.set_state(Gst.State.PLAYING)
    
    # 대기 시간 추가
    time.sleep(1)  # 1초 대기
    
    # Wait until error or EOS (End of Stream)
    bus = pipeline.get_bus()
    msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)
    
    if msg:
        if msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            print(f"Error: {err}, {debug}", file=sys.stderr)
        elif msg.type == Gst.MessageType.EOS:
            print("End of stream")
    
    # Free resources
    pipeline.set_state(Gst.State.NULL)
    print(f"Conversion finished. Saved as {output_file}")

def conversion_thread(output_dir, flip):
    """Thread function that processes the conversion queue."""
    while not stop_event.is_set() or not conversion_queue.empty():
        # Get the next file to convert from the queue
        try:
            h265_file = conversion_queue.get(timeout=1)  # Wait for a short time to allow the loop to check stop_event
        except queue.Empty:
            continue

        if h265_file is None:
            # Stop signal received
            break

        # Define output MP4 file path
        mp4_file = h265_file.replace(".h265", ".mp4")

        # Convert the H.265 file to MP4
        print(f"Converting {h265_file} to {mp4_file}...")
        convert_h265_to_mp4(h265_file, mp4_file, flip)

        # Delete the original H.265 file after conversion
        if os.path.exists(h265_file):
            os.remove(h265_file)
            print(f"Deleted {h265_file}")

        # Notify the queue that the task is done
        conversion_queue.task_done()

def convert_remaining_h265_files(output_dir, flip):
    """Converts and deletes remaining .h265 files in the output directory."""
    for filename in os.listdir(output_dir):
        if filename.endswith(".h265"):
            h265_file = os.path.join(output_dir, filename)
            mp4_file = h265_file.replace(".h265", ".mp4")
            print(f"Converting remaining file: {h265_file} to {mp4_file}...")
            convert_h265_to_mp4(h265_file, mp4_file, flip)
            if os.path.exists(h265_file):
                os.remove(h265_file)
                print(f"Deleted remaining file: {h265_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="DepthAI Video Recording and Conversion")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for segmented H.265 files")
    parser.add_argument("--flip", type=bool, default=False, help="Set True for vertical flip, False for no flip")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Start the conversion thread
    threading.Thread(target=conversion_thread, args=(args.output_dir, args.flip), daemon=True).start()

    # Start recording H.265 video in segments and enqueue them for conversion
    record_h265_segmented(args.output_dir, segment_duration=300, flip=args.flip)  # Adjust the duration as needed

    # Wait for the conversion queue to be empty
    conversion_queue.join()

    # Check for remaining .h265 files and convert them
    convert_remaining_h265_files(args.output_dir, args.flip)

if __name__ == "__main__":
    main()

