#!/usr/bin/env python3
"""Field recorder entry point.

Despite the historical filename, this now records fast per-stream event manifests
and compressed images first. Run build_synced_dataset.py afterwards to create the
synchronized timestamps.csv.
"""
from geonova_depthai.capture.raw_event_recorder import main

if __name__ == "__main__":
    main()
