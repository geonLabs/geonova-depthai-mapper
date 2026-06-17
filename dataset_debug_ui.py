#!/usr/bin/env python3

import argparse
import csv
import json
import mimetypes
import os
import threading
import time
import urllib.parse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np


HOST = "127.0.0.1"
DEFAULT_PORT = 8088
WIDTH = 1280
HEIGHT = 720


DATASET_CACHE = OrderedDict()
DEPTH_CACHE = OrderedDict()
CACHE_LOCK = threading.Lock()
MAX_DATASET_CACHE = 6
MAX_DEPTH_CACHE = 12


def resolve_path(path_text):
    if not path_text:
        raise ValueError("Dataset path is empty.")
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def json_bytes(payload):
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def read_csv(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value, low, high):
    return max(low, min(high, value))


class Dataset:
    def __init__(self, root):
        self.root = root
        timestamps_path = root / "timestamps.csv"
        imu_path = root / "imu.csv"
        metadata_path = root / "metadata.json"

        if not timestamps_path.exists():
            raise ValueError(f"timestamps.csv not found in {root}")
        if not imu_path.exists():
            raise ValueError(f"imu.csv not found in {root}")

        self.timestamps = read_csv(timestamps_path)
        self.imu_by_frame = {}
        for row in read_csv(imu_path):
            frame_index = safe_int(row.get("frame_index"), -1)
            self.imu_by_frame.setdefault(frame_index, []).append(row)

        self.metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as file:
                self.metadata = json.load(file)

        if not self.timestamps:
            raise ValueError(f"No frames listed in {timestamps_path}")

    @property
    def frame_count(self):
        return len(self.timestamps)

    def frame(self, index):
        index = clamp(index, 0, self.frame_count - 1)
        row = self.timestamps[index]
        rgb_file = row.get("rgb_file")
        depth_file = row.get("depth_file")
        if not rgb_file or not depth_file:
            raise ValueError("timestamps.csv must include rgb_file and depth_file columns.")
        return {
            "index": index,
            "row": row,
            "rgb_path": self.root / rgb_file,
            "depth_path": self.root / depth_file,
            "imu": self.imu_by_frame.get(index, []),
        }


def get_dataset(path_text):
    root = resolve_path(path_text)
    key = str(root)
    with CACHE_LOCK:
        cached = DATASET_CACHE.get(key)
        if cached is not None:
            DATASET_CACHE.move_to_end(key)
            return cached

    dataset = Dataset(root)
    with CACHE_LOCK:
        DATASET_CACHE[key] = dataset
        DATASET_CACHE.move_to_end(key)
        while len(DATASET_CACHE) > MAX_DATASET_CACHE:
            DATASET_CACHE.popitem(last=False)
    return dataset


def get_depth_frame(dataset, index):
    frame = dataset.frame(index)
    depth_path = frame["depth_path"]
    key = str(depth_path)
    with CACHE_LOCK:
        cached = DEPTH_CACHE.get(key)
        if cached is not None:
            DEPTH_CACHE.move_to_end(key)
            return cached

    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Failed to read depth image: {depth_path}")
    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16)

    with CACHE_LOCK:
        DEPTH_CACHE[key] = depth
        DEPTH_CACHE.move_to_end(key)
        while len(DEPTH_CACHE) > MAX_DEPTH_CACHE:
            DEPTH_CACHE.popitem(last=False)
    return depth


def latest_dataset_under(root_text):
    root = resolve_path(root_text)
    if (root / "timestamps.csv").exists():
        return root
    if not root.exists():
        raise ValueError(f"Path does not exist: {root}")

    candidates = [
        path for path in root.iterdir()
        if path.is_dir() and (path / "timestamps.csv").exists()
    ]
    if not candidates:
        raise ValueError(f"No dataset folders found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def summarize_imu(rows):
    if not rows:
        return None
    row = rows[-1]
    return {
        "packet_count": len(rows),
        "accel": {
            "x": safe_float(row.get("accel_x_m_s2")),
            "y": safe_float(row.get("accel_y_m_s2")),
            "z": safe_float(row.get("accel_z_m_s2")),
        },
        "gyro": {
            "x": safe_float(row.get("gyro_x_rad_s")),
            "y": safe_float(row.get("gyro_y_rad_s")),
            "z": safe_float(row.get("gyro_z_rad_s")),
        },
    }


def robust_depth_value(depth, x, y, radius):
    exact = int(depth[y, x])
    if exact > 0 or radius <= 0:
        return {
            "depth_mm": exact,
            "exact_depth_mm": exact,
            "median_depth_mm": exact if exact > 0 else None,
            "sample_count": 1 if exact > 0 else 0,
            "radius": radius,
            "source": "exact" if exact > 0 else "invalid",
        }

    y0 = clamp(y - radius, 0, depth.shape[0] - 1)
    y1 = clamp(y + radius + 1, 1, depth.shape[0])
    x0 = clamp(x - radius, 0, depth.shape[1] - 1)
    x1 = clamp(x + radius + 1, 1, depth.shape[1])
    patch = depth[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return {
            "depth_mm": 0,
            "exact_depth_mm": exact,
            "median_depth_mm": None,
            "sample_count": 0,
            "radius": radius,
            "source": "invalid",
        }

    median = int(np.median(valid))
    return {
        "depth_mm": median,
        "exact_depth_mm": exact,
        "median_depth_mm": median,
        "sample_count": int(valid.size),
        "radius": radius,
        "source": "median",
    }


def make_depth_preview(depth, max_mm):
    max_mm = max(1, int(max_mm))
    clipped = np.clip(depth, 0, max_mm)
    scaled = (clipped * (255.0 / max_mm)).astype(np.uint8)
    color = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    ok, encoded = cv2.imencode(".png", color)
    if not ok:
        raise ValueError("Failed to encode depth preview PNG.")
    return encoded.tobytes()


def estimate_sequence_fps(dataset, index, window=30):
    rows = dataset.timestamps
    if len(rows) < 2:
        return None
    start = clamp(index - window, 0, len(rows) - 2)
    end = clamp(index + window, 1, len(rows) - 1)
    first = safe_int(rows[start].get("rgb_device_ts_ns"), None)
    last = safe_int(rows[end].get("rgb_device_ts_ns"), None)
    if first is None or last is None or last <= first:
        return None
    return (end - start) / ((last - first) / 1_000_000_000.0)


INDEX_HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DepthAI Dataset Debugger</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101414;
      --panel: #171d1c;
      --panel-2: #202827;
      --line: #31403d;
      --text: #edf4ef;
      --muted: #aab8b2;
      --accent: #58d68d;
      --accent-2: #67b7ff;
      --warn: #ffd166;
      --bad: #ff6b6b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, input, select {
      font: inherit;
      color: inherit;
    }
    .app {
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 100vh;
    }
    .topbar {
      display: grid;
      grid-template-columns: minmax(280px, 1fr) auto auto auto auto auto;
      gap: 8px;
      align-items: center;
      padding: 10px 12px;
      background: #111817;
      border-bottom: 1px solid var(--line);
    }
    .pathInput, .numberInput, select {
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      padding: 0 10px;
      outline: none;
    }
    .pathInput:focus, .numberInput:focus, select:focus {
      border-color: var(--accent);
    }
    .numberInput {
      width: 92px;
    }
    button {
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
      padding: 0 12px;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    button:disabled {
      opacity: 0.45;
      cursor: default;
    }
    .main {
      display: grid;
      grid-template-columns: 1fr 360px;
      min-height: 0;
    }
    .viewer {
      display: grid;
      grid-template-rows: 1fr auto;
      min-width: 0;
      min-height: 0;
    }
    .panes {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1px;
      min-height: 0;
      background: var(--line);
    }
    .pane {
      position: relative;
      min-width: 0;
      min-height: 0;
      background: #070909;
      overflow: hidden;
    }
    .paneTitle {
      position: absolute;
      top: 10px;
      left: 10px;
      z-index: 2;
      height: 28px;
      display: inline-flex;
      align-items: center;
      padding: 0 9px;
      border-radius: 5px;
      background: rgba(10, 14, 13, 0.8);
      border: 1px solid rgba(255,255,255,0.12);
      font-size: 13px;
      color: var(--muted);
    }
    .imageWrap {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
    }
    .debugImage {
      max-width: 100%;
      max-height: 100%;
      width: auto;
      height: auto;
      object-fit: contain;
      image-rendering: auto;
      user-select: none;
      -webkit-user-drag: none;
    }
    .crosshair {
      position: absolute;
      width: 13px;
      height: 13px;
      border: 2px solid var(--accent);
      border-radius: 50%;
      transform: translate(-50%, -50%);
      pointer-events: none;
      display: none;
      box-shadow: 0 0 0 2px rgba(0,0,0,0.65);
    }
    .crosshair::before,
    .crosshair::after {
      content: "";
      position: absolute;
      background: var(--accent);
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
    }
    .crosshair::before { width: 22px; height: 2px; }
    .crosshair::after { width: 2px; height: 22px; }
    .strip {
      display: grid;
      grid-template-columns: auto auto auto 1fr auto auto auto;
      align-items: center;
      gap: 8px;
      padding: 10px 12px;
      border-top: 1px solid var(--line);
      background: #111817;
    }
    .range {
      width: 100%;
      accent-color: var(--accent);
    }
    .side {
      border-left: 1px solid var(--line);
      background: var(--panel);
      padding: 12px;
      overflow: auto;
    }
    .section {
      padding: 12px 0;
      border-bottom: 1px solid var(--line);
    }
    .section:first-child { padding-top: 0; }
    .sectionTitle {
      margin: 0 0 9px 0;
      font-size: 13px;
      color: var(--muted);
      font-weight: 650;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    .metricGrid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .metric {
      min-height: 58px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      background: #141b1a;
    }
    .label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 5px;
    }
    .value {
      font-variant-numeric: tabular-nums;
      font-size: 18px;
      line-height: 1.2;
    }
    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.55;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      color: #d7e2dc;
    }
    .muted { color: var(--muted); }
    .accent { color: var(--accent); }
    .bad { color: var(--bad); }
    .warn { color: var(--warn); }
    .pair {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 3px 0;
      font-variant-numeric: tabular-nums;
    }
    .kbd {
      display: inline-grid;
      place-items: center;
      min-width: 22px;
      height: 22px;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #101514;
      color: var(--muted);
      font-size: 12px;
      padding: 0 6px;
    }
    .error {
      color: var(--bad);
      min-height: 20px;
      font-size: 13px;
    }
    @media (max-width: 1100px) {
      .main { grid-template-columns: 1fr; }
      .side {
        border-left: 0;
        border-top: 1px solid var(--line);
      }
    }
    @media (max-width: 760px) {
      .topbar { grid-template-columns: 1fr auto; }
      .topbar > select, .topbar > .numberInput { display: none; }
      .panes { grid-template-columns: 1fr; }
      .strip { grid-template-columns: auto auto 1fr auto; }
      #frameText, #saveMode { display: none; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <input id="pathInput" class="pathInput" placeholder="dataset folder path, e.g. image_records/2026-06-17_11-27-13" />
      <button id="latestBtn" title="Open latest dataset under this path">Latest</button>
      <button id="openBtn" title="Open dataset">Open</button>
      <select id="depthMaxSelect" title="Depth color range">
        <option value="3000">3m</option>
        <option value="5000">5m</option>
        <option value="8000" selected>8m</option>
        <option value="12000">12m</option>
      </select>
      <select id="sampleRadiusSelect" title="Depth sample radius">
        <option value="0">1 px</option>
        <option value="2">5 px</option>
        <option value="4" selected>9 px</option>
        <option value="7">15 px</option>
      </select>
      <div class="error" id="errorText"></div>
    </header>
    <main class="main">
      <section class="viewer">
        <div class="panes">
          <div class="pane" id="rgbPane">
            <div class="paneTitle">RGB</div>
            <div class="imageWrap"><img id="rgbImage" class="debugImage" alt="RGB frame" /></div>
            <div id="rgbCrosshair" class="crosshair"></div>
          </div>
          <div class="pane" id="depthPane">
            <div class="paneTitle">Depth mm</div>
            <div class="imageWrap"><img id="depthImage" class="debugImage" alt="Depth frame" /></div>
            <div id="depthCrosshair" class="crosshair"></div>
          </div>
        </div>
        <div class="strip">
          <button id="prevBtn" title="Previous frame (D)">Back</button>
          <button id="nextBtn" title="Next frame (F)">Next</button>
          <button id="firstValidBtn" title="Jump to first frame with valid depth">First Valid</button>
          <input id="frameRange" class="range" type="range" min="0" max="0" value="0" />
          <input id="frameInput" class="numberInput" type="number" min="0" value="0" />
          <span id="frameText" class="mono muted">0 / 0</span>
          <span id="saveMode" class="mono muted"><span class="kbd">D</span> back <span class="kbd">F</span> next</span>
        </div>
      </section>
      <aside class="side">
        <div class="section">
          <h2 class="sectionTitle">Point</h2>
          <div class="metricGrid">
            <div class="metric"><span class="label">Hover XY</span><span id="hoverXY" class="value">-</span></div>
            <div class="metric"><span class="label">Distance</span><span id="hoverDepth" class="value accent">-</span></div>
            <div class="metric"><span class="label">Clicked XY</span><span id="clickXY" class="value">-</span></div>
            <div class="metric"><span class="label">Clicked Distance</span><span id="clickDepth" class="value accent">-</span></div>
          </div>
        </div>
        <div class="section">
          <h2 class="sectionTitle">Frame</h2>
          <div id="frameInfo" class="mono">No dataset loaded.</div>
        </div>
        <div class="section">
          <h2 class="sectionTitle">Sync</h2>
          <div id="syncInfo" class="mono">-</div>
        </div>
        <div class="section">
          <h2 class="sectionTitle">Alignment</h2>
          <div id="alignInfo" class="mono">-</div>
        </div>
        <div class="section">
          <h2 class="sectionTitle">IMU</h2>
          <div id="imuInfo" class="mono">-</div>
        </div>
        <div class="section">
          <h2 class="sectionTitle">Files</h2>
          <div id="fileInfo" class="mono">-</div>
        </div>
      </aside>
    </main>
  </div>
  <script>
    const WIDTH = 1280;
    const HEIGHT = 720;

    const state = {
      datasetPath: "",
      frameCount: 0,
      index: 0,
      depthMaxMm: 8000,
      sampleRadius: 4,
      hoverRequest: null,
      lastHover: null,
      frame: null
    };

    const el = id => document.getElementById(id);
    const pathInput = el("pathInput");
    const errorText = el("errorText");
    const rgbImage = el("rgbImage");
    const depthImage = el("depthImage");
    const frameRange = el("frameRange");
    const frameInput = el("frameInput");
    const frameText = el("frameText");
    const rgbCrosshair = el("rgbCrosshair");
    const depthCrosshair = el("depthCrosshair");

    function qs(params) {
      return new URLSearchParams(params).toString();
    }

    async function api(path, params) {
      const res = await fetch(`${path}?${qs(params)}`);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || res.statusText);
      }
      return res.json();
    }

    function setError(message) {
      errorText.textContent = message || "";
    }

    function mediaUrl(kind, index = state.index) {
      const path = kind === "rgb" ? "/media/rgb" : "/media/depth_preview";
      return `${path}?${qs({ path: state.datasetPath, index, max_mm: state.depthMaxMm, t: Date.now() })}`;
    }

    async function openDataset(useLatest=false) {
      try {
        setError("");
        let path = pathInput.value.trim();
        if (useLatest) {
          const latest = await api("/api/latest", { path });
          path = latest.path;
          pathInput.value = path;
        }
        const data = await api("/api/dataset", { path });
        state.datasetPath = data.path;
        state.frameCount = data.frame_count;
        state.index = 0;
        frameRange.max = Math.max(0, state.frameCount - 1);
        frameRange.value = 0;
        frameInput.max = Math.max(0, state.frameCount - 1);
        frameInput.value = 0;
        await loadFrame(0);
        if (state.frame && state.frame.valid_depth_pixels === 0) {
          try {
            const firstValid = await api("/api/first_valid_depth", { path: state.datasetPath });
            if (firstValid.index > 0) await loadFrame(firstValid.index);
          } catch (err) {
            setError(`Loaded, but no valid depth frame was found: ${err.message}`);
          }
        }
      } catch (err) {
        setError(err.message);
      }
    }

    async function loadFrame(index) {
      if (!state.datasetPath) return;
      index = Math.max(0, Math.min(state.frameCount - 1, Number(index) || 0));
      state.index = index;
      frameRange.value = index;
      frameInput.value = index;
      frameText.textContent = `${index + 1} / ${state.frameCount}`;
      try {
        const frame = await api("/api/frame", { path: state.datasetPath, index });
        state.frame = frame;
        rgbImage.src = mediaUrl("rgb", index);
        depthImage.src = mediaUrl("depth", index);
        renderFrame(frame);
        clearPoint(false);
      } catch (err) {
        setError(err.message);
      }
    }

    function mmText(value, source, sampleCount) {
      if (value === null || value === undefined || Number.isNaN(value)) return "-";
      if (value === 0) return source === "invalid" ? "invalid depth" : "0 mm";
      const suffix = source === "median" ? ` median/${sampleCount}px` : "";
      return `${value} mm (${(value / 1000).toFixed(3)} m)${suffix}`;
    }

    function formatNumber(value, digits=4) {
      if (value === null || value === undefined || Number.isNaN(value)) return "-";
      return Number(value).toFixed(digits);
    }

    function renderFrame(frame) {
      const row = frame.row || {};
      el("frameInfo").textContent =
        `frame: ${frame.index}\n` +
        `stem: ${row.stem || "-"}\n` +
        `sequence rgb/depth: ${row.rgb_sequence || "-"} / ${row.depth_sequence || "-"}\n` +
        `valid depth pixels: ${frame.valid_depth_pixels ?? "-"}\n` +
        `estimated fps: ${formatNumber(frame.estimated_fps, 2)}`;

      el("syncInfo").textContent =
        `rgb-depth: ${row.rgb_depth_delta_ms || "-"} ms\n` +
        `rgb-imu: ${row.rgb_imu_delta_ms || "-"} ms\n` +
        `depth-imu: ${row.depth_imu_delta_ms || "-"} ms\n` +
        `imu packets in group: ${row.imu_packets || "-"}`;

      const alignment = frame.metadata?.depth_alignment;
      const sockets = frame.metadata?.camera_sockets;
      el("alignInfo").textContent = alignment ? (
        `enabled: ${alignment.enabled}\n` +
        `aligned to: ${alignment.aligned_to} (${alignment.aligned_to_socket})\n` +
        `rgb socket: ${sockets?.rgb || "-"}\n` +
        `stereo: ${sockets?.stereo_left || "-"} / ${sockets?.stereo_right || "-"}\n` +
        `method: ${alignment.method}\n` +
        `same pixel coords: ${alignment.depth_pixel_coordinates_match_rgb}`
      ) : "No alignment metadata in this dataset.";

      const imu = frame.imu_summary;
      el("imuInfo").textContent = imu ? (
        `packets: ${imu.packet_count}\n` +
        `accel m/s^2\n` +
        `  x ${formatNumber(imu.accel.x, 6)}\n` +
        `  y ${formatNumber(imu.accel.y, 6)}\n` +
        `  z ${formatNumber(imu.accel.z, 6)}\n` +
        `gyro rad/s\n` +
        `  x ${formatNumber(imu.gyro.x, 6)}\n` +
        `  y ${formatNumber(imu.gyro.y, 6)}\n` +
        `  z ${formatNumber(imu.gyro.z, 6)}`
      ) : "-";

      el("fileInfo").textContent =
        `${row.rgb_file || "-"}\n${row.depth_file || "-"}`;
    }

    function eventToPixel(event, image, surface) {
      let rect = image.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) {
        rect = surface.getBoundingClientRect();
      }
      if (rect.width <= 0 || rect.height <= 0) return null;
      const rawX = (event.clientX - rect.left) * WIDTH / rect.width;
      const rawY = (event.clientY - rect.top) * HEIGHT / rect.height;
      const x = Math.max(0, Math.min(WIDTH - 1, Math.floor(rawX)));
      const y = Math.max(0, Math.min(HEIGHT - 1, Math.floor(rawY)));
      return { x, y, inside: rawX >= 0 && rawX < WIDTH && rawY >= 0 && rawY < HEIGHT };
    }

    function setCrosshair(point) {
      for (const [image, crosshair] of [[rgbImage, rgbCrosshair], [depthImage, depthCrosshair]]) {
        const rect = image.getBoundingClientRect();
        const x = rect.left + point.x * rect.width / WIDTH;
        const y = rect.top + point.y * rect.height / HEIGHT;
        const parentRect = crosshair.parentElement.getBoundingClientRect();
        crosshair.style.left = `${x - parentRect.left}px`;
        crosshair.style.top = `${y - parentRect.top}px`;
        crosshair.style.display = "block";
      }
    }

    async function updatePoint(point, mode) {
      if (!point || !state.datasetPath) return;
      setCrosshair(point);
      const xyEl = mode === "click" ? el("clickXY") : el("hoverXY");
      const depthEl = mode === "click" ? el("clickDepth") : el("hoverDepth");
      xyEl.textContent = `${point.x}, ${point.y}`;
      try {
        const value = await api("/api/depth_value", {
          path: state.datasetPath,
          index: state.index,
          x: point.x,
          y: point.y,
          radius: state.sampleRadius
        });
        depthEl.textContent = mmText(value.depth_mm, value.source, value.sample_count);
        depthEl.className = value.depth_mm === 0 ? "value warn" : "value accent";
      } catch (err) {
        depthEl.textContent = "API error";
        setError(err.message);
      }
    }

    function scheduleHover(point) {
      state.lastHover = point;
      if (state.hoverRequest) return;
      state.hoverRequest = setTimeout(() => {
        state.hoverRequest = null;
        updatePoint(state.lastHover, "hover");
      }, 35);
    }

    function clearPoint(clearClick=true) {
      el("hoverXY").textContent = "-";
      el("hoverDepth").textContent = "-";
      rgbCrosshair.style.display = "none";
      depthCrosshair.style.display = "none";
      if (clearClick) {
        el("clickXY").textContent = "-";
        el("clickDepth").textContent = "-";
      }
    }

    function bindPointerSurface(surface, image) {
      surface.addEventListener("pointermove", event => {
        const point = eventToPixel(event, image, surface);
        if (point) scheduleHover(point);
      });
      surface.addEventListener("pointerleave", () => clearPoint(false));
      surface.addEventListener("pointerdown", event => {
        const point = eventToPixel(event, image, surface);
        if (point) updatePoint(point, "click");
      });
    }

    el("openBtn").addEventListener("click", () => openDataset(false));
    el("latestBtn").addEventListener("click", () => openDataset(true));
    pathInput.addEventListener("keydown", event => {
      if (event.key === "Enter") openDataset(false);
    });
    el("prevBtn").addEventListener("click", () => loadFrame(state.index - 1));
    el("nextBtn").addEventListener("click", () => loadFrame(state.index + 1));
    el("firstValidBtn").addEventListener("click", async () => {
      if (!state.datasetPath) return;
      try {
        const data = await api("/api/first_valid_depth", { path: state.datasetPath });
        await loadFrame(data.index);
      } catch (err) {
        setError(err.message);
      }
    });
    frameRange.addEventListener("input", () => loadFrame(frameRange.value));
    frameInput.addEventListener("change", () => loadFrame(frameInput.value));
    el("depthMaxSelect").addEventListener("change", event => {
      state.depthMaxMm = Number(event.target.value);
      if (state.datasetPath) depthImage.src = mediaUrl("depth");
    });
    el("sampleRadiusSelect").addEventListener("change", event => {
      state.sampleRadius = Number(event.target.value);
    });
    document.addEventListener("keydown", event => {
      const tag = event.target.tagName.toLowerCase();
      if (tag === "input" || tag === "select") return;
      if (event.key === "d" || event.key === "D" || event.key === "ArrowLeft") {
        loadFrame(state.index - 1);
      } else if (event.key === "f" || event.key === "F" || event.key === "ArrowRight") {
        loadFrame(state.index + 1);
      } else if (event.key === "Home") {
        loadFrame(0);
      } else if (event.key === "End") {
        loadFrame(state.frameCount - 1);
      }
    });
    bindPointerSurface(el("rgbPane"), rgbImage);
    bindPointerSurface(el("depthPane"), depthImage);

    const initialParams = new URLSearchParams(location.search);
    pathInput.value = initialParams.get("path") || "image_records";
    window.addEventListener("load", () => {
      if (initialParams.get("path")) {
        openDataset(false);
      } else {
        openDataset(true);
      }
    });
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, data, content_type, status=200):
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        except BrokenPipeError:
            pass

    def send_json(self, payload, status=200):
        self.send_bytes(json_bytes(payload), "application/json; charset=utf-8", status)

    def send_error_text(self, message, status=400):
        self.send_bytes(str(message).encode("utf-8"), "text/plain; charset=utf-8", status)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = {key: values[-1] for key, values in urllib.parse.parse_qs(parsed.query).items()}
        try:
            if parsed.path == "/":
                self.send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/latest":
                root = latest_dataset_under(params.get("path", "."))
                self.send_json({"path": str(root)})
            elif parsed.path == "/api/dataset":
                dataset = get_dataset(params.get("path"))
                self.send_json({
                    "path": str(dataset.root),
                    "frame_count": dataset.frame_count,
                    "metadata": dataset.metadata,
                })
            elif parsed.path == "/api/frame":
                dataset = get_dataset(params.get("path"))
                index = safe_int(params.get("index"), 0)
                frame = dataset.frame(index)
                depth = get_depth_frame(dataset, frame["index"])
                self.send_json({
                    "index": frame["index"],
                    "row": frame["row"],
                    "imu": frame["imu"],
                    "imu_summary": summarize_imu(frame["imu"]),
                    "estimated_fps": estimate_sequence_fps(dataset, frame["index"]),
                    "valid_depth_pixels": int((depth > 0).sum()),
                    "metadata": dataset.metadata,
                })
            elif parsed.path == "/api/first_valid_depth":
                dataset = get_dataset(params.get("path"))
                min_valid_pixels = clamp(safe_int(params.get("min_valid_pixels"), 1000), 1, WIDTH * HEIGHT)
                found = None
                for index in range(dataset.frame_count):
                    depth = get_depth_frame(dataset, index)
                    valid_pixels = int((depth > 0).sum())
                    if valid_pixels >= min_valid_pixels:
                        found = {"index": index, "valid_depth_pixels": valid_pixels}
                        break
                if found is None:
                    raise ValueError("No frame with enough valid depth pixels was found.")
                self.send_json(found)
            elif parsed.path == "/api/depth_value":
                dataset = get_dataset(params.get("path"))
                index = safe_int(params.get("index"), 0)
                x = clamp(safe_int(params.get("x"), 0), 0, WIDTH - 1)
                y = clamp(safe_int(params.get("y"), 0), 0, HEIGHT - 1)
                radius = clamp(safe_int(params.get("radius"), 4), 0, 20)
                depth = get_depth_frame(dataset, index)
                self.send_json({"x": x, "y": y, **robust_depth_value(depth, x, y, radius)})
            elif parsed.path == "/media/rgb":
                dataset = get_dataset(params.get("path"))
                frame = dataset.frame(safe_int(params.get("index"), 0))
                data = frame["rgb_path"].read_bytes()
                content_type = mimetypes.guess_type(frame["rgb_path"].name)[0] or "image/png"
                self.send_bytes(data, content_type)
            elif parsed.path == "/media/depth_preview":
                dataset = get_dataset(params.get("path"))
                index = safe_int(params.get("index"), 0)
                max_mm = safe_int(params.get("max_mm"), 8000)
                depth = get_depth_frame(dataset, index)
                self.send_bytes(make_depth_preview(depth, max_mm), "image/png")
            else:
                self.send_error_text("Not found", 404)
        except BrokenPipeError:
            pass
        except Exception as exc:
            self.send_error_text(exc, 400)

    def log_message(self, fmt, *args):
        return


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the DepthAI dataset debug UI.")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


def main():
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"DepthAI dataset debug UI: {url}")
    print("Open a dataset folder containing rgb/, depth_mm/, timestamps.csv, and imu.csv.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
