#!/usr/bin/env python3
"""Optional local dataset inspection UI; never started by capture or batch jobs."""
import argparse
import mimetypes
import os
import socket
import subprocess
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from geonova_common.config_cli import parse_args_with_yaml
from .dataset import *  # Preserve the existing inspection API.

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
      grid-template-columns: minmax(280px, 1fr) auto auto auto auto auto auto;
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
    .sectionTitleRow {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 9px;
    }
    .sectionTitleRow .sectionTitle { margin: 0; }
    .statusBadge {
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 2px 6px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
    }
    .statusBadge.good { border-color: var(--accent); color: var(--accent); }
    .statusBadge.warn { border-color: var(--warn); color: var(--warn); }
    .statusBadge.bad { border-color: var(--bad); color: var(--bad); }
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
    .metric.wide {
      grid-column: span 2;
      min-height: 96px;
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
    .coordValue {
      display: block;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-variant-numeric: tabular-nums;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
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
    .calibGrid {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
    }
    .calibField {
      display: grid;
      gap: 4px;
    }
    .calibField label {
      color: var(--muted);
      font-size: 12px;
    }
    .smallInput {
      width: 100%;
      height: 32px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #101514;
      padding: 0 8px;
      outline: none;
    }
    .smallInput:focus { border-color: var(--accent); }
    .buttonRow {
      display: flex;
      gap: 8px;
      margin-top: 8px;
      flex-wrap: wrap;
    }
    .buttonRow button {
      height: 32px;
      padding: 0 10px;
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
      <input id="pathInput" class="pathInput" placeholder="dataset folder path, e.g. /data/collections/2026-06-17-11-27-13_raw" />
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
      <select id="orientationSourceSelect" title="World coordinate orientation source">
        <option value="compare" selected>Compare</option>
        <option value="ebimu">EBIMU</option>
        <option value="gps-course">GPS Course + Tilt</option>
        <option value="gps-course-level">GPS Course Level</option>
      </select>
      <div class="error" id="errorText"></div>
    </header>
    <main class="main">
      <section class="viewer">
        <div class="panes">
          <div class="pane" id="rgbPane">
            <div class="paneTitle" id="rgbPaneTitle">RGB</div>
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
            <div class="metric wide"><span class="label">Hover Absolute Coord</span><span id="hoverCoord" class="coordValue">-</span></div>
            <div class="metric wide"><span class="label">Clicked Absolute Coord</span><span id="clickCoord" class="coordValue">-</span></div>
          </div>
        </div>
        <div class="section">
          <h2 class="sectionTitle">Frame</h2>
          <div id="frameInfo" class="mono">No dataset loaded.</div>
        </div>
        <div class="section">
          <h2 class="sectionTitle">YOLO Segmentation</h2>
          <div id="yoloInfo" class="mono">Run tests/test_yolo_seg_shp.py to create results.</div>
        </div>
        <div class="section">
          <h2 class="sectionTitle">Dataset</h2>
          <div id="datasetInfo" class="mono">-</div>
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
          <div class="sectionTitleRow">
            <h2 class="sectionTitle">GPS / EBIMU</h2>
            <span id="rtkBadge" class="statusBadge">UNKNOWN</span>
          </div>
          <div id="externalInfo" class="mono">-</div>
        </div>
        <div class="section">
          <h2 class="sectionTitle">World Config</h2>
          <div id="worldInfo" class="mono">-</div>
        </div>
        <div class="section">
          <h2 class="sectionTitle">Mount Calibration</h2>
          <div class="calibGrid">
            <div class="calibField">
              <label for="mountRoll">Roll</label>
              <input id="mountRoll" class="smallInput" type="number" step="0.1" value="0" />
            </div>
            <div class="calibField">
              <label for="mountPitch">Pitch</label>
              <input id="mountPitch" class="smallInput" type="number" step="0.1" value="0" />
            </div>
            <div class="calibField">
              <label for="mountYaw">Yaw</label>
              <input id="mountYaw" class="smallInput" type="number" step="0.1" value="0" />
            </div>
          </div>
          <div class="buttonRow">
            <button id="applyMountBtn" title="Apply mount values for this UI session">Apply</button>
            <button id="saveMountBtn" title="Save mount values to metadata.json">Save</button>
          </div>
          <div class="calibGrid" style="margin-top:10px;">
            <div class="calibField">
              <label for="actualLat">Actual Lat</label>
              <input id="actualLat" class="smallInput" type="number" step="0.00000001" placeholder="37.x" />
            </div>
            <div class="calibField">
              <label for="actualLon">Actual Lon</label>
              <input id="actualLon" class="smallInput" type="number" step="0.00000001" placeholder="126.x" />
            </div>
            <div class="calibField">
              <label for="actualAlt">Actual Alt</label>
              <input id="actualAlt" class="smallInput" type="number" step="0.01" placeholder="m" />
            </div>
          </div>
          <div class="buttonRow">
            <button id="solveMountBtn" title="Use clicked point and actual coordinate to solve yaw/pitch">Calibrate & Save</button>
          </div>
          <div id="calibrationInfo" class="mono muted" style="margin-top:8px;">Click a point, enter its actual coordinate, then calibrate.</div>
        </div>
        <div class="section">
          <h2 class="sectionTitle">Files</h2>
          <div id="fileInfo" class="mono">-</div>
        </div>
      </aside>
    </main>
  </div>
  <script>
    const state = {
      datasetPath: "",
      frameCount: 0,
      index: 0,
      imageWidth: 1280,
      imageHeight: 720,
      depthMaxMm: 8000,
      sampleRadius: 4,
      orientationSource: "compare",
      hoverRequest: null,
      lastHover: null,
      lastClick: null,
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
      let path = "/media/depth_preview";
      if (kind === "rgb") path = "/media/rgb";
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
        setImageSize(data.image_size);
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
        setImageSize(frame.metadata?.image_size);
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

    function setImageSize(size) {
      if (!size || typeof size !== "object") return;
      const width = Number(size.width || size[0] || 0);
      const height = Number(size.height || size[1] || 0);
      if (width > 0 && height > 0) {
        state.imageWidth = width;
        state.imageHeight = height;
      }
    }

    function vectorText(values, digits=2) {
      if (!Array.isArray(values) || values.length === 0) return "-";
      return values.map(value => formatNumber(value, digits)).join(", ");
    }

    function gpsFixCountsText(counts) {
      if (!counts || typeof counts !== "object") return "-";
      const labels = {"0": "invalid", "1": "standalone", "2": "DGPS", "4": "fixed", "5": "float"};
      const entries = Object.entries(counts);
      if (!entries.length) return "-";
      return entries
        .sort(([a], [b]) => Number(a) - Number(b))
        .map(([quality, count]) => `${quality}:${labels[quality] || "other"}=${count}`)
        .join(", ");
    }

    function setMountInputs(values) {
      const rpy = Array.isArray(values) ? values : [0, 0, 0];
      el("mountRoll").value = Number(rpy[0] || 0).toFixed(2);
      el("mountPitch").value = Number(rpy[1] || 0).toFixed(2);
      el("mountYaw").value = Number(rpy[2] || 0).toFixed(2);
    }

    function getMountInputs() {
      return {
        roll: Number(el("mountRoll").value || 0),
        pitch: Number(el("mountPitch").value || 0),
        yaw: Number(el("mountYaw").value || 0)
      };
    }

    function singleCoordText(label, world) {
      if (!world || world.status !== "ok") {
        return `${label}\nunavailable: ${world?.reason || "-"}`;
      }
      const enu = world.enu_offset_m || {};
      const cam = world.camera_point_m || {};
      const cameraOrigin = world.camera_position_enu_m || {};
      const orientation = world.orientation || {};
      const assumptions = (world.assumptions || []).length ? `\n${world.assumptions.join("\n")}` : "";
      const courseLine = world.orientation_source?.startsWith("gps-course")
        ? `\ncourse ${formatNumber(orientation.course_deg, 2)} deg, speed ${formatNumber(orientation.speed_m_s, 2)} m/s (${orientation.course_source || "-"})`
        : "";
      const courseDeltaLine = orientation.course_frame_delta_ms !== null && orientation.course_frame_delta_ms !== undefined
        ? `\ncourse sample delta ${formatNumber(orientation.course_frame_delta_ms, 1)} ms`
        : "";
      const opticalLine =
        `\noptical heading/elev ${formatNumber(orientation.optical_heading_deg, 2)} deg, ${formatNumber(orientation.optical_elevation_deg, 2)} deg`;
      return (
        `${label}\n` +
        `lat ${formatNumber(world.latitude_deg, 8)}\n` +
        `lon ${formatNumber(world.longitude_deg, 8)}\n` +
        `alt ${formatNumber(world.altitude_m, 3)} m\n` +
        `ENU e/n/u ${formatNumber(enu.east, 3)}, ${formatNumber(enu.north, 3)}, ${formatNumber(enu.up, 3)} m\n` +
        `camera origin e/n/u ${formatNumber(cameraOrigin.east, 3)}, ${formatNumber(cameraOrigin.north, 3)}, ${formatNumber(cameraOrigin.up, 3)} m\n` +
        `cam x/y/z ${formatNumber(cam.x, 3)}, ${formatNumber(cam.y, 3)}, ${formatNumber(cam.z, 3)} m` +
        courseLine +
        courseDeltaLine +
        opticalLine +
        assumptions
      );
    }

    function coordDeltaText(a, b, label="Delta GPS-EBIMU") {
      if (!a || !b || a.status !== "ok" || b.status !== "ok") return "";
      const ea = a.enu_offset_m || {};
      const eb = b.enu_offset_m || {};
      const de = (eb.east ?? 0) - (ea.east ?? 0);
      const dn = (eb.north ?? 0) - (ea.north ?? 0);
      const du = (eb.up ?? 0) - (ea.up ?? 0);
      const norm = Math.sqrt(de * de + dn * dn + du * du);
      return `\n${label} e/n/u ${formatNumber(de, 3)}, ${formatNumber(dn, 3)}, ${formatNumber(du, 3)} m | ${formatNumber(norm, 3)} m`;
    }

    function coordText(world, worlds, source) {
      if (source === "compare" && worlds) {
        const ebimu = worlds.ebimu;
        const gpsCourse = worlds.gps_course;
        const gpsCourseLevel = worlds.gps_course_level;
        return (
          singleCoordText("EBIMU", ebimu) +
          "\n\n" +
          singleCoordText("GPS Course + Tilt", gpsCourse) +
          "\n\n" +
          singleCoordText("GPS Course Level", gpsCourseLevel) +
          coordDeltaText(ebimu, gpsCourse, "Delta Tilt-EBIMU") +
          coordDeltaText(gpsCourse, gpsCourseLevel, "Delta Level-Tilt")
        );
      }
      let label = "EBIMU";
      if (source === "gps-course") label = "GPS Course + Tilt";
      if (source === "gps-course-level") label = "GPS Course Level";
      return singleCoordText(label, world);
    }

    function renderFrame(frame) {
      const row = frame.row || {};
      const metadata = frame.metadata || {};
      el("frameInfo").textContent =
        `frame: ${frame.index}\n` +
        `stem: ${row.stem || "-"}\n` +
        `sequence rgb/depth: ${row.rgb_sequence || "-"} / ${row.depth_sequence || "-"}\n` +
        `valid depth pixels: ${frame.valid_depth_pixels ?? "-"}\n` +
        `estimated fps: ${formatNumber(frame.estimated_fps, 2)}`;

      const yolo = frame.yolo;
      const yoloDetections = yolo?.detections || [];
      const yoloPoints = yoloDetections.flatMap(item => item.points || []);
      const worldPoints = yoloPoints.filter(point => point.world_status === "ok").length;
      el("yoloInfo").textContent = yolo ? (
        `detections: ${yolo.detection_count ?? yoloDetections.length}\n` +
        `points: ${yoloPoints.length} (world: ${worldPoints})\n` +
        yoloDetections.map(item =>
          `#${item.detection_id} ${item.class_name} conf=${formatNumber(item.confidence, 3)}\n` +
          (item.points || []).map(point =>
            `  ${point.role}: (${point.pixel_x}, ${point.pixel_y}) ${point.depth_mm}mm ${point.coordinate_quality}`
          ).join("\n")
        ).join("\n")
      ) : "No YOLO result for this frame.";

      const transport = metadata.host_transport || {};
      const confidenceMeta = metadata.confidence_map || {};
      const usbSpeed = metadata.usb_speed || "legacy/unknown";
      const datasetInfo = el("datasetInfo");
      datasetInfo.textContent =
        `USB: ${usbSpeed}\n` +
        `transport rgb/depth/conf: ${transport.rgb || "legacy"} / ${transport.depth || "RAW16"} / ${transport.confidence || (confidenceMeta.saved ? "legacy" : "disabled")}\n` +
        `confidence lossy transport: ${confidenceMeta.transport_is_lossy ?? "-"}\n` +
        `requested / saved fps: ${metadata.requested_fps ?? "-"} / ${formatNumber(metadata.average_saved_fps, 2)}\n` +
        `frames rgb/conf: ${metadata.frame_count ?? state.frameCount} / ${metadata.confidence_frame_count ?? "-"}\n` +
        `GGA fixes: ${gpsFixCountsText(metadata.gps_gga_fix_quality_counts)}\n` +
        `RGB save: ${metadata.rgb_format || "-"}, depth: ${metadata.depth_format || "-"} (${metadata.depth_units || "-"})`;
      datasetInfo.classList.toggle("warn", ["HIGH", "FULL", "LOW"].includes(usbSpeed));

      el("syncInfo").textContent =
        `rgb-depth: ${row.rgb_depth_delta_ms || "-"} ms\n` +
        `rgb-imu: ${row.rgb_imu_delta_ms || "-"} ms\n` +
        `depth-imu: ${row.depth_imu_delta_ms || "-"} ms\n` +
        `camera queue lag: ${row.frame_queue_lag_ms || "-"} ms\n` +
        `gps measurement-frame: ${row.gps_frame_delta_ms || "-"} ms\n` +
        `gps receive latency: ${row.gps_receive_latency_ms || "-"} ms\n` +
        `external IMU-frame: ${row.external_imu_frame_delta_ms || "-"} ms\n` +
        `imu packets in group: ${row.imu_packets || "-"}`;

      const alignment = frame.metadata?.depth_alignment;
      const sockets = frame.metadata?.camera_sockets;
      const transform = frame.metadata?.image_transform;
      el("alignInfo").textContent = alignment ? (
        `enabled: ${alignment.enabled}\n` +
        `aligned to: ${alignment.aligned_to} (${alignment.aligned_to_socket})\n` +
        `rgb socket: ${sockets?.rgb || "-"}\n` +
        `stereo: ${sockets?.stereo_left || "-"} / ${sockets?.stereo_right || "-"}\n` +
        `method: ${alignment.method}\n` +
        `same pixel coords: ${alignment.depth_pixel_coordinates_match_rgb}\n` +
        `rotate 180: ${transform?.rotate_180 ?? "-"}\n` +
        `flip vertical: ${transform?.flip_vertical ?? "-"}`
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

      const gps = frame.gps_summary;
      const ext = frame.external_imu_summary;
      const mag = ext?.mag || {};
      const fixQuality = String(gps?.fix_quality ?? row.gps_fix_quality ?? "");
      const rtkStatus = gps?.rtk_status ?? row.gps_rtk_status ?? "unknown";
      const correctionAge = Number(gps?.differential_age_s ?? row.gps_differential_age_s);
      const maxCorrectionAge = Number(frame.metadata?.world_coordinates?.rtk_max_correction_age_s ?? 2.0);
      const hdop = Number(gps?.hdop ?? row.gps_hdop);
      const maxHdop = Number(frame.metadata?.world_coordinates?.rtk_max_hdop ?? 2.0);
      const rtkTrusted = fixQuality === "4"
        && Number.isFinite(correctionAge) && correctionAge <= maxCorrectionAge
        && Number.isFinite(hdop) && hdop <= maxHdop;
      const rtkBadge = el("rtkBadge");
      rtkBadge.textContent = fixQuality === "4"
        ? (rtkTrusted ? "RTK FIXED" : "RTK FIXED / STALE")
        : (fixQuality === "5" ? "RTK FLOAT" : String(rtkStatus).toUpperCase());
      rtkBadge.classList.toggle("good", rtkTrusted);
      rtkBadge.classList.toggle(
        "warn",
        (fixQuality === "4" && !rtkTrusted) || fixQuality === "5" || fixQuality === "2"
      );
      rtkBadge.classList.toggle("bad", !["2", "4", "5"].includes(fixQuality));
      el("externalInfo").textContent =
        `GPS sample: ${gps?.sample_index ?? row.gps_sample_index ?? "-"}\n` +
        `lat/lon: ${formatNumber(gps?.latitude_deg ?? row.gps_latitude_deg, 8)}, ${formatNumber(gps?.longitude_deg ?? row.gps_longitude_deg, 8)}\n` +
        `alt: ${formatNumber(gps?.altitude_m ?? row.gps_altitude_m, 3)} m\n` +
        `fix: ${gps?.fix_quality ?? row.gps_fix_quality ?? "-"} (${gps?.fix_quality_name ?? row.gps_fix_quality_name ?? "unknown"})\n` +
        `RTK: ${gps?.rtk_status ?? row.gps_rtk_status ?? "unknown"}, corrected=${gps?.rtk_corrected ?? row.gps_rtk_corrected ?? "-"}\n` +
        `position valid: ${gps?.position_valid ?? row.gps_position_valid ?? "-"}\n` +
        `sats/base/age: ${gps?.satellites ?? row.gps_satellites ?? "-"} / ${gps?.reference_station_id ?? row.gps_reference_station_id ?? "-"} / ${formatNumber(gps?.differential_age_s ?? row.gps_differential_age_s, 1)} s\n` +
        `course/speed: ${formatNumber(gps?.course_deg, 2)} deg / ${formatNumber((gps?.speed_knots ?? 0) * 0.514444, 2)} m/s\n` +
        `hdop: ${formatNumber(gps?.hdop, 2)}\n` +
        `EBIMU sample: ${ext?.sample_index ?? row.external_imu_sample_index ?? "-"}\n` +
        `orientation: ${ext?.orientation_format ?? "-"}\n` +
        `q x/y/z/w: ${formatNumber(ext?.q_x, 4)}, ${formatNumber(ext?.q_y, 4)}, ${formatNumber(ext?.q_z, 4)}, ${formatNumber(ext?.q_w, 4)}\n` +
        `mag x/y/z: ${formatNumber(mag.x, 2)}, ${formatNumber(mag.y, 2)}, ${formatNumber(mag.z, 2)}`;

      const world = frame.metadata?.world_coordinates;
      const cameraModel = frame.metadata?.camera_model;
      const sensorHeights = world?.sensor_heights_above_ground_m;
      el("worldInfo").textContent = world ? (
        `camera intrinsics: ${cameraModel?.intrinsics ? "ok" : "missing"}\n` +
        `height GPS/camera/IMU: ${formatNumber(sensorHeights?.gps_antenna, 2)} / ${formatNumber(sensorHeights?.camera, 2)} / ${formatNumber(sensorHeights?.external_imu, 2)} m\n` +
        `imu_from_camera r/p/y: ${vectorText(world.imu_from_camera_rpy_deg, 2)} deg\n` +
        `camera_mount r/p/y: ${vectorText(world.camera_mount_rpy_deg, 2)} deg\n` +
        `gps_to_camera e/n/u: ${vectorText(world.gps_to_camera_enu_m, 2)} m\n` +
        `gps_from_camera x/y/z: ${vectorText(world.gps_from_camera_camera_m, 2)} m\n` +
        `gps_to_camera x/y/z: ${vectorText(world.gps_to_camera_camera_m, 2)} m\n` +
        `external_imu_from_camera x/y/z: ${vectorText(world.external_imu_from_camera_camera_m, 2)} m\n` +
        `camera_from_external_imu x/y/z: ${vectorText(world.camera_from_external_imu_camera_m, 2)} m\n` +
        `mag declination: ${world.magnetic_declination_deg ?? 0} deg\n` +
        `orientation source: ${state.orientationSource}\n` +
        `world frame: ${world.local_frame || "-"}\n` +
        `lever frame: ${world.lever_arm_frame || world.camera_frame || "-"}`
      ) : "No world coordinate metadata in this dataset.";
      if (world) setMountInputs(world.camera_mount_rpy_deg || [0, 0, 0]);

      el("fileInfo").textContent =
        `RGB: ${row.rgb_file || "-"}\n` +
        `depth: ${row.depth_file || "-"}\n` +
        `confidence: ${row.confidence_file || "-"}`;
    }

    function eventToPixel(event, image, surface) {
      let rect = image.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) {
        rect = surface.getBoundingClientRect();
      }
      if (rect.width <= 0 || rect.height <= 0) return null;
      const width = state.imageWidth;
      const height = state.imageHeight;
      const rawX = (event.clientX - rect.left) * width / rect.width;
      const rawY = (event.clientY - rect.top) * height / rect.height;
      const x = Math.max(0, Math.min(width - 1, Math.floor(rawX)));
      const y = Math.max(0, Math.min(height - 1, Math.floor(rawY)));
      return { x, y, inside: rawX >= 0 && rawX < width && rawY >= 0 && rawY < height };
    }

    function setCrosshair(point) {
      for (const [image, crosshair] of [[rgbImage, rgbCrosshair], [depthImage, depthCrosshair]]) {
        const rect = image.getBoundingClientRect();
        const x = rect.left + point.x * rect.width / state.imageWidth;
        const y = rect.top + point.y * rect.height / state.imageHeight;
        const parentRect = crosshair.parentElement.getBoundingClientRect();
        crosshair.style.left = `${x - parentRect.left}px`;
        crosshair.style.top = `${y - parentRect.top}px`;
        crosshair.style.display = "block";
      }
    }

    async function updatePoint(point, mode) {
      if (!point || !state.datasetPath) return;
      setCrosshair(point);
      if (mode === "click") state.lastClick = point;
      const xyEl = mode === "click" ? el("clickXY") : el("hoverXY");
      const depthEl = mode === "click" ? el("clickDepth") : el("hoverDepth");
      const coordEl = mode === "click" ? el("clickCoord") : el("hoverCoord");
      xyEl.textContent = `${point.x}, ${point.y}`;
      try {
        const value = await api("/api/depth_value", {
          path: state.datasetPath,
          index: state.index,
          x: point.x,
          y: point.y,
          radius: state.sampleRadius,
          orientation_source: state.orientationSource
        });
        depthEl.textContent = mmText(value.depth_mm, value.source, value.sample_count);
        depthEl.className = value.depth_mm === 0 ? "value warn" : "value accent";
        coordEl.textContent = coordText(value.world, value.worlds, value.orientation_source);
        coordEl.className = value.world?.status === "ok" ? "coordValue accent" : "coordValue warn";
      } catch (err) {
        depthEl.textContent = "API error";
        coordEl.textContent = "API error";
        setError(err.message);
      }
    }

    async function refreshActivePoints() {
      if (state.lastHover) await updatePoint(state.lastHover, "hover");
      if (state.lastClick) await updatePoint(state.lastClick, "click");
    }

    async function updateMount(save=false) {
      if (!state.datasetPath) return;
      const mount = getMountInputs();
      const data = await api("/api/update_mount_rpy", {
        path: state.datasetPath,
        roll_deg: mount.roll,
        pitch_deg: mount.pitch,
        yaw_deg: mount.yaw,
        save: save ? "true" : "false"
      });
      if (state.frame) {
        state.frame.metadata = data.metadata;
        renderFrame(state.frame);
      }
      el("calibrationInfo").textContent =
        `${save ? "Saved" : "Applied"} camera_mount_rpy_deg: ` +
        data.camera_mount_rpy_deg.map(value => formatNumber(value, 3)).join(", ");
      await refreshActivePoints();
    }

    async function solveMountFromClick() {
      if (!state.datasetPath || !state.lastClick) {
        setError("Click a point before calibration.");
        return;
      }
      const lat = Number(el("actualLat").value);
      const lon = Number(el("actualLon").value);
      const alt = Number(el("actualAlt").value || 0);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
        setError("Actual latitude and longitude are required.");
        return;
      }
      const data = await api("/api/solve_mount_calibration", {
        path: state.datasetPath,
        index: state.index,
        x: state.lastClick.x,
        y: state.lastClick.y,
        radius: state.sampleRadius,
        latitude_deg: lat,
        longitude_deg: lon,
        altitude_m: alt,
        apply: "true",
        save: "true"
      });
      const proposed = data.solution.proposed_rpy_deg;
      setMountInputs(proposed);
      if (state.frame) {
        state.frame.metadata = data.metadata;
        renderFrame(state.frame);
      }
      el("calibrationInfo").textContent =
        `Saved r/p/y: ${proposed.map(value => formatNumber(value, 3)).join(", ")} deg\n` +
        `error: ${formatNumber(data.solution.error_before_m, 3)} m -> ${formatNumber(data.solution.error_after_m, 3)} m\n` +
        `course: ${formatNumber(data.solution.course.course_deg, 2)} deg (${data.solution.course.source})`;
      await refreshActivePoints();
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
      el("hoverCoord").textContent = "-";
      rgbCrosshair.style.display = "none";
      depthCrosshair.style.display = "none";
      if (clearClick) {
        el("clickXY").textContent = "-";
        el("clickDepth").textContent = "-";
        el("clickCoord").textContent = "-";
        state.lastClick = null;
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
    el("orientationSourceSelect").addEventListener("change", event => {
      state.orientationSource = event.target.value;
      if (state.frame) renderFrame(state.frame);
      if (state.lastHover) updatePoint(state.lastHover, "hover");
    });
    el("applyMountBtn").addEventListener("click", async () => {
      try {
        setError("");
        await updateMount(false);
      } catch (err) {
        setError(err.message);
      }
    });
    el("saveMountBtn").addEventListener("click", async () => {
      try {
        setError("");
        await updateMount(true);
      } catch (err) {
        setError(err.message);
      }
    });
    el("solveMountBtn").addEventListener("click", async () => {
      try {
        setError("");
        await solveMountFromClick();
      } catch (err) {
        setError(err.message);
      }
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
    pathInput.value = initialParams.get("path") || "../data";
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
                    "image_size": {"width": dataset.image_width, "height": dataset.image_height},
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
                    "gps_summary": summarize_gps(frame.get("gps")),
                    "external_imu_summary": summarize_external_imu(frame.get("external_imu")),
                    "estimated_fps": estimate_sequence_fps(dataset, frame["index"]),
                    "valid_depth_pixels": int((depth > 0).sum()),
                    "yolo": dataset.yolo_result(frame["index"]),
                    "metadata": dataset.metadata,
                })
            elif parsed.path == "/api/first_valid_depth":
                dataset = get_dataset(params.get("path"))
                min_valid_pixels = clamp(
                    safe_int(params.get("min_valid_pixels"), 1000),
                    1,
                    dataset.image_width * dataset.image_height,
                )
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
                x = clamp(safe_int(params.get("x"), 0), 0, dataset.image_width - 1)
                y = clamp(safe_int(params.get("y"), 0), 0, dataset.image_height - 1)
                radius = clamp(safe_int(params.get("radius"), 4), 0, 20)
                orientation_source = params.get("orientation_source", "compare")
                if orientation_source not in ("compare", "ebimu", "gps-course", "gps-course-level"):
                    orientation_source = "compare"
                frame = dataset.frame(index)
                depth = get_depth_frame(dataset, index)
                depth_value = robust_depth_value(depth, x, y, radius)
                worlds = {}
                if orientation_source in ("compare", "ebimu"):
                    worlds["ebimu"] = compute_world_coordinate_for_source(
                        dataset, frame, x, y, depth_value["depth_mm"], "ebimu"
                    )
                if orientation_source in ("compare", "gps-course"):
                    worlds["gps_course"] = compute_world_coordinate_for_source(
                        dataset, frame, x, y, depth_value["depth_mm"], "gps-course"
                    )
                if orientation_source in ("compare", "gps-course-level"):
                    worlds["gps_course_level"] = compute_world_coordinate_for_source(
                        dataset, frame, x, y, depth_value["depth_mm"], "gps-course-level"
                    )

                if orientation_source == "gps-course":
                    world = worlds.get("gps_course")
                elif orientation_source == "gps-course-level":
                    world = worlds.get("gps_course_level")
                elif orientation_source == "ebimu":
                    world = worlds.get("ebimu")
                else:
                    gps_world = worlds.get("gps_course_level") or worlds.get("gps_course")
                    ebimu_world = worlds.get("ebimu")
                    world = gps_world if gps_world and gps_world.get("status") == "ok" else ebimu_world

                self.send_json({
                    "x": x,
                    "y": y,
                    **depth_value,
                    "orientation_source": orientation_source,
                    "world": world,
                    "worlds": worlds,
                })
            elif parsed.path == "/api/update_mount_rpy":
                dataset = get_dataset(params.get("path"))
                roll = safe_float(params.get("roll_deg"))
                pitch = safe_float(params.get("pitch_deg"))
                yaw = safe_float(params.get("yaw_deg"))
                if roll is None or pitch is None or yaw is None:
                    raise ValueError("roll_deg, pitch_deg, and yaw_deg are required.")
                save = params.get("save", "false").lower() in ("1", "true", "yes")
                rpy = set_camera_mount_rpy(dataset, roll, pitch, yaw, save=save)
                self.send_json({
                    "status": "ok",
                    "saved": save,
                    "camera_mount_rpy_deg": rpy,
                    "metadata": dataset.metadata,
                })
            elif parsed.path == "/api/solve_mount_calibration":
                dataset = get_dataset(params.get("path"))
                index = safe_int(params.get("index"), 0)
                x = clamp(safe_int(params.get("x"), 0), 0, dataset.image_width - 1)
                y = clamp(safe_int(params.get("y"), 0), 0, dataset.image_height - 1)
                radius = clamp(safe_int(params.get("radius"), 4), 0, 20)
                actual_lat = safe_float(params.get("latitude_deg"))
                actual_lon = safe_float(params.get("longitude_deg"))
                actual_alt = safe_float(params.get("altitude_m"), 0.0)
                if actual_lat is None or actual_lon is None:
                    raise ValueError("latitude_deg and longitude_deg are required.")
                frame = dataset.frame(index)
                depth = get_depth_frame(dataset, index)
                depth_value = robust_depth_value(depth, x, y, radius)
                solution = solve_camera_mount_from_point(
                    dataset,
                    frame,
                    x,
                    y,
                    depth_value["depth_mm"],
                    actual_lat,
                    actual_lon,
                    actual_alt,
                )
                save = params.get("save", "false").lower() in ("1", "true", "yes")
                apply_solution = params.get("apply", "true").lower() in ("1", "true", "yes")
                if apply_solution:
                    calibration = {
                        "source": "single_clicked_point",
                        "updated_wall_time": datetime.now().isoformat(timespec="seconds"),
                        "frame_index": index,
                        "x": x,
                        "y": y,
                        "actual_latitude_deg": actual_lat,
                        "actual_longitude_deg": actual_lon,
                        "actual_altitude_m": actual_alt,
                        "error_before_m": solution["error_before_m"],
                        "error_after_m": solution["error_after_m"],
                        "note": "Single-point calibration updates yaw/pitch and keeps roll unchanged.",
                    }
                    set_camera_mount_rpy(dataset, *solution["proposed_rpy_deg"], save=save, calibration=calibration)
                self.send_json({
                    "status": "ok",
                    "saved": save and apply_solution,
                    "applied": apply_solution,
                    **depth_value,
                    "solution": solution,
                    "metadata": dataset.metadata,
                })
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Launch the DepthAI dataset debug UI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=HOST, help="Bind address. Use 127.0.0.1 for local-only access.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="HTTP server TCP port")
    return parse_args_with_yaml(parser, argv)


def discover_ipv4_addresses():
    addresses = []
    ignored_prefixes = ("docker", "br-", "veth")

    try:
        output = subprocess.check_output(
            ["ip", "-4", "-o", "addr", "show", "scope", "global"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in output.splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            interface = parts[1]
            if interface.startswith(ignored_prefixes):
                continue
            if "inet" not in parts:
                continue
            address = parts[parts.index("inet") + 1].split("/", 1)[0]
            if address and address not in addresses:
                addresses.append(address)
    except Exception:
        pass

    if not addresses:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                address = sock.getsockname()[0]
                if address and not address.startswith("127.") and address not in addresses:
                    addresses.append(address)
        except Exception:
            pass

    return addresses


def print_server_urls(host, port):
    print(f"DepthAI dataset debug UI bind: http://{host}:{port}")
    if host in ("0.0.0.0", ""):
        print(f"Local URL: http://127.0.0.1:{port}")
        addresses = discover_ipv4_addresses()
        if addresses:
            print("External/LAN URL candidates:")
            for address in addresses:
                print(f"  http://{address}:{port}")
        else:
            print("External/LAN URL candidates: unavailable; check `ip -4 addr`.")
        print("External access requires the client to be on a reachable network and the firewall to allow this port.")
    else:
        print(f"URL: http://{host}:{port}")


def main(argv=None):
    args = parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print_server_urls(args.host, args.port)
    print("Open a dataset folder containing rgb/, depth_mm/, timestamps.csv, and imu.csv.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
