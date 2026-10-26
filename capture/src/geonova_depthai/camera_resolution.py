"""Device-independent camera resolution policy (compatible with Python 3.8)."""

import math


DEFAULT_RGB_MAX_PIXELS = 1920 * 1200
# GEO-X-1/Jetson storage and encoding share the USB capture path. These budgets
# stand in for both limits, not theoretical USB bandwidth: OAK-D-LR RGB-D at
# 1920x1200 reached ~149 MB/s on 2026-08-19, but only ~18.5 FPS at 30 requested.
DEFAULT_SUPER_BUDGET_MB_S = 150.0
DEFAULT_OTHER_BUDGET_MB_S = 30.0
# RVC2 stereo matching accepts at most 1280 horizontal pixels (AR0234 example):
# https://docs.luxonis.com/hardware/platform/depth/depth-accuracy
MAX_RVC2_STEREO_WIDTH = 1280
# MJPEG limits, distinct from the H.26x 248 MPix/s limit on the same page:
# https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/video_encoder
RVC2_MJPEG_MAX_WIDTH = 16384
RVC2_MJPEG_MAX_HEIGHT = 8192
RVC2_MJPEG_MAX_PIXELS_S = 450_000_000
# The width multiple is documented under H.264/H.265; apply it conservatively
# to MJPEG as well.
ENCODER_WIDTH_MULTIPLE = 32


def enum_name(value):
    return str(getattr(value, "name", str(value).split(".")[-1]))


def _fov_area(value):
    if isinstance(value, (int, float)):
        return float(value)
    size = getattr(value, "size", value)
    if callable(size):
        size = size()
    return float(getattr(size, "width", 0) or 0) * float(getattr(size, "height", 0) or 0)


def describe_camera(feature):
    """Return JSON-safe sensor capabilities, including deduplicated configs."""
    configs = []
    for config in getattr(feature, "configs", []) or []:
        item = {
            "width": int(getattr(config, "width", 0) or 0),
            "height": int(getattr(config, "height", 0) or 0),
            "min_fps": float(getattr(config, "minFps", 0) or 0),
            "max_fps": float(getattr(config, "maxFps", 0) or 0),
            "type": enum_name(getattr(config, "type", "")),
            "fov": _fov_area(getattr(config, "fov", None)),
        }
        if item["width"] > 0 and item["height"] > 0 and item not in configs:
            configs.append(item)
    configs.sort(key=lambda c: (-c["width"] * c["height"], -c["max_fps"], c["type"], -c["fov"]))
    return {
        "socket": enum_name(getattr(feature, "socket", "")),
        "name": str(getattr(feature, "sensorName", "") or ""),
        "width": int(getattr(feature, "width", 0) or 0),
        "height": int(getattr(feature, "height", 0) or 0),
        "types": sorted(set(enum_name(t) for t in (getattr(feature, "supportedTypes", []) or []))),
        "configs": configs,
    }


def nonnegative_integer(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("{0} must be a finite integer >= 0".format(name))
    if not math.isfinite(number) or not number.is_integer() or number < 0:
        raise ValueError("{0} must be a finite integer >= 0".format(name))
    return int(number)


def validate_frame_rate(value, name="fps", allow_zero=False):
    message = "{0} must be finite and {1}".format(name, ">= 0" if allow_zero else "> 0")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(message)
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise ValueError(message)
    return number


def validate_size(width, height, prefix):
    width = nonnegative_integer(width, prefix + "_width")
    height = nonnegative_integer(height, prefix + "_height")
    if bool(width) != bool(height):
        raise ValueError("Set both {0}_width and {0}_height, or set both to 0 for auto".format(prefix))


def resolution_limits(rgb_max_pixels=0, stream_budget_mb_s=0, usb_speed="UNKNOWN"):
    rgb_max_pixels = nonnegative_integer(rgb_max_pixels, "rgb_max_pixels")
    try:
        stream_budget_mb_s = float(stream_budget_mb_s)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("camera_stream_budget_mb_s must be finite and >= 0")
    if not math.isfinite(stream_budget_mb_s) or stream_budget_mb_s < 0:
        raise ValueError("rgb_max_pixels/camera_stream_budget_mb_s must be finite and >= 0")
    return {
        "rgb_max_pixels": int(rgb_max_pixels or DEFAULT_RGB_MAX_PIXELS),
        "rgb_max_pixels_source": "configured" if rgb_max_pixels else "default",
        "stream_budget_mb_s": float(stream_budget_mb_s or (
            DEFAULT_SUPER_BUDGET_MB_S if enum_name(usb_speed) in ("SUPER", "SUPER_PLUS")
            else DEFAULT_OTHER_BUDGET_MB_S
        )),
        "stream_budget_mb_s_source": "configured" if stream_budget_mb_s else "default",
    }


def estimate_stream_mb_s(width, height, fps, depth_fps=None, depth_enabled=True,
                         rgb_transport="raw", save_confidence_map=False, confidence_transport="raw",
                         confidence_size=None):
    """Estimate decimal MB/s; mono inputs stay on-device and are not counted."""
    fps = validate_frame_rate(fps)
    depth_fps = fps if depth_fps is None else validate_frame_rate(depth_fps, "depth_fps", allow_zero=True)
    depth_fps = depth_fps or fps
    per_pixel = 1.5 * fps * (0.25 if rgb_transport == "mjpeg" else 1.0)
    confidence_rate = 0.0
    if depth_enabled:
        per_pixel += 2.0 * depth_fps
        if save_confidence_map:
            cw, ch = confidence_size if confidence_size is not None else (width, height)
            confidence_rate = cw * ch * depth_fps * (0.25 if confidence_transport == "mjpeg" else 1.0)
    return (width * height * per_pixel + confidence_rate) / 1_000_000.0


def _configs(camera, preferred_type):
    preferred = [c for c in camera["configs"] if c["type"] == preferred_type]
    return preferred or camera["configs"]


def _fps_message(camera, configs, requested, role):
    return (
        "{0} socket={1} sensor={2}: requested {3:g} FPS, supported maximum {4:g} FPS; "
        "lower {5}"
    ).format(role, camera["socket"] or "unknown", camera["name"] or "unknown", requested,
             max(c["max_fps"] for c in configs), "fps" if role == "rgb" else "depth_fps")


def _configured_fps_warnings(camera, configs, width, height, fps, role):
    if not any(c["max_fps"] for c in configs):
        return []
    matching = [c for c in configs if c["width"] >= width and c["height"] >= height]
    if any(not c["max_fps"] or c["max_fps"] >= fps for c in matching):
        return []
    if matching:
        return [_fps_message(camera, matching, fps, role)]
    return [(
        "{0} socket={1} sensor={2}: requested {3:g} FPS at configured {4}x{5}, "
        "but no reported sensor mode can supply this size"
    ).format(role, camera["socket"] or "unknown", camera["name"] or "unknown", fps, width, height)]


def _candidate(width, height, source, fov=0):
    return {"width": width, "height": height, "source": source, "fov": fov, "rejected_reasons": []}


def _largest_scaled(width, height, max_pixels, max_width=None, max_height=None, width_multiple=16):
    """Search the aligned grid; maximize area with <=1% relative aspect error."""
    aspect = width / height
    best = None
    for w in range(min(width, max_width or width) // width_multiple * width_multiple, 0, -width_multiple):
        h = min(height, max_height or height, int(max_pixels // w), int(w / (aspect * 0.99))) // 8 * 8
        if h <= 0 or abs(w / h / aspect - 1.0) > 0.01 + 1e-12:
            continue
        score = (w * h, -abs(w / h / aspect - 1.0))
        if best is None or score > best[0]:
            best = (score, (w, h))
    return best[1] if best else None


def _finish(candidates, aspect, warnings):
    fitting = [c for c in candidates if not c["rejected_reasons"]]
    if not fitting:
        raise ValueError("No camera resolution fits the limits: " + str(candidates))
    selected = max(fitting, key=lambda c: (
        c["width"] * c["height"], -abs(c["width"] / c["height"] / aspect - 1), c["fov"]
    ))
    for candidate in fitting:
        candidate["selection_reason"] = (
            "selected" if candidate is selected else "Lower area, aspect-ratio match, or FOV rank"
        )
    result = dict(selected)
    result.pop("rejected_reasons")
    result["reason"] = "Largest supported area within limits; ties prefer sensor aspect ratio then wider FOV"
    result["warnings"] = warnings
    result["candidates"] = candidates
    result["rejected_candidates"] = [c for c in candidates if c["rejected_reasons"]]
    return result


def select_rgb_size(feature, fps=15.0, depth_fps=None, width=0, height=0,
                    rgb_max_pixels=0, stream_budget_mb_s=0, usb_speed="UNKNOWN",
                    rgb_transport="raw", confidence_transport="raw", depth_enabled=True,
                    save_confidence_map=False, platform="RVC2", confidence_size=None):
    validate_size(width, height, "rgb")
    fps = validate_frame_rate(fps)
    depth_fps = fps if depth_fps is None else validate_frame_rate(depth_fps, "depth_fps", allow_zero=True)
    depth_fps = depth_fps or fps
    limits = resolution_limits(rgb_max_pixels, stream_budget_mb_s, usb_speed)
    camera = describe_camera(feature)
    sw, sh = camera["width"], camera["height"]
    known_sensor = sw > 0 and sh > 0
    if not known_sensor:
        sw, sh = 1280, 720
    configs = _configs(camera, "COLOR")
    warnings = []
    candidates = []
    stream_options = dict(fps=fps, depth_fps=depth_fps, depth_enabled=depth_enabled,
                          rgb_transport=rgb_transport, confidence_transport=confidence_transport,
                          save_confidence_map=save_confidence_map, confidence_size=confidence_size)
    rvc2 = enum_name(platform) == "RVC2"
    mjpeg = rgb_transport == "mjpeg" and rvc2
    confidence_mjpeg = rvc2 and depth_enabled and save_confidence_map and confidence_transport == "mjpeg"
    encoder_fps = (fps if mjpeg else 0) + (depth_fps if confidence_mjpeg and confidence_size is None else 0)
    fixed_encoder_rate = (confidence_size[0] * confidence_size[1] * depth_fps
                          if confidence_mjpeg and confidence_size is not None else 0)

    def encoder_frame_rejections(w, h, prefix):
        reasons = []
        if w % ENCODER_WIDTH_MULTIPLE or h % 2:
            reasons.append(prefix + "_alignment: width must be a multiple of 32 and height even")
        if w > RVC2_MJPEG_MAX_WIDTH or h > RVC2_MJPEG_MAX_HEIGHT:
            reasons.append(prefix + "_frame_size: exceeds 16384x8192")
        return reasons

    def rejections(w, h, check_sensor=True):
        reasons = []
        if check_sensor and (w > sw or h > sh):
            reasons.append("sensor_max: automatic upscaling is forbidden")
        # This is a conservative rule for the RVC2 alignment/NV12 path,
        # not a documented hardware limit.
        if depth_enabled and (w % 16 or h % 2):
            reasons.append("alignment_grid: width must be a multiple of 16 and height even")
        if w * h > limits["rgb_max_pixels"]:
            reasons.append("rgb_max_pixels: exceeds {0}".format(limits["rgb_max_pixels"]))
        rate = estimate_stream_mb_s(w, h, **stream_options)
        if rate > limits["stream_budget_mb_s"] + 1e-9:
            reasons.append("stream_budget_mb_s: {0:.3f} exceeds {1:g}".format(rate, limits["stream_budget_mb_s"]))
        if mjpeg:
            reasons.extend(encoder_frame_rejections(w, h, "mjpeg"))
        if confidence_mjpeg:
            reasons.extend(encoder_frame_rejections(*(confidence_size or (w, h)), prefix="confidence_mjpeg"))
        if (mjpeg or confidence_mjpeg) and w * h * encoder_fps + fixed_encoder_rate > RVC2_MJPEG_MAX_PIXELS_S:
            reasons.append("mjpeg_throughput: exceeds shared 450 MPix/s")
        return reasons

    if width:
        item = _candidate(width, height, "configured")
        warnings.extend(rejections(width, height, check_sensor=False))
        warnings.extend(_configured_fps_warnings(camera, configs, width, height, fps, "rgb"))
        if known_sensor and (width > sw or height > sh):
            warnings.append("Configured RGB exceeds sensor maximum; on-device upscaling may be required")
        candidates.append(item)
    else:
        for config in configs:
            w, h = config["width"], config["height"]
            source = "sensor_max" if (w, h) == (sw, sh) else "sensor_config"
            item = _candidate(w, h, source, config["fov"])
            item["max_fps"] = config["max_fps"]
            if config["max_fps"] and config["max_fps"] < fps:
                item["rejected_reasons"].append("max_fps: {0:g} < requested {1:g}".format(config["max_fps"], fps))
            candidates.append(item)
        eligible = [c for c in candidates if not c["rejected_reasons"]]
        if configs and not eligible:
            raise ValueError(_fps_message(camera, configs, fps, "rgb"))
        if not configs:
            item = _candidate(sw, sh, "sensor_max" if known_sensor else "fallback_unknown_sensor")
            candidates.append(item)
            eligible = [item]
        for item in candidates:
            item["rejected_reasons"].extend(rejections(item["width"], item["height"]))
        fixed_mb_s = estimate_stream_mb_s(0, 0, **stream_options)
        variable_options = dict(stream_options)
        if confidence_size is not None:
            variable_options["confidence_size"] = (0, 0)
        per_pixel_mb_s = estimate_stream_mb_s(1, 1, **variable_options)
        max_pixels = min(limits["rgb_max_pixels"],
                         (limits["stream_budget_mb_s"] - fixed_mb_s) / per_pixel_mb_s)
        if encoder_fps:
            max_pixels = min(max_pixels, (RVC2_MJPEG_MAX_PIXELS_S - fixed_encoder_rate) / encoder_fps)
        # Every FPS-compatible mode can supply a scaled output with its own FOV.
        for mode in eligible:
            if not mode["rejected_reasons"]:
                continue
            mw, mh = mode["width"], mode["height"]
            output_mjpeg = mjpeg or (confidence_mjpeg and confidence_size is None)
            size = _largest_scaled(mw, mh, max_pixels,
                                   RVC2_MJPEG_MAX_WIDTH if output_mjpeg else None,
                                   RVC2_MJPEG_MAX_HEIGHT if output_mjpeg else None,
                                   ENCODER_WIDTH_MULTIPLE if output_mjpeg else 16)
            if size:
                item = _candidate(
                    size[0], size[1], mode["source"] + "_scaled_to_limit" if known_sensor else "fallback_unknown_sensor",
                    mode["fov"],
                )
                item["sensor_mode"] = {"width": mw, "height": mh, "max_fps": mode.get("max_fps", 0)}
                item["rejected_reasons"] = rejections(*size)
                candidates.append(item)
    result = _finish(candidates, sw / sh, warnings)
    if width:
        result["reason"] = "Explicit RGB size retained; safety limit violations are warnings"
    result["policy"] = "configured" if width else "auto"
    result["limits"] = limits
    result["estimated_stream_mb_s"] = estimate_stream_mb_s(result["width"], result["height"], **stream_options)
    return result


def select_stereo_size(left_feature, right_feature, platform="RVC2", depth_fps=15.0, width=0, height=0):
    validate_size(width, height, "depth_input")
    depth_fps = validate_frame_rate(depth_fps, "depth_fps", allow_zero=True)
    rvc2 = enum_name(platform) == "RVC2"
    left, right = describe_camera(left_feature), describe_camera(right_feature)
    warnings, candidates = [], []
    if width:
        if rvc2 and width > MAX_RVC2_STEREO_WIDTH:
            raise ValueError("depth_input_width exceeds the RVC2 StereoDepth limit of 1280")
        for camera in (left, right):
            warnings.extend(_configured_fps_warnings(camera, _configs(camera, "MONO"), width, height,
                                                   depth_fps, "stereo"))
        result = _finish([_candidate(width, height, "configured")], width / height, warnings)
        result.update(policy="configured", reason="Explicit stereo input size")
        return result

    def sizes(camera):
        configs = _configs(camera, "MONO")
        result = set()
        for config in configs:
            if not config["max_fps"] or config["max_fps"] >= depth_fps:
                result.add((config["width"], config["height"]))
            else:
                item = _candidate(config["width"], config["height"], "sensor_config")
                item["socket"] = camera["socket"]
                item["rejected_reasons"] = ["max_fps: below requested depth FPS"]
                candidates.append(item)
        if not configs and camera["width"] > 0 and camera["height"] > 0:
            result.add((camera["width"], camera["height"]))
        if configs and not result:
            raise ValueError(_fps_message(camera, configs, depth_fps, "stereo"))
        return result

    left_sizes, right_sizes = sizes(left), sizes(right)
    # An empty set here means unknown capabilities; reported but slow configs
    # have already raised above and must never enter the unknown-sensor fallback.
    known_sizes = [modes for modes in (left_sizes, right_sizes) if modes]
    common = set.intersection(*known_sizes) if known_sizes else {(1280, 800)}
    lmax = (left["width"], left["height"]) if left_sizes else (0, 0)
    rmax = (right["width"], right["height"]) if right_sizes else (0, 0)
    source = "sensor_config" if all(v > 0 for v in lmax + rmax) else "fallback_unknown_sensor"
    if not common:
        largest = [max(modes, key=lambda size: (size[0] * size[1], size)) for modes in known_sizes]
        # Both dimensions must fit a single FPS-compatible mode on each sensor.
        fallback_size = (min(size[0] for size in largest), min(size[1] for size in largest))
        common.add(fallback_size)
        source = "fps_common_fallback"
        warnings.append((
            "No common FPS-compatible stereo config at requested {0:g} FPS; using {1}x{2} "
            "scaled/cropped from each camera's largest FPS-compatible mode"
        ).format(depth_fps, *fallback_size))
    # Missing maxima impose no extra limit on the FPS-compatible candidates.
    maximum = tuple(min((v for v in values if v > 0), default=max(size[i] for size in common))
                    for i, values in enumerate(zip(lmax, rmax)))
    for w, h in sorted(common, reverse=True):
        item_source = "sensor_max" if (w, h) == lmax == rmax else source
        item = _candidate(w, h, item_source)
        if w > maximum[0] or h > maximum[1]:
            item["rejected_reasons"].append("sensor_max: exceeds one of the stereo sensors")
        if rvc2 and w > MAX_RVC2_STEREO_WIDTH:
            item["rejected_reasons"].append("rvc2_stereo_width: exceeds 1280")
            size = _largest_scaled(w, h, w * h, MAX_RVC2_STEREO_WIDTH,
                                   int(MAX_RVC2_STEREO_WIDTH * h / w))
            if size and w <= maximum[0] and h <= maximum[1]:
                if source == "fallback_unknown_sensor":
                    scaled_source = source
                else:
                    scaled_source = (source + "_scaled_to_platform_limit" if source == "fps_common_fallback"
                                     else "scaled_to_platform_limit")
                candidates.append(_candidate(size[0], size[1], scaled_source))
        candidates.append(item)
    result = _finish(candidates, maximum[0] / maximum[1], warnings)
    result["policy"] = "auto"
    return result


def build_resolution_metadata(args):
    """Assemble additive metadata without changing legacy camera model fields."""
    rgb = dict(getattr(args, "rgb_resolution_decision", {}) or {})
    stereo = dict(getattr(args, "stereo_resolution_decision", {}) or {})
    width, height = int(getattr(args, "rgb_width", 0)), int(getattr(args, "rgb_height", 0))
    camera_width = int(getattr(args, "rgb_camera_width", width))
    camera_height = int(getattr(args, "rgb_camera_height", height))
    rgb.update(camera_output_size={"width": camera_width, "height": camera_height},
               upscaled=camera_width < width or camera_height < height)
    return {
        "policy": {"rgb": rgb.get("policy", "unknown"), "stereo": stereo.get("policy", "disabled")},
        "usb_speed": getattr(args, "usb_speed", "UNKNOWN"),
        "platform": getattr(args, "depthai_platform", "unknown"),
        "limits": rgb.get("limits", resolution_limits(
            getattr(args, "rgb_max_pixels", 0), getattr(args, "camera_stream_budget_mb_s", 0),
            getattr(args, "usb_speed", "UNKNOWN"))),
        "cameras": list(getattr(args, "connected_camera_metadata", []) or []),
        "rgb": rgb,
        "stereo": stereo,
        "aligned_depth_output": None if getattr(args, "monitor_only", False) else {"width": width, "height": height},
        "estimated_stream_mb_s": rgb.get("estimated_stream_mb_s"),
    }
