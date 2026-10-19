"""StereoDepth tuning values that can be validated without a DepthAI device."""

from __future__ import annotations

from typing import Optional, Tuple

Region = Tuple[float, float, float, float]


def parse_ae_region(text) -> Optional[Region]:
    """Return a normalised ``(x0, y0, x1, y1)`` region, or None for the whole frame.

    Coordinates follow the saved image (after ``flip``/``rotate_180``), so the
    region can be read straight off the controller preview.
    """
    value = str(text or "").strip()
    if not value:
        return None
    try:
        parts = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError(f"stereo_ae_region must be 'x0,y0,x1,y1', got {value!r}") from error
    if len(parts) != 4:
        raise ValueError(f"stereo_ae_region must have four values, got {value!r}")
    x0, y0, x1, y1 = parts
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"stereo_ae_region needs 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, got {value!r}")
    return parts


def sensor_ae_region(region: Region, size, rotate_180=False, flip=False) -> Tuple[int, int, int, int]:
    """Map a saved-image region to ``(x, y, width, height)`` sensor pixels.

    The host flips/rotates frames only when saving them, so the sensor still
    sees the unrotated scene. Both operations are their own inverse.
    """
    x0, y0, x1, y1 = region
    if flip:
        y0, y1 = 1.0 - y1, 1.0 - y0
    if rotate_180:
        x0, x1 = 1.0 - x1, 1.0 - x0
        y0, y1 = 1.0 - y1, 1.0 - y0
    width, height = (int(value) for value in size)
    start_x = min(int(round(x0 * width)), width - 1)
    start_y = min(int(round(y0 * height)), height - 1)
    end_x = max(int(round(x1 * width)), start_x + 1)
    end_y = max(int(round(y1 * height)), start_y + 1)
    return start_x, start_y, end_x - start_x, end_y - start_y


def validate_stereo_args(args) -> None:
    """Reject combinations that only fail once the camera pipeline starts."""
    shaves = int(getattr(args, "stereo_postprocessing_shaves", 0) or 0)
    slices = int(getattr(args, "stereo_postprocessing_memory_slices", 0) or 0)
    if bool(shaves) != bool(slices):
        raise ValueError(
            "Set stereo_postprocessing_shaves and stereo_postprocessing_memory_slices together, "
            "or leave both at 0 for the DepthAI default"
        )
    parse_ae_region(getattr(args, "stereo_ae_region", ""))
