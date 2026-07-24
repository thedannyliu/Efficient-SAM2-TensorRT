from __future__ import annotations

from collections.abc import Sequence


def display_to_image_point(
    x: float,
    y: float,
    scale: float,
    width: int,
    height: int,
) -> tuple[float, float] | None:
    if scale <= 0:
        raise ValueError("scale must be positive")
    image_x = x / scale
    image_y = y / scale
    if image_x < 0 or image_y < 0 or image_x >= width or image_y >= height:
        return None
    return image_x, image_y


def drag_to_box(
    start: tuple[float, float],
    end: tuple[float, float],
    width: int,
    height: int,
    min_size: float = 5.0,
) -> tuple[float, float, float, float] | None:
    x0 = max(0.0, min(float(width - 1), min(start[0], end[0])))
    y0 = max(0.0, min(float(height - 1), min(start[1], end[1])))
    x1 = max(0.0, min(float(width - 1), max(start[0], end[0])))
    y1 = max(0.0, min(float(height - 1), max(start[1], end[1])))
    if x1 - x0 < min_size or y1 - y0 < min_size:
        return None
    return x0, y0, x1, y1


def event_rate_hz(timestamps: Sequence[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    duration = timestamps[-1] - timestamps[0]
    return (len(timestamps) - 1) / duration if duration > 0 else 0.0

