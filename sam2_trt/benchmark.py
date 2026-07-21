from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def summarize_trace(path: str | Path) -> dict[str, object]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("trace has no rows")
    fields = (
        "capture_ms",
        "preprocess_ms",
        "encoder_ms",
        "tail_ms",
        "postprocess_ms",
        "queue_wait_ms",
        "color_convert_ms",
        "inference_ms",
        "mask_publish_ms",
        "callback_total_ms",
        "source_age_ms",
        "end_to_end_ms",
        "tracking_fps",
    )
    summary: dict[str, object] = {"frames": len(rows), "dropped_frames": sum(int(row.get("dropped", 0)) for row in rows)}
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows if field in row])
        if values.size:
            summary[field] = {
                "mean": float(values.mean()),
                "p50": float(np.percentile(values, 50)),
                "p90": float(np.percentile(values, 90)),
                "p99": float(np.percentile(values, 99)),
            }
    intervals = [float(row["frame_interval_ms"]) for row in rows if float(row.get("frame_interval_ms", 0.0)) > 0]
    duration = sum(intervals) / 1000.0
    summary["throughput_fps"] = len(intervals) / duration if duration > 0 else None
    return summary
