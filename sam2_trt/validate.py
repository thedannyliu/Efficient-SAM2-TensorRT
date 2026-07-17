from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GateResult:
    passed: bool
    metric_drops: dict[str, float]
    minimum_frame_iou: float
    failed_frames: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "metric_drops": self.metric_drops,
            "minimum_frame_iou": self.minimum_frame_iou,
            "failed_frames": list(self.failed_frames),
        }


def binary_iou(reference: np.ndarray, candidate: np.ndarray) -> float:
    if reference.shape != candidate.shape:
        raise ValueError(f"mask shape mismatch: {reference.shape} != {candidate.shape}")
    lhs = np.asarray(reference, dtype=bool)
    rhs = np.asarray(candidate, dtype=bool)
    union = np.logical_or(lhs, rhs).sum(dtype=np.int64)
    if union == 0:
        return 1.0
    return float(np.logical_and(lhs, rhs).sum(dtype=np.int64) / union)


def accuracy_gate(
    baseline_report: str | Path,
    candidate_report: str | Path,
    *,
    maximum_metric_drop: float = 0.1,
    minimum_frame_iou: float = 0.999,
) -> GateResult:
    baseline = json.loads(Path(baseline_report).read_text(encoding="utf-8"))
    candidate = json.loads(Path(candidate_report).read_text(encoding="utf-8"))
    if baseline.get("metric_unit") != "percentage_points" or candidate.get("metric_unit") != "percentage_points":
        raise ValueError("both reports must declare metric_unit='percentage_points'")
    required = ("sav_jf", "image_miou")
    drops = {
        metric: float(baseline["metrics"][metric]) - float(candidate["metrics"][metric])
        for metric in required
    }

    baseline_masks_path = Path(baseline["binary_masks_npz"])
    candidate_masks_path = Path(candidate["binary_masks_npz"])
    if not baseline_masks_path.is_absolute():
        baseline_masks_path = Path(baseline_report).parent / baseline_masks_path
    if not candidate_masks_path.is_absolute():
        candidate_masks_path = Path(candidate_report).parent / candidate_masks_path

    failed: list[str] = []
    observed_minimum = 1.0
    with np.load(baseline_masks_path) as baseline_masks, np.load(candidate_masks_path) as candidate_masks:
        if set(baseline_masks.files) != set(candidate_masks.files):
            raise ValueError("baseline and candidate frame/object mask keys differ")
        for key in sorted(baseline_masks.files):
            score = binary_iou(baseline_masks[key], candidate_masks[key])
            observed_minimum = min(observed_minimum, score)
            if score < minimum_frame_iou:
                failed.append(key)

    passed = all(drop <= maximum_metric_drop for drop in drops.values()) and not failed
    return GateResult(passed, drops, observed_minimum, tuple(failed))


def write_gate_result(result: GateResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
