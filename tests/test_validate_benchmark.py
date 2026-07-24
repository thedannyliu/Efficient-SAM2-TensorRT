import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sam2_trt.benchmark import summarize_trace
from sam2_trt.validate import accuracy_gate, binary_iou


class ValidateBenchmarkTest(unittest.TestCase):
    def test_accuracy_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:6, 2:6] = 1
            np.savez_compressed(root / "baseline.npz", frame_000_object_1=mask)
            np.savez_compressed(root / "candidate.npz", frame_000_object_1=mask)
            baseline = {
                "metric_unit": "percentage_points",
                "metrics": {"sav_jf": 80.0, "image_miou": 75.0},
                "binary_masks_npz": "baseline.npz",
            }
            candidate = {
                "metric_unit": "percentage_points",
                "metrics": {"sav_jf": 79.95, "image_miou": 74.91},
                "binary_masks_npz": "candidate.npz",
            }
            (root / "baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
            (root / "candidate.json").write_text(json.dumps(candidate), encoding="utf-8")
            result = accuracy_gate(root / "baseline.json", root / "candidate.json")
            self.assertTrue(result.passed)
            self.assertEqual(result.minimum_frame_iou, 1.0)

    def test_empty_masks_have_perfect_iou(self):
        self.assertEqual(binary_iou(np.zeros((2, 2)), np.zeros((2, 2))), 1.0)

    def test_trace_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            rows = [
                {
                    "end_to_end_ms": 10,
                    "inference_ms": 6,
                    "callback_total_ms": 10,
                    "queue_wait_ms": 1,
                    "frame_interval_ms": 0,
                    "dropped": 0,
                },
                {
                    "end_to_end_ms": 20,
                    "inference_ms": 8,
                    "callback_total_ms": 20,
                    "queue_wait_ms": 2,
                    "frame_interval_ms": 20,
                    "dropped": 1,
                },
            ]
            trace.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            summary = summarize_trace(trace)
            self.assertEqual(summary["frames"], 2)
            self.assertEqual(summary["dropped_frames"], 1)
            self.assertEqual(summary["measurement_duration_s"], 0.02)
            self.assertEqual(summary["interval_count"], 1)
            self.assertEqual(summary["throughput_fps"], 50.0)
            self.assertAlmostEqual(
                summary["processing_capacity_fps_from_mean_latency"], 1000.0 / 15.0
            )
            self.assertEqual(summary["inference_ms"]["mean"], 7.0)
            self.assertEqual(summary["queue_wait_ms"]["p50"], 1.5)


if __name__ == "__main__":
    unittest.main()
