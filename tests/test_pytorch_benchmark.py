import unittest

from sam2_trt.pytorch_benchmark import _timing_summary


class PytorchBenchmarkTest(unittest.TestCase):
    def test_timing_summary(self):
        result = _timing_summary([1.0, 2.0, 3.0])
        self.assertEqual(result["mean_ms"], 2.0)
        self.assertEqual(result["median_ms"], 2.0)
        self.assertEqual(result["enqueue_fps"], 500.0)


if __name__ == "__main__":
    unittest.main()
