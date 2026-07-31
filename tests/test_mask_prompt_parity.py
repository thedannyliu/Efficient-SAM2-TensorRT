import tempfile
import unittest
from pathlib import Path

import numpy as np

from sam2_trt.mask_prompt_parity import sample_names


class MaskPromptParityTest(unittest.TestCase):
    def test_discovers_matching_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.npz"
            np.savez(
                path,
                image__b=np.zeros((2, 3, 3), dtype=np.uint8),
                mask__b=np.zeros((2, 3), dtype=np.uint8),
                image__a=np.zeros((2, 3, 3), dtype=np.uint8),
                mask__a=np.zeros((2, 3), dtype=np.uint8),
                ignored=np.zeros(1),
            )
            with np.load(path) as payload:
                self.assertEqual(sample_names(payload), ("a", "b"))

    def test_rejects_unpaired_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.npz"
            np.savez(path, image__a=np.zeros((2, 3, 3), dtype=np.uint8))
            with np.load(path) as payload:
                with self.assertRaisesRegex(ValueError, "no matching"):
                    sample_names(payload)


if __name__ == "__main__":
    unittest.main()
