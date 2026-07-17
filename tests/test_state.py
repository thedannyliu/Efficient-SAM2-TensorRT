import unittest

from sam2_trt.state import (
    bucket_by_memory_length,
    object_batch_size,
    select_closest_conditioning,
    select_memory,
)


class StateSelectionTest(unittest.TestCase):
    def test_matches_recent_memory_and_pointer_contract(self):
        selected = select_memory(
            frame_idx=20,
            num_frames=21,
            conditioning={0: "c0"},
            non_conditioning={index: f"n{index}" for index in range(1, 20)},
        )
        self.assertEqual([frame for _, frame, _ in selected.memories], [0, 14, 15, 16, 17, 18, 19])
        self.assertEqual(len(selected.pointers), 16)
        self.assertEqual(selected.pointers[0][:2], (20, 0))
        self.assertEqual(selected.pointers[-1][:2], (15, 5))

    def test_stride_keeps_last_frame(self):
        selected = select_memory(
            frame_idx=20,
            num_frames=21,
            conditioning={0: 0},
            non_conditioning={index: index for index in range(1, 20)},
            stride=3,
        )
        self.assertEqual([frame for _, frame, _ in selected.memories], [0, 6, 9, 12, 15, 18, 19])

    def test_closest_conditioning_and_batch_buckets(self):
        selected, unselected = select_closest_conditioning(10, {0: 0, 4: 4, 8: 8, 12: 12}, 2)
        self.assertEqual(set(selected), {8, 12})
        self.assertEqual(set(unselected), {0, 4})
        self.assertEqual(object_batch_size(3), 4)
        selection = select_memory(
            frame_idx=3, num_frames=4, conditioning={0: 0}, non_conditioning={1: 1, 2: 2}
        )
        self.assertEqual(bucket_by_memory_length({7: selection}), {(3, 3): [7]})

    def test_rejects_out_of_range_object_count(self):
        with self.assertRaises(ValueError):
            object_batch_size(9)


if __name__ == "__main__":
    unittest.main()
