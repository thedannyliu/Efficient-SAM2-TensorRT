import unittest

from sam2_trt.interactive import (
    all_masks_ready,
    display_to_image_point,
    drag_to_box,
    event_rate_hz,
)


class InteractiveTest(unittest.TestCase):
    def test_display_to_image_point_accounts_for_scale(self):
        self.assertEqual(display_to_image_point(320, 180, 0.5, 1280, 720), (640, 360))
        self.assertIsNone(display_to_image_point(640, 180, 0.5, 1280, 720))

    def test_drag_to_box_normalizes_and_clamps(self):
        self.assertEqual(
            drag_to_box((900, 600), (-10, 100), 1280, 720),
            (0.0, 100, 900, 600),
        )
        self.assertIsNone(drag_to_box((10, 10), (12, 30), 1280, 720))

    def test_event_rate(self):
        self.assertEqual(event_rate_hz([]), 0.0)
        self.assertAlmostEqual(event_rate_hz([1.0, 1.1, 1.2]), 10.0)

    def test_all_masks_must_arrive_before_display(self):
        self.assertFalse(all_masks_ready([1, 2], [1]))
        self.assertTrue(all_masks_ready([1, 2], [2, 1]))
        self.assertTrue(all_masks_ready([], []))


if __name__ == "__main__":
    unittest.main()
