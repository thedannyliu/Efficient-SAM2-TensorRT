import unittest
from types import SimpleNamespace

from sam2_trt.export import _dynamic_input_shapes


class DynamicInputShapesTest(unittest.TestCase):
    def test_reuses_named_dimensions_and_bounds_them(self):
        def dimension(name, *, min, max):
            return SimpleNamespace(name=name, minimum=min, maximum=max)

        torch = SimpleNamespace(export=SimpleNamespace(Dim=dimension))
        result = _dynamic_input_shapes(
            torch,
            ("embedding", "memory", "pointer"),
            {
                "embedding": {0: "batch"},
                "memory": {0: "memory_frames", 2: "batch"},
                "pointer": {0: "pointer_frames", 1: "batch"},
            },
        )

        self.assertIs(result[0][0], result[1][2])
        self.assertIs(result[0][0], result[2][1])
        self.assertEqual((result[0][0].minimum, result[0][0].maximum), (1, 8))
        self.assertEqual((result[1][0].minimum, result[1][0].maximum), (1, 7))
        self.assertEqual((result[2][0].minimum, result[2][0].maximum), (1, 16))


if __name__ == "__main__":
    unittest.main()
