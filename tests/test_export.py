import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from sam2_trt.export import _dynamic_input_shapes, _export_one


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

    def test_legacy_export_uses_fixed_shape_options(self):
        export = Mock()
        torch = SimpleNamespace(onnx=SimpleNamespace(export=export))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "encoder.onnx"
            _export_one(
                torch,
                object(),
                (object(),),
                output,
                ["image"],
                ["embedding"],
                {},
                exporter="legacy",
            )

        options = export.call_args.kwargs
        self.assertFalse(options["dynamo"])
        self.assertIsNone(options["dynamic_axes"])
        self.assertNotIn("dynamic_shapes", options)


if __name__ == "__main__":
    unittest.main()
