import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from sam2_trt.export import _dynamic_input_shapes, _export_one, patch_onnx_stability_scores


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

    def test_export_enters_requested_autocast_context(self):
        export = Mock()
        autocast = Mock()
        autocast.return_value.__enter__ = Mock()
        autocast.return_value.__exit__ = Mock(return_value=False)
        torch = SimpleNamespace(autocast=autocast, onnx=SimpleNamespace(export=export))
        tensor = SimpleNamespace(device=SimpleNamespace(type="cuda"))
        with tempfile.TemporaryDirectory() as directory:
            _export_one(
                torch,
                object(),
                (tensor,),
                Path(directory) / "prompt.onnx",
                ["embedding"],
                ["mask"],
                {},
                exporter="legacy",
                autocast_dtype="fp16",
            )

        autocast.assert_called_once_with(device_type="cuda", dtype="fp16")
        autocast.return_value.__enter__.assert_called_once_with()
        autocast.return_value.__exit__.assert_called_once()

    def test_onnx_stability_patch_matches_pixel_count_formula(self):
        import torch

        decoder = SimpleNamespace(dynamic_multimask_stability_delta=0.05)
        model = SimpleNamespace(sam_mask_decoder=decoder)
        patch_onnx_stability_scores(torch, model)
        logits = torch.linspace(-1, 1, 4 * 256 * 256).reshape(1, 4, 256, 256)
        flattened = logits.flatten(-2)
        area_i = torch.sum(flattened > 0.05, dim=-1).float()
        area_u = torch.sum(flattened > -0.05, dim=-1).float()
        reference = torch.where(area_u > 0, area_i / area_u, 1.0)

        actual = decoder._get_stability_scores(logits)

        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
