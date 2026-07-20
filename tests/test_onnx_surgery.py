import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from sam2_trt.onnx_surgery import rewrite_dynamic_batch_resize


class OnnxResizeSurgeryTest(unittest.TestCase):
    def test_rewrites_sizes_to_spatial_scales(self):
        graph_input = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, ["batch", 3, 4, 4]
        )
        graph_output = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, ["batch", 3, 8, 8]
        )
        sizes = numpy_helper.from_array(np.asarray([1, 3, 8, 8], dtype=np.int64), "sizes")
        resize = helper.make_node(
            "Resize",
            ["input", "", "", "sizes"],
            ["output"],
            mode="linear",
            coordinate_transformation_mode="half_pixel",
        )
        model = helper.make_model(
            helper.make_graph([resize], "resize", [graph_input], [graph_output], [sizes]),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resize.onnx"
            onnx.save(model, path)
            self.assertEqual(rewrite_dynamic_batch_resize(path), 1)
            rewritten = onnx.load(path)
        node = rewritten.graph.node[0]
        self.assertEqual(len(node.input), 3)
        scales = {
            item.name: numpy_helper.to_array(item) for item in rewritten.graph.initializer
        }[node.input[2]]
        np.testing.assert_array_equal(scales, np.asarray([1, 1, 2, 2], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
