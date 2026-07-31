import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from sam2_trt.onnx_surgery import (
    rewrite_dynamic_batch_resize,
    rewrite_track_shared_image_batch,
)


class OnnxResizeSurgeryTest(unittest.TestCase):
    def test_rewrites_track_image_inputs_to_batch_one(self):
        feature_shapes = {
            "high_res_s0": ["batch", 32, 256, 256],
            "high_res_s1": ["batch", 64, 128, 128],
            "image_embedding": ["batch", 256, 64, 64],
            "image_position": ["batch", 256, 64, 64],
        }
        graph_inputs = [
            helper.make_tensor_value_info(
                name, TensorProto.FLOAT, shape
            )
            for name, shape in feature_shapes.items()
        ]
        graph_inputs.append(
            helper.make_tensor_value_info(
                "mask_memory",
                TensorProto.FLOAT,
                [7, 4096, "batch", 64],
            )
        )
        outputs = []
        nodes = []
        for name in feature_shapes:
            output = f"{name}_output"
            outputs.append(
                helper.make_tensor_value_info(
                    output, TensorProto.FLOAT, feature_shapes[name]
                )
            )
            nodes.append(
                helper.make_node("Identity", [name], [output])
            )
        model = helper.make_model(
            helper.make_graph(
                nodes, "track", graph_inputs, outputs
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.onnx"
            destination = Path(directory) / "shared.onnx"
            onnx.save(model, source)
            self.assertEqual(
                rewrite_track_shared_image_batch(source, destination), 4
            )
            rewritten = onnx.load(destination)
        inputs = {item.name: item for item in rewritten.graph.input}
        for name in feature_shapes:
            dimensions = inputs[name].type.tensor_type.shape.dim
            self.assertEqual(dimensions[0].dim_value, 1)
            consumer = next(
                node
                for node in rewritten.graph.node
                if node.output == [f"{name}_output"]
            )
            self.assertEqual(
                consumer.input[0],
                f"{name}_sam2_trt_shared_image_expanded",
            )
        self.assertEqual(
            sum(node.op_type == "Expand" for node in rewritten.graph.node),
            4,
        )

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
