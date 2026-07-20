from __future__ import annotations

from pathlib import Path


def rewrite_dynamic_batch_resize(path: str | Path) -> int:
    """Rewrite sizes-based 4-D ONNX Resize nodes to batch-safe scales.

    PyTorch exports ``interpolate(..., size=(H, W))`` with a sampled sizes
    initializer such as ``[1, C, H, W]``. That incorrectly fixes the batch to
    one when the graph batch axis is dynamic, and TensorRT then treats linear
    Resize as operating over four dimensions. Equivalent constant scales keep
    batch and channel at 1.0 and resize only the two spatial dimensions.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    destination = Path(path)
    model = onnx.load(destination)
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {}
    for value in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dimensions = []
        for dimension in tensor_type.shape.dim:
            dimensions.append(dimension.dim_value if dimension.HasField("dim_value") else None)
        shapes[value.name] = dimensions

    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    rewritten = 0
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Resize" or len(node.input) < 4 or not node.input[3]:
            continue
        target = initializers.get(node.input[3])
        source = shapes.get(node.input[0])
        if target is None or source is None or len(target) != 4 or len(source) != 4:
            continue
        if source[2] in (None, 0) or source[3] in (None, 0):
            continue
        scales = np.asarray(
            [1.0, 1.0, float(target[2]) / source[2], float(target[3]) / source[3]],
            dtype=np.float32,
        )
        source_name = node.input[0]
        scale_name = f"sam2_trt_resize_scales_{index}"
        model.graph.initializer.append(numpy_helper.from_array(scales, scale_name))
        del node.input[:]
        node.input.extend((source_name, "", scale_name))
        rewritten += 1

    if rewritten:
        onnx.checker.check_model(model)
        onnx.save(model, destination)
    return rewritten
