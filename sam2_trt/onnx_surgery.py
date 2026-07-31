from __future__ import annotations

from pathlib import Path


_TRACK_IMAGE_FEATURES = {
    "high_res_s0": (32, 256, 256),
    "high_res_s1": (64, 128, 128),
    "image_embedding": (256, 64, 64),
    "image_position": (256, 64, 64),
}


def rewrite_track_shared_image_batch(
    source_path: str | Path, destination_path: str | Path
) -> int:
    """Keep per-frame image features at batch one and broadcast in the graph."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(Path(source_path))
    inputs = {value.name: value for value in model.graph.input}
    missing = set(_TRACK_IMAGE_FEATURES).difference(inputs)
    if missing or "mask_memory" not in inputs:
        raise ValueError(
            f"track graph is missing inputs: {sorted(missing | ({'mask_memory'} - inputs.keys()))}"
        )

    prefix = "sam2_trt_shared_image"
    existing_names = {
        name
        for node in model.graph.node
        for name in (*node.input, *node.output)
        if name
    }
    if any(name.startswith(prefix) for name in existing_names):
        raise ValueError("track graph already has shared-image broadcast nodes")

    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name in _TRACK_IMAGE_FEATURES:
                node.input[index] = f"{name}_{prefix}_expanded"

    nodes = []
    batch_shape = f"{prefix}_memory_shape"
    batch_scalar = f"{prefix}_batch_scalar"
    batch_vector = f"{prefix}_batch_vector"
    gather_index = f"{prefix}_batch_axis"
    unsqueeze_axes = f"{prefix}_unsqueeze_axes"
    model.graph.initializer.extend(
        (
            numpy_helper.from_array(
                np.asarray(2, dtype=np.int64), gather_index
            ),
            numpy_helper.from_array(
                np.asarray([0], dtype=np.int64), unsqueeze_axes
            ),
        )
    )
    nodes.extend(
        (
            helper.make_node(
                "Shape", ["mask_memory"], [batch_shape],
                name=f"{prefix}_Shape",
            ),
            helper.make_node(
                "Gather",
                [batch_shape, gather_index],
                [batch_scalar],
                axis=0,
                name=f"{prefix}_Gather",
            ),
            helper.make_node(
                "Unsqueeze",
                [batch_scalar, unsqueeze_axes],
                [batch_vector],
                name=f"{prefix}_Unsqueeze",
            ),
        )
    )
    for name, feature_shape in _TRACK_IMAGE_FEATURES.items():
        tensor_type = inputs[name].type.tensor_type
        first_dimension = tensor_type.shape.dim[0]
        first_dimension.ClearField("dim_param")
        first_dimension.dim_value = 1
        suffix = f"{prefix}_{name}_suffix"
        target = f"{prefix}_{name}_target"
        model.graph.initializer.append(
            numpy_helper.from_array(
                np.asarray(feature_shape, dtype=np.int64), suffix
            )
        )
        nodes.extend(
            (
                helper.make_node(
                    "Concat",
                    [batch_vector, suffix],
                    [target],
                    axis=0,
                    name=f"{prefix}_{name}_Concat",
                ),
                helper.make_node(
                    "Expand",
                    [name, target],
                    [f"{name}_{prefix}_expanded"],
                    name=f"{prefix}_{name}_Expand",
                ),
            )
        )
    original_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend((*nodes, *original_nodes))
    onnx.checker.check_model(model)
    onnx.save(model, Path(destination_path))
    return len(_TRACK_IMAGE_FEATURES)


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
