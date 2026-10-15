#!/usr/bin/env python3
"""Rewrite a LightGlue-ONNX v3.0 model so ONNX Runtime's WebGPU EP runs it whole.

The published RaCo-ALIKED-LightGlue+ models target the TensorRT EP. Under the
WebGPU EP some of their ops have no kernel and fall back to the CPU, which
costs a GPU -> CPU readback around every such node (~145 per run for k1024):

  Selu        no WebGPU kernel   -> Mul(Elu(x, alpha), gamma)
  LogSoftmax  no WebGPU kernel   -> Sub(x, ReduceLogSumExp(x, axis, keepdims=1))
  GridSample  WebGPU kernel is opset 16-19 only, the model is opset 20
              -> the model is lowered to opset 19 (mode linear -> bilinear),
              which forces
  Gelu        (ai.onnx since opset 20) -> com.microsoft Gelu / FastGelu

The rewrites are exact: on a V1_01_easy pair the patched model returns the
same matches and keypoints within 6e-5 px, about 11% faster per pair.

Usage: python patch_onnx_for_webgpu.py INPUT.onnx OUTPUT.onnx
Needs the `onnx` and `numpy` packages.
"""

import argparse
import collections
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

# ONNX Selu defaults
SELU_ALPHA = 1.67326319217681884765625
SELU_GAMMA = 1.05070102214813232421875

TARGET_OPSET = 19
# Ops whose semantics changed between opset 19 and the source opset, and
# which this script does not rewrite: lowering the opset would silently
# reinterpret them, so their presence aborts the patch
OPSET_20_CHANGED_OPS = {"ConstantOfShape", "DFT", "AffineGrid", "IsInf", "IsNaN"}


def string_attribute(value):
    return value.decode() if isinstance(value, bytes) else value


def rewrite_node(node, graph):
    """Return the replacement nodes for `node`, or None to keep it."""
    attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    output = node.output[0]

    if node.op_type == "Selu":
        gamma_name = output + "_gamma"
        graph.initializer.append(numpy_helper.from_array(
            np.array(attributes.get("gamma", SELU_GAMMA), np.float32), gamma_name))
        return [
            helper.make_node("Elu", [node.input[0]], [output + "_elu"],
                             alpha=attributes.get("alpha", SELU_ALPHA)),
            helper.make_node("Mul", [output + "_elu", gamma_name], [output]),
        ]

    if node.op_type == "LogSoftmax":
        axes_name = output + "_axes"
        graph.initializer.append(numpy_helper.from_array(
            np.array([attributes.get("axis", -1)], np.int64), axes_name))
        return [
            helper.make_node("ReduceLogSumExp", [node.input[0], axes_name],
                             [output + "_logsumexp"], keepdims=1),
            helper.make_node("Sub", [node.input[0], output + "_logsumexp"], [output]),
        ]

    if node.op_type == "GridSample":
        # Opset 20 renamed the modes; opset 16-19 spell them bilinear/bicubic
        mode = string_attribute(attributes.get("mode", "linear"))
        mode = {"linear": "bilinear", "cubic": "bicubic"}.get(mode, mode)
        return [helper.make_node("GridSample", list(node.input), list(node.output),
                                 name=node.name, **{**attributes, "mode": mode})]

    if node.op_type == "Gelu":
        approximate = string_attribute(attributes.get("approximate", "none"))
        op_type = "FastGelu" if approximate == "tanh" else "Gelu"
        return [helper.make_node(op_type, list(node.input), list(node.output),
                                 domain="com.microsoft")]

    return None


def patch(model):
    graph = model.graph
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx") and opset.version != 20:
            sys.exit(f"expected an opset 20 model, got opset {opset.version}")
    blocking = OPSET_20_CHANGED_OPS & {node.op_type for node in graph.node}
    if blocking:
        sys.exit(f"cannot lower to opset {TARGET_OPSET}: model uses {sorted(blocking)}")

    rewritten = collections.Counter()
    nodes = []
    for node in graph.node:
        replacement = rewrite_node(node, graph)
        if replacement is None:
            nodes.append(node)
        else:
            nodes.extend(replacement)
            rewritten[node.op_type] += 1
    del graph.node[:]
    graph.node.extend(nodes)

    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            opset.version = TARGET_OPSET
    if not any(opset.domain == "com.microsoft" for opset in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    return rewritten


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="published model (.onnx)")
    parser.add_argument("output", help="patched model to write (.onnx)")
    arguments = parser.parse_args()

    model = onnx.load(arguments.input)
    rewritten = patch(model)
    onnx.save(model, arguments.output)
    print(f"rewrote {dict(rewritten)}, wrote {arguments.output}")


if __name__ == "__main__":
    main()
