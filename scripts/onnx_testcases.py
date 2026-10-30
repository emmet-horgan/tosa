#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["onnx>=1.15", "onnxruntime>=1.17", "numpy"]
# ///
"""Build the small ONNX models checked in under ``crates/tosa-convert/tests/onnx``.

Each case is built with ``onnx.helper``, checked with the ONNX checker, then
run through onnxruntime on seeded inputs to record the expected outputs, in
the same case format as ``scripts/models.py`` produces (``desc.json`` with
``onnx_file``, one ``.npy`` per input and ``ref-output-<i>.npy`` per output).

    uv run scripts/onnx_testcases.py             # rebuild every case
    uv run scripts/onnx_testcases.py conv_bn_relu
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import record_outputs  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "crates" / "tosa-convert" / "tests" / "onnx"
OPSET = 13


class Builder:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.nodes, self.inits, self.inputs, self.outputs = [], [], [], []

    def const(self, name: str, value: np.ndarray) -> str:
        self.inits.append(numpy_helper.from_array(np.ascontiguousarray(value), name))
        return name

    def input(self, name: str, shape, dtype=TensorProto.FLOAT) -> str:
        self.inputs.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    def output(self, name: str, shape, dtype=TensorProto.FLOAT) -> str:
        self.outputs.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    def node(self, op: str, inputs, outputs, **attrs) -> str:
        self.nodes.append(helper.make_node(op, inputs, outputs if isinstance(outputs, list) else [outputs], **attrs))
        return outputs if isinstance(outputs, str) else outputs[0]

    def model(self, name: str) -> onnx.ModelProto:
        graph = helper.make_graph(self.nodes, name, self.inputs, self.outputs, initializer=self.inits)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="tosa-convert tests")
        model.ir_version = 8
        onnx.checker.check_model(model)
        return model


def conv_bn_relu(b: Builder) -> None:
    """Conv(pads) -> BatchNormalization -> Relu -> MaxPool -> Conv(SAME_UPPER, stride 2, bias)
    -> Relu -> GlobalAveragePool -> Flatten -> Gemm -> Softmax."""
    r = b.rng
    x = b.input("input", [1, 3, 8, 8])
    w1 = b.const("w1", r.normal(0, 0.3, (4, 3, 3, 3)).astype(np.float32))
    c1 = b.node("Conv", [x, w1], "conv1", kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
    bn = b.node("BatchNormalization", [c1, b.const("scale", r.uniform(0.5, 1.5, 4).astype(np.float32)),
                                        b.const("bias", r.normal(0, 0.1, 4).astype(np.float32)),
                                        b.const("mean", r.normal(0, 0.1, 4).astype(np.float32)),
                                        b.const("var", r.uniform(0.5, 1.0, 4).astype(np.float32))], "bn", epsilon=1e-5)
    relu = b.node("Relu", [bn], "relu1")
    mp = b.node("MaxPool", [relu], "pool", kernel_shape=[2, 2], strides=[2, 2])
    w2 = b.const("w2", r.normal(0, 0.3, (6, 4, 3, 3)).astype(np.float32))
    b2 = b.const("b2", r.normal(0, 0.1, 6).astype(np.float32))
    c2 = b.node("Conv", [mp, w2, b2], "conv2", kernel_shape=[3, 3], auto_pad="SAME_UPPER", strides=[2, 2])
    relu2 = b.node("Relu", [c2], "relu2")
    gap = b.node("GlobalAveragePool", [relu2], "gap")
    flat = b.node("Flatten", [gap], "flat", axis=1)
    fw = b.const("fc_w", r.normal(0, 0.5, (5, 6)).astype(np.float32))
    fb = b.const("fc_b", r.normal(0, 0.1, 5).astype(np.float32))
    fc = b.node("Gemm", [flat, fw, fb], "fc", transB=1)
    b.node("Softmax", [fc], b.output("output", [1, 5]), axis=1)


def elementwise(b: Builder) -> None:
    """Add (broadcast), Mul, Sub, Div, Sigmoid, Tanh, Clip (inputs), LeakyRelu, Sqrt, Exp,
    Concat, Transpose, ReduceMean, Unsqueeze, and the Shape/Gather/Concat/Reshape flatten idiom."""
    r = b.rng
    x = b.input("a", [1, 2, 4, 4])
    y = b.input("b", [1, 2, 1, 1])
    s = b.node("Add", [x, y], "sum")
    k = b.const("k", np.array([[[[0.5]], [[-2.0]]]], np.float32))
    m = b.node("Mul", [s, k], "mul")
    d = b.node("Sub", [m, x], "diff")
    q = b.node("Div", [d, b.const("two", np.array(2.0, np.float32))], "quot")
    sig = b.node("Sigmoid", [q], "sig")
    th = b.node("Tanh", [q], "tanh")
    cl = b.node("Clip", [th, b.const("lo", np.array(-0.5, np.float32)), b.const("hi", np.array(0.5, np.float32))], "clip")
    lr = b.node("LeakyRelu", [d], "leaky", alpha=0.1)
    ab = b.node("Abs", [lr], "abs")
    sq = b.node("Sqrt", [ab], "sqrt")
    ex = b.node("Exp", [cl], "exp")
    cat = b.node("Concat", [sig, ex, sq], "cat", axis=1)          # [1, 6, 4, 4]
    tr = b.node("Transpose", [cat], "tr", perm=[0, 2, 3, 1])     # [1, 4, 4, 6]
    mean = b.node("ReduceMean", [tr], "mean", axes=[1, 2], keepdims=0)  # [1, 6]
    un = b.node("Unsqueeze", [mean, b.const("axes", np.array([2], np.int64))], "un")  # [1, 6, 1]
    b.output("out_a", [1, 6, 1])
    b.nodes[-1].output[0] = "out_a"
    # Flatten idiom: Reshape(cat, Concat(Gather(Shape(cat), 0), [-1]))
    sh = b.node("Shape", [cat], "shape")
    g = b.node("Gather", [sh, b.const("zero", np.array([0], np.int64))], "g", axis=0)
    cc = b.node("Concat", [g, b.const("minus_one", np.array([-1], np.int64))], "newshape", axis=0)
    b.node("Reshape", [cat, cc], b.output("out_b", [1, 96]))


def depthwise(b: Builder) -> None:
    """Grouped (depthwise) Conv, Pad, AveragePool with VALID-like floor sizes,
    Softmax over axis 1 of a 4-D tensor, MatMul with a 2-D weight, Squeeze."""
    r = b.rng
    x = b.input("input", [1, 4, 9, 9])
    w = b.const("dw_w", r.normal(0, 0.3, (8, 1, 3, 3)).astype(np.float32))   # depth multiplier 2
    dw = b.node("Conv", [x, w], "dw", kernel_shape=[3, 3], group=4, strides=[2, 2])   # [1,8,4,4]
    pad = b.node("Pad", [dw, b.const("pads", np.array([0, 0, 1, 0, 0, 0, 0, 1], np.int64))], "pad", mode="constant")  # [1,8,5,5]
    ap = b.node("AveragePool", [pad], "ap", kernel_shape=[2, 2], strides=[2, 2])   # [1,8,2,2]
    sm = b.node("Softmax", [ap], "sm", axis=1)
    rs = b.node("Reshape", [sm, b.const("shape", np.array([1, 8, 4], np.int64))], "rs")
    tr = b.node("Transpose", [rs], "tr", perm=[0, 2, 1])      # [1,4,8]
    mm = b.node("MatMul", [tr, b.const("mw", r.normal(0, 0.5, (8, 3)).astype(np.float32))], "mm")  # [1,4,3]
    b.node("Squeeze", [mm, b.const("sq_axes", np.array([0], np.int64))], b.output("output", [4, 3]))


CASES = {"conv_bn_relu": conv_bn_relu, "elementwise": elementwise, "depthwise": depthwise}


def build(name: str, seed: int) -> None:
    case = OUT / name
    case.mkdir(parents=True, exist_ok=True)
    for old in case.glob("*.npy"):
        old.unlink()
    b = Builder(np.random.default_rng(seed))
    CASES[name](b)
    model = b.model(name)
    path = case / "model.onnx"
    onnx.save(model, str(path))
    meta = {"generator": "scripts/onnx_testcases.py", "seed": seed, "description": CASES[name].__doc__, "opset": OPSET}
    desc = record_outputs(path, case, seed + 1, meta)
    print(f"{name}: {len(b.nodes)} nodes, {len(desc['ofm_name'])} output(s)")


def main() -> None:
    names = sys.argv[1:] or list(CASES)
    for name in names:
        if name not in CASES:
            sys.exit(f"error: unknown case {name}; choose from {', '.join(CASES)}")
        build(name, seed=11)


if __name__ == "__main__":
    main()
