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
        # (np.ascontiguousarray would turn a 0-d scalar into a 1-element vector.)
        self.inits.append(numpy_helper.from_array(np.require(np.asarray(value), requirements="C"), name))
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

    def model(self, name: str, opset: int = OPSET, extra_domains: dict[str, int] | None = None) -> onnx.ModelProto:
        graph = helper.make_graph(self.nodes, name, self.inputs, self.outputs, initializer=self.inits)
        imports = [helper.make_opsetid("", opset)] + [helper.make_opsetid(d, v) for d, v in (extra_domains or {}).items()]
        model = helper.make_model(graph, opset_imports=imports, producer_name="tosa-convert tests")
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




def shapes(b: Builder) -> None:
    """Data movement and the remaining float operators: Split (sizes input), Transpose,
    Tile, Expand, DepthToSpace (DCR and CRD), SpaceToDepth, Resize (linear half-pixel,
    nearest asymmetric/floor, align_corners, and from a 1x1 map), ConvTranspose
    (stride 2, pads, output_padding), Pad (reflect, edge), Where with comparisons and
    the logical operators, ArgMax, ReduceMax/Min/Sum/Prod, LogSoftmax, Sum and Mean
    (n-ary), Sin, Cos, HardSigmoid, HardSwish, Softplus, Elu, PRelu, LRN and
    InstanceNormalization."""
    r = b.rng
    x = b.input("input", [1, 4, 4, 4])
    b.node("Split", [x, b.const("split_sizes", np.array([1, 3], np.int64))], ["split0", "split1"], axis=1)
    s0, s1 = "split0", "split1"
    tr = b.node("Transpose", [s1], "tr", perm=[0, 1, 3, 2])                        # [1,3,4,4]
    tl = b.node("Tile", [s0, b.const("repeats", np.array([1, 3, 1, 1], np.int64))], "tile")  # [1,3,4,4]
    ex = b.node("Expand", [b.const("row", r.normal(0, 1, (1, 4)).astype(np.float32)),
                           b.const("expand_shape", np.array([1, 3, 4, 4], np.int64))], "expand")
    sm = b.node("Sum", [tr, tl, ex], "sum3")
    mn = b.node("Mean", [tr, tl], "mean2")
    cat = b.node("Concat", [sm, mn, s1, s0], "cat12", axis=1)                        # [1,10,4,4]
    cat = b.node("Concat", [cat, s0, s0], "cat12b", axis=1)                         # [1,12,4,4]
    d2s = b.node("DepthToSpace", [cat], "d2s", blocksize=2, mode="DCR")             # [1,3,8,8]
    d2s_crd = b.node("DepthToSpace", [cat], "d2s_crd", blocksize=2, mode="CRD")
    s2d = b.node("SpaceToDepth", [d2s_crd], "s2d", blocksize=2)                     # [1,12,4,4]
    b.node("Sub", [s2d, cat], b.output("out_s2d", [1, 12, 4, 4]))                   # zeros
    b.node("Resize", [x, "", b.const("scales2", np.array([1, 1, 2, 2], np.float32))], b.output("out_linear", [1, 4, 8, 8]),
           mode="linear", coordinate_transformation_mode="half_pixel")
    b.node("Resize", [x, "", "", b.const("sizes6", np.array([1, 4, 6, 6], np.int64))], b.output("out_nearest", [1, 4, 6, 6]),
           mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor")
    b.node("Resize", [x, "", b.const("scales_ac", np.array([1, 1, 1.75, 1.75], np.float32))], b.output("out_align", [1, 4, 7, 7]),
           mode="linear", coordinate_transformation_mode="align_corners")
    gap = b.node("GlobalAveragePool", [x], "gap")                                    # [1,4,1,1]
    b.node("Resize", [gap, "", b.const("scales3", np.array([1, 1, 3, 3], np.float32))], b.output("out_from_1x1", [1, 4, 3, 3]),
           mode="linear", coordinate_transformation_mode="half_pixel")
    w = b.const("tconv_w", r.normal(0, 0.3, (4, 2, 3, 3)).astype(np.float32))        # [C_in, C_out, kH, kW]
    tc = b.node("ConvTranspose", [x, w, b.const("tconv_b", r.normal(0, 0.1, 2).astype(np.float32))], "tconv",
                kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1], output_padding=[1, 1])  # [1,2,8,8]
    refl = b.node("Pad", [tc, b.const("pads_r", np.array([0, 0, 2, 1, 0, 0, 1, 2], np.int64))], "pad_reflect", mode="reflect")
    edge = b.node("Pad", [tc, b.const("pads_e", np.array([0, 0, 1, 2, 0, 0, 2, 1], np.int64))], "pad_edge", mode="edge")
    b.node("Add", [refl, edge], b.output("out_pads", [1, 2, 11, 11]))
    sn = b.node("Sin", [x], "sin")
    cs = b.node("Cos", [x], "cos")
    gt = b.node("Greater", [sn, cs], "gt")
    lt = b.node("Less", [x, b.const("zero", np.array(0.0, np.float32))], "lt")
    eq = b.node("Equal", [b.node("Floor", [x], "floor"), b.node("Ceil", [x], "ceil")], "eq")
    ge = b.node("GreaterOrEqual", [x, sn], "ge")
    le = b.node("LessOrEqual", [x, cs], "le")
    cond = b.node("Or", [b.node("And", [gt, lt], "and"), b.node("Not", [eq], "not")], "or")
    cond = b.node("Xor", [cond, b.node("And", [ge, le], "and2")], "xor")
    wh = b.node("Where", [cond, sn, cs], "where")
    hs = b.node("HardSigmoid", [wh], "hard_sigmoid", alpha=0.2, beta=0.5)
    hw = b.node("HardSwish", [x], "hard_swish")
    sp = b.node("Softplus", [x], "softplus")
    el = b.node("Elu", [x], "elu", alpha=0.7)
    pr = b.node("PRelu", [x, b.const("slope", np.array([[[0.1]], [[0.2]], [[-0.3]], [[0.4]]], np.float32))], "prelu")
    lrn = b.node("LRN", [x], "lrn", size=3, alpha=1e-2, beta=0.75, bias=1.0)
    inn = b.node("InstanceNormalization", [x, b.const("in_scale", r.uniform(0.5, 1.5, 4).astype(np.float32)),
                                           b.const("in_bias", r.normal(0, 0.1, 4).astype(np.float32))], "instnorm", epsilon=1e-5)
    b.node("Concat", [hs, hw, sp, el, pr, lrn, inn], b.output("out_acts", [1, 28, 4, 4]), axis=1)
    am = b.node("ArgMax", [x], "argmax", axis=1, keepdims=0)                         # int64 [1,4,4]
    amf = b.node("Cast", [am], "argmax_f", to=TensorProto.FLOAT)
    rmax = b.node("ReduceMax", [x], "rmax", axes=[1], keepdims=0)                    # [1,4,4]
    rmin = b.node("ReduceMin", [x], "rmin", axes=[2, 3], keepdims=1)                 # [1,4,1,1]
    rsum = b.node("ReduceSum", [x, b.const("sum_axes", np.array([1], np.int64))], "rsum", keepdims=1)  # [1,1,4,4]
    rprod = b.node("ReduceProd", [b.node("Sigmoid", [x], "sig")], "rprod", axes=[1], keepdims=0)
    ls = b.node("LogSoftmax", [x], "log_softmax", axis=1)
    red = b.node("Add", [b.node("Add", [amf, rmax], "r1"), rprod], "r2")            # [1,4,4]
    red = b.node("Add", [b.node("Unsqueeze", [red, b.const("u_axes", np.array([1], np.int64))], "r3"), rsum], "r4")  # [1,1,4,4]
    red = b.node("Add", [red, rmin], "r5")                                            # [1,4,4,4]
    b.node("Add", [red, ls], b.output("out_reduce", [1, 4, 4, 4]))


def upsample(b: Builder) -> None:
    """Opset 9: Upsample nearest and linear with a scales input, as older exporters emit."""
    x = b.input("input", [1, 2, 3, 3])
    near = b.node("Upsample", [x, b.const("scales_n", np.array([1, 1, 2, 2], np.float32))], b.output("out_nearest", [1, 2, 6, 6]), mode="nearest")
    b.node("Upsample", [x, b.const("scales_l", np.array([1, 1, 3, 2], np.float32))], b.output("out_linear", [1, 2, 9, 6]), mode="linear")


def qlinear(b: Builder) -> None:
    """A quantized (uint8 activations, int8 per-channel weights) CNN as the ONNX
    quantizer emits: QuantizeLinear -> QLinearConv -> depthwise QLinearConv ->
    com.microsoft.QLinearAdd -> com.microsoft.QLinearGlobalAveragePool -> Reshape ->
    QLinearMatMul (per-column) -> DequantizeLinear. Bit-exact against onnxruntime."""
    r = b.rng
    x = b.input("input", [1, 3, 8, 8])
    xs = b.const("x_scale", np.array(0.02, np.float32))
    xzp = b.const("x_zp", np.array(118, np.uint8))
    xq = b.node("QuantizeLinear", [x, xs, xzp], "xq")
    # conv 1: 3 -> 8 channels, per-channel int8 weights, int32 bias
    w1 = b.const("w1", r.integers(-127, 128, (8, 3, 3, 3)).astype(np.int8))
    w1s = b.const("w1_scale", r.uniform(0.002, 0.01, 8).astype(np.float32))
    w1zp = b.const("w1_zp", np.zeros(8, np.int8))
    b1 = b.const("b1", r.integers(-3000, 3000, 8).astype(np.int32))
    y1s, y1zp = b.const("y1_scale", np.array(0.05, np.float32)), b.const("y1_zp", np.array(100, np.uint8))
    c1 = b.node("QLinearConv", [xq, xs, xzp, w1, w1s, w1zp, y1s, y1zp, b1], "c1", kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
    # depthwise conv: group 8
    w2 = b.const("w2", r.integers(-127, 128, (8, 1, 3, 3)).astype(np.int8))
    w2s = b.const("w2_scale", r.uniform(0.002, 0.01, 8).astype(np.float32))
    w2zp = b.const("w2_zp", np.zeros(8, np.int8))
    b2 = b.const("b2", r.integers(-3000, 3000, 8).astype(np.int32))
    y2s, y2zp = b.const("y2_scale", np.array(0.04, np.float32)), b.const("y2_zp", np.array(130, np.uint8))
    c2 = b.node("QLinearConv", [c1, y1s, y1zp, w2, w2s, w2zp, y2s, y2zp, b2], "c2", kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2], group=8)
    # pointwise conv back to 8 channels at the same resolution, to add
    w3 = b.const("w3", r.integers(-127, 128, (8, 8, 1, 1)).astype(np.int8))
    w3s = b.const("w3_scale", r.uniform(0.002, 0.01, 8).astype(np.float32))
    w3zp = b.const("w3_zp", np.zeros(8, np.int8))
    y3s, y3zp = b.const("y3_scale", np.array(0.03, np.float32)), b.const("y3_zp", np.array(120, np.uint8))
    c3 = b.node("QLinearConv", [c2, y2s, y2zp, w3, w3s, w3zp, y3s, y3zp], "c3", kernel_shape=[1, 1])
    as_, azp = b.const("add_scale", np.array(0.06, np.float32)), b.const("add_zp", np.array(125, np.uint8))
    add = b.node("QLinearAdd", [c2, y2s, y2zp, c3, y3s, y3zp, as_, azp], "qadd", domain="com.microsoft")
    gs, gzp = b.const("gap_scale", np.array(0.045, np.float32)), b.const("gap_zp", np.array(122, np.uint8))
    gap = b.node("QLinearGlobalAveragePool", [add, as_, azp, gs, gzp], "qgap", domain="com.microsoft", channels_last=0)
    flat = b.node("Reshape", [gap, b.const("flat_shape", np.array([1, 8], np.int64))], "flat")
    wm = b.const("wm", r.integers(-127, 128, (8, 5)).astype(np.int8))
    wms = b.const("wm_scale", r.uniform(0.002, 0.01, 5).astype(np.float32))
    wmzp = b.const("wm_zp", np.zeros(5, np.int8))
    ms, mzp = b.const("mm_scale", np.array(0.08, np.float32)), b.const("mm_zp", np.array(110, np.uint8))
    mm = b.node("QLinearMatMul", [flat, gs, gzp, wm, wms, wmzp, ms, mzp], "qmm")
    b.node("DequantizeLinear", [mm, ms, mzp], b.output("output", [1, 5]))


CASE_OPTIONS = {
    "shapes": {"opset": 14},
    "upsample": {"opset": 9},
    "qlinear": {"opset": 13, "extra_domains": {"com.microsoft": 1}},
}

CASES = {
    "conv_bn_relu": conv_bn_relu,
    "elementwise": elementwise,
    "depthwise": depthwise,
    "shapes": shapes,
    "upsample": upsample,
    "qlinear": qlinear,
}


def build(name: str, seed: int) -> None:
    case = OUT / name
    case.mkdir(parents=True, exist_ok=True)
    for old in case.glob("*.npy"):
        old.unlink()
    b = Builder(np.random.default_rng(seed))
    CASES[name](b)
    options = CASE_OPTIONS.get(name, {})
    model = b.model(name, **options)
    path = case / "model.onnx"
    onnx.save(model, str(path))
    meta = {"generator": "scripts/onnx_testcases.py", "seed": seed, "description": CASES[name].__doc__,
            "opset": options.get("opset", OPSET)}
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
