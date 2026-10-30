#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["ai-edge-litert>=1.0", "numpy"]
# ///
"""Build the small TFLite models checked in under ``crates/tosa-convert/tests/tflite``.

Each case is written as flatc JSON, compiled to ``model.tflite`` with
``flatc -b`` (so no TensorFlow is needed), then run through LiteRT on seeded
inputs to record the expected outputs, in the same case format as
``scripts/models.py`` produces for the downloaded models.

    uv run scripts/tflite_testcases.py            # rebuild every case
    uv run scripts/tflite_testcases.py conv_f32   # one case

``flatc`` must be the release matching the ``flatbuffers`` crate (see
``xtask/src/main.rs``); ``FLATC`` overrides the binary used.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import record_outputs  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SCHEMA = REPO / "crates" / "tosa-convert" / "schema" / "tflite.fbs"
OUT = REPO / "crates" / "tosa-convert" / "tests" / "tflite"

# BuiltinOperator values for deprecated_builtin_code (older readers).
BUILTIN_CODES = {
    "ADD": 0, "AVERAGE_POOL_2D": 1, "CONCATENATION": 2, "CONV_2D": 3, "DEPTHWISE_CONV_2D": 4,
    "DEQUANTIZE": 6, "FULLY_CONNECTED": 9, "LOGISTIC": 14, "MAX_POOL_2D": 17, "MUL": 18, "RELU": 19,
    "RELU6": 21, "RESHAPE": 22, "SOFTMAX": 25, "SUB": 41, "SQUEEZE": 43, "PAD": 34, "MEAN": 40,
    "QUANTIZE": 114, "TANH": 28,
    "DEPTH_TO_SPACE": 5, "FLOOR": 8, "L2_NORMALIZATION": 11, "RESIZE_BILINEAR": 23, "SPACE_TO_DEPTH": 26,
    "GATHER": 36, "TRANSPOSE": 39, "DIV": 42, "STRIDED_SLICE": 45, "EXP": 47, "SPLIT": 49, "LOG_SOFTMAX": 50,
    "CAST": 53, "PRELU": 54, "MAXIMUM": 55, "ARG_MAX": 56, "MINIMUM": 57, "LESS": 58, "NEG": 59, "GREATER": 61,
    "GREATER_EQUAL": 62, "LESS_EQUAL": 63, "SELECT": 64, "SLICE": 65, "SIN": 66, "TRANSPOSE_CONV": 67, "TILE": 69,
    "EXPAND_DIMS": 70, "EQUAL": 71, "NOT_EQUAL": 72, "LOG": 73, "SUM": 74, "SQRT": 75, "RSQRT": 76, "POW": 78,
    "ARG_MIN": 79, "REDUCE_PROD": 81, "REDUCE_MAX": 82, "PACK": 83, "LOGICAL_OR": 84, "LOGICAL_AND": 86,
    "LOGICAL_NOT": 87, "UNPACK": 88, "REDUCE_MIN": 89, "RESIZE_NEAREST_NEIGHBOR": 97, "LEAKY_RELU": 98,
    "SQUARED_DIFFERENCE": 99, "ABS": 101, "SPLIT_V": 102, "CEIL": 104, "COS": 108, "ELU": 111, "HARD_SWISH": 117,
    "SELECT_V2": 123, "BATCH_MATMUL": 126, "GELU": 150,
}

TYPE_NP = {"FLOAT32": np.float32, "INT32": np.int32, "UINT8": np.uint8, "INT8": np.int8, "BOOL": np.bool_}


class Builder:
    """Accumulates tensors, buffers and operators of a one-subgraph model."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.tensors: list[dict] = []
        self.buffers: list[dict] = [{}]  # buffer 0 is the empty buffer
        self.operators: list[dict] = []
        self.codes: list[str] = []
        self.inputs: list[int] = []
        self.outputs: list[int] = []

    def tensor(self, name: str, shape: list[int], ttype: str, data: np.ndarray | None = None,
               scale=None, zero_point=None, quantized_dimension: int = 0) -> int:
        buffer = 0
        if data is not None:
            data = np.ascontiguousarray(data.astype(TYPE_NP[ttype]))
            self.buffers.append({"data": list(data.tobytes())})
            buffer = len(self.buffers) - 1
        t = {"shape": shape, "type": ttype, "buffer": buffer, "name": name}
        if scale is not None:
            scales = np.atleast_1d(np.asarray(scale, dtype=np.float64)).tolist()
            zps = np.atleast_1d(np.asarray(zero_point if zero_point is not None else 0)).astype(int).tolist()
            if len(zps) == 1 and len(scales) > 1:
                zps = zps * len(scales)
            t["quantization"] = {"scale": scales, "zero_point": zps, "quantized_dimension": quantized_dimension}
        self.tensors.append(t)
        return len(self.tensors) - 1

    def op(self, code: str, inputs: list[int], outputs: list[int], options_type: str | None = None,
           options: dict | None = None) -> None:
        if code not in self.codes:
            self.codes.append(code)
        o = {"opcode_index": self.codes.index(code), "inputs": inputs, "outputs": outputs}
        if options_type:
            o["builtin_options_type"] = options_type
            o["builtin_options"] = options or {}
        self.operators.append(o)

    def model(self, description: str) -> dict:
        return {
            "version": 3,
            "operator_codes": [
                {"deprecated_builtin_code": min(BUILTIN_CODES[c], 127), "version": 1, "builtin_code": c}
                for c in self.codes
            ],
            "subgraphs": [{
                "tensors": self.tensors,
                "inputs": self.inputs,
                "outputs": self.outputs,
                "operators": self.operators,
                "name": "main",
            }],
            "description": description,
            "buffers": self.buffers,
        }


def act(name: str) -> dict:
    return {"fused_activation_function": name}


def conv_f32(b: Builder) -> None:
    """Float CNN head to tail: CONV_2D(RELU6) -> DEPTHWISE_CONV_2D(RELU) ->
    AVERAGE_POOL_2D -> RESHAPE -> FULLY_CONNECTED -> SOFTMAX."""
    r = b.rng
    x = b.tensor("input", [1, 8, 8, 3], "FLOAT32")
    w1 = b.tensor("conv_w", [4, 3, 3, 3], "FLOAT32", r.normal(0, 0.3, (4, 3, 3, 3)))
    b1 = b.tensor("conv_b", [4], "FLOAT32", r.normal(0, 0.1, 4))
    c1 = b.tensor("conv", [1, 8, 8, 4], "FLOAT32")
    b.op("CONV_2D", [x, w1, b1], [c1], "Conv2DOptions",
         {"padding": "SAME", "stride_w": 1, "stride_h": 1, "dilation_w_factor": 1, "dilation_h_factor": 1, **act("RELU6")})
    w2 = b.tensor("dw_w", [1, 3, 3, 4], "FLOAT32", r.normal(0, 0.3, (1, 3, 3, 4)))
    b2 = b.tensor("dw_b", [4], "FLOAT32", r.normal(0, 0.1, 4))
    c2 = b.tensor("dw", [1, 3, 3, 4], "FLOAT32")
    b.op("DEPTHWISE_CONV_2D", [c1, w2, b2], [c2], "DepthwiseConv2DOptions",
         {"padding": "VALID", "stride_w": 2, "stride_h": 2, "depth_multiplier": 1, "dilation_w_factor": 1,
          "dilation_h_factor": 1, **act("RELU")})
    p = b.tensor("pool", [1, 1, 1, 4], "FLOAT32")
    b.op("AVERAGE_POOL_2D", [c2], [p], "Pool2DOptions",
         {"padding": "VALID", "stride_w": 1, "stride_h": 1, "filter_width": 3, "filter_height": 3, **act("NONE")})
    shp = b.tensor("shape", [2], "INT32", np.array([1, 4], np.int32))
    flat = b.tensor("flat", [1, 4], "FLOAT32")
    b.op("RESHAPE", [p, shp], [flat], "ReshapeOptions", {"new_shape": [1, 4]})
    fw = b.tensor("fc_w", [5, 4], "FLOAT32", r.normal(0, 0.5, (5, 4)))
    fb_ = b.tensor("fc_b", [5], "FLOAT32", r.normal(0, 0.1, 5))
    fc = b.tensor("fc", [1, 5], "FLOAT32")
    b.op("FULLY_CONNECTED", [flat, fw, fb_], [fc], "FullyConnectedOptions",
         {**act("NONE"), "weights_format": "DEFAULT", "keep_num_dims": False})
    y = b.tensor("output", [1, 5], "FLOAT32")
    b.op("SOFTMAX", [fc], [y], "SoftmaxOptions", {"beta": 1.0})
    b.inputs, b.outputs = [x], [y]


def elementwise_f32(b: Builder) -> None:
    """Two inputs: ADD(RELU) with broadcasting, MUL by a constant, SUB, LOGISTIC,
    TANH, MAX_POOL_2D, PAD, MEAN, CONCATENATION, SQUEEZE, RELU6."""
    r = b.rng
    x = b.tensor("a", [1, 4, 4, 2], "FLOAT32")
    y = b.tensor("b", [1, 1, 1, 2], "FLOAT32")
    s = b.tensor("sum", [1, 4, 4, 2], "FLOAT32")
    b.op("ADD", [x, y], [s], "AddOptions", act("RELU"))
    k = b.tensor("k", [2], "FLOAT32", np.array([0.5, -2.0], np.float32))
    m = b.tensor("mul", [1, 4, 4, 2], "FLOAT32")
    b.op("MUL", [s, k], [m], "MulOptions", act("NONE"))
    d = b.tensor("diff", [1, 4, 4, 2], "FLOAT32")
    b.op("SUB", [m, x], [d], "SubOptions", act("NONE"))
    sig = b.tensor("sigmoid", [1, 4, 4, 2], "FLOAT32")
    b.op("LOGISTIC", [d], [sig])
    th = b.tensor("tanh", [1, 4, 4, 2], "FLOAT32")
    b.op("TANH", [d], [th])
    mp = b.tensor("maxpool", [1, 2, 2, 2], "FLOAT32")
    b.op("MAX_POOL_2D", [th], [mp], "Pool2DOptions",
         {"padding": "SAME", "stride_w": 2, "stride_h": 2, "filter_width": 3, "filter_height": 3, **act("NONE")})
    pads = b.tensor("pads", [4, 2], "INT32", np.array([[0, 0], [1, 0], [0, 1], [0, 0]], np.int32))
    padded = b.tensor("padded", [1, 3, 3, 2], "FLOAT32")
    b.op("PAD", [mp, pads], [padded], "PadOptions", {})
    axes = b.tensor("axes", [2], "INT32", np.array([1, 2], np.int32))
    mean = b.tensor("mean", [1, 1, 1, 2], "FLOAT32")
    b.op("MEAN", [padded, axes], [mean], "ReducerOptions", {"keep_dims": True})
    cat = b.tensor("concat", [1, 1, 1, 4], "FLOAT32")
    b.op("CONCATENATION", [mean, y], [cat], "ConcatenationOptions", {"axis": 3, **act("NONE")})
    sq = b.tensor("squeezed", [1, 4], "FLOAT32")
    b.op("SQUEEZE", [cat], [sq], "SqueezeOptions", {"squeeze_dims": [1, 2]})
    out = b.tensor("output", [1, 4], "FLOAT32")
    b.op("RELU6", [sq], [out])
    out2 = b.tensor("output_sigmoid", [1, 4, 4, 2], "FLOAT32")
    b.op("RELU", [sig], [out2])
    b.inputs, b.outputs = [x, y], [out, out2]


def quant_params(r, n: int | None = None, lo=0.001, hi=0.05):
    """Random per-tensor (n=None) or per-channel scales."""
    return r.uniform(lo, hi) if n is None else r.uniform(lo, hi, n)


def conv_uint8(b: Builder) -> None:
    """uint8 per-tensor quantized CNN, as the classic TFLite MobileNets:
    CONV_2D(RELU6) -> DEPTHWISE_CONV_2D -> AVERAGE_POOL_2D -> RESHAPE ->
    FULLY_CONNECTED -> SOFTMAX, uint8 input and output."""
    r = b.rng
    in_s, in_zp = 0.0078125, 128
    x = b.tensor("input", [1, 8, 8, 3], "UINT8", scale=in_s, zero_point=in_zp)
    w1_s, w1_zp = 0.02, 120
    w1 = b.tensor("conv_w", [4, 3, 3, 3], "UINT8", r.integers(0, 256, (4, 3, 3, 3)), scale=w1_s, zero_point=w1_zp)
    b1 = b.tensor("conv_b", [4], "INT32", r.integers(-2000, 2000, 4), scale=in_s * w1_s, zero_point=0)
    c1_s, c1_zp = 0.03, 0
    c1 = b.tensor("conv", [1, 8, 8, 4], "UINT8", scale=c1_s, zero_point=c1_zp)
    b.op("CONV_2D", [x, w1, b1], [c1], "Conv2DOptions",
         {"padding": "SAME", "stride_w": 1, "stride_h": 1, "dilation_w_factor": 1, "dilation_h_factor": 1, **act("RELU6")})
    w2_s, w2_zp = 0.015, 130
    w2 = b.tensor("dw_w", [1, 3, 3, 4], "UINT8", r.integers(0, 256, (1, 3, 3, 4)), scale=w2_s, zero_point=w2_zp)
    b2 = b.tensor("dw_b", [4], "INT32", r.integers(-3000, 3000, 4), scale=c1_s * w2_s, zero_point=0)
    c2_s, c2_zp = 0.05, 110
    c2 = b.tensor("dw", [1, 3, 3, 4], "UINT8", scale=c2_s, zero_point=c2_zp)
    b.op("DEPTHWISE_CONV_2D", [c1, w2, b2], [c2], "DepthwiseConv2DOptions",
         {"padding": "VALID", "stride_w": 2, "stride_h": 2, "depth_multiplier": 1, "dilation_w_factor": 1,
          "dilation_h_factor": 1, **act("NONE")})
    p = b.tensor("pool", [1, 1, 1, 4], "UINT8", scale=c2_s, zero_point=c2_zp)
    b.op("AVERAGE_POOL_2D", [c2], [p], "Pool2DOptions",
         {"padding": "VALID", "stride_w": 1, "stride_h": 1, "filter_width": 3, "filter_height": 3, **act("NONE")})
    shp = b.tensor("shape", [2], "INT32", np.array([1, 4], np.int32))
    flat = b.tensor("flat", [1, 4], "UINT8", scale=c2_s, zero_point=c2_zp)
    b.op("RESHAPE", [p, shp], [flat], "ReshapeOptions", {"new_shape": [1, 4]})
    fw_s, fw_zp = 0.01, 125
    fw = b.tensor("fc_w", [5, 4], "UINT8", r.integers(0, 256, (5, 4)), scale=fw_s, zero_point=fw_zp)
    fb_ = b.tensor("fc_b", [5], "INT32", r.integers(-500, 500, 5), scale=c2_s * fw_s, zero_point=0)
    fc = b.tensor("fc", [1, 5], "UINT8", scale=0.1, zero_point=100)
    b.op("FULLY_CONNECTED", [flat, fw, fb_], [fc], "FullyConnectedOptions",
         {**act("NONE"), "weights_format": "DEFAULT", "keep_num_dims": False})
    y = b.tensor("output", [1, 5], "UINT8", scale=1 / 256, zero_point=0)
    b.op("SOFTMAX", [fc], [y], "SoftmaxOptions", {"beta": 1.0})
    b.inputs, b.outputs = [x], [y]


def conv_int8(b: Builder) -> None:
    """int8 per-channel quantized model with float edges, as post-training
    quantization produces: QUANTIZE -> CONV_2D (per-channel, RELU) and
    DEPTHWISE_CONV_2D (per-channel) -> ADD (different scales) -> MAX_POOL_2D ->
    DEQUANTIZE."""
    r = b.rng
    xf = b.tensor("input", [1, 6, 6, 3], "FLOAT32")
    in_s, in_zp = 0.02, -3
    x = b.tensor("input_q", [1, 6, 6, 3], "INT8", scale=in_s, zero_point=in_zp)
    b.op("QUANTIZE", [xf], [x], "QuantizeOptions", {})
    w1_s = quant_params(r, 4)
    w1 = b.tensor("conv_w", [4, 3, 3, 3], "INT8", r.integers(-127, 128, (4, 3, 3, 3)), scale=w1_s, zero_point=0,
                  quantized_dimension=0)
    b1 = b.tensor("conv_b", [4], "INT32", r.integers(-2000, 2000, 4), scale=in_s * w1_s, zero_point=0,
                  quantized_dimension=0)
    c1_s, c1_zp = 0.04, -128
    c1 = b.tensor("conv", [1, 6, 6, 4], "INT8", scale=c1_s, zero_point=c1_zp)
    b.op("CONV_2D", [x, w1, b1], [c1], "Conv2DOptions",
         {"padding": "SAME", "stride_w": 1, "stride_h": 1, "dilation_w_factor": 1, "dilation_h_factor": 1, **act("RELU")})
    w2_s = quant_params(r, 4)
    w2 = b.tensor("dw_w", [1, 3, 3, 4], "INT8", r.integers(-127, 128, (1, 3, 3, 4)), scale=w2_s, zero_point=0,
                  quantized_dimension=3)
    b2 = b.tensor("dw_b", [4], "INT32", r.integers(-3000, 3000, 4), scale=c1_s * w2_s, zero_point=0,
                  quantized_dimension=0)
    c2_s, c2_zp = 0.06, 7
    c2 = b.tensor("dw", [1, 6, 6, 4], "INT8", scale=c2_s, zero_point=c2_zp)
    b.op("DEPTHWISE_CONV_2D", [c1, w2, b2], [c2], "DepthwiseConv2DOptions",
         {"padding": "SAME", "stride_w": 1, "stride_h": 1, "depth_multiplier": 1, "dilation_w_factor": 1,
          "dilation_h_factor": 1, **act("NONE")})
    s_s, s_zp = 0.09, -10
    s = b.tensor("sum", [1, 6, 6, 4], "INT8", scale=s_s, zero_point=s_zp)
    b.op("ADD", [c1, c2], [s], "AddOptions", act("NONE"))
    mp = b.tensor("maxpool", [1, 3, 3, 4], "INT8", scale=s_s, zero_point=s_zp)
    b.op("MAX_POOL_2D", [s], [mp], "Pool2DOptions",
         {"padding": "SAME", "stride_w": 2, "stride_h": 2, "filter_width": 2, "filter_height": 2, **act("NONE")})
    y = b.tensor("output", [1, 3, 3, 4], "FLOAT32")
    b.op("DEQUANTIZE", [mp], [y], "DequantizeOptions", {})
    b.inputs, b.outputs = [xf], [y]



def i32(b: Builder, name: str, values) -> int:
    """An INT32 constant (shape [] for a Python scalar)."""
    arr = np.asarray(values, np.int32)
    return b.tensor(name, list(arr.shape), "INT32", arr)


def math_f32(b: Builder) -> None:
    """Float element-wise operators on two inputs: HARD_SWISH, PRELU, LEAKY_RELU,
    GELU (exact and approximate), MAXIMUM, MINIMUM, EXP, LOG, SQRT, RSQRT, SIN, COS, FLOOR, NEG,
    POW, DIV, SQUARED_DIFFERENCE, L2_NORMALIZATION, LOG_SOFTMAX; reductions
    REDUCE_MAX/MIN, SUM, REDUCE_PROD, ARG_MAX/MIN with CAST; comparisons, the
    logical operators and SELECT/SELECT_V2. Transcendental functions follow each
    runtime's libm, so results agree to a few float32 ulps, not bit for bit."""
    shape = [1, 4, 4, 2]
    a = b.tensor("a", shape, "FLOAT32")
    c = b.tensor("b", shape, "FLOAT32")

    def unary(code, x, name, options_type=None, options=None):
        y = b.tensor(name, shape, "FLOAT32")
        b.op(code, [x], [y], options_type, options)
        return y

    def binary(code, x, z, name, options_type=None, options=None):
        y = b.tensor(name, shape, "FLOAT32")
        b.op(code, [x, z], [y], options_type, options)
        return y

    hs = unary("HARD_SWISH", a, "hard_swish", "HardSwishOptions", {})
    alpha = b.tensor("prelu_alpha", [2], "FLOAT32", np.array([0.1, -0.3], np.float32))
    pr = binary("PRELU", hs, alpha, "prelu")
    lr = unary("LEAKY_RELU", c, "leaky_relu", "LeakyReluOptions", {"alpha": 0.2})
    el = unary("GELU", lr, "gelu_approx", "GeluOptions", {"approximate": True})  # ELU is absent from LiteRT's reference resolver
    ge = unary("GELU", a, "gelu", "GeluOptions", {"approximate": False})
    mx = binary("MAXIMUM", pr, el, "maximum", "MaximumMinimumOptions", {})
    mn = binary("MINIMUM", pr, el, "minimum", "MaximumMinimumOptions", {})
    ex = unary("EXP", mn, "exp", "ExpOptions", {})
    lg = unary("LOG", ex, "log")
    ab = unary("ABS", mx, "abs", "AbsOptions", {})
    sq = unary("SQRT", ab, "sqrt")
    rs = unary("RSQRT", ex, "rsqrt")
    sn = unary("SIN", a, "sin")
    cs = unary("COS", a, "cos", "CosOptions", {})
    three = b.tensor("three", [1], "FLOAT32", np.array([3.0], np.float32))
    a3 = binary("MUL", a, three, "a3", "MulOptions", act("NONE"))
    c3 = binary("MUL", c, three, "c3", "MulOptions", act("NONE"))
    fl = unary("FLOOR", a3, "floor")
    ce = unary("FLOOR", c3, "floor_b")  # CEIL (like ELU) is absent from LiteRT's reference resolver
    ng = unary("NEG", a, "neg", "NegOptions", {})
    pw = binary("POW", ex, c, "pow", "PowOptions", {})
    dv = binary("DIV", a, ex, "div", "DivOptions", act("NONE"))
    sd = binary("SQUARED_DIFFERENCE", a, c, "squared_difference", "SquaredDifferenceOptions", {})
    l2 = unary("L2_NORMALIZATION", a, "l2_normalization", "L2NormOptions", act("NONE"))
    ls = unary("LOG_SOFTMAX", a, "log_softmax", "LogSoftmaxOptions", {})
    parts = [hs, pr, lr, el, ge, mx, mn, ex, lg, sq, rs, sn, cs, fl, ce, ng, pw, dv, sd, l2, ls]
    out_elementwise = b.tensor("output_elementwise", [1, 4, 4, 2 * len(parts)], "FLOAT32")
    b.op("CONCATENATION", parts, [out_elementwise], "ConcatenationOptions", {"axis": 3, **act("NONE")})

    axes12 = i32(b, "axes12", [1, 2])
    reductions = []
    for code, x in (("REDUCE_MAX", a), ("REDUCE_MIN", c), ("SUM", a)):
        y = b.tensor(code.lower(), [1, 2], "FLOAT32")
        b.op(code, [x, axes12], [y], "ReducerOptions", {"keep_dims": False})
        reductions.append(y)
    out_reduce = b.tensor("output_reductions", [3, 1, 2], "FLOAT32")
    b.op("PACK", reductions, [out_reduce], "PackOptions", {"values_count": 3, "axis": 0})
    axis3 = i32(b, "axis3", [3])
    out_prod = b.tensor("output_prod", [1, 4, 4, 1], "FLOAT32")
    b.op("REDUCE_PROD", [ex, axis3], [out_prod], "ReducerOptions", {"keep_dims": True})

    axis_scalar = i32(b, "axis_scalar", 3)
    am = b.tensor("argmax", [1, 4, 4], "INT32")
    b.op("ARG_MAX", [a, axis_scalar], [am], "ArgMaxOptions", {"output_type": "INT32"})
    an = b.tensor("argmin", [1, 4, 4], "INT32")
    b.op("ARG_MIN", [c, axis_scalar], [an], "ArgMinOptions", {"output_type": "INT32"})
    am_f = b.tensor("argmax_f", [1, 4, 4], "FLOAT32")
    b.op("CAST", [am], [am_f], "CastOptions", {"in_data_type": "INT32", "out_data_type": "FLOAT32"})
    an_f = b.tensor("argmin_f", [1, 4, 4], "FLOAT32")
    b.op("CAST", [an], [an_f], "CastOptions", {"in_data_type": "INT32", "out_data_type": "FLOAT32"})
    two = b.tensor("two", [1], "FLOAT32", np.array([2.0], np.float32))
    am2 = b.tensor("argmax_x2", [1, 4, 4], "FLOAT32")
    b.op("MUL", [am_f, two], [am2], "MulOptions", act("NONE"))
    out_arg = b.tensor("output_arg", [1, 4, 4], "FLOAT32")
    b.op("ADD", [am2, an_f], [out_arg], "AddOptions", act("NONE"))

    def compare(code, x, z, name, options_type):
        y = b.tensor(name, shape, "BOOL")
        b.op(code, [x, z], [y], options_type, {})
        return y

    gt = compare("GREATER", a, c, "greater", "GreaterOptions")
    le = compare("LESS_EQUAL", sn, cs, "less_equal", "LessEqualOptions")
    eq = compare("EQUAL", fl, ce, "equal", "EqualOptions")
    ne = compare("NOT_EQUAL", fl, ce, "not_equal", "NotEqualOptions")
    ge_ = compare("GREATER_EQUAL", a, ng, "greater_equal", "GreaterEqualOptions")
    lt = compare("LESS", c, ng, "less", "LessOptions")
    both = b.tensor("and", shape, "BOOL")
    b.op("LOGICAL_AND", [gt, le], [both], "LogicalAndOptions", {})
    not_eq = b.tensor("not", shape, "BOOL")
    b.op("LOGICAL_NOT", [eq], [not_eq], "LogicalNotOptions", {})
    either = b.tensor("or", shape, "BOOL")
    b.op("LOGICAL_OR", [both, not_eq], [either], "LogicalOrOptions", {})
    sel = b.tensor("select", shape, "FLOAT32")
    b.op("SELECT", [either, a, c], [sel], "SelectOptions", {})
    sel2 = b.tensor("select_v2", shape, "FLOAT32")
    b.op("SELECT_V2", [ne, sel, hs], [sel2], "SelectV2Options", {})
    sel3 = b.tensor("select_v2_b", shape, "FLOAT32")
    b.op("SELECT_V2", [ge_, sel2, c], [sel3], "SelectV2Options", {})
    out_select = b.tensor("output_select", shape, "FLOAT32")
    b.op("SELECT_V2", [lt, sel3, a], [out_select], "SelectV2Options", {})
    b.inputs, b.outputs = [a, c], [out_elementwise, out_reduce, out_prod, out_arg, out_select]


def shapes_f32(b: Builder) -> None:
    """Float data-movement operators: TRANSPOSE, SLICE, STRIDED_SLICE (shrink axis),
    SPLIT, SPLIT_V, PACK, UNPACK, TILE, GATHER, EXPAND_DIMS, DEPTH_TO_SPACE,
    SPACE_TO_DEPTH, RESIZE_BILINEAR (half-pixel centres, and from a 1x1 map),
    RESIZE_NEAREST_NEIGHBOR, TRANSPOSE_CONV (SAME, stride 2) and BATCH_MATMUL."""
    r = b.rng
    x = b.tensor("input", [1, 4, 4, 3], "FLOAT32")
    tr = b.tensor("output_transpose", [1, 3, 4, 4], "FLOAT32")
    b.op("TRANSPOSE", [x, i32(b, "perm", [0, 3, 1, 2])], [tr], "TransposeOptions", {})
    sl = b.tensor("slice", [1, 2, 2, 3], "FLOAT32")
    b.op("SLICE", [x, i32(b, "begin", [0, 1, 1, 0]), i32(b, "size", [1, 2, 2, -1])], [sl], "SliceOptions", {})
    ss = b.tensor("output_strided", [2, 2, 2], "FLOAT32")
    b.op("STRIDED_SLICE", [x, i32(b, "ss_begin", [0, 0, 0, 1]), i32(b, "ss_end", [1, 4, 4, 3]),
                           i32(b, "ss_strides", [1, 2, 2, 1])], [ss], "StridedSliceOptions",
         {"begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 1})
    splits = [b.tensor(f"split{i}", [1, 4, 4, 1], "FLOAT32") for i in range(3)]
    b.op("SPLIT", [i32(b, "split_axis", 3), x], splits, "SplitOptions", {"num_splits": 3})
    packed = b.tensor("packed", [1, 4, 4, 3, 1], "FLOAT32")
    b.op("PACK", splits, [packed], "PackOptions", {"values_count": 3, "axis": 3})
    repacked = b.tensor("repacked", [1, 4, 4, 3], "FLOAT32")
    b.op("RESHAPE", [packed, i32(b, "repack_shape", [1, 4, 4, 3])], [repacked], "ReshapeOptions",
         {"new_shape": [1, 4, 4, 3]})
    rows = [b.tensor(f"row{i}", [1, 4, 3], "FLOAT32") for i in range(4)]
    b.op("UNPACK", [x], rows, "UnpackOptions", {"num": 4, "axis": 1})
    restacked = b.tensor("restacked", [1, 4, 4, 3], "FLOAT32")
    b.op("PACK", rows, [restacked], "PackOptions", {"values_count": 4, "axis": 1})
    out_pack = b.tensor("output_pack", [1, 4, 4, 3], "FLOAT32")
    b.op("ADD", [repacked, restacked], [out_pack], "AddOptions", act("NONE"))
    sv0 = b.tensor("splitv0", [1, 4, 4, 1], "FLOAT32")
    sv1 = b.tensor("output_splitv", [1, 4, 4, 2], "FLOAT32")
    b.op("SPLIT_V", [x, i32(b, "size_splits", [1, -1]), i32(b, "splitv_axis", 3)], [sv0, sv1], "SplitVOptions",
         {"num_splits": 2})
    tl = b.tensor("tiled", [1, 4, 4, 3], "FLOAT32")
    b.op("TILE", [sl, i32(b, "multiples", [1, 2, 2, 1])], [tl], "TileOptions", {})
    ga = b.tensor("gathered", [1, 2, 4, 3], "FLOAT32")
    b.op("GATHER", [tl, i32(b, "indices", [2, 0])], [ga], "GatherOptions", {"axis": 1, "batch_dims": 0})
    out_expand = b.tensor("output_expand", [1, 1, 2, 4, 3], "FLOAT32")
    b.op("EXPAND_DIMS", [ga, i32(b, "expand_axis", 0)], [out_expand], "ExpandDimsOptions", {})
    wide = b.tensor("wide", [1, 4, 4, 12], "FLOAT32")
    b.op("CONCATENATION", [x, sv0, x, tl, sv1], [wide], "ConcatenationOptions", {"axis": 3, **act("NONE")})
    d2s = b.tensor("output_d2s", [1, 8, 8, 3], "FLOAT32")
    b.op("DEPTH_TO_SPACE", [wide], [d2s], "DepthToSpaceOptions", {"block_size": 2})
    s2d = b.tensor("output_s2d", [1, 4, 4, 12], "FLOAT32")
    b.op("SPACE_TO_DEPTH", [d2s], [s2d], "SpaceToDepthOptions", {"block_size": 2})
    rb = b.tensor("output_bilinear", [1, 8, 8, 3], "FLOAT32")
    b.op("RESIZE_BILINEAR", [x, i32(b, "size88", [8, 8])], [rb], "ResizeBilinearOptions",
         {"align_corners": False, "half_pixel_centers": True})
    rn = b.tensor("output_nearest", [1, 6, 6, 3], "FLOAT32")
    b.op("RESIZE_NEAREST_NEIGHBOR", [x, i32(b, "size66", [6, 6])], [rn], "ResizeNearestNeighborOptions",
         {"align_corners": False, "half_pixel_centers": False})
    mean = b.tensor("mean", [1, 1, 1, 3], "FLOAT32")
    b.op("MEAN", [x, i32(b, "mean_axes", [1, 2])], [mean], "ReducerOptions", {"keep_dims": True})
    rb1 = b.tensor("output_bilinear_from_1x1", [1, 3, 3, 3], "FLOAT32")
    b.op("RESIZE_BILINEAR", [mean, i32(b, "size33", [3, 3])], [rb1], "ResizeBilinearOptions",
         {"align_corners": True, "half_pixel_centers": False})
    w = b.tensor("tconv_w", [2, 3, 3, 3], "FLOAT32", r.normal(0, 0.3, (2, 3, 3, 3)))
    bias = b.tensor("tconv_b", [2], "FLOAT32", r.normal(0, 0.1, 2))
    tc = b.tensor("output_tconv", [1, 8, 8, 2], "FLOAT32")
    b.op("TRANSPOSE_CONV", [i32(b, "tconv_shape", [1, 8, 8, 2]), w, x, bias], [tc], "TransposeConvOptions",
         {"padding": "SAME", "stride_w": 2, "stride_h": 2, **act("NONE")})
    mw = b.tensor("matmul_w", [1, 1, 3, 5], "FLOAT32", r.normal(0, 0.5, (1, 1, 3, 5)))
    bm = b.tensor("output_matmul", [1, 4, 4, 5], "FLOAT32")
    b.op("BATCH_MATMUL", [x, mw], [bm], "BatchMatMulOptions", {"adj_x": False, "adj_y": False})
    b.inputs = [x]
    b.outputs = [tr, ss, out_pack, sv1, out_expand, d2s, s2d, rb, rn, rb1, tc, bm]


def shapes_uint8(b: Builder) -> None:
    """uint8 data movement in the quantized domain: RESIZE_NEAREST_NEIGHBOR,
    TRANSPOSE, STRIDED_SLICE, SPLIT, PACK, UNPACK, TILE, GATHER, DEPTH_TO_SPACE,
    MAXIMUM, REDUCE_MAX, GREATER + SELECT_V2, MEAN, PAD and a per-tensor
    TRANSPOSE_CONV; every operator is bit-exact against TFLite."""
    r = b.rng
    s, zp = 0.02, 128
    x = b.tensor("input", [1, 4, 4, 4], "UINT8", scale=s, zero_point=zp)

    def q(name, shape):
        return b.tensor(name, shape, "UINT8", scale=s, zero_point=zp)

    rn = q("output_nearest", [1, 8, 8, 4])
    b.op("RESIZE_NEAREST_NEIGHBOR", [x, i32(b, "size88", [8, 8])], [rn], "ResizeNearestNeighborOptions",
         {"align_corners": False, "half_pixel_centers": True})
    tr = q("transposed", [1, 4, 4, 4])
    b.op("TRANSPOSE", [x, i32(b, "perm", [0, 2, 1, 3])], [tr], "TransposeOptions", {})
    ss = q("strided", [1, 2, 4, 4])
    b.op("STRIDED_SLICE", [tr, i32(b, "ss_begin", [0, 0, 0, 0]), i32(b, "ss_end", [1, 4, 4, 4]),
                           i32(b, "ss_strides", [1, 2, 1, 1])], [ss], "StridedSliceOptions",
         {"begin_mask": 0, "end_mask": 0, "ellipsis_mask": 0, "new_axis_mask": 0, "shrink_axis_mask": 0})
    halves = [q(f"split{i}", [1, 4, 4, 2]) for i in range(2)]
    b.op("SPLIT", [i32(b, "split_axis", 3), x], halves, "SplitOptions", {"num_splits": 2})
    mx = q("maximum", [1, 4, 4, 2])
    b.op("MAXIMUM", halves, [mx], "MaximumMinimumOptions", {})
    packed = q("packed", [2, 1, 4, 4, 2])
    b.op("PACK", halves, [packed], "PackOptions", {"values_count": 2, "axis": 0})
    unpacked = [q(f"unpacked{i}", [1, 4, 4, 2]) for i in range(2)]
    b.op("UNPACK", [packed], unpacked, "UnpackOptions", {"num": 2, "axis": 0})
    tl = q("tiled", [1, 4, 4, 4])
    b.op("TILE", [unpacked[1], i32(b, "multiples", [1, 1, 1, 2])], [tl], "TileOptions", {})
    ga = q("output_gather", [1, 3, 4, 4])
    b.op("GATHER", [tl, i32(b, "indices", [3, 1, 1])], [ga], "GatherOptions", {"axis": 1, "batch_dims": 0})
    d2s = q("output_d2s", [1, 8, 8, 1])
    b.op("DEPTH_TO_SPACE", [x], [d2s], "DepthToSpaceOptions", {"block_size": 2})
    rmax = q("reduce_max", [1, 1, 1, 4])
    b.op("REDUCE_MAX", [x, i32(b, "axes12", [1, 2])], [rmax], "ReducerOptions", {"keep_dims": True})
    gt = b.tensor("greater", [1, 4, 4, 4], "BOOL")
    b.op("GREATER", [x, rmax], [gt], "GreaterOptions", {})
    sel = q("output_select", [1, 4, 4, 4])
    b.op("SELECT_V2", [gt, x, tr], [sel], "SelectV2Options", {})
    mean = q("mean", [1, 1, 1, 4])
    b.op("MEAN", [ss, i32(b, "mean_axes", [1, 2])], [mean], "ReducerOptions", {"keep_dims": True})
    pads = i32(b, "pads", [[0, 0], [1, 1], [1, 1], [0, 0]])
    padded = q("output_pad", [1, 3, 3, 4])
    b.op("PAD", [mean, pads], [padded], "PadOptions", {})
    w_s, w_zp = 0.01, 125
    w = b.tensor("tconv_w", [2, 3, 3, 4], "UINT8", r.integers(0, 256, (2, 3, 3, 4)), scale=w_s, zero_point=w_zp)
    bias = b.tensor("tconv_b", [2], "INT32", r.integers(-500, 500, 2), scale=s * w_s, zero_point=0)
    tc = b.tensor("output_tconv", [1, 8, 8, 2], "UINT8", scale=0.05, zero_point=120)
    b.op("TRANSPOSE_CONV", [i32(b, "tconv_shape", [1, 8, 8, 2]), w, x, bias], [tc], "TransposeConvOptions",
         {"padding": "SAME", "stride_w": 2, "stride_h": 2, **act("NONE")})
    b.inputs = [x]
    b.outputs = [rn, mx, ga, d2s, sel, padded, tc]


# Float cases whose operators follow libm or accumulate in another order
# agree to float32 rounding, not bit for bit (see docs/CONVERSION.md).
TOLERANCES = {
    "math_f32": {"max_abs": 1e-6, "rel": 4e-6},
    "shapes_f32": {"max_abs": 1e-6, "rel": 2e-6},
}

CASES = {
    "conv_f32": conv_f32,
    "elementwise_f32": elementwise_f32,
    "conv_uint8": conv_uint8,
    "conv_int8": conv_int8,
    "math_f32": math_f32,
    "shapes_f32": shapes_f32,
    "shapes_uint8": shapes_uint8,
}


def flatc() -> str:
    exe = os.environ.get("FLATC") or shutil.which("flatc")
    if not exe:
        sys.exit("error: flatc not found; install it or set FLATC")
    return exe


def build(name: str, seed: int) -> None:
    case = OUT / name
    case.mkdir(parents=True, exist_ok=True)
    for old in case.glob("*.npy"):
        old.unlink()
    b = Builder(np.random.default_rng(seed))
    CASES[name](b)
    json_path = case / "model.json"
    json_path.write_text(json.dumps(b.model(f"tosa-convert test case {name}"), indent=1) + "\n", newline="\n")
    subprocess.run([flatc(), "-b", "-o", str(case), str(SCHEMA), str(json_path)], check=True)
    meta = {"generator": "scripts/tflite_testcases.py", "seed": seed, "description": CASES[name].__doc__}
    desc = record_outputs(case / "model.tflite", case, seed + 1, meta, TOLERANCES.get(name))
    print(f"{name}: {len(b.operators)} operators, {len(desc['ofm_name'])} output(s)")


def main() -> None:
    names = sys.argv[1:] or list(CASES)
    for name in names:
        if name not in CASES:
            sys.exit(f"error: unknown case {name}; choose from {', '.join(CASES)}")
        build(name, seed=7)


if __name__ == "__main__":
    main()
