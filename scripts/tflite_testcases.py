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


CASES = {
    "conv_f32": conv_f32,
    "elementwise_f32": elementwise_f32,
    "conv_uint8": conv_uint8,
    "conv_int8": conv_int8,
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
    desc = record_outputs(case / "model.tflite", case, seed + 1, meta)
    print(f"{name}: {len(b.operators)} operators, {len(desc['ofm_name'])} output(s)")


def main() -> None:
    names = sys.argv[1:] or list(CASES)
    for name in names:
        if name not in CASES:
            sys.exit(f"error: unknown case {name}; choose from {', '.join(CASES)}")
        build(name, seed=7)


if __name__ == "__main__":
    main()
