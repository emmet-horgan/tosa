#!/usr/bin/env python3
"""Generate equivalence test fixtures by running the TOSA reference model.

For each op x dtype combination, this script:
1. Builds a minimal TOSA graph using TosaSerializer
2. Generates random input data
3. Runs through tosa_reference_model.run() to get ground-truth output
4. Saves graph.tosa + input .npy + expected output .npy

The Rust integration tests in equivalence.rs then load these and verify.

Usage:
    cd /home/ehorgan/tosa-tools
    .venv/bin/python tosa-interp/tests/test_equivalence.py
"""

import sys
import json
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "serialization" / "python"))

import tosa_serializer as ts
from tosa.Op import Op
from tosa.DType import DType
import tosa_reference_model

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# Map DType enum to numpy dtype
DTYPE_TO_NP = {
    DType.FP32: np.float32,
    DType.FP16: np.float16,
    DType.INT8: np.int8,
    DType.INT16: np.int16,
    DType.INT32: np.int32,
    DType.INT64: np.int64,
    DType.BOOL: np.bool_,
}

DTYPE_NAMES = {
    DType.FP32: "f32",
    DType.FP16: "f16",
    DType.INT8: "i8",
    DType.INT16: "i16",
    DType.INT32: "i32",
    DType.INT64: "i64",
    DType.BOOL: "bool",
}


def np_dtype(dtype):
    return DTYPE_TO_NP[dtype]


def random_tensor(shape, dtype, low=-5, high=5):
    ndt = np_dtype(dtype)
    if ndt == np.bool_:
        return np.random.choice([True, False], size=shape).astype(np.bool_)
    elif np.issubdtype(ndt, np.integer):
        info = np.iinfo(ndt)
        lo = max(low, info.min + 1)  # avoid INT_MIN for abs safety
        hi = min(high, info.max)
        return np.random.randint(lo, hi, size=shape).astype(ndt)
    else:
        return np.random.uniform(low, high, size=shape).astype(ndt)


def save_fixture(name, graph_bytes, inputs, expected_outputs, metadata=None):
    """Save a test fixture directory."""
    d = FIXTURES_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "graph.tosa").write_bytes(graph_bytes)

    input_names = []
    for i, (tensor_name, arr) in enumerate(inputs.items()):
        fname = f"input_{i}.npy"
        np.save(d / fname, np.ascontiguousarray(arr))
        input_names.append(tensor_name)

    output_names = []
    for i, (tensor_name, arr) in enumerate(expected_outputs.items()):
        fname = f"expected_{i}.npy"
        np.save(d / fname, np.ascontiguousarray(arr))
        output_names.append(tensor_name)

    meta = {
        "input_names": input_names,
        "output_names": output_names,
        "dtype": metadata.get("dtype", "") if metadata else "",
        "op": metadata.get("op", "") if metadata else "",
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


def run_ref_model(graph_bytes, input_arrays):
    """Run graph through reference model, return (outputs, status)."""
    outputs, status = tosa_reference_model.run(
        graph_bytes,
        input_arrays,
        verbosity="NONE",
    )
    return outputs, status


# ---------------------------------------------------------------------------
# Op-specific graph builders
# ---------------------------------------------------------------------------

def build_binary_op(op_enum, attr_method, dtype, shape=(4, 4), attr_args=None):
    """Build a graph: input_0 OP input_1 -> output."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    inp0 = ts.TosaSerializerTensor("input_0", list(shape), dtype)
    inp1 = ts.TosaSerializerTensor("input_1", list(shape), dtype)
    ser.addInputTensor(inp0)
    ser.addInputTensor(inp1)
    out = ser.addOutput(list(shape), dtype)
    attr = ts.TosaSerializerAttribute()
    if attr_args:
        getattr(attr, attr_method)(*attr_args)
    else:
        getattr(attr, attr_method)()
    ser.addOperator(op_enum, ["input_0", "input_1"], [out.name], attr)
    return ser.serialize(), ["input_0", "input_1"], [out.name]


def build_mul_op(dtype, shape=(4, 4)):
    """MUL has a special shift input (rank-1, i8)."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    inp0 = ts.TosaSerializerTensor("input_0", list(shape), dtype)
    inp1 = ts.TosaSerializerTensor("input_1", list(shape), dtype)
    ser.addInputTensor(inp0)
    ser.addInputTensor(inp1)
    shift = ts.TosaSerializerTensor("shift", [1], DType.INT8)
    ser.addInputTensor(shift)
    out = ser.addOutput(list(shape), dtype)
    attr = ts.TosaSerializerAttribute()
    attr.MulAttribute()
    ser.addOperator(Op.MUL, ["input_0", "input_1", "shift"], [out.name], attr)
    return ser.serialize(), ["input_0", "input_1", "shift"], [out.name]


def build_unary_op(op_enum, attr_method, dtype, shape=(8,)):
    """Build a graph: OP(input_0) -> output."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    inp = ts.TosaSerializerTensor("input_0", list(shape), dtype)
    ser.addInputTensor(inp)
    out = ser.addOutput(list(shape), dtype)
    attr = ts.TosaSerializerAttribute()
    getattr(attr, attr_method)()
    ser.addOperator(op_enum, ["input_0"], [out.name], attr)
    return ser.serialize(), ["input_0"], [out.name]


def build_negate_op(dtype, shape=(8,)):
    """NEGATE requires 3 inputs: [input, input1_zp, output_zp]."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    inp = ts.TosaSerializerTensor("input_0", list(shape), dtype)
    inp_zp = ts.TosaSerializerTensor("input_zp", [1], dtype)
    out_zp = ts.TosaSerializerTensor("output_zp", [1], dtype)
    ser.addInputTensor(inp)
    ser.addInputTensor(inp_zp)
    ser.addInputTensor(out_zp)
    out = ser.addOutput(list(shape), dtype)
    attr = ts.TosaSerializerAttribute()
    attr.NegateAttribute()
    ser.addOperator(Op.NEGATE, ["input_0", "input_zp", "output_zp"], [out.name], attr)
    return ser.serialize(), ["input_0", "input_zp", "output_zp"], [out.name]


def build_clamp_op(dtype, shape=(8,), min_val=None, max_val=None):
    """Build CLAMP with serialized min/max bytes."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    inp = ts.TosaSerializerTensor("input_0", list(shape), dtype)
    ser.addInputTensor(inp)
    out = ser.addOutput(list(shape), dtype)

    if dtype in (DType.FP32, DType.FP16):
        min_v = min_val if min_val is not None else -1.0
        max_v = max_val if max_val is not None else 1.0
    else:
        min_v = min_val if min_val is not None else -2
        max_v = max_val if max_val is not None else 2

    min_bytes = ts.TosaSerializer.convertDataToUint8Vec(dtype, [min_v])
    max_bytes = ts.TosaSerializer.convertDataToUint8Vec(dtype, [max_v])
    # Align to 8 bytes
    while len(min_bytes) % 8 != 0:
        min_bytes.append(0)
    while len(max_bytes) % 8 != 0:
        max_bytes.append(0)

    attr = ts.TosaSerializerAttribute()
    attr.ClampAttribute(min_bytes, max_bytes, ts.NanPropagationMode.PROPAGATE)
    ser.addOperator(Op.CLAMP, ["input_0"], [out.name], attr)
    return ser.serialize(), ["input_0"], [out.name]


def build_comparison_op(op_enum, attr_method, dtype, shape=(6,)):
    """Build a comparison graph: input_0 CMP input_1 -> bool output."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    inp0 = ts.TosaSerializerTensor("input_0", list(shape), dtype)
    inp1 = ts.TosaSerializerTensor("input_1", list(shape), dtype)
    ser.addInputTensor(inp0)
    ser.addInputTensor(inp1)
    out = ser.addOutput(list(shape), DType.BOOL)
    attr = ts.TosaSerializerAttribute()
    getattr(attr, attr_method)()
    ser.addOperator(op_enum, ["input_0", "input_1"], [out.name], attr)
    return ser.serialize(), ["input_0", "input_1"], [out.name]


def build_select_op(dtype, shape=(6,)):
    """SELECT: 3 inputs [condition(bool), a, b] -> output."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    cond = ts.TosaSerializerTensor("condition", list(shape), DType.BOOL)
    a = ts.TosaSerializerTensor("input_a", list(shape), dtype)
    b = ts.TosaSerializerTensor("input_b", list(shape), dtype)
    ser.addInputTensor(cond)
    ser.addInputTensor(a)
    ser.addInputTensor(b)
    out = ser.addOutput(list(shape), dtype)
    attr = ts.TosaSerializerAttribute()
    attr.SelectAttribute()
    ser.addOperator(Op.SELECT, ["condition", "input_a", "input_b"], [out.name], attr)
    return ser.serialize(), ["condition", "input_a", "input_b"], [out.name]


def build_reduce_op(op_enum, attr_method, dtype, shape=(3, 4), axis=1, extra_args=None):
    """Build a reduction graph."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    inp = ts.TosaSerializerTensor("input_0", list(shape), dtype)
    ser.addInputTensor(inp)
    out_shape = list(shape)
    out_shape[axis] = 1
    # Determine output dtype
    if "All" in attr_method or "Any" in attr_method:
        out_dtype = DType.BOOL
    else:
        out_dtype = dtype
    out = ser.addOutput(out_shape, out_dtype)
    attr = ts.TosaSerializerAttribute()
    args = [axis] + (extra_args or [])
    getattr(attr, attr_method)(*args)
    ser.addOperator(op_enum, ["input_0"], [out.name], attr)
    return ser.serialize(), ["input_0"], [out.name]


def build_transpose_op(dtype, shape=(2, 3, 4), perms=(1, 2, 0)):
    """Build TRANSPOSE."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    inp = ts.TosaSerializerTensor("input_0", list(shape), dtype)
    ser.addInputTensor(inp)
    out_shape = [shape[p] for p in perms]
    out = ser.addOutput(out_shape, dtype)
    attr = ts.TosaSerializerAttribute()
    attr.TransposeAttribute(list(perms))
    ser.addOperator(Op.TRANSPOSE, ["input_0"], [out.name], attr)
    return ser.serialize(), ["input_0"], [out.name]


def build_concat_op(dtype, shape=(2, 3), axis=0):
    """Build CONCAT of two tensors along axis."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    inp0 = ts.TosaSerializerTensor("input_0", list(shape), dtype)
    inp1 = ts.TosaSerializerTensor("input_1", list(shape), dtype)
    ser.addInputTensor(inp0)
    ser.addInputTensor(inp1)
    out_shape = list(shape)
    out_shape[axis] *= 2
    out = ser.addOutput(out_shape, dtype)
    attr = ts.TosaSerializerAttribute()
    attr.ConcatAttribute(axis)
    ser.addOperator(Op.CONCAT, ["input_0", "input_1"], [out.name], attr)
    return ser.serialize(), ["input_0", "input_1"], [out.name]


def build_pad_op(dtype, shape=(3, 3)):
    """Build PAD: pad with 1 on each side -> [5,5].
    3 inputs: [input, padding(SHAPE const), pad_value(scalar)]."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    inp = ts.TosaSerializerTensor("input_0", list(shape), dtype)
    ser.addInputTensor(inp)
    # Padding as SHAPE const: [before_dim0, after_dim0, before_dim1, after_dim1]
    pad_const = ser.addConst([4], DType.SHAPE, np.array([1, 1, 1, 1], dtype=np.int64))
    # Pad value scalar
    pad_val = ts.TosaSerializerTensor("pad_val", [1], dtype)
    ser.addInputTensor(pad_val)
    out_shape = [s + 2 for s in shape]
    out = ser.addOutput(out_shape, dtype)
    attr = ts.TosaSerializerAttribute()
    attr.PadAttribute()
    ser.addOperator(Op.PAD, ["input_0", pad_const.name, "pad_val"], [out.name], attr)
    return ser.serialize(), ["input_0", "pad_val"], [out.name]


def build_tile_op(dtype, shape=(2, 3), multiples=(2, 1)):
    """Build TILE: 2 inputs [input, multiples(SHAPE const)]."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    inp = ts.TosaSerializerTensor("input_0", list(shape), dtype)
    ser.addInputTensor(inp)
    mult_const = ser.addConst([len(multiples)], DType.SHAPE,
                              np.array(multiples, dtype=np.int64))
    out_shape = [s * m for s, m in zip(shape, multiples)]
    out = ser.addOutput(out_shape, dtype)
    attr = ts.TosaSerializerAttribute()
    attr.TileAttribute()
    ser.addOperator(Op.TILE, ["input_0", mult_const.name], [out.name], attr)
    return ser.serialize(), ["input_0"], [out.name]


def build_reverse_op(dtype, shape=(4,), axis=0):
    """Build REVERSE."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    inp = ts.TosaSerializerTensor("input_0", list(shape), dtype)
    ser.addInputTensor(inp)
    out = ser.addOutput(list(shape), dtype)
    attr = ts.TosaSerializerAttribute()
    attr.ReverseAttribute(axis)
    ser.addOperator(Op.REVERSE, ["input_0"], [out.name], attr)
    return ser.serialize(), ["input_0"], [out.name]


def build_gather_op(dtype, N=2, K=4, C=3, W=2):
    """Build GATHER: values [N, K, C], indices [N, W] -> output [N, W, C]."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    values = ts.TosaSerializerTensor("input_0", [N, K, C], dtype)
    indices = ts.TosaSerializerTensor("indices", [N, W], DType.INT32)
    ser.addInputTensor(values)
    ser.addInputTensor(indices)
    out = ser.addOutput([N, W, C], dtype)
    attr = ts.TosaSerializerAttribute()
    attr.GatherAttribute()
    ser.addOperator(Op.GATHER, ["input_0", "indices"], [out.name], attr)
    return ser.serialize(), ["input_0", "indices"], [out.name]


def build_scatter_op(dtype, N=2, K=4, C=3, W=2):
    """Build SCATTER: values_in [N, K, C], indices [N, W], input [N, W, C] -> output [N, K, C]."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    values_in = ts.TosaSerializerTensor("values_in", [N, K, C], dtype)
    indices = ts.TosaSerializerTensor("indices", [N, W], DType.INT32)
    inp = ts.TosaSerializerTensor("input_0", [N, W, C], dtype)
    ser.addInputTensor(values_in)
    ser.addInputTensor(indices)
    ser.addInputTensor(inp)
    out = ser.addOutput([N, K, C], dtype)
    attr = ts.TosaSerializerAttribute()
    attr.ScatterAttribute()
    ser.addOperator(Op.SCATTER, ["values_in", "indices", "input_0"], [out.name], attr)
    return ser.serialize(), ["values_in", "indices", "input_0"], [out.name]


def build_conv2d_op(in_dtype):
    """Build CONV2D: [1,4,4,2] * [3,2,2,2] + bias -> [1,3,3,3].
    5 inputs: input, weight, bias, input_zp, weight_zp."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    inp = ts.TosaSerializerTensor("input_0", [1, 4, 4, 2], in_dtype)
    weight = ts.TosaSerializerTensor("weight", [3, 2, 2, 2], in_dtype)
    bias = ts.TosaSerializerTensor("bias", [3], in_dtype)
    input_zp = ts.TosaSerializerTensor("input_zp", [1], in_dtype)
    weight_zp = ts.TosaSerializerTensor("weight_zp", [1], in_dtype)
    ser.addInputTensor(inp)
    ser.addInputTensor(weight)
    ser.addInputTensor(bias)
    ser.addInputTensor(input_zp)
    ser.addInputTensor(weight_zp)
    out = ser.addOutput([1, 3, 3, 3], in_dtype)
    attr = ts.TosaSerializerAttribute()
    acc_type = DType.FP32 if in_dtype in (DType.FP32, DType.FP16) else DType.INT32
    attr.Conv2dAttribute([0, 0, 0, 0], [1, 1], [1, 1], False, acc_type)
    ser.addOperator(Op.CONV2D, ["input_0", "weight", "bias", "input_zp", "weight_zp"],
                    [out.name], attr)
    return ser.serialize(), ["input_0", "weight", "bias", "input_zp", "weight_zp"], [out.name]


def build_depthwise_conv2d_op(in_dtype):
    """Build DEPTHWISE_CONV2D: input [1,4,4,2], weight [2,2,2,1], bias [2], output [1,3,3,2].
    5 inputs: input, weight, bias, input_zp, weight_zp."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    inp = ts.TosaSerializerTensor("input_0", [1, 4, 4, 2], in_dtype)
    weight = ts.TosaSerializerTensor("weight", [2, 2, 2, 1], in_dtype)
    bias = ts.TosaSerializerTensor("bias", [2], in_dtype)
    input_zp = ts.TosaSerializerTensor("input_zp", [1], in_dtype)
    weight_zp = ts.TosaSerializerTensor("weight_zp", [1], in_dtype)
    ser.addInputTensor(inp)
    ser.addInputTensor(weight)
    ser.addInputTensor(bias)
    ser.addInputTensor(input_zp)
    ser.addInputTensor(weight_zp)
    out = ser.addOutput([1, 3, 3, 2], in_dtype)
    attr = ts.TosaSerializerAttribute()
    acc_type = DType.FP32 if in_dtype in (DType.FP32, DType.FP16) else DType.INT32
    attr.DepthwiseConv2dAttribute([0, 0, 0, 0], [1, 1], [1, 1], False, acc_type)
    ser.addOperator(Op.DEPTHWISE_CONV2D,
                    ["input_0", "weight", "bias", "input_zp", "weight_zp"],
                    [out.name], attr)
    return ser.serialize(), ["input_0", "weight", "bias", "input_zp", "weight_zp"], [out.name]


def build_transpose_conv2d_op(in_dtype):
    """Build TRANSPOSE_CONV2D: input [1,2,2,1], weight [1,2,2,1], bias [1], output [1,3,3,1].
    5 inputs: input, weight, bias, input_zp, weight_zp."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    inp = ts.TosaSerializerTensor("input_0", [1, 2, 2, 1], in_dtype)
    weight = ts.TosaSerializerTensor("weight", [1, 2, 2, 1], in_dtype)
    bias = ts.TosaSerializerTensor("bias", [1], in_dtype)
    input_zp = ts.TosaSerializerTensor("input_zp", [1], in_dtype)
    weight_zp = ts.TosaSerializerTensor("weight_zp", [1], in_dtype)
    ser.addInputTensor(inp)
    ser.addInputTensor(weight)
    ser.addInputTensor(bias)
    ser.addInputTensor(input_zp)
    ser.addInputTensor(weight_zp)
    out = ser.addOutput([1, 3, 3, 1], in_dtype)
    attr = ts.TosaSerializerAttribute()
    acc_type = DType.FP32 if in_dtype in (DType.FP32, DType.FP16) else DType.INT32
    attr.TransposeConv2dAttribute([0, 0, 0, 0], [1, 1], False, acc_type)
    ser.addOperator(Op.TRANSPOSE_CONV2D,
                    ["input_0", "weight", "bias", "input_zp", "weight_zp"],
                    [out.name], attr)
    return ser.serialize(), ["input_0", "weight", "bias", "input_zp", "weight_zp"], [out.name]


def build_conv3d_op(in_dtype):
    """Build CONV3D: input [1,3,3,3,1], weight [1,2,2,2,1], bias [1], output [1,2,2,2,1].
    5 inputs: input, weight, bias, input_zp, weight_zp."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    inp = ts.TosaSerializerTensor("input_0", [1, 3, 3, 3, 1], in_dtype)
    weight = ts.TosaSerializerTensor("weight", [1, 2, 2, 2, 1], in_dtype)
    bias = ts.TosaSerializerTensor("bias", [1], in_dtype)
    input_zp = ts.TosaSerializerTensor("input_zp", [1], in_dtype)
    weight_zp = ts.TosaSerializerTensor("weight_zp", [1], in_dtype)
    ser.addInputTensor(inp)
    ser.addInputTensor(weight)
    ser.addInputTensor(bias)
    ser.addInputTensor(input_zp)
    ser.addInputTensor(weight_zp)
    out = ser.addOutput([1, 2, 2, 2, 1], in_dtype)
    attr = ts.TosaSerializerAttribute()
    acc_type = DType.FP32 if in_dtype in (DType.FP32, DType.FP16) else DType.INT32
    attr.Conv3dAttribute([0, 0, 0, 0, 0, 0], [1, 1, 1], [1, 1, 1], False, acc_type)
    ser.addOperator(Op.CONV3D,
                    ["input_0", "weight", "bias", "input_zp", "weight_zp"],
                    [out.name], attr)
    return ser.serialize(), ["input_0", "weight", "bias", "input_zp", "weight_zp"], [out.name]


def build_matmul_op(dtype):
    """Build MATMUL: [1,3,4] x [1,4,2] -> [1,3,2].
    4 inputs: A, B, A_zp, B_zp."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    inp0 = ts.TosaSerializerTensor("input_0", [1, 3, 4], dtype)
    inp1 = ts.TosaSerializerTensor("input_1", [1, 4, 2], dtype)
    a_zp = ts.TosaSerializerTensor("a_zp", [1], dtype)
    b_zp = ts.TosaSerializerTensor("b_zp", [1], dtype)
    ser.addInputTensor(inp0)
    ser.addInputTensor(inp1)
    ser.addInputTensor(a_zp)
    ser.addInputTensor(b_zp)
    out = ser.addOutput([1, 3, 2], dtype)
    attr = ts.TosaSerializerAttribute()
    attr.MatMulAttribute()
    ser.addOperator(Op.MATMUL, ["input_0", "input_1", "a_zp", "b_zp"], [out.name], attr)
    return ser.serialize(), ["input_0", "input_1", "a_zp", "b_zp"], [out.name]


def build_avg_pool2d_op(dtype):
    """Build AVG_POOL2D: [1,4,4,2] -> [1,3,3,2] with kernel=2, stride=1, pad=0.
    3 inputs: input, input_zp, output_zp."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    inp = ts.TosaSerializerTensor("input_0", [1, 4, 4, 2], dtype)
    input_zp = ts.TosaSerializerTensor("input_zp", [1], dtype)
    output_zp = ts.TosaSerializerTensor("output_zp", [1], dtype)
    ser.addInputTensor(inp)
    ser.addInputTensor(input_zp)
    ser.addInputTensor(output_zp)
    out = ser.addOutput([1, 3, 3, 2], dtype)
    attr = ts.TosaSerializerAttribute()
    acc_type = DType.FP32 if dtype in (DType.FP32, DType.FP16) else DType.INT32
    attr.AvgPool2dAttribute([2, 2], [1, 1], [0, 0, 0, 0], acc_type)
    ser.addOperator(Op.AVG_POOL2D, ["input_0", "input_zp", "output_zp"], [out.name], attr)
    return ser.serialize(), ["input_0", "input_zp", "output_zp"], [out.name]


def build_max_pool2d_op(dtype):
    """Build MAX_POOL2D: [1,4,4,2] -> [1,3,3,2] with kernel=2, stride=1, pad=0.
    1 input only."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    inp = ts.TosaSerializerTensor("input_0", [1, 4, 4, 2], dtype)
    ser.addInputTensor(inp)
    out = ser.addOutput([1, 3, 3, 2], dtype)
    attr = ts.TosaSerializerAttribute()
    attr.MaxPool2dAttribute([2, 2], [1, 1], [0, 0, 0, 0], ts.NanPropagationMode.PROPAGATE)
    ser.addOperator(Op.MAX_POOL2D, ["input_0"], [out.name], attr)
    return ser.serialize(), ["input_0"], [out.name]


def build_argmax_op(dtype, shape=(2, 5, 3), axis=1):
    """Build ARGMAX."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    inp = ts.TosaSerializerTensor("input_0", list(shape), dtype)
    ser.addInputTensor(inp)
    out_shape = list(shape)
    del out_shape[axis]
    out = ser.addOutput(out_shape, DType.INT32)
    attr = ts.TosaSerializerAttribute()
    attr.ArgMaxAttribute(axis, ts.NanPropagationMode.PROPAGATE)
    ser.addOperator(Op.ARGMAX, ["input_0"], [out.name], attr)
    return ser.serialize(), ["input_0"], [out.name]


def build_cast_op(in_dtype, out_dtype, shape=(8,)):
    """Build CAST."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    inp = ts.TosaSerializerTensor("input_0", list(shape), in_dtype)
    ser.addInputTensor(inp)
    out = ser.addOutput(list(shape), out_dtype)
    attr = ts.TosaSerializerAttribute()
    attr.CastAttribute()
    ser.addOperator(Op.CAST, ["input_0"], [out.name], attr)
    return ser.serialize(), ["input_0"], [out.name]


def build_rescale_op(in_dtype, out_dtype, shape=(8,)):
    """Build RESCALE: 5 inputs [input, multiplier, shift, input_zp, output_zp]."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR))
    inp = ts.TosaSerializerTensor("input_0", list(shape), in_dtype)
    mult = ts.TosaSerializerTensor("multiplier", [1], DType.INT32)
    shift = ts.TosaSerializerTensor("shift", [1], DType.INT8)
    inp_zp = ts.TosaSerializerTensor("input_zp", [1], in_dtype)
    out_zp = ts.TosaSerializerTensor("output_zp", [1], out_dtype)
    ser.addInputTensor(inp)
    ser.addInputTensor(mult)
    ser.addInputTensor(shift)
    ser.addInputTensor(inp_zp)
    ser.addInputTensor(out_zp)
    out = ser.addOutput(list(shape), out_dtype)
    attr = ts.TosaSerializerAttribute()
    # RescaleAttribute(scale32, rounding_mode, per_channel, input_unsigned, output_unsigned)
    # RoundingMode.SINGLE_ROUND = 1
    attr.RescaleAttribute(True, 1, False, False, False)
    ser.addOperator(Op.RESCALE, ["input_0", "multiplier", "shift", "input_zp", "output_zp"],
                    [out.name], attr)
    return ser.serialize(), ["input_0", "multiplier", "shift", "input_zp", "output_zp"], [out.name]


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    name: str
    op: str
    build_fn: object
    input_gen: object  # function(input_names) -> dict of name -> np.array


def gen_binary_tests():
    """Generate all binary op test cases."""
    nan_propagate = ts.NanPropagationMode.PROPAGATE
    cases = []

    # Standard binary ops: ADD, SUB
    for op_name, op_enum, attr_method in [
        ("add", Op.ADD, "AddAttribute"),
        ("sub", Op.SUB, "SubAttribute"),
    ]:
        for dtype in [DType.FP32, DType.INT32, DType.INT64]:
            def make_build(oe=op_enum, am=attr_method, dt=dtype):
                return lambda: build_binary_op(oe, am, dt)

            def make_inputs(dt=dtype):
                def gen(input_names):
                    d = {}
                    for name in input_names:
                        d[name] = random_tensor((4, 4), dt)
                    return d
                return gen

            cases.append(TestCase(
                name=f"equiv_{op_name}_{DTYPE_NAMES[dtype]}",
                op=op_name,
                build_fn=make_build(),
                input_gen=make_inputs(),
            ))

    # MAXIMUM, MINIMUM (need nan_mode)
    for op_name, op_enum, attr_method in [
        ("maximum", Op.MAXIMUM, "MaximumAttribute"),
        ("minimum", Op.MINIMUM, "MinimumAttribute"),
    ]:
        for dtype in [DType.FP32, DType.INT32]:
            def make_build(oe=op_enum, am=attr_method, dt=dtype):
                return lambda: build_binary_op(oe, am, dt, attr_args=[nan_propagate])

            def make_inputs(dt=dtype):
                def gen(input_names):
                    d = {}
                    for name in input_names:
                        d[name] = random_tensor((4, 4), dt)
                    return d
                return gen

            cases.append(TestCase(
                name=f"equiv_{op_name}_{DTYPE_NAMES[dtype]}",
                op=op_name,
                build_fn=make_build(),
                input_gen=make_inputs(),
            ))

    # POW: f32 only (base must be >= 0)
    def make_pow_build():
        return lambda: build_binary_op(Op.POW, "PowAttribute", DType.FP32)

    def make_pow_inputs():
        def gen(input_names):
            return {
                "input_0": np.random.uniform(0.1, 3.0, (4, 4)).astype(np.float32),
                "input_1": np.random.uniform(0.5, 2.0, (4, 4)).astype(np.float32),
            }
        return gen

    cases.append(TestCase(name="equiv_pow_f32", op="pow",
                          build_fn=make_pow_build(), input_gen=make_pow_inputs()))

    # ARITHMETIC_RIGHT_SHIFT: i8, i16, i32
    for dtype in [DType.INT8, DType.INT16, DType.INT32]:
        def make_ars_build(dt=dtype):
            return lambda: build_binary_op(Op.ARITHMETIC_RIGHT_SHIFT,
                                           "ArithmeticRightShiftAttribute", dt,
                                           attr_args=[True])

        def make_ars_inputs(dt=dtype):
            def gen(input_names):
                ndt = np_dtype(dt)
                bitwidth = np.iinfo(ndt).bits
                return {
                    "input_0": np.random.randint(-100, 100, (4, 4)).astype(ndt),
                    "input_1": np.random.randint(0, bitwidth, (4, 4)).astype(ndt),
                }
            return gen

        cases.append(TestCase(
            name=f"equiv_arithmetic_right_shift_{DTYPE_NAMES[dtype]}",
            op="arithmetic_right_shift",
            build_fn=make_ars_build(),
            input_gen=make_ars_inputs(),
        ))

    # INTDIV: i32
    def make_intdiv_build():
        return lambda: build_binary_op(Op.INTDIV, "IntDivAttribute", DType.INT32)

    def make_intdiv_inputs():
        def gen(input_names):
            return {
                "input_0": np.random.randint(-20, 20, (4, 4)).astype(np.int32),
                "input_1": np.random.randint(1, 5, (4, 4)).astype(np.int32),
            }
        return gen

    cases.append(TestCase(name="equiv_intdiv_i32", op="intdiv",
                          build_fn=make_intdiv_build(), input_gen=make_intdiv_inputs()))

    # LOGICAL_AND, LOGICAL_OR, LOGICAL_XOR: bool
    for op_name, op_enum, attr_method in [
        ("logical_and", Op.LOGICAL_AND, "LogicalAndAttribute"),
        ("logical_or", Op.LOGICAL_OR, "LogicalOrAttribute"),
        ("logical_xor", Op.LOGICAL_XOR, "LogicalXorAttribute"),
    ]:
        def make_logic_build(oe=op_enum, am=attr_method):
            return lambda: build_binary_op(oe, am, DType.BOOL)

        def make_logic_inputs():
            def gen(input_names):
                d = {}
                for name in input_names:
                    d[name] = np.random.choice([True, False], (4, 4)).astype(np.bool_)
                return d
            return gen

        cases.append(TestCase(
            name=f"equiv_{op_name}_bool",
            op=op_name,
            build_fn=make_logic_build(),
            input_gen=make_logic_inputs(),
        ))

    # LOGICAL_LEFT_SHIFT, LOGICAL_RIGHT_SHIFT: i8, i16, i32
    for op_name, op_enum, attr_method in [
        ("logical_left_shift", Op.LOGICAL_LEFT_SHIFT, "LogicalLeftShiftAttribute"),
        ("logical_right_shift", Op.LOGICAL_RIGHT_SHIFT, "LogicalRightShiftAttribute"),
    ]:
        for dtype in [DType.INT8, DType.INT16, DType.INT32]:
            def make_shift_build(oe=op_enum, am=attr_method, dt=dtype):
                return lambda: build_binary_op(oe, am, dt)

            def make_shift_inputs(dt=dtype):
                def gen(input_names):
                    ndt = np_dtype(dt)
                    bitwidth = np.iinfo(ndt).bits
                    return {
                        "input_0": np.random.randint(0, 50, (4, 4)).astype(ndt),
                        "input_1": np.random.randint(0, bitwidth, (4, 4)).astype(ndt),
                    }
                return gen

            cases.append(TestCase(
                name=f"equiv_{op_name}_{DTYPE_NAMES[dtype]}",
                op=op_name,
                build_fn=make_shift_build(),
                input_gen=make_shift_inputs(),
            ))

    # BITWISE_AND, BITWISE_OR, BITWISE_XOR: i8, i16, i32
    for op_name, op_enum, attr_method in [
        ("bitwise_and", Op.BITWISE_AND, "BitwiseAndAttribute"),
        ("bitwise_or", Op.BITWISE_OR, "BitwiseOrAttribute"),
        ("bitwise_xor", Op.BITWISE_XOR, "BitwiseXorAttribute"),
    ]:
        for dtype in [DType.INT8, DType.INT16, DType.INT32]:
            def make_bw_build(oe=op_enum, am=attr_method, dt=dtype):
                return lambda: build_binary_op(oe, am, dt)

            def make_bw_inputs(dt=dtype):
                def gen(input_names):
                    d = {}
                    for name in input_names:
                        d[name] = random_tensor((4, 4), dt)
                    return d
                return gen

            cases.append(TestCase(
                name=f"equiv_{op_name}_{DTYPE_NAMES[dtype]}",
                op=op_name,
                build_fn=make_bw_build(),
                input_gen=make_bw_inputs(),
            ))

    # MUL: f32, i32 (with shift=0 tensor)
    for dtype in [DType.FP32, DType.INT32]:
        def make_mul_build(dt=dtype):
            return lambda: build_mul_op(dt)

        def make_mul_inputs(dt=dtype):
            def gen(input_names):
                d = {}
                for name in input_names:
                    if name == "shift":
                        d[name] = np.array([0], dtype=np.int8)
                    elif dt in (DType.FP32, DType.FP16):
                        d[name] = np.random.uniform(-3, 3, (4, 4)).astype(np_dtype(dt))
                    else:
                        d[name] = np.random.randint(-5, 5, (4, 4)).astype(np_dtype(dt))
                return d
            return gen

        cases.append(TestCase(
            name=f"equiv_mul_{DTYPE_NAMES[dtype]}",
            op="mul",
            build_fn=make_mul_build(),
            input_gen=make_mul_inputs(),
        ))

    return cases


def gen_unary_tests():
    """Generate unary op test cases."""
    cases = []

    # Float-only unary ops
    float_ops = [
        ("ceil", Op.CEIL, "CeilAttribute", None),
        ("floor", Op.FLOOR, "FloorAttribute", None),
        ("exp", Op.EXP, "ExpAttribute", "exp"),
        ("log", Op.LOG, "LogAttribute", "log"),
        ("sin", Op.SIN, "SinAttribute", None),
        ("cos", Op.COS, "CosAttribute", None),
        ("reciprocal", Op.RECIPROCAL, "ReciprocalAttribute", "reciprocal"),
        ("rsqrt", Op.RSQRT, "RsqrtAttribute", "rsqrt"),
        ("sigmoid", Op.SIGMOID, "SigmoidAttribute", None),
        ("tanh", Op.TANH, "TanhAttribute", None),
        ("erf", Op.ERF, "ErfAttribute", None),
    ]

    for op_name, op_enum, attr_method, special in float_ops:
        def make_build(oe=op_enum, am=attr_method):
            return lambda: build_unary_op(oe, am, DType.FP32, shape=(8,))

        def make_inputs(sp=special):
            def gen(input_names):
                if sp == "log" or sp == "rsqrt":
                    return {"input_0": np.random.uniform(0.1, 5.0, (8,)).astype(np.float32)}
                elif sp == "reciprocal":
                    # Non-zero inputs
                    vals = np.random.uniform(0.5, 5.0, (8,)).astype(np.float32)
                    vals[::2] *= -1  # mix of positive and negative
                    return {"input_0": vals}
                elif sp == "exp":
                    return {"input_0": np.random.uniform(-2, 2, (8,)).astype(np.float32)}
                else:
                    return {"input_0": np.random.uniform(-3, 3, (8,)).astype(np.float32)}
            return gen

        cases.append(TestCase(
            name=f"equiv_{op_name}_f32",
            op=op_name,
            build_fn=make_build(),
            input_gen=make_inputs(),
        ))

    # ABS: f32, i32
    for dtype in [DType.FP32, DType.INT32]:
        def make_abs_build(dt=dtype):
            return lambda: build_unary_op(Op.ABS, "AbsAttribute", dt, shape=(8,))

        def make_abs_inputs(dt=dtype):
            def gen(input_names):
                if dt == DType.FP32:
                    return {"input_0": np.random.uniform(-5, 5, (8,)).astype(np.float32)}
                else:
                    # Avoid INT_MIN for safety
                    ndt = np_dtype(dt)
                    return {"input_0": np.random.randint(-100, 100, (8,)).astype(ndt)}
            return gen

        cases.append(TestCase(
            name=f"equiv_abs_{DTYPE_NAMES[dtype]}",
            op="abs",
            build_fn=make_abs_build(),
            input_gen=make_abs_inputs(),
        ))

    # NEGATE: f32, i32, i8, i16
    for dtype in [DType.FP32, DType.INT32, DType.INT8, DType.INT16]:
        def make_negate_build(dt=dtype):
            return lambda: build_negate_op(dt, shape=(8,))

        def make_negate_inputs(dt=dtype):
            def gen(input_names):
                ndt = np_dtype(dt)
                if dt == DType.FP32:
                    return {
                        "input_0": np.random.uniform(-5, 5, (8,)).astype(ndt),
                        "input_zp": np.zeros(1, dtype=ndt),
                        "output_zp": np.zeros(1, dtype=ndt),
                    }
                else:
                    return {
                        "input_0": np.random.randint(-100, 100, (8,)).astype(ndt),
                        "input_zp": np.zeros(1, dtype=ndt),
                        "output_zp": np.zeros(1, dtype=ndt),
                    }
            return gen

        cases.append(TestCase(
            name=f"equiv_negate_{DTYPE_NAMES[dtype]}",
            op="negate",
            build_fn=make_negate_build(),
            input_gen=make_negate_inputs(),
        ))

    # CLZ: i32
    def make_clz_build():
        return lambda: build_unary_op(Op.CLZ, "ClzAttribute", DType.INT32, shape=(8,))

    def make_clz_inputs():
        def gen(input_names):
            # Mix of positive, zero, and negative values
            return {"input_0": np.array([0, 1, 2, 7, 16, 255, 1024, -1], dtype=np.int32)}
        return gen

    cases.append(TestCase(name="equiv_clz_i32", op="clz",
                          build_fn=make_clz_build(), input_gen=make_clz_inputs()))

    # BITWISE_NOT: i8, i16, i32
    for dtype in [DType.INT8, DType.INT16, DType.INT32]:
        def make_bnot_build(dt=dtype):
            return lambda: build_unary_op(Op.BITWISE_NOT, "BitwiseNotAttribute", dt, shape=(8,))

        def make_bnot_inputs(dt=dtype):
            def gen(input_names):
                return {"input_0": random_tensor((8,), dt)}
            return gen

        cases.append(TestCase(
            name=f"equiv_bitwise_not_{DTYPE_NAMES[dtype]}",
            op="bitwise_not",
            build_fn=make_bnot_build(),
            input_gen=make_bnot_inputs(),
        ))

    # LOGICAL_NOT: bool
    def make_lnot_build():
        return lambda: build_unary_op(Op.LOGICAL_NOT, "LogicalNotAttribute", DType.BOOL, shape=(8,))

    def make_lnot_inputs():
        def gen(input_names):
            return {"input_0": np.random.choice([True, False], (8,)).astype(np.bool_)}
        return gen

    cases.append(TestCase(name="equiv_logical_not_bool", op="logical_not",
                          build_fn=make_lnot_build(), input_gen=make_lnot_inputs()))

    # CLAMP: f32
    def make_clamp_build():
        return lambda: build_clamp_op(DType.FP32, shape=(8,), min_val=-1.0, max_val=1.0)

    def make_clamp_inputs():
        def gen(input_names):
            return {"input_0": np.random.uniform(-5, 5, (8,)).astype(np.float32)}
        return gen

    cases.append(TestCase(
        name="equiv_clamp_f32",
        op="clamp",
        build_fn=make_clamp_build(),
        input_gen=make_clamp_inputs(),
    ))

    # IDENTITY: f32, i32, i8
    for dtype in [DType.FP32, DType.INT32, DType.INT8]:
        def make_identity_build(dt=dtype):
            return lambda: build_unary_op(Op.IDENTITY, "IdentityAttribute", dt, shape=(8,))

        def make_identity_inputs(dt=dtype):
            def gen(input_names):
                return {"input_0": random_tensor((8,), dt)}
            return gen

        cases.append(TestCase(
            name=f"equiv_identity_{DTYPE_NAMES[dtype]}",
            op="identity",
            build_fn=make_identity_build(),
            input_gen=make_identity_inputs(),
        ))

    return cases


def gen_comparison_tests():
    """Generate comparison op test cases."""
    cases = []
    for op_name, op_enum, attr_method in [
        ("equal", Op.EQUAL, "EqualAttribute"),
        ("greater", Op.GREATER, "GreaterAttribute"),
        ("greater_equal", Op.GREATER_EQUAL, "GreaterEqualAttribute"),
    ]:
        for dtype in [DType.FP32, DType.INT32]:
            def make_build(oe=op_enum, am=attr_method, dt=dtype):
                return lambda: build_comparison_op(oe, am, dt, shape=(6,))

            def make_inputs(dt=dtype):
                def gen(input_names):
                    d = {}
                    for name in input_names:
                        d[name] = random_tensor((6,), dt)
                    return d
                return gen

            cases.append(TestCase(
                name=f"equiv_{op_name}_{DTYPE_NAMES[dtype]}",
                op=op_name,
                build_fn=make_build(),
                input_gen=make_inputs(),
            ))
    return cases


def gen_select_tests():
    """Generate SELECT test cases."""
    cases = []
    for dtype in [DType.FP32, DType.INT32, DType.INT8, DType.INT16]:
        def make_build(dt=dtype):
            return lambda: build_select_op(dt, shape=(6,))

        def make_inputs(dt=dtype):
            def gen(input_names):
                return {
                    "condition": np.random.choice([True, False], (6,)).astype(np.bool_),
                    "input_a": random_tensor((6,), dt),
                    "input_b": random_tensor((6,), dt),
                }
            return gen

        cases.append(TestCase(
            name=f"equiv_select_{DTYPE_NAMES[dtype]}",
            op="select",
            build_fn=make_build(),
            input_gen=make_inputs(),
        ))
    return cases


def gen_reduce_tests():
    """Generate reduction op test cases."""
    nan_propagate = ts.NanPropagationMode.PROPAGATE
    cases = []

    # REDUCE_SUM: f32, i32
    for dtype in [DType.FP32, DType.INT32]:
        def make_build(dt=dtype):
            return lambda: build_reduce_op(Op.REDUCE_SUM, "ReduceSumAttribute", dt,
                                           shape=(3, 4), axis=1)

        def make_inputs(dt=dtype):
            def gen(input_names):
                return {"input_0": random_tensor((3, 4), dt, low=-3, high=3)}
            return gen

        cases.append(TestCase(
            name=f"equiv_reduce_sum_{DTYPE_NAMES[dtype]}",
            op="reduce_sum",
            build_fn=make_build(),
            input_gen=make_inputs(),
        ))

    # REDUCE_MAX: f32, i32 (nan_mode)
    for dtype in [DType.FP32, DType.INT32]:
        def make_build(dt=dtype):
            return lambda: build_reduce_op(Op.REDUCE_MAX, "ReduceMaxAttribute", dt,
                                           shape=(3, 4), axis=1, extra_args=[nan_propagate])

        def make_inputs(dt=dtype):
            def gen(input_names):
                return {"input_0": random_tensor((3, 4), dt)}
            return gen

        cases.append(TestCase(
            name=f"equiv_reduce_max_{DTYPE_NAMES[dtype]}",
            op="reduce_max",
            build_fn=make_build(),
            input_gen=make_inputs(),
        ))

    # REDUCE_MIN: f32, i32 (nan_mode)
    for dtype in [DType.FP32, DType.INT32]:
        def make_build(dt=dtype):
            return lambda: build_reduce_op(Op.REDUCE_MIN, "ReduceMinAttribute", dt,
                                           shape=(3, 4), axis=1, extra_args=[nan_propagate])

        def make_inputs(dt=dtype):
            def gen(input_names):
                return {"input_0": random_tensor((3, 4), dt)}
            return gen

        cases.append(TestCase(
            name=f"equiv_reduce_min_{DTYPE_NAMES[dtype]}",
            op="reduce_min",
            build_fn=make_build(),
            input_gen=make_inputs(),
        ))

    # REDUCE_PRODUCT: f32 only
    def make_rp_build():
        return lambda: build_reduce_op(Op.REDUCE_PRODUCT, "ReduceProductAttribute",
                                       DType.FP32, shape=(3, 4), axis=1)

    def make_rp_inputs():
        def gen(input_names):
            # Use small values to avoid overflow
            return {"input_0": np.random.uniform(0.5, 2.0, (3, 4)).astype(np.float32)}
        return gen

    cases.append(TestCase(name="equiv_reduce_product_f32", op="reduce_product",
                          build_fn=make_rp_build(), input_gen=make_rp_inputs()))

    # REDUCE_ALL: bool
    def make_ra_build():
        return lambda: build_reduce_op(Op.REDUCE_ALL, "ReduceAllAttribute",
                                       DType.BOOL, shape=(3, 4), axis=1)

    def make_ra_inputs():
        def gen(input_names):
            return {"input_0": np.random.choice([True, False], (3, 4)).astype(np.bool_)}
        return gen

    cases.append(TestCase(name="equiv_reduce_all_bool", op="reduce_all",
                          build_fn=make_ra_build(), input_gen=make_ra_inputs()))

    # REDUCE_ANY: bool
    def make_rany_build():
        return lambda: build_reduce_op(Op.REDUCE_ANY, "ReduceAnyAttribute",
                                       DType.BOOL, shape=(3, 4), axis=1)

    def make_rany_inputs():
        def gen(input_names):
            return {"input_0": np.random.choice([True, False], (3, 4)).astype(np.bool_)}
        return gen

    cases.append(TestCase(name="equiv_reduce_any_bool", op="reduce_any",
                          build_fn=make_rany_build(), input_gen=make_rany_inputs()))

    return cases


def gen_layout_tests():
    """Generate data layout op test cases."""
    cases = []

    # TRANSPOSE: f32, i32, i8
    for dtype in [DType.FP32, DType.INT32, DType.INT8]:
        def make_build(dt=dtype):
            return lambda: build_transpose_op(dt, (2, 3, 4), (1, 2, 0))

        def make_inputs(dt=dtype):
            def gen(input_names):
                return {"input_0": random_tensor((2, 3, 4), dt)}
            return gen

        cases.append(TestCase(
            name=f"equiv_transpose_{DTYPE_NAMES[dtype]}",
            op="transpose",
            build_fn=make_build(),
            input_gen=make_inputs(),
        ))

    # CONCAT: f32, i32, i8, i16
    for dtype in [DType.FP32, DType.INT32, DType.INT8, DType.INT16]:
        def make_build(dt=dtype):
            return lambda: build_concat_op(dt, (2, 3), axis=0)

        def make_inputs(dt=dtype):
            def gen(input_names):
                d = {}
                for name in input_names:
                    d[name] = random_tensor((2, 3), dt)
                return d
            return gen

        cases.append(TestCase(
            name=f"equiv_concat_{DTYPE_NAMES[dtype]}",
            op="concat",
            build_fn=make_build(),
            input_gen=make_inputs(),
        ))

    # PAD: f32, i32, i8
    for dtype in [DType.FP32, DType.INT32, DType.INT8]:
        def make_build(dt=dtype):
            return lambda: build_pad_op(dt, shape=(3, 3))

        def make_inputs(dt=dtype):
            def gen(input_names):
                return {
                    "input_0": random_tensor((3, 3), dt),
                    "pad_val": np.zeros(1, dtype=np_dtype(dt)),
                }
            return gen

        cases.append(TestCase(
            name=f"equiv_pad_{DTYPE_NAMES[dtype]}",
            op="pad",
            build_fn=make_build(),
            input_gen=make_inputs(),
        ))

    # TILE: f32, i32, i8
    for dtype in [DType.FP32, DType.INT32, DType.INT8]:
        def make_build(dt=dtype):
            return lambda: build_tile_op(dt, shape=(2, 3), multiples=(2, 1))

        def make_inputs(dt=dtype):
            def gen(input_names):
                return {"input_0": random_tensor((2, 3), dt)}
            return gen

        cases.append(TestCase(
            name=f"equiv_tile_{DTYPE_NAMES[dtype]}",
            op="tile",
            build_fn=make_build(),
            input_gen=make_inputs(),
        ))

    # REVERSE: f32, i32, i8
    for dtype in [DType.FP32, DType.INT32, DType.INT8]:
        def make_build(dt=dtype):
            return lambda: build_reverse_op(dt, shape=(4,), axis=0)

        def make_inputs(dt=dtype):
            def gen(input_names):
                return {"input_0": random_tensor((4,), dt)}
            return gen

        cases.append(TestCase(
            name=f"equiv_reverse_{DTYPE_NAMES[dtype]}",
            op="reverse",
            build_fn=make_build(),
            input_gen=make_inputs(),
        ))

    # GATHER: f32, i32, i8
    for dtype in [DType.FP32, DType.INT32, DType.INT8]:
        def make_build(dt=dtype):
            return lambda: build_gather_op(dt, N=2, K=4, C=3, W=2)

        def make_inputs(dt=dtype):
            def gen(input_names):
                return {
                    "input_0": random_tensor((2, 4, 3), dt),
                    "indices": np.random.randint(0, 4, (2, 2)).astype(np.int32),
                }
            return gen

        cases.append(TestCase(
            name=f"equiv_gather_{DTYPE_NAMES[dtype]}",
            op="gather",
            build_fn=make_build(),
            input_gen=make_inputs(),
        ))

    # SCATTER: f32, i32
    for dtype in [DType.FP32, DType.INT32]:
        def make_scatter_build(dt=dtype):
            return lambda: build_scatter_op(dt, N=2, K=4, C=3, W=2)

        def make_scatter_inputs(dt=dtype):
            def gen(input_names):
                ndt = np_dtype(dt)
                if dt == DType.FP32:
                    return {
                        "values_in": np.random.uniform(-3, 3, (2, 4, 3)).astype(ndt),
                        "indices": np.random.randint(0, 4, (2, 2)).astype(np.int32),
                        "input_0": np.random.uniform(-3, 3, (2, 2, 3)).astype(ndt),
                    }
                else:
                    return {
                        "values_in": np.random.randint(-10, 10, (2, 4, 3)).astype(ndt),
                        "indices": np.random.randint(0, 4, (2, 2)).astype(np.int32),
                        "input_0": np.random.randint(-10, 10, (2, 2, 3)).astype(ndt),
                    }
            return gen

        cases.append(TestCase(
            name=f"equiv_scatter_{DTYPE_NAMES[dtype]}",
            op="scatter",
            build_fn=make_scatter_build(),
            input_gen=make_scatter_inputs(),
        ))

    return cases


def gen_compute_tests():
    """Generate compute op test cases (convolutions, matmul, pooling)."""
    cases = []

    # CONV2D: f32
    def make_conv2d_build():
        return lambda: build_conv2d_op(DType.FP32)

    def make_conv2d_inputs():
        def gen(input_names):
            return {
                "input_0": np.random.uniform(-1, 1, (1, 4, 4, 2)).astype(np.float32),
                "weight": np.random.uniform(-1, 1, (3, 2, 2, 2)).astype(np.float32),
                "bias": np.random.uniform(-1, 1, (3,)).astype(np.float32),
                "input_zp": np.zeros(1, dtype=np.float32),
                "weight_zp": np.zeros(1, dtype=np.float32),
            }
        return gen

    cases.append(TestCase(name="equiv_conv2d_f32", op="conv2d",
                          build_fn=make_conv2d_build(), input_gen=make_conv2d_inputs()))

    # DEPTHWISE_CONV2D: f32
    def make_dwconv_build():
        return lambda: build_depthwise_conv2d_op(DType.FP32)

    def make_dwconv_inputs():
        def gen(input_names):
            return {
                "input_0": np.random.uniform(-1, 1, (1, 4, 4, 2)).astype(np.float32),
                "weight": np.random.uniform(-1, 1, (2, 2, 2, 1)).astype(np.float32),
                "bias": np.random.uniform(-1, 1, (2,)).astype(np.float32),
                "input_zp": np.zeros(1, dtype=np.float32),
                "weight_zp": np.zeros(1, dtype=np.float32),
            }
        return gen

    cases.append(TestCase(name="equiv_depthwise_conv2d_f32", op="depthwise_conv2d",
                          build_fn=make_dwconv_build(), input_gen=make_dwconv_inputs()))

    # TRANSPOSE_CONV2D: f32
    def make_tconv_build():
        return lambda: build_transpose_conv2d_op(DType.FP32)

    def make_tconv_inputs():
        def gen(input_names):
            return {
                "input_0": np.random.uniform(-1, 1, (1, 2, 2, 1)).astype(np.float32),
                "weight": np.random.uniform(-1, 1, (1, 2, 2, 1)).astype(np.float32),
                "bias": np.random.uniform(-1, 1, (1,)).astype(np.float32),
                "input_zp": np.zeros(1, dtype=np.float32),
                "weight_zp": np.zeros(1, dtype=np.float32),
            }
        return gen

    cases.append(TestCase(name="equiv_transpose_conv2d_f32", op="transpose_conv2d",
                          build_fn=make_tconv_build(), input_gen=make_tconv_inputs()))

    # CONV3D: f32
    def make_conv3d_build():
        return lambda: build_conv3d_op(DType.FP32)

    def make_conv3d_inputs():
        def gen(input_names):
            return {
                "input_0": np.random.uniform(-1, 1, (1, 3, 3, 3, 1)).astype(np.float32),
                "weight": np.random.uniform(-1, 1, (1, 2, 2, 2, 1)).astype(np.float32),
                "bias": np.random.uniform(-1, 1, (1,)).astype(np.float32),
                "input_zp": np.zeros(1, dtype=np.float32),
                "weight_zp": np.zeros(1, dtype=np.float32),
            }
        return gen

    cases.append(TestCase(name="equiv_conv3d_f32", op="conv3d",
                          build_fn=make_conv3d_build(), input_gen=make_conv3d_inputs()))

    # MATMUL: f32
    def make_matmul_build():
        return lambda: build_matmul_op(DType.FP32)

    def make_matmul_inputs():
        def gen(input_names):
            return {
                "input_0": np.random.uniform(-2, 2, (1, 3, 4)).astype(np.float32),
                "input_1": np.random.uniform(-2, 2, (1, 4, 2)).astype(np.float32),
                "a_zp": np.zeros(1, dtype=np.float32),
                "b_zp": np.zeros(1, dtype=np.float32),
            }
        return gen

    cases.append(TestCase(name="equiv_matmul_f32", op="matmul",
                          build_fn=make_matmul_build(), input_gen=make_matmul_inputs()))

    # AVG_POOL2D: f32
    def make_avgpool_build():
        return lambda: build_avg_pool2d_op(DType.FP32)

    def make_avgpool_inputs():
        def gen(input_names):
            return {
                "input_0": np.random.uniform(-3, 3, (1, 4, 4, 2)).astype(np.float32),
                "input_zp": np.zeros(1, dtype=np.float32),
                "output_zp": np.zeros(1, dtype=np.float32),
            }
        return gen

    cases.append(TestCase(name="equiv_avg_pool2d_f32", op="avg_pool2d",
                          build_fn=make_avgpool_build(), input_gen=make_avgpool_inputs()))

    # MAX_POOL2D: f32, i8
    for dtype in [DType.FP32, DType.INT8]:
        def make_maxpool_build(dt=dtype):
            return lambda: build_max_pool2d_op(dt)

        def make_maxpool_inputs(dt=dtype):
            def gen(input_names):
                if dt == DType.FP32:
                    return {"input_0": np.random.uniform(-3, 3, (1, 4, 4, 2)).astype(np.float32)}
                else:
                    return {"input_0": np.random.randint(-50, 50, (1, 4, 4, 2)).astype(np.int8)}
            return gen

        cases.append(TestCase(
            name=f"equiv_max_pool2d_{DTYPE_NAMES[dtype]}",
            op="max_pool2d",
            build_fn=make_maxpool_build(),
            input_gen=make_maxpool_inputs(),
        ))

    # ARGMAX: f32
    def make_argmax_build():
        return lambda: build_argmax_op(DType.FP32)

    def make_argmax_inputs():
        def gen(input_names):
            return {"input_0": np.random.uniform(-5, 5, (2, 5, 3)).astype(np.float32)}
        return gen

    cases.append(TestCase(name="equiv_argmax_f32", op="argmax",
                          build_fn=make_argmax_build(), input_gen=make_argmax_inputs()))

    return cases


def gen_cast_tests():
    """Generate type conversion test cases."""
    cases = []
    conversions = [
        (DType.FP32, DType.INT32, "f32_to_i32"),
        (DType.INT32, DType.FP32, "i32_to_f32"),
        (DType.INT8, DType.INT32, "i8_to_i32"),
        (DType.INT32, DType.INT8, "i32_to_i8"),
        (DType.FP32, DType.FP16, "f32_to_f16"),
        (DType.FP16, DType.FP32, "f16_to_f32"),
        (DType.INT16, DType.INT32, "i16_to_i32"),
        (DType.INT32, DType.INT16, "i32_to_i16"),
        (DType.INT8, DType.FP32, "i8_to_f32"),
        (DType.FP32, DType.INT8, "f32_to_i8"),
    ]
    for in_dt, out_dt, suffix in conversions:
        def make_build(idt=in_dt, odt=out_dt):
            return lambda: build_cast_op(idt, odt, (8,))

        def make_inputs(dt=in_dt):
            def gen(input_names):
                if dt in (DType.FP32, DType.FP16):
                    return {"input_0": np.random.uniform(-5, 5, (8,)).astype(np_dtype(dt))}
                else:
                    lo, hi = (-50, 50) if dt != DType.INT8 else (-100, 100)
                    return {"input_0": np.random.randint(lo, hi, (8,)).astype(np_dtype(dt))}
            return gen

        cases.append(TestCase(
            name=f"equiv_cast_{suffix}",
            op="cast",
            build_fn=make_build(),
            input_gen=make_inputs(),
        ))
    return cases


def gen_rescale_tests():
    """Generate RESCALE test cases."""
    cases = []

    # RESCALE i8 -> i8 with identity-ish multiplier
    def make_build():
        return lambda: build_rescale_op(DType.INT8, DType.INT8, shape=(8,))

    def make_inputs():
        def gen(input_names):
            return {
                "input_0": np.random.randint(-50, 50, (8,)).astype(np.int8),
                "multiplier": np.array([1073741824], dtype=np.int32),  # 2^30
                "shift": np.array([30], dtype=np.int8),
                "input_zp": np.array([0], dtype=np.int8),
                "output_zp": np.array([0], dtype=np.int8),
            }
        return gen

    cases.append(TestCase(name="equiv_rescale_i8_to_i8", op="rescale",
                          build_fn=make_build(), input_gen=make_inputs()))

    # RESCALE i8 -> i8 with non-zero zero-points
    def make_build_zp():
        return lambda: build_rescale_op(DType.INT8, DType.INT8, shape=(8,))

    def make_inputs_zp():
        def gen(input_names):
            return {
                "input_0": np.random.randint(-50, 50, (8,)).astype(np.int8),
                "multiplier": np.array([1073741824], dtype=np.int32),  # 2^30
                "shift": np.array([30], dtype=np.int8),
                "input_zp": np.array([5], dtype=np.int8),
                "output_zp": np.array([-3], dtype=np.int8),
            }
        return gen

    cases.append(TestCase(name="equiv_rescale_i8_to_i8_zp", op="rescale",
                          build_fn=make_build_zp(), input_gen=make_inputs_zp()))

    return cases


def gen_extra_tests():
    """Generate additional test cases to reach 150+ coverage."""
    cases = []

    # Additional TRANSPOSE permutations
    for perms, pname in [((2, 0, 1), "201"), ((0, 2, 1), "021")]:
        def make_build(p=perms):
            return lambda: build_transpose_op(DType.FP32, (2, 3, 4), p)

        def make_inputs():
            def gen(input_names):
                return {"input_0": np.random.uniform(-5, 5, (2, 3, 4)).astype(np.float32)}
            return gen

        cases.append(TestCase(
            name=f"equiv_transpose_f32_perm{pname}",
            op="transpose",
            build_fn=make_build(),
            input_gen=make_inputs(),
        ))

    # CONCAT along axis=1
    for dtype in [DType.FP32, DType.INT32]:
        def make_build(dt=dtype):
            return lambda: build_concat_op(dt, (2, 3), axis=1)

        def make_inputs(dt=dtype):
            def gen(input_names):
                d = {}
                for name in input_names:
                    d[name] = random_tensor((2, 3), dt)
                return d
            return gen

        cases.append(TestCase(
            name=f"equiv_concat_axis1_{DTYPE_NAMES[dtype]}",
            op="concat",
            build_fn=make_build(),
            input_gen=make_inputs(),
        ))

    # REDUCE_SUM along axis=0
    for dtype in [DType.FP32, DType.INT32]:
        def make_build(dt=dtype):
            return lambda: build_reduce_op(Op.REDUCE_SUM, "ReduceSumAttribute", dt,
                                           shape=(3, 4), axis=0)

        def make_inputs(dt=dtype):
            def gen(input_names):
                return {"input_0": random_tensor((3, 4), dt)}
            return gen

        cases.append(TestCase(
            name=f"equiv_reduce_sum_axis0_{DTYPE_NAMES[dtype]}",
            op="reduce_sum",
            build_fn=make_build(),
            input_gen=make_inputs(),
        ))

    # REDUCE_MAX along axis=0
    nan_propagate = ts.NanPropagationMode.PROPAGATE
    for dtype in [DType.FP32, DType.INT32]:
        def make_build(dt=dtype):
            return lambda: build_reduce_op(Op.REDUCE_MAX, "ReduceMaxAttribute", dt,
                                           shape=(3, 4), axis=0, extra_args=[nan_propagate])

        def make_inputs(dt=dtype):
            def gen(input_names):
                return {"input_0": random_tensor((3, 4), dt)}
            return gen

        cases.append(TestCase(
            name=f"equiv_reduce_max_axis0_{DTYPE_NAMES[dtype]}",
            op="reduce_max",
            build_fn=make_build(),
            input_gen=make_inputs(),
        ))

    # TILE with different multiples
    def make_tile2_build():
        return lambda: build_tile_op(DType.FP32, shape=(2, 3), multiples=(3, 2))

    def make_tile2_inputs():
        def gen(input_names):
            return {"input_0": np.random.uniform(-5, 5, (2, 3)).astype(np.float32)}
        return gen

    cases.append(TestCase(name="equiv_tile_f32_3x2", op="tile",
                          build_fn=make_tile2_build(), input_gen=make_tile2_inputs()))

    # PAD with different padding values
    def make_pad2_build():
        """PAD [4,4] with pad=2 on each side -> [8,8]."""
        def builder():
            ser = ts.TosaSerializer(str(FIXTURES_DIR))
            inp = ts.TosaSerializerTensor("input_0", [4, 4], DType.FP32)
            ser.addInputTensor(inp)
            pad_const = ser.addConst([4], DType.SHAPE, np.array([2, 2, 2, 2], dtype=np.int64))
            pad_val = ts.TosaSerializerTensor("pad_val", [1], DType.FP32)
            ser.addInputTensor(pad_val)
            out = ser.addOutput([8, 8], DType.FP32)
            attr = ts.TosaSerializerAttribute()
            attr.PadAttribute()
            ser.addOperator(Op.PAD, ["input_0", pad_const.name, "pad_val"], [out.name], attr)
            return ser.serialize(), ["input_0", "pad_val"], [out.name]
        return builder

    def make_pad2_inputs():
        def gen(input_names):
            return {
                "input_0": np.random.uniform(-5, 5, (4, 4)).astype(np.float32),
                "pad_val": np.array([0.0], dtype=np.float32),
            }
        return gen

    cases.append(TestCase(name="equiv_pad_f32_2x2", op="pad",
                          build_fn=make_pad2_build(), input_gen=make_pad2_inputs()))

    # GATHER with larger K
    def make_gather2_build():
        return lambda: build_gather_op(DType.FP32, N=1, K=8, C=4, W=3)

    def make_gather2_inputs():
        def gen(input_names):
            return {
                "input_0": np.random.uniform(-5, 5, (1, 8, 4)).astype(np.float32),
                "indices": np.random.randint(0, 8, (1, 3)).astype(np.int32),
            }
        return gen

    cases.append(TestCase(name="equiv_gather_f32_large", op="gather",
                          build_fn=make_gather2_build(), input_gen=make_gather2_inputs()))

    # REVERSE with larger shape
    def make_rev2_build():
        return lambda: build_reverse_op(DType.FP32, shape=(8,), axis=0)

    def make_rev2_inputs():
        def gen(input_names):
            return {"input_0": np.random.uniform(-5, 5, (8,)).astype(np.float32)}
        return gen

    cases.append(TestCase(name="equiv_reverse_f32_8", op="reverse",
                          build_fn=make_rev2_build(), input_gen=make_rev2_inputs()))

    # MATMUL with larger matrices
    def make_matmul2_build():
        def builder():
            ser = ts.TosaSerializer(str(FIXTURES_DIR))
            inp0 = ts.TosaSerializerTensor("input_0", [2, 4, 6], DType.FP32)
            inp1 = ts.TosaSerializerTensor("input_1", [2, 6, 3], DType.FP32)
            a_zp = ts.TosaSerializerTensor("a_zp", [1], DType.FP32)
            b_zp = ts.TosaSerializerTensor("b_zp", [1], DType.FP32)
            ser.addInputTensor(inp0)
            ser.addInputTensor(inp1)
            ser.addInputTensor(a_zp)
            ser.addInputTensor(b_zp)
            out = ser.addOutput([2, 4, 3], DType.FP32)
            attr = ts.TosaSerializerAttribute()
            attr.MatMulAttribute()
            ser.addOperator(Op.MATMUL, ["input_0", "input_1", "a_zp", "b_zp"], [out.name], attr)
            return ser.serialize(), ["input_0", "input_1", "a_zp", "b_zp"], [out.name]
        return builder

    def make_matmul2_inputs():
        def gen(input_names):
            return {
                "input_0": np.random.uniform(-2, 2, (2, 4, 6)).astype(np.float32),
                "input_1": np.random.uniform(-2, 2, (2, 6, 3)).astype(np.float32),
                "a_zp": np.zeros(1, dtype=np.float32),
                "b_zp": np.zeros(1, dtype=np.float32),
            }
        return gen

    cases.append(TestCase(name="equiv_matmul_f32_large", op="matmul",
                          build_fn=make_matmul2_build(), input_gen=make_matmul2_inputs()))

    # SUB with INT64
    def make_sub_i64_build():
        return lambda: build_binary_op(Op.SUB, "SubAttribute", DType.INT64)

    def make_sub_i64_inputs():
        def gen(input_names):
            d = {}
            for name in input_names:
                d[name] = np.random.randint(-100, 100, (4, 4)).astype(np.int64)
            return d
        return gen

    cases.append(TestCase(name="equiv_sub_i64_extra", op="sub",
                          build_fn=make_sub_i64_build(), input_gen=make_sub_i64_inputs()))

    # MAXIMUM/MINIMUM with INT64
    for op_name, op_enum, attr_method in [
        ("maximum", Op.MAXIMUM, "MaximumAttribute"),
        ("minimum", Op.MINIMUM, "MinimumAttribute"),
    ]:
        def make_build(oe=op_enum, am=attr_method):
            return lambda: build_binary_op(oe, am, DType.INT64,
                                           attr_args=[ts.NanPropagationMode.PROPAGATE])

        def make_inputs():
            def gen(input_names):
                d = {}
                for name in input_names:
                    d[name] = np.random.randint(-100, 100, (4, 4)).astype(np.int64)
                return d
            return gen

        cases.append(TestCase(
            name=f"equiv_{op_name}_i64",
            op=op_name,
            build_fn=make_build(),
            input_gen=make_inputs(),
        ))

    # NEGATE with INT64
    def make_negate_i64_build():
        def builder():
            ser = ts.TosaSerializer(str(FIXTURES_DIR))
            inp = ts.TosaSerializerTensor("input_0", [8], DType.INT64)
            inp_zp = ts.TosaSerializerTensor("input_zp", [1], DType.INT64)
            out_zp = ts.TosaSerializerTensor("output_zp", [1], DType.INT64)
            ser.addInputTensor(inp)
            ser.addInputTensor(inp_zp)
            ser.addInputTensor(out_zp)
            out = ser.addOutput([8], DType.INT64)
            attr = ts.TosaSerializerAttribute()
            attr.NegateAttribute()
            ser.addOperator(Op.NEGATE, ["input_0", "input_zp", "output_zp"], [out.name], attr)
            return ser.serialize(), ["input_0", "input_zp", "output_zp"], [out.name]
        return builder

    def make_negate_i64_inputs():
        def gen(input_names):
            return {
                "input_0": np.random.randint(-100, 100, (8,)).astype(np.int64),
                "input_zp": np.zeros(1, dtype=np.int64),
                "output_zp": np.zeros(1, dtype=np.int64),
            }
        return gen

    cases.append(TestCase(name="equiv_negate_i64", op="negate",
                          build_fn=make_negate_i64_build(), input_gen=make_negate_i64_inputs()))

    # CONV2D with different configuration (larger stride)
    # Input 7x7, kernel 3x3, stride 2 -> output (7-3)/2+1 = 3
    def make_conv2d_stride2_build():
        def builder():
            ser = ts.TosaSerializer(str(FIXTURES_DIR))
            inp = ts.TosaSerializerTensor("input_0", [1, 7, 7, 1], DType.FP32)
            weight = ts.TosaSerializerTensor("weight", [2, 3, 3, 1], DType.FP32)
            bias = ts.TosaSerializerTensor("bias", [2], DType.FP32)
            input_zp = ts.TosaSerializerTensor("input_zp", [1], DType.FP32)
            weight_zp = ts.TosaSerializerTensor("weight_zp", [1], DType.FP32)
            ser.addInputTensor(inp)
            ser.addInputTensor(weight)
            ser.addInputTensor(bias)
            ser.addInputTensor(input_zp)
            ser.addInputTensor(weight_zp)
            out = ser.addOutput([1, 3, 3, 2], DType.FP32)
            attr = ts.TosaSerializerAttribute()
            attr.Conv2dAttribute([0, 0, 0, 0], [2, 2], [1, 1], False, DType.FP32)
            ser.addOperator(Op.CONV2D,
                            ["input_0", "weight", "bias", "input_zp", "weight_zp"],
                            [out.name], attr)
            return ser.serialize(), ["input_0", "weight", "bias", "input_zp", "weight_zp"], [out.name]
        return builder

    def make_conv2d_stride2_inputs():
        def gen(input_names):
            return {
                "input_0": np.random.uniform(-1, 1, (1, 7, 7, 1)).astype(np.float32),
                "weight": np.random.uniform(-1, 1, (2, 3, 3, 1)).astype(np.float32),
                "bias": np.random.uniform(-1, 1, (2,)).astype(np.float32),
                "input_zp": np.zeros(1, dtype=np.float32),
                "weight_zp": np.zeros(1, dtype=np.float32),
            }
        return gen

    cases.append(TestCase(name="equiv_conv2d_f32_stride2", op="conv2d",
                          build_fn=make_conv2d_stride2_build(), input_gen=make_conv2d_stride2_inputs()))

    # AVG_POOL2D with stride=2
    def make_avgpool_stride2_build():
        def builder():
            ser = ts.TosaSerializer(str(FIXTURES_DIR))
            inp = ts.TosaSerializerTensor("input_0", [1, 6, 6, 1], DType.FP32)
            input_zp = ts.TosaSerializerTensor("input_zp", [1], DType.FP32)
            output_zp = ts.TosaSerializerTensor("output_zp", [1], DType.FP32)
            ser.addInputTensor(inp)
            ser.addInputTensor(input_zp)
            ser.addInputTensor(output_zp)
            out = ser.addOutput([1, 3, 3, 1], DType.FP32)
            attr = ts.TosaSerializerAttribute()
            attr.AvgPool2dAttribute([2, 2], [2, 2], [0, 0, 0, 0], DType.FP32)
            ser.addOperator(Op.AVG_POOL2D, ["input_0", "input_zp", "output_zp"], [out.name], attr)
            return ser.serialize(), ["input_0", "input_zp", "output_zp"], [out.name]
        return builder

    def make_avgpool_stride2_inputs():
        def gen(input_names):
            return {
                "input_0": np.random.uniform(-3, 3, (1, 6, 6, 1)).astype(np.float32),
                "input_zp": np.zeros(1, dtype=np.float32),
                "output_zp": np.zeros(1, dtype=np.float32),
            }
        return gen

    cases.append(TestCase(name="equiv_avg_pool2d_f32_stride2", op="avg_pool2d",
                          build_fn=make_avgpool_stride2_build(), input_gen=make_avgpool_stride2_inputs()))

    # LOGICAL_LEFT_SHIFT with INT64
    def make_lls_i64_build():
        return lambda: build_binary_op(Op.LOGICAL_LEFT_SHIFT, "LogicalLeftShiftAttribute", DType.INT64)

    def make_lls_i64_inputs():
        def gen(input_names):
            return {
                "input_0": np.random.randint(0, 50, (4, 4)).astype(np.int64),
                "input_1": np.random.randint(0, 32, (4, 4)).astype(np.int64),
            }
        return gen

    cases.append(TestCase(name="equiv_logical_left_shift_i64", op="logical_left_shift",
                          build_fn=make_lls_i64_build(), input_gen=make_lls_i64_inputs()))

    # LOGICAL_RIGHT_SHIFT with INT64
    def make_lrs_i64_build():
        return lambda: build_binary_op(Op.LOGICAL_RIGHT_SHIFT, "LogicalRightShiftAttribute", DType.INT64)

    def make_lrs_i64_inputs():
        def gen(input_names):
            return {
                "input_0": np.random.randint(0, 1000, (4, 4)).astype(np.int64),
                "input_1": np.random.randint(0, 32, (4, 4)).astype(np.int64),
            }
        return gen

    cases.append(TestCase(name="equiv_logical_right_shift_i64", op="logical_right_shift",
                          build_fn=make_lrs_i64_build(), input_gen=make_lrs_i64_inputs()))

    # BITWISE_AND/OR/XOR with INT64
    for op_name, op_enum, attr_method in [
        ("bitwise_and", Op.BITWISE_AND, "BitwiseAndAttribute"),
        ("bitwise_or", Op.BITWISE_OR, "BitwiseOrAttribute"),
        ("bitwise_xor", Op.BITWISE_XOR, "BitwiseXorAttribute"),
    ]:
        def make_bw64_build(oe=op_enum, am=attr_method):
            return lambda: build_binary_op(oe, am, DType.INT64)

        def make_bw64_inputs():
            def gen(input_names):
                d = {}
                for name in input_names:
                    d[name] = np.random.randint(-100, 100, (4, 4)).astype(np.int64)
                return d
            return gen

        cases.append(TestCase(
            name=f"equiv_{op_name}_i64",
            op=op_name,
            build_fn=make_bw64_build(),
            input_gen=make_bw64_inputs(),
        ))

    # EXP with different input range
    def make_exp2_build():
        return lambda: build_unary_op(Op.EXP, "ExpAttribute", DType.FP32, shape=(16,))

    def make_exp2_inputs():
        def gen(input_names):
            return {"input_0": np.random.uniform(-5, 3, (16,)).astype(np.float32)}
        return gen

    cases.append(TestCase(name="equiv_exp_f32_wide", op="exp",
                          build_fn=make_exp2_build(), input_gen=make_exp2_inputs()))

    # SIGMOID with larger tensor
    def make_sigmoid2_build():
        return lambda: build_unary_op(Op.SIGMOID, "SigmoidAttribute", DType.FP32, shape=(4, 4))

    def make_sigmoid2_inputs():
        def gen(input_names):
            return {"input_0": np.random.uniform(-6, 6, (4, 4)).astype(np.float32)}
        return gen

    cases.append(TestCase(name="equiv_sigmoid_f32_4x4", op="sigmoid",
                          build_fn=make_sigmoid2_build(), input_gen=make_sigmoid2_inputs()))

    # TANH with larger tensor
    def make_tanh2_build():
        return lambda: build_unary_op(Op.TANH, "TanhAttribute", DType.FP32, shape=(4, 4))

    def make_tanh2_inputs():
        def gen(input_names):
            return {"input_0": np.random.uniform(-3, 3, (4, 4)).astype(np.float32)}
        return gen

    cases.append(TestCase(name="equiv_tanh_f32_4x4", op="tanh",
                          build_fn=make_tanh2_build(), input_gen=make_tanh2_inputs()))

    # SELECT with larger shape
    def make_select2_build():
        return lambda: build_select_op(DType.FP32, shape=(4, 4))

    def make_select2_inputs():
        def gen(input_names):
            return {
                "condition": np.random.choice([True, False], (4, 4)).astype(np.bool_),
                "input_a": np.random.uniform(-5, 5, (4, 4)).astype(np.float32),
                "input_b": np.random.uniform(-5, 5, (4, 4)).astype(np.float32),
            }
        return gen

    cases.append(TestCase(name="equiv_select_f32_4x4", op="select",
                          build_fn=make_select2_build(), input_gen=make_select2_inputs()))

    # CLAMP with wider range
    def make_clamp2_build():
        return lambda: build_clamp_op(DType.FP32, shape=(16,), min_val=-2.5, max_val=2.5)

    def make_clamp2_inputs():
        def gen(input_names):
            return {"input_0": np.random.uniform(-10, 10, (16,)).astype(np.float32)}
        return gen

    cases.append(TestCase(name="equiv_clamp_f32_wide", op="clamp",
                          build_fn=make_clamp2_build(), input_gen=make_clamp2_inputs()))

    return cases


# ---------------------------------------------------------------------------
# Control flow tests
# ---------------------------------------------------------------------------

def gen_control_flow_tests():
    """Generate COND_IF and WHILE_LOOP tests using separate regions."""
    cases = []

    # COND_IF: if cond then a+b else a-b
    def build_cond_if_true():
        ser = ts.TosaSerializer(str(FIXTURES_DIR))
        cond = ts.TosaSerializerTensor('cond', [1], DType.BOOL)
        a = ts.TosaSerializerTensor('a', [4], DType.FP32)
        b = ts.TosaSerializerTensor('b', [4], DType.FP32)
        ser.addInputTensor(cond)
        ser.addInputTensor(a)
        ser.addInputTensor(b)
        out = ser.addOutput([4], DType.FP32)
        attr = ts.TosaSerializerAttribute()
        attr.CondIfAttribute('then_region', 'else_region')
        ser.addOperator(Op.COND_IF, ['cond', 'a', 'b'], [out.name], attr)

        ser.startRegion('then_region', str(FIXTURES_DIR))
        ser.currRegion.addBasicBlock('then_block')
        then_a = ts.TosaSerializerTensor('then_a', [4], DType.FP32)
        then_b = ts.TosaSerializerTensor('then_b', [4], DType.FP32)
        ser.addInputTensor(then_a)
        ser.addInputTensor(then_b)
        then_out = ser.addOutput([4], DType.FP32)
        attr2 = ts.TosaSerializerAttribute(); attr2.AddAttribute()
        ser.addOperator(Op.ADD, ['then_a', 'then_b'], [then_out.name], attr2)

        ser.startRegion('else_region', str(FIXTURES_DIR))
        ser.currRegion.addBasicBlock('else_block')
        else_a = ts.TosaSerializerTensor('else_a', [4], DType.FP32)
        else_b = ts.TosaSerializerTensor('else_b', [4], DType.FP32)
        ser.addInputTensor(else_a)
        ser.addInputTensor(else_b)
        else_out = ser.addOutput([4], DType.FP32)
        attr3 = ts.TosaSerializerAttribute(); attr3.SubAttribute()
        ser.addOperator(Op.SUB, ['else_a', 'else_b'], [else_out.name], attr3)

        return ser.serialize(), ['cond', 'a', 'b'], [out.name]

    def cond_if_true_inputs(input_names):
        return {
            'cond': np.array([True], dtype=np.bool_),
            'a': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            'b': np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        }

    cases.append(TestCase(
        name="equiv_cond_if_true",
        op="cond_if",
        build_fn=build_cond_if_true,
        input_gen=cond_if_true_inputs,
    ))

    # COND_IF with cond=False
    def cond_if_false_inputs(input_names):
        return {
            'cond': np.array([False], dtype=np.bool_),
            'a': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            'b': np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        }

    cases.append(TestCase(
        name="equiv_cond_if_false",
        op="cond_if",
        build_fn=build_cond_if_true,  # same graph, different input
        input_gen=cond_if_false_inputs,
    ))

    # WHILE_LOOP: count down from 5 to 0
    def build_while_loop():
        ser = ts.TosaSerializer(str(FIXTURES_DIR))
        counter = ts.TosaSerializerTensor('counter', [1], DType.INT32)
        ser.addInputTensor(counter)
        out = ser.addOutput([1], DType.INT32)
        attr = ts.TosaSerializerAttribute()
        attr.WhileLoopAttribute('cond_region', 'body_region')
        ser.addOperator(Op.WHILE_LOOP, ['counter'], [out.name], attr)

        # Cond: counter > 0
        ser.startRegion('cond_region', str(FIXTURES_DIR))
        ser.currRegion.addBasicBlock('cond_block')
        cond_in = ts.TosaSerializerTensor('cond_in', [1], DType.INT32)
        ser.addInputTensor(cond_in)
        cond_out = ser.addOutput([1], DType.BOOL)
        zero = ser.addConst([1], DType.INT32, np.array([0], dtype=np.int32), 'zero')
        attr_gt = ts.TosaSerializerAttribute(); attr_gt.GreaterAttribute()
        ser.addOperator(Op.GREATER, ['cond_in', zero.name], [cond_out.name], attr_gt)

        # Body: counter - 1
        ser.startRegion('body_region', str(FIXTURES_DIR))
        ser.currRegion.addBasicBlock('body_block')
        body_in = ts.TosaSerializerTensor('body_in', [1], DType.INT32)
        ser.addInputTensor(body_in)
        body_out = ser.addOutput([1], DType.INT32)
        one = ser.addConst([1], DType.INT32, np.array([1], dtype=np.int32), 'one')
        attr_sub = ts.TosaSerializerAttribute(); attr_sub.SubAttribute()
        ser.addOperator(Op.SUB, ['body_in', one.name], [body_out.name], attr_sub)

        return ser.serialize(), ['counter'], [out.name]

    def while_loop_inputs(input_names):
        return {'counter': np.array([5], dtype=np.int32)}

    cases.append(TestCase(
        name="equiv_while_loop",
        op="while_loop",
        build_fn=build_while_loop,
        input_gen=while_loop_inputs,
    ))

    return cases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    all_cases = []
    all_cases.extend(gen_binary_tests())
    all_cases.extend(gen_unary_tests())
    all_cases.extend(gen_comparison_tests())
    all_cases.extend(gen_select_tests())
    all_cases.extend(gen_reduce_tests())
    all_cases.extend(gen_layout_tests())
    all_cases.extend(gen_compute_tests())
    all_cases.extend(gen_cast_tests())
    all_cases.extend(gen_rescale_tests())
    all_cases.extend(gen_extra_tests())
    all_cases.extend(gen_control_flow_tests())

    print(f"Generating {len(all_cases)} equivalence test fixtures...")

    passed = 0
    failed = 0
    errors = []

    for tc in all_cases:
        try:
            graph_bytes, input_names, output_names = tc.build_fn()
            inputs = tc.input_gen(input_names)

            # Run reference model
            input_arrays = [inputs[name] for name in input_names]
            outputs, status = run_ref_model(graph_bytes, input_arrays)

            if status != tosa_reference_model.GraphStatus.TOSA_VALID:
                errors.append(f"{tc.name}: ref model returned status {status}")
                failed += 1
                continue

            # Save fixture
            expected = {}
            for i, out_name in enumerate(output_names):
                expected[out_name] = np.ascontiguousarray(outputs[i])

            save_fixture(tc.name, graph_bytes, inputs, expected,
                         metadata={"op": tc.op, "dtype": tc.name.split("_")[-1]})
            passed += 1

        except Exception as e:
            errors.append(f"{tc.name}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {len(all_cases)}")
    if errors:
        print("\nFailures:")
        for e in errors:
            print(f"  {e}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
