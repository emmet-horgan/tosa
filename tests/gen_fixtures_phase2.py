#!/usr/bin/env python3
"""Generate Phase 2 test fixtures for tosa-interp.

Run from the tosa-tools root:
    .venv/bin/python tosa-interp/tests/gen_fixtures_phase2.py
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "serialization" / "python"))

import tosa_serializer as ts
from tosa.Op import Op
from tosa.DType import DType

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def save(name, tosa_bytes, inputs, expected):
    d = FIXTURES_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "graph.tosa").write_bytes(tosa_bytes)
    for k, v in inputs.items():
        np.save(d / f"input_{k}.npy", v)
    for k, v in expected.items():
        np.save(d / f"expected_{k}.npy", v)


def gen_maximum_f32():
    ser = ts.TosaSerializer(str(FIXTURES_DIR / "maximum_f32"))
    a_data = np.array([1.0, 5.0, 3.0, -2.0], dtype=np.float32)
    b_data = np.array([2.0, 4.0, 3.0, 0.0], dtype=np.float32)
    a = ser.addConst([4], DType.FP32, a_data, "a")
    b = ser.addConst([4], DType.FP32, b_data, "b")
    out = ser.addOutput([4], DType.FP32)
    attr = ts.TosaSerializerAttribute()
    attr.MaximumAttribute(1)  # nan_mode=PROPAGATE
    ser.addOperator(Op.MAXIMUM, [a.name, b.name], [out.name], attr)
    save("maximum_f32", ser.serialize(), {}, {out.name: np.maximum(a_data, b_data)})


def gen_matmul_f32():
    ser = ts.TosaSerializer(str(FIXTURES_DIR / "matmul_f32"))
    # [1, 2, 3] x [1, 3, 2] -> [1, 2, 2]
    a_data = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
    b_data = np.array([[[1, 2], [3, 4], [5, 6]]], dtype=np.float32)
    a = ser.addConst([1, 2, 3], DType.FP32, a_data, "a")
    b = ser.addConst([1, 3, 2], DType.FP32, b_data, "b")
    a_zp = ser.addConst([1], DType.FP32, np.zeros(1, dtype=np.float32), "a_zp")
    b_zp = ser.addConst([1], DType.FP32, np.zeros(1, dtype=np.float32), "b_zp")
    out = ser.addOutput([1, 2, 2], DType.FP32)
    attr = ts.TosaSerializerAttribute()
    attr.MatMulAttribute()
    ser.addOperator(Op.MATMUL, [a.name, b.name, a_zp.name, b_zp.name], [out.name], attr)
    expected = a_data @ b_data
    save("matmul_f32", ser.serialize(), {}, {out.name: expected})


def gen_conv2d_f32():
    ser = ts.TosaSerializer(str(FIXTURES_DIR / "conv2d_f32"))
    # Input: [1, 3, 3, 1], Weight: [1, 2, 2, 1], Bias: [1]
    inp = np.arange(9, dtype=np.float32).reshape(1, 3, 3, 1)
    weight = np.ones((1, 2, 2, 1), dtype=np.float32)
    bias = np.array([0.0], dtype=np.float32)

    a = ser.addConst([1, 3, 3, 1], DType.FP32, inp, "input")
    w = ser.addConst([1, 2, 2, 1], DType.FP32, weight, "weight")
    b = ser.addConst([1], DType.FP32, bias, "bias")
    in_zp = ser.addConst([1], DType.FP32, np.zeros(1, dtype=np.float32), "input_zp")
    w_zp = ser.addConst([1], DType.FP32, np.zeros(1, dtype=np.float32), "weight_zp")
    out = ser.addOutput([1, 2, 2, 1], DType.FP32)

    attr = ts.TosaSerializerAttribute()
    # pad=[0,0,0,0], stride=[1,1], dilation=[1,1], local_bound=False, acc_type=FP32
    attr.Conv2dAttribute([0, 0, 0, 0], [1, 1], [1, 1], False, DType.FP32)
    ser.addOperator(Op.CONV2D, [a.name, w.name, b.name, in_zp.name, w_zp.name], [out.name], attr)

    # Manual conv2d: sum of 2x2 windows
    # [0+1+3+4, 1+2+4+5, 3+4+6+7, 4+5+7+8] = [8, 12, 20, 24]
    expected = np.array([[[[8], [12]], [[20], [24]]]], dtype=np.float32)
    save("conv2d_f32", ser.serialize(), {}, {out.name: expected})


def gen_transpose_f32():
    ser = ts.TosaSerializer(str(FIXTURES_DIR / "transpose_f32"))
    a_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)  # [2, 3]
    a = ser.addConst([2, 3], DType.FP32, a_data, "a")
    out = ser.addOutput([3, 2], DType.FP32)
    attr = ts.TosaSerializerAttribute()
    attr.TransposeAttribute([1, 0])
    ser.addOperator(Op.TRANSPOSE, [a.name], [out.name], attr)
    save("transpose_f32", ser.serialize(), {}, {out.name: np.ascontiguousarray(a_data.T)})


def gen_reduce_sum_f32():
    ser = ts.TosaSerializer(str(FIXTURES_DIR / "reduce_sum_f32"))
    a_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)  # [2, 3]
    a = ser.addConst([2, 3], DType.FP32, a_data, "a")
    out = ser.addOutput([2, 1], DType.FP32)  # reduce along axis=1
    attr = ts.TosaSerializerAttribute()
    attr.ReduceSumAttribute(1)
    ser.addOperator(Op.REDUCE_SUM, [a.name], [out.name], attr)
    expected = a_data.sum(axis=1, keepdims=True)
    save("reduce_sum_f32", ser.serialize(), {}, {out.name: expected})


def gen_equal_i32():
    ser = ts.TosaSerializer(str(FIXTURES_DIR / "equal_i32"))
    a_data = np.array([1, 2, 3, 4], dtype=np.int32)
    b_data = np.array([1, 0, 3, 5], dtype=np.int32)
    a = ser.addConst([4], DType.INT32, a_data, "a")
    b = ser.addConst([4], DType.INT32, b_data, "b")
    out = ser.addOutput([4], DType.BOOL)
    attr = ts.TosaSerializerAttribute()
    attr.EqualAttribute()
    ser.addOperator(Op.EQUAL, [a.name, b.name], [out.name], attr)
    expected = (a_data == b_data).astype(np.bool_)
    save("equal_i32", ser.serialize(), {}, {out.name: expected})


def gen_cast_f32_to_i32():
    ser = ts.TosaSerializer(str(FIXTURES_DIR / "cast_f32_to_i32"))
    a_data = np.array([1.5, -2.7, 3.0, 0.0], dtype=np.float32)
    a = ser.addConst([4], DType.FP32, a_data, "a")
    out = ser.addOutput([4], DType.INT32)
    attr = ts.TosaSerializerAttribute()
    attr.CastAttribute()
    ser.addOperator(Op.CAST, [a.name], [out.name], attr)
    # TOSA CAST float->int uses round-to-nearest
    expected = np.round(a_data).astype(np.int32)
    save("cast_f32_to_i32", ser.serialize(), {}, {out.name: expected})


def gen_sigmoid_f32():
    ser = ts.TosaSerializer(str(FIXTURES_DIR / "sigmoid_f32"))
    a_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    a = ser.addConst([5], DType.FP32, a_data, "a")
    out = ser.addOutput([5], DType.FP32)
    attr = ts.TosaSerializerAttribute()
    attr.SigmoidAttribute()
    ser.addOperator(Op.SIGMOID, [a.name], [out.name], attr)
    expected = 1.0 / (1.0 + np.exp(-a_data))
    save("sigmoid_f32", ser.serialize(), {}, {out.name: expected.astype(np.float32)})


def gen_concat_f32():
    ser = ts.TosaSerializer(str(FIXTURES_DIR / "concat_f32"))
    a_data = np.array([[1, 2], [3, 4]], dtype=np.float32)  # [2, 2]
    b_data = np.array([[5, 6], [7, 8]], dtype=np.float32)  # [2, 2]
    a = ser.addConst([2, 2], DType.FP32, a_data, "a")
    b = ser.addConst([2, 2], DType.FP32, b_data, "b")
    out = ser.addOutput([4, 2], DType.FP32)  # concat along axis=0
    attr = ts.TosaSerializerAttribute()
    attr.ConcatAttribute(0)
    ser.addOperator(Op.CONCAT, [a.name, b.name], [out.name], attr)
    expected = np.concatenate([a_data, b_data], axis=0)
    save("concat_f32", ser.serialize(), {}, {out.name: expected})


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating Phase 2 fixtures in {FIXTURES_DIR}")

    generators = {
        "maximum_f32": gen_maximum_f32,
        "matmul_f32": gen_matmul_f32,
        "conv2d_f32": gen_conv2d_f32,
        "transpose_f32": gen_transpose_f32,
        "reduce_sum_f32": gen_reduce_sum_f32,
        "equal_i32": gen_equal_i32,
        "cast_f32_to_i32": gen_cast_f32_to_i32,
        "sigmoid_f32": gen_sigmoid_f32,
        "concat_f32": gen_concat_f32,
    }
    # Optionally regenerate only the fixtures named on the command line.
    for name in sys.argv[1:] or generators:
        generators[name]()

    print("Done! Generated fixtures:")
    for d in sorted(FIXTURES_DIR.iterdir()):
        if d.is_dir() and (d / "graph.tosa").exists():
            files = [f.name for f in d.iterdir()]
            print(f"  {d.name}/: {', '.join(sorted(files))}")


if __name__ == "__main__":
    main()
