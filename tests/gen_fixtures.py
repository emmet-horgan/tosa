#!/usr/bin/env python3
"""Generate .tosa test fixtures for the tosa-interp integration tests.

Run from the tosa-tools root after `pip install -e .`:
    python tosa-interp/tests/gen_fixtures.py

Generates files in tosa-interp/tests/fixtures/
"""

import os
import sys
from pathlib import Path

import numpy as np

# Add the serialization python paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "serialization" / "python"))

import tosa_serializer as ts  # noqa: E402
from tosa.Op import Op  # noqa: E402
from tosa.DType import DType  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def save_fixture(name: str, tosa_bytes: bytes, inputs: dict, expected_outputs: dict):
    """Save a test fixture: .tosa file + input/output .npy files."""
    fixture_dir = FIXTURES_DIR / name
    fixture_dir.mkdir(parents=True, exist_ok=True)

    with open(fixture_dir / "graph.tosa", "wb") as f:
        f.write(tosa_bytes)

    for tensor_name, arr in inputs.items():
        np.save(fixture_dir / f"input_{tensor_name}.npy", arr)

    for tensor_name, arr in expected_outputs.items():
        np.save(fixture_dir / f"expected_{tensor_name}.npy", arr)


def gen_const_add_f32():
    """CONST(a) + CONST(b) -> output, all f32."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR / "const_add_f32"))

    a_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b_data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

    a = ser.addConst([4], DType.FP32, a_data, "const_a")
    b = ser.addConst([4], DType.FP32, b_data, "const_b")
    out = ser.addOutput([4], DType.FP32)

    attr = ts.TosaSerializerAttribute()
    attr.AddAttribute()
    ser.addOperator(Op.ADD, [a.name, b.name], [out.name], attr)

    expected = a_data + b_data
    save_fixture("const_add_f32", ser.serialize(), {}, {out.name: expected})


def gen_input_add_f32():
    """input_a + input_b -> output, f32 with external inputs."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR / "input_add_f32"))

    shape = [2, 3]
    a = ser.addPlaceholder(shape, DType.FP32, None)
    b = ser.addPlaceholder(shape, DType.FP32, None)
    out = ser.addOutput(shape, DType.FP32)

    attr = ts.TosaSerializerAttribute()
    attr.AddAttribute()
    ser.addOperator(Op.ADD, [a.name, b.name], [out.name], attr)

    a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b_data = np.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]], dtype=np.float32)
    expected = a_data + b_data

    save_fixture(
        "input_add_f32",
        ser.serialize(),
        {a.name: a_data, b.name: b_data},
        {out.name: expected},
    )


def gen_sub_i32():
    """CONST(a) - CONST(b) -> output, i32."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR / "sub_i32"))

    a_data = np.array([100, 200, 300, 400], dtype=np.int32)
    b_data = np.array([10, 20, 30, 40], dtype=np.int32)

    a = ser.addConst([4], DType.INT32, a_data, "const_a")
    b = ser.addConst([4], DType.INT32, b_data, "const_b")
    out = ser.addOutput([4], DType.INT32)

    attr = ts.TosaSerializerAttribute()
    attr.SubAttribute()
    ser.addOperator(Op.SUB, [a.name, b.name], [out.name], attr)

    expected = a_data - b_data
    save_fixture("sub_i32", ser.serialize(), {}, {out.name: expected})


def gen_abs_negate_f32():
    """ABS(NEGATE(CONST(a))) -> output, f32."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR / "abs_negate_f32"))

    a_data = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32)
    a = ser.addConst([4], DType.FP32, a_data, "const_a")
    zp_in = ser.addConst([1], DType.FP32, np.zeros(1, dtype=np.float32), "input1_zp")
    zp_out = ser.addConst([1], DType.FP32, np.zeros(1, dtype=np.float32), "output_zp")
    neg = ser.addIntermediate([4], DType.FP32)
    out = ser.addOutput([4], DType.FP32)

    attr_neg = ts.TosaSerializerAttribute()
    attr_neg.NegateAttribute()
    ser.addOperator(Op.NEGATE, [a.name, zp_in.name, zp_out.name], [neg.name], attr_neg)

    attr_abs = ts.TosaSerializerAttribute()
    attr_abs.AbsAttribute()
    ser.addOperator(Op.ABS, [neg.name], [out.name], attr_abs)

    expected = np.abs(-a_data)
    save_fixture("abs_negate_f32", ser.serialize(), {}, {out.name: expected})


def gen_reshape_f32():
    """RESHAPE CONST [2,3] -> [3,2], f32."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR / "reshape_f32"))

    a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    a = ser.addConst([2, 3], DType.FP32, a_data, "const_a")
    new_shape = ser.addConst([2], DType.SHAPE, np.array([3, 2], dtype=np.int64), "new_shape")
    out = ser.addOutput([3, 2], DType.FP32)

    attr = ts.TosaSerializerAttribute()
    attr.ReshapeAttribute()
    ser.addOperator(Op.RESHAPE, [a.name, new_shape.name], [out.name], attr)

    expected = a_data.reshape(3, 2)
    save_fixture("reshape_f32", ser.serialize(), {}, {out.name: expected})


def gen_broadcast_add_f32():
    """ADD with broadcasting: [2, 3] + [1, 3] -> [2, 3]."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR / "broadcast_add_f32"))

    a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b_data = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)

    a = ser.addConst([2, 3], DType.FP32, a_data, "const_a")
    b = ser.addConst([1, 3], DType.FP32, b_data, "const_b")
    out = ser.addOutput([2, 3], DType.FP32)

    attr = ts.TosaSerializerAttribute()
    attr.AddAttribute()
    ser.addOperator(Op.ADD, [a.name, b.name], [out.name], attr)

    expected = a_data + b_data
    save_fixture("broadcast_add_f32", ser.serialize(), {}, {out.name: expected})


def gen_mul_f32():
    """MUL of two f32 constants with shift=0."""
    ser = ts.TosaSerializer(str(FIXTURES_DIR / "mul_f32"))

    a_data = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    b_data = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)

    a = ser.addConst([4], DType.FP32, a_data, "const_a")
    b = ser.addConst([4], DType.FP32, b_data, "const_b")
    # MUL requires a shift input for int, for float it's a 1-element i8 tensor = 0
    shift = ser.addConst([1], DType.INT8, np.array([0], dtype=np.int8), "shift")
    out = ser.addOutput([4], DType.FP32)

    attr = ts.TosaSerializerAttribute()
    attr.MulAttribute()
    ser.addOperator(Op.MUL, [a.name, b.name, shift.name], [out.name], attr)

    expected = a_data * b_data
    save_fixture("mul_f32", ser.serialize(), {}, {out.name: expected})


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating fixtures in {FIXTURES_DIR}")

    generators = {
        "const_add_f32": gen_const_add_f32,
        "input_add_f32": gen_input_add_f32,
        "sub_i32": gen_sub_i32,
        "abs_negate_f32": gen_abs_negate_f32,
        "reshape_f32": gen_reshape_f32,
        "broadcast_add_f32": gen_broadcast_add_f32,
        "mul_f32": gen_mul_f32,
    }
    # Optionally regenerate only the fixtures named on the command line.
    for name in sys.argv[1:] or generators:
        generators[name]()

    print("Done! Generated fixtures:")
    for d in sorted(FIXTURES_DIR.iterdir()):
        if d.is_dir():
            files = [f.name for f in d.iterdir()]
            print(f"  {d.name}/: {', '.join(sorted(files))}")


if __name__ == "__main__":
    main()
