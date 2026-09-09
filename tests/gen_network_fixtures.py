#!/usr/bin/env python3
"""Generate multi-op network model fixtures for end-to-end testing.

Builds synthetic network-like TOSA graphs, runs them through the reference
model, and saves inputs + expected outputs for Rust integration tests.

Usage:
    cd /home/ehorgan/tosa-tools
    .venv/bin/python tosa-interp/tests/gen_network_fixtures.py
"""

import json
import sys
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "serialization" / "python"))

import tosa_serializer as ts
from tosa.Op import Op
from tosa.DType import DType
import tosa_reference_model

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def save_fixture(name, tosa_bytes, input_arrays, input_names, output_names):
    """Run through ref model and save fixture."""
    outputs, status = tosa_reference_model.run(tosa_bytes, input_arrays, verbosity="NONE")
    if status != tosa_reference_model.GraphStatus.TOSA_VALID:
        raise RuntimeError(f"Reference model returned status {status}")

    d = FIXTURES_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "graph.tosa").write_bytes(tosa_bytes)

    for i, arr in enumerate(input_arrays):
        np.save(d / f"input_{i}.npy", np.ascontiguousarray(arr))

    for i, arr in enumerate(outputs):
        np.save(d / f"expected_{i}.npy", np.ascontiguousarray(arr))

    meta = {
        "input_names": input_names,
        "output_names": output_names,
        "op": "network",
        "dtype": "f32",
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


def _clamp_attr(ser, min_val, max_val):
    attr = ts.TosaSerializerAttribute()
    min_bytes = ts.TosaSerializer.convertDataToUint8Vec(DType.FP32, [min_val])
    max_bytes = ts.TosaSerializer.convertDataToUint8Vec(DType.FP32, [max_val])
    while len(min_bytes) % 8 != 0:
        min_bytes.append(0)
    while len(max_bytes) % 8 != 0:
        max_bytes.append(0)
    attr.ClampAttribute(min_bytes, max_bytes, ts.NanPropagationMode.PROPAGATE)
    return attr


def gen_perf_model():
    """Generate a scaled-down version of the CI performance model.

    Structure: CONV2D -> CLAMP -> RESHAPE -> MATMUL -> CLAMP (x2 repeats)
    Uses small spatial dims (8x8) for fast testing.
    """
    height, width, cin, cout, matmul_w, repeat = 8, 8, 4, 8, 4, 2
    rng = np.random.default_rng(42)

    ser = ts.TosaSerializer(str(FIXTURES_DIR))

    input_shape = [1, height, width, cin]
    conv_w_shape = [cout, 3, 3, cin]
    conv_b_shape = [cout]
    matmul_w_shape = [1, cout, matmul_w]

    input_data = rng.standard_normal(input_shape).astype(np.float32) * 0.1
    conv_w_data = rng.standard_normal(conv_w_shape).astype(np.float32) * 0.1
    conv_b_data = rng.standard_normal(conv_b_shape).astype(np.float32) * 0.01
    matmul_w_data = rng.standard_normal(matmul_w_shape).astype(np.float32) * 0.1

    ser.addInputTensor(ts.TosaSerializerTensor("input_0", input_shape, DType.FP32))
    ser.addInputTensor(ts.TosaSerializerTensor("conv_w", conv_w_shape, DType.FP32))
    ser.addInputTensor(ts.TosaSerializerTensor("conv_b", conv_b_shape, DType.FP32))
    ser.addInputTensor(ts.TosaSerializerTensor("matmul_w", matmul_w_shape, DType.FP32))

    conv_zp = np.zeros((1,), dtype=np.float32)
    matmul_zp = np.zeros((1,), dtype=np.float32)
    ser.addConst([1], DType.FP32, conv_zp, "conv_zp")
    ser.addConst([1], DType.FP32, matmul_zp, "matmul_zp")

    prev = "input_0"
    for idx in range(repeat):
        conv_out = ser.addIntermediate([1, height, width, cout], DType.FP32)
        conv_attr = ts.TosaSerializerAttribute()
        conv_attr.Conv2dAttribute([1, 1, 1, 1], [1, 1], [1, 1], False, DType.FP32)
        ser.addOperator(
            Op.CONV2D,
            [prev, "conv_w", "conv_b", "conv_zp", "conv_zp"],
            [conv_out.name],
            conv_attr,
        )

        clamp1_out = ser.addIntermediate([1, height, width, cout], DType.FP32)
        ser.addOperator(Op.CLAMP, [conv_out.name], [clamp1_out.name],
                        _clamp_attr(ser, -1.0, 1.0))

        reshape_shape = [1, height * width, cout]
        reshape_name = f"reshape_shape_{idx}"
        ser.addConst([3], DType.SHAPE, np.array(reshape_shape, dtype=np.int64), reshape_name)
        reshape_out = ser.addIntermediate(reshape_shape, DType.FP32)
        reshape_attr = ts.TosaSerializerAttribute()
        reshape_attr.ReshapeAttribute()
        ser.addOperator(Op.RESHAPE, [clamp1_out.name, reshape_name], [reshape_out.name], reshape_attr)

        matmul_out = ser.addIntermediate([1, height * width, matmul_w], DType.FP32)
        matmul_attr = ts.TosaSerializerAttribute()
        matmul_attr.MatMulAttribute()
        ser.addOperator(
            Op.MATMUL,
            [reshape_out.name, "matmul_w", "matmul_zp", "matmul_zp"],
            [matmul_out.name],
            matmul_attr,
        )

        clamp2_out = ser.addIntermediate([1, height * width, matmul_w], DType.FP32)
        ser.addOperator(Op.CLAMP, [matmul_out.name], [clamp2_out.name],
                        _clamp_attr(ser, -0.5, 0.5))

        if idx < repeat - 1:
            reshape_back = [1, height, width, cin]
            reshape_back_name = f"reshape_back_{idx}"
            ser.addConst([4], DType.SHAPE, np.array(reshape_back, dtype=np.int64), reshape_back_name)
            reshape_back_out = ser.addIntermediate(reshape_back, DType.FP32)
            rb_attr = ts.TosaSerializerAttribute()
            rb_attr.ReshapeAttribute()
            ser.addOperator(Op.RESHAPE, [clamp2_out.name, reshape_back_name],
                            [reshape_back_out.name], rb_attr)
            prev = reshape_back_out.name
        else:
            prev = clamp2_out.name

    output = ser.addOutput([1, height * width, matmul_w], DType.FP32)
    id_attr = ts.TosaSerializerAttribute()
    id_attr.IdentityAttribute()
    ser.addOperator(Op.IDENTITY, [prev], [output.name], id_attr)

    graph = ser.serialize()
    input_arrays = [input_data, conv_w_data, conv_b_data, matmul_w_data]
    input_names = ["input_0", "conv_w", "conv_b", "matmul_w"]

    save_fixture("network_perf_conv_matmul", graph, input_arrays, input_names, [output.name])
    print("  network_perf_conv_matmul: OK (CONV2D->CLAMP->RESHAPE->MATMUL->CLAMP x2)")


def gen_mobilenet_like():
    """Generate a synthetic MobileNet-like model.

    Structure: CONV2D -> CLAMP(ReLU6) -> DEPTHWISE_CONV2D -> CLAMP -> CONV2D ->
               AVG_POOL2D -> RESHAPE -> MATMUL (classifier)
    Small dims: [1, 8, 8, 4] input
    """
    rng = np.random.default_rng(123)
    N, H, W, C = 1, 8, 8, 4
    M = 1  # depthwise multiplier
    COUT = 8  # pointwise conv output channels
    NUM_CLASSES = 4

    ser = ts.TosaSerializer(str(FIXTURES_DIR))

    # Inputs
    input_shape = [N, H, W, C]
    input_data = rng.standard_normal(input_shape).astype(np.float32) * 0.5

    ser.addInputTensor(ts.TosaSerializerTensor("input", input_shape, DType.FP32))
    ser.addConst([1], DType.FP32, np.zeros(1, dtype=np.float32), "zp")

    # --- Block 1: Depthwise Conv (3x3) ---
    dw_w = rng.standard_normal([3, 3, C, M]).astype(np.float32) * 0.1
    dw_b = np.zeros(C * M, dtype=np.float32)
    ser.addConst([3, 3, C, M], DType.FP32, dw_w, "dw_weight")
    ser.addConst([C * M], DType.FP32, dw_b, "dw_bias")

    dw_out = ser.addIntermediate([N, H, W, C * M], DType.FP32)
    dw_attr = ts.TosaSerializerAttribute()
    dw_attr.DepthwiseConv2dAttribute([1, 1, 1, 1], [1, 1], [1, 1], False, DType.FP32)
    ser.addOperator(Op.DEPTHWISE_CONV2D,
                    ["input", "dw_weight", "dw_bias", "zp", "zp"],
                    [dw_out.name], dw_attr)

    # ReLU6 (clamp 0..6)
    relu1_out = ser.addIntermediate([N, H, W, C * M], DType.FP32)
    ser.addOperator(Op.CLAMP, [dw_out.name], [relu1_out.name], _clamp_attr(ser, 0.0, 6.0))

    # --- Block 2: Pointwise Conv (1x1) ---
    pw_w = rng.standard_normal([COUT, 1, 1, C * M]).astype(np.float32) * 0.1
    pw_b = np.zeros(COUT, dtype=np.float32)
    ser.addConst([COUT, 1, 1, C * M], DType.FP32, pw_w, "pw_weight")
    ser.addConst([COUT], DType.FP32, pw_b, "pw_bias")

    pw_out = ser.addIntermediate([N, H, W, COUT], DType.FP32)
    pw_attr = ts.TosaSerializerAttribute()
    pw_attr.Conv2dAttribute([0, 0, 0, 0], [1, 1], [1, 1], False, DType.FP32)
    ser.addOperator(Op.CONV2D,
                    [relu1_out.name, "pw_weight", "pw_bias", "zp", "zp"],
                    [pw_out.name], pw_attr)

    # ReLU6
    relu2_out = ser.addIntermediate([N, H, W, COUT], DType.FP32)
    ser.addOperator(Op.CLAMP, [pw_out.name], [relu2_out.name], _clamp_attr(ser, 0.0, 6.0))

    # --- Global Average Pool ---
    pool_out = ser.addIntermediate([N, 1, 1, COUT], DType.FP32)
    pool_attr = ts.TosaSerializerAttribute()
    pool_attr.AvgPool2dAttribute([H, W], [H, W], [0, 0, 0, 0], DType.FP32)
    ser.addConst([1], DType.FP32, np.zeros(1, dtype=np.float32), "pool_zp_in")
    ser.addConst([1], DType.FP32, np.zeros(1, dtype=np.float32), "pool_zp_out")
    ser.addOperator(Op.AVG_POOL2D,
                    [relu2_out.name, "pool_zp_in", "pool_zp_out"],
                    [pool_out.name], pool_attr)

    # --- Reshape to [1, 1, COUT] for matmul ---
    rs_shape = [1, 1, COUT]
    rs_name = "reshape_shape"
    ser.addConst([3], DType.SHAPE, np.array(rs_shape, dtype=np.int64), rs_name)
    rs_out = ser.addIntermediate(rs_shape, DType.FP32)
    rs_attr = ts.TosaSerializerAttribute()
    rs_attr.ReshapeAttribute()
    ser.addOperator(Op.RESHAPE, [pool_out.name, rs_name], [rs_out.name], rs_attr)

    # --- Classifier: MATMUL ---
    fc_w = rng.standard_normal([1, COUT, NUM_CLASSES]).astype(np.float32) * 0.1
    ser.addConst([1, COUT, NUM_CLASSES], DType.FP32, fc_w, "fc_weight")
    ser.addConst([1], DType.FP32, np.zeros(1, dtype=np.float32), "fc_zp_a")
    ser.addConst([1], DType.FP32, np.zeros(1, dtype=np.float32), "fc_zp_b")

    fc_out = ser.addIntermediate([1, 1, NUM_CLASSES], DType.FP32)
    fc_attr = ts.TosaSerializerAttribute()
    fc_attr.MatMulAttribute()
    ser.addOperator(Op.MATMUL,
                    [rs_out.name, "fc_weight", "fc_zp_a", "fc_zp_b"],
                    [fc_out.name], fc_attr)

    # Output
    output = ser.addOutput([1, 1, NUM_CLASSES], DType.FP32)
    id_attr = ts.TosaSerializerAttribute()
    id_attr.IdentityAttribute()
    ser.addOperator(Op.IDENTITY, [fc_out.name], [output.name], id_attr)

    graph = ser.serialize()
    input_arrays = [input_data]
    input_names = ["input"]

    save_fixture("network_mobilenet_like", graph, input_arrays, input_names, [output.name])
    print("  network_mobilenet_like: OK (DW_CONV->ReLU6->PW_CONV->ReLU6->AVG_POOL->MATMUL)")


def gen_lstm_model():
    """Test the pre-existing LSTM model from the repo examples."""
    lstm_path = REPO_ROOT / "reference_model" / "examples" / "test_lstm_stateful_13x21x3_f32" / "flatbuffer-tflite" / "test_lstm_stateful_13x21x3_f32.tosa"
    if not lstm_path.exists():
        print("  LSTM model not found, skipping")
        return

    tosa_bytes = lstm_path.read_bytes()

    # Parse graph to get input info
    from tosa.TosaGraph import TosaGraph
    g = TosaGraph.GetRootAs(bytearray(tosa_bytes))
    block = g.Regions(0).Blocks(0)

    input_names = []
    input_arrays = []

    rng = np.random.default_rng(99)
    for i in range(block.InputsLength()):
        name = block.Inputs(i).decode()
        input_names.append(name)
        # Find tensor def
        for j in range(block.TensorsLength()):
            t = block.Tensors(j)
            if t.Name().decode() == name:
                shape = [t.Shape(k) for k in range(t.ShapeLength())]
                # Generate small random inputs
                arr = rng.standard_normal(shape).astype(np.float32) * 0.1
                input_arrays.append(arr)
                break

    if not input_arrays:
        print("  LSTM: could not determine inputs, skipping")
        return

    # Get output names
    output_names = [block.Outputs(i).decode() for i in range(block.OutputsLength())]

    try:
        save_fixture("network_lstm", tosa_bytes, input_arrays, input_names, output_names)
        print(f"  network_lstm: OK ({len(input_names)} inputs, {len(output_names)} outputs)")
    except Exception as e:
        print(f"  network_lstm: SKIPPED - TOSA 1.0 model incompatible with 1.1 ref model ({e})")


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating network model fixtures...")

    gen_perf_model()
    gen_mobilenet_like()
    gen_lstm_model()

    print("\nDone!")


if __name__ == "__main__":
    main()
