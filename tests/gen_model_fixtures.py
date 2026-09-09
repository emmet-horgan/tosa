#!/usr/bin/env python3
"""Generate reference model outputs for pre-existing .tosa model files.

For each model: loads the graph, generates random inputs matching the input shapes,
runs through the reference model, and saves inputs + expected outputs.

Usage:
    cd /home/ehorgan/tosa-tools
    .venv/bin/python tosa-interp/tests/gen_model_fixtures.py
"""

import json
import sys
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "serialization" / "python"))

import tosa_reference_model
from tosa.TosaGraph import TosaGraph
from tosa.DType import DType

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

DTYPE_TO_NP = {
    DType.BOOL: np.bool_,
    DType.INT8: np.int8,
    DType.INT16: np.int16,
    DType.INT32: np.int32,
    DType.INT64: np.int64,
    DType.FP16: np.float16,
    DType.FP32: np.float32,
    DType.BF16: np.float32,  # use f32 for bf16 inputs
}


def get_graph_info(tosa_bytes):
    """Extract input tensor names, shapes, and dtypes from a TOSA graph."""
    g = TosaGraph.GetRootAs(bytearray(tosa_bytes))
    block = g.Regions(0).Blocks(0)

    inputs = []
    for i in range(block.InputsLength()):
        name = block.Inputs(i).decode()
        # Find matching tensor def
        for j in range(block.TensorsLength()):
            t = block.Tensors(j)
            if t.Name().decode() == name:
                shape = [t.Shape(k) for k in range(t.ShapeLength())]
                dtype = t.Type()
                inputs.append({"name": name, "shape": shape, "dtype": dtype})
                break

    outputs = []
    for i in range(block.OutputsLength()):
        name = block.Outputs(i).decode()
        outputs.append(name)

    return inputs, outputs


def generate_input(shape, dtype, seed=42):
    """Generate random input data for a given shape and dtype."""
    rng = np.random.RandomState(seed)
    np_dtype = DTYPE_TO_NP.get(dtype)
    if np_dtype is None:
        return None

    # Use small shapes if dimensions are very large (for testing speed)
    effective_shape = [min(d, 64) if d > 0 else 1 for d in shape]

    if np_dtype == np.bool_:
        return rng.choice([True, False], size=effective_shape).astype(np.bool_)
    elif np.issubdtype(np_dtype, np.integer):
        return rng.randint(-5, 5, size=effective_shape).astype(np_dtype)
    else:
        return rng.uniform(-1.0, 1.0, size=effective_shape).astype(np_dtype)


def run_model_test(name, tosa_path, npy_inputs=None):
    """Run a model through the reference model and save fixtures.

    Args:
        name: fixture name (e.g., "model_add_f32")
        tosa_path: path to .tosa file
        npy_inputs: optional dict of {tensor_name: path_to_npy} for pre-existing inputs
    """
    tosa_bytes = tosa_path.read_bytes()

    input_infos, output_names = get_graph_info(tosa_bytes)
    if not input_infos:
        print(f"  {name}: no inputs found, skipping")
        return False

    # Prepare inputs
    input_arrays = []
    input_dict = {}

    for i, info in enumerate(input_infos):
        if npy_inputs and info["name"] in npy_inputs:
            arr = np.load(npy_inputs[info["name"]])
            arr = np.ascontiguousarray(arr)
        else:
            arr = generate_input(info["shape"], info["dtype"], seed=42 + i)
            if arr is None:
                print(f"  {name}: unsupported dtype {info['dtype']} for input '{info['name']}', skipping")
                return False

        input_arrays.append(arr)
        input_dict[info["name"]] = arr

    # Run reference model
    try:
        outputs, status = tosa_reference_model.run(
            tosa_bytes, input_arrays, verbosity="NONE"
        )
    except Exception as e:
        print(f"  {name}: reference model error: {e}")
        return False

    if status != tosa_reference_model.GraphStatus.TOSA_VALID:
        print(f"  {name}: reference model returned status {status}")
        return False

    # Save fixture
    fixture_dir = FIXTURES_DIR / f"model_{name}"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    # Copy .tosa file
    (fixture_dir / "graph.tosa").write_bytes(tosa_bytes)

    # Save inputs
    for i, (info, arr) in enumerate(zip(input_infos, input_arrays)):
        np.save(fixture_dir / f"input_{i}.npy", np.ascontiguousarray(arr))

    # Save expected outputs
    for i, out_arr in enumerate(outputs):
        np.save(fixture_dir / f"expected_{i}.npy", np.ascontiguousarray(out_arr))

    # Save metadata
    meta = {
        "input_names": [info["name"] for info in input_infos],
        "output_names": output_names,
        "op": "model",
        "dtype": "mixed",
        "source": str(tosa_path.relative_to(REPO_ROOT)),
    }
    (fixture_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return True


def main():
    np.random.seed(42)
    print("Generating model test fixtures...")

    models = []

    # Reference model examples
    examples_dir = REPO_ROOT / "reference_model" / "examples"

    # ADD f32 (TF)
    add_tf = examples_dir / "test_add_1x4x4x4_f32" / "flatbuffer-tf" / "test_add_1x4x4x4_f32.tosa"
    if add_tf.exists():
        # Use the provided .npy inputs
        parent = add_tf.parent.parent
        npy_inputs = {}
        p0 = parent / "placeholder_0.npy"
        p1 = parent / "placeholder_1.npy"
        if p0.exists() and p1.exists():
            npy_inputs = {"TosaInput_0": str(p0), "TosaInput_1": str(p1)}
        models.append(("add_f32_tf", add_tf, npy_inputs))

    # ADD f32 (TFLite)
    add_tflite = examples_dir / "test_add_1x4x4x4_f32" / "flatbuffer-tflite" / "test_add_1x4x4x4_f32.tosa"
    if add_tflite.exists():
        parent = add_tflite.parent.parent
        npy_inputs = {}
        p0 = parent / "placeholder_0.npy"
        p1 = parent / "placeholder_1.npy"
        if p0.exists() and p1.exists():
            npy_inputs = {"TosaInput_0": str(p0), "TosaInput_1": str(p1)}
        models.append(("add_f32_tflite", add_tflite, npy_inputs))

    # CONV2D f32 (TF)
    conv_tf = examples_dir / "test_conv2d_1x1_1x32x32x8_f32_st11_padSAME_dilat11" / "flatbuffer-tf" / "test_conv2d_1x1_1x32x32x8_f32_st11_padSAME_dilat11.tosa"
    if conv_tf.exists():
        models.append(("conv2d_f32_tf", conv_tf, {}))

    # CONV2D f32 (TFLite)
    conv_tflite = examples_dir / "test_conv2d_1x1_1x32x32x8_f32_st11_padSAME_dilat11" / "flatbuffer-tflite" / "test_conv2d_1x1_1x32x32x8_f32_st11_padSAME_dilat11.tosa"
    if conv_tflite.exists():
        models.append(("conv2d_f32_tflite", conv_tflite, {}))

    # CONV2D qi8 (TFLite)
    conv_qi8 = examples_dir / "test_conv2d_1x1_1x32x32x8_qi8_st11_padSAME_dilat11" / "flatbuffer-tflite" / "test_conv2d_1x1_1x32x32x8_qi8_st11_padSAME_dilat11.tosa"
    if conv_qi8.exists():
        models.append(("conv2d_qi8_tflite", conv_qi8, {}))

    # Serialization test examples
    ser_dir = REPO_ROOT / "serialization" / "test" / "examples"

    ser_add = ser_dir / "test_add_1x4x4x4_f32.tosa"
    if ser_add.exists():
        models.append(("ser_add_f32", ser_add, {}))

    passed = 0
    failed = 0

    for name, path, npy_inputs in models:
        print(f"  Processing {name}...")
        try:
            if run_model_test(name, path, npy_inputs or None):
                passed += 1
                print(f"    OK")
            else:
                failed += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {len(models)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
