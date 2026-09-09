# tosa-interp

A Rust implementation of a TOSA (Tensor Operator Set Architecture) interpreter, designed for use as an IR interpreter in compiler toolchains with built-in statistics collection hooks.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Public API: run_graph(), Interpreter::run()     │
├─────────────────────────────────────────────────┤
│  Interpreter Engine (ready-list scheduler)       │
│  - Dataflow-driven execution                    │
│  - Recursive sub-block eval for control flow    │
├─────────────────────────────────────────────────┤
│  Op Dispatch (match on Op enum)                 │
├──────────┬──────────┬──────────┬────────────────┤
│ binary   │ unary    │ compare  │ tensor_ops     │
│ 16 ops   │ 17 ops   │ 4 ops    │ conv/matmul/.. │
├──────────┼──────────┼──────────┼────────────────┤
│ reduction│ layout   │ type_conv│ shape_ops      │
│ 6 ops    │ 9 ops    │ 2 ops    │ 15 ops         │
├──────────┴──────────┴──────────┴────────────────┤
│  IR Layer (owned types from FlatBuffer)          │
├─────────────────────────────────────────────────┤
│  FlatBuffers Deserialization (flatc --rust)      │
└─────────────────────────────────────────────────┘
```

- **Runtime shape**: All tensor shapes are dynamic `Vec<usize>` — no compile-time rank specialization
- **Dtype dispatch**: `TensorData` enum with per-element-type variants, matched at each op boundary
- **Statistics hooks**: `InterpreterHook` trait allows composable collectors via `HookSet`

## Supported Operators

### Fully Implemented & Equivalence-Tested (88 ops)

| Category | Operators | Tested Dtypes |
|----------|-----------|---------------|
| **Element-wise binary** | ADD, SUB, MUL, MAXIMUM, MINIMUM, POW, ARITHMETIC_RIGHT_SHIFT, INTDIV, LOGICAL_{AND,OR,XOR}, LOGICAL_{LEFT,RIGHT}_SHIFT, BITWISE_{AND,OR,XOR} | F32, I32, I8, I16, I64, Bool |
| **Element-wise unary** | ABS, NEGATE, CEIL, FLOOR, EXP, LOG, SIN, COS, RECIPROCAL, RSQRT, ERF, SIGMOID, TANH, CLZ, BITWISE_NOT, LOGICAL_NOT, CLAMP | F32, I32, I8, I16, Bool |
| **Comparison** | EQUAL, GREATER, GREATER_EQUAL | F32, I32, I8 |
| **Ternary** | SELECT | F32, I32, I8, I16 |
| **Reduction** | REDUCE_SUM, REDUCE_MAX, REDUCE_MIN, REDUCE_PRODUCT, REDUCE_ALL, REDUCE_ANY | F32, I32, Bool |
| **Data layout** | RESHAPE, TRANSPOSE, CONCAT, SLICE, PAD, TILE, REVERSE, GATHER, SCATTER | F32, I32, I8 |
| **Tensor compute** | CONV2D, DEPTHWISE_CONV2D, TRANSPOSE_CONV2D, CONV3D, MATMUL, AVG_POOL2D, MAX_POOL2D, ARGMAX | F32, I8 |
| **Type conversion** | CAST, RESCALE | F32↔I32, I8↔I32, F32↔F16 |
| **Data nodes** | CONST, CONST_SHAPE, IDENTITY, VARIABLE, VARIABLE_READ, VARIABLE_WRITE | All |
| **Control flow** | COND_IF, WHILE_LOOP | F32, I32 |
| **Table** | TABLE | I8 |
| **Resize** | RESIZE | F32 |
| **Shape ops** | ADD_SHAPE, SUB_SHAPE, MUL_SHAPE, CONCAT_SHAPE, SLICE_SHAPE, EXP2_SHAPE, LOG2_{CEIL,FLOOR}_SHAPE, MAX_SHAPE, MIN_SHAPE, MOD_SHAPE, DIV_{CEIL,FLOOR}_SHAPE, DIM, ASSERT_EQUAL_SHAPE | I64 (SHAPE) |
| **Adaptive pooling** | AVG_POOL2D_ADAPTIVE, MAX_POOL2D_ADAPTIVE | F32 |

### Excluded Operators (9 ops)

| Operator | Justification |
|----------|--------------|
| **FFT2D** | Specialized signal processing op. Requires complex number support not yet implemented. Rarely used in standard inference workloads. |
| **RFFT2D** | Real-valued FFT. Same reasoning as FFT2D. |
| **CUSTOM** | Vendor extension mechanism. By definition implementation-specific — cannot be interpreted generically. Returns `UnsupportedOp` error. |
| **MATMUL_T_BLOCK_SCALED** | MX (Microscaling) format operation for block-scaled quantization. Requires specialized MX floating-point format support (FP4/FP6/FP8 with per-block E8M0 scales). |
| **CAST_FROM_BLOCK_SCALED** | Conversion from MX block-scaled format to standard types. Same MX format dependency. |
| **CAST_TO_BLOCK_SCALED** | Conversion to MX block-scaled format. Same MX format dependency. |
| **CONV2D_BLOCK_SCALED** | Convolution with MX block-scaled weights. Same MX format dependency. |
| **RESHAPE_BLOCK_SCALED** | Reshape maintaining block scaling metadata. Same MX format dependency. |
| **ROW_GATHER_BLOCK_SCALED** | Gather with block-scaled format. Same MX format dependency. |

The MX (Microscaling) block-scaled ops are part of the `tosa-ext-mxfp` extension and require support for MXINT8, FP4E2M1, FP6E2M3, FP6E3M2, FP8E4M3, and FP8UE8M0 formats with per-block scale factors. These can be added when MX format support is needed for a specific compiler target.

## Testing

### Methodology

The interpreter is tested at three levels:

1. **Equivalence tests** (154 cases): For each op × dtype combination, a Python script builds a minimal TOSA graph using `TosaSerializer`, generates random inputs, runs the graph through the official ARM TOSA reference model (`tosa_reference_model.run()`), and saves the outputs as ground truth. The Rust test harness loads each fixture and verifies our interpreter produces bitwise-identical (integer) or ULP-accurate (float) results.

2. **Integration tests** (20 cases): Hand-crafted test graphs built with the FlatBuffers Rust API, testing specific features like broadcasting, multi-op pipelines, and the stats collection system.

3. **Model tests** (5 cases): Real `.tosa` model files from the tosa-tools repository (produced by tf2tosa and tflite2tosa converters) are loaded, executed with random inputs, and outputs are compared element-wise against the reference model. Float models achieve **exact bitwise parity** (max_diff=0). The quantized INT8 CONV2D model shows ≤1 LSB difference in ~10% of elements due to RESCALE rounding mode nuance.

### Running Tests

```bash
# Generate equivalence test fixtures (requires Python venv with tosa-tools installed)
cd /path/to/tosa-tools
.venv/bin/python tosa-interp/tests/test_equivalence.py

# Run all Rust tests
cd tosa-interp
cargo test

# Run only equivalence tests with output
cargo test --test equivalence -- --nocapture
```

### Test Coverage Summary

| Test Suite | Cases | Description |
|------------|-------|-------------|
| Unit tests | 5 | Broadcasting logic |
| Equivalence | 154 | Reference model parity across all ops and dtypes |
| Integration | 20 | Feature-level tests (broadcasting, stats, etc.) |
| Model tests | 5 | Real .tosa models with output comparison against ref model |
| **Total** | **184** | All passing |

## Usage

```rust
use tosa_interp::{run_graph, run_graph_with_hooks, Tensor, TensorData, HookSet, StatsCollector, FlopCounter};
use std::collections::HashMap;

// Load a .tosa file
let data = std::fs::read("model.tosa")?;

// Prepare inputs
let mut inputs = HashMap::new();
inputs.insert(
    "input_0".to_string(),
    Tensor::new("input_0".to_string(), vec![1, 224, 224, 3], TensorData::F32(vec![0.0; 150528])),
);

// Run with statistics collection
let mut hooks = HookSet::new();
hooks.add(StatsCollector::new());
hooks.add(FlopCounter::new());

let outputs = run_graph_with_hooks(&data, inputs, &mut hooks)?;
```

## Statistics Collection

The `InterpreterHook` trait enables pluggable analysis:

```rust
pub trait InterpreterHook {
    fn pre_op(&mut self, op: &OpInfo, inputs: &[&Tensor]) {}
    fn post_op(&mut self, op: &OpInfo, inputs: &[&Tensor], outputs: &[&Tensor]) {}
    fn on_tensor_alloc(&mut self, name: &str, shape: &[usize], dtype: DType, bytes: usize) {}
    fn on_tensor_free(&mut self, name: &str, bytes: usize) {}
}
```

Built-in collectors:
- **`StatsCollector`** — op counts, per-op timing, peak memory usage
- **`FlopCounter`** — estimated FLOPs for compute ops (MATMUL, CONV2D, element-wise)
- **`OpTrace`** — records every op execution with input/output shapes

Multiple collectors compose via `HookSet` or `Vec<Box<dyn InterpreterHook>>`.

## Spec Compliance Notes

- **CAST float→int**: Uses round-to-nearest-even (matching TOSA spec), not C-style truncation
- **Integer overflow**: All integer arithmetic uses wrapping semantics (matching TOSA spec)
- **F16/BF16 compute**: Promoted to F32 for computation, truncated back (matching reference model behavior)
- **Control flow**: Sub-blocks looked up by region name (TOSA 1.1 format)
- **NaN propagation**: Always propagates (PROPAGATE mode) for MAXIMUM/MINIMUM/REDUCE_MAX/REDUCE_MIN
- **RESCALE**: Supports per-tensor multiplier/shift with zero-points

## Dependencies

- `flatbuffers` — FlatBuffer deserialization (schema compiled at build time via `flatc`)
- `half` — F16/BF16 types with num-traits support
- `thiserror` — Error type derivation
- `npyz` — NumPy .npy file reading (for test fixtures)
