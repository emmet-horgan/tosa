//! A readable text rendering of a graph, for `tosa dump` and debugging.
//! It is not parsed back; the flatbuffer is the interchange format.

use std::fmt;

use crate::ir::{BasicBlock, DType, Graph, Op, OpAttribute, TensorDef};

fn dtype_name(d: DType) -> &'static str {
    match d {
        DType::Bool => "bool",
        DType::I4 => "i4",
        DType::I8 => "i8",
        DType::I16 => "i16",
        DType::I32 => "i32",
        DType::I48 => "i48",
        DType::I64 => "i64",
        DType::Fp8E4M3 => "fp8e4m3",
        DType::Fp8E5M2 => "fp8e5m2",
        DType::F16 => "f16",
        DType::Bf16 => "bf16",
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::Shape => "shape",
        DType::Fp6E2M3 => "fp6e2m3",
        DType::Fp6E3M2 => "fp6e3m2",
        DType::Fp4E2M1 => "fp4e2m1",
        DType::Fp8Ue8M0 => "fp8ue8m0",
        DType::MxInt8 => "mxint8",
        DType::Unknown => "unknown",
    }
}

fn type_of(def: Option<&TensorDef>) -> String {
    match def {
        Some(t) if t.is_unranked => format!("{}[*]", dtype_name(t.dtype)),
        Some(t) => {
            let dims: Vec<String> = t.shape.iter().map(|d| if *d < 0 { "?".into() } else { d.to_string() }).collect();
            format!("{}[{}]", dtype_name(t.dtype), dims.join(","))
        }
        None => "<undeclared>".into(),
    }
}

fn write_block(f: &mut fmt::Formatter<'_>, block: &BasicBlock) -> fmt::Result {
    let def = |name: &str| block.tensors.iter().find(|t| t.name == name);
    let sig =
        |names: &[String]| names.iter().map(|n| format!("%{n}: {}", type_of(def(n)))).collect::<Vec<_>>().join(", ");
    writeln!(f, "  block \"{}\"({}) -> ({}) {{", block.name, sig(&block.inputs), sig(&block.outputs))?;
    for op in &block.operators {
        let outputs = sig(&op.outputs);
        let inputs = op.inputs.iter().map(|n| format!("%{n}")).collect::<Vec<_>>().join(", ");
        write!(f, "    ")?;
        if !outputs.is_empty() {
            write!(f, "{outputs} = ")?;
        }
        write!(f, "{}({inputs})", op.op.name())?;
        if op.attribute != OpAttribute::None {
            write!(f, " {:?}", op.attribute)?;
        }
        if matches!(op.op, Op::Const | Op::ConstShape | Op::Variable)
            && let Some(bytes) = op.outputs.first().and_then(|n| def(n)).and_then(|t| t.data.as_ref())
        {
            write!(f, " <{} bytes>", bytes.len())?;
        }
        if let Some(loc) = &op.location {
            write!(f, "  // {loc}")?;
        }
        writeln!(f)?;
    }
    writeln!(f, "  }}")
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = &self.version;
        writeln!(f, "tosa v{}.{}.{}{}", v.major, v.minor, v.patch, if v.draft { "-draft" } else { "" })?;
        for region in &self.regions {
            writeln!(f, "region \"{}\" {{", region.name)?;
            for block in &region.blocks {
                write_block(f, block)?;
            }
            writeln!(f, "}}")?;
        }
        Ok(())
    }
}
