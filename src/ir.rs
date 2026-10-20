use crate::error::{Error, Result};
use crate::generated;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    I4,
    I8,
    I16,
    I32,
    I48,
    I64,
    Fp8E4M3,
    Fp8E5M2,
    F16,
    Bf16,
    F32,
    F64,
    Shape,
    // Microscaling (MX) formats: recognised but not yet executable.
    Fp6E2M3,
    Fp6E3M2,
    Fp4E2M1,
    Fp8Ue8M0,
    MxInt8,
    Unknown,
}

impl DType {
    pub fn from_fb(dt: generated::DType) -> Self {
        match dt {
            generated::DType::BOOL => DType::Bool,
            generated::DType::INT4 => DType::I4,
            generated::DType::INT8 => DType::I8,
            generated::DType::INT16 => DType::I16,
            generated::DType::INT32 => DType::I32,
            generated::DType::INT48 => DType::I48,
            generated::DType::INT64 => DType::I64,
            generated::DType::FP8E4M3 => DType::Fp8E4M3,
            generated::DType::FP8E5M2 => DType::Fp8E5M2,
            generated::DType::FP16 => DType::F16,
            generated::DType::BF16 => DType::Bf16,
            generated::DType::FP32 => DType::F32,
            generated::DType::SHAPE => DType::Shape,
            generated::DType::FP6E2M3 => DType::Fp6E2M3,
            generated::DType::FP6E3M2 => DType::Fp6E3M2,
            generated::DType::FP4E2M1 => DType::Fp4E2M1,
            generated::DType::FP8UE8M0 => DType::Fp8Ue8M0,
            generated::DType::MXINT8 => DType::MxInt8,
            _ => DType::Unknown,
        }
    }

    /// Bytes per element in memory (INT4 is held unpacked, one per byte).
    pub fn element_size_bytes(self) -> usize {
        match self {
            DType::Bool | DType::I4 | DType::I8 | DType::Fp8E4M3 | DType::Fp8E5M2 => 1,
            DType::Fp6E2M3 | DType::Fp6E3M2 | DType::Fp4E2M1 | DType::Fp8Ue8M0 | DType::MxInt8 => 1,
            DType::I16 | DType::F16 | DType::Bf16 => 2,
            DType::I32 | DType::F32 => 4,
            DType::I48 | DType::I64 | DType::F64 | DType::Shape => 8,
            DType::Unknown => 0,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, DType::Fp8E4M3 | DType::Fp8E5M2 | DType::F16 | DType::Bf16 | DType::F32 | DType::F64)
    }

    pub fn is_int(self) -> bool {
        matches!(self, DType::I4 | DType::I8 | DType::I16 | DType::I32 | DType::I48 | DType::I64)
    }
}

/// RESCALE rounding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundingMode {
    /// Round half up once.
    Single,
    /// Round half up, with an extra rounding step for shifts above 31.
    Double,
    /// Implementation-defined; the reference evaluates it in FP32.
    Inexact,
    Unknown,
}

impl RoundingMode {
    fn from_fb(m: generated::RoundingMode) -> Self {
        match m {
            generated::RoundingMode::SINGLE_ROUND => RoundingMode::Single,
            generated::RoundingMode::DOUBLE_ROUND => RoundingMode::Double,
            generated::RoundingMode::INEXACT_ROUND => RoundingMode::Inexact,
            _ => RoundingMode::Unknown,
        }
    }
}

/// NaN handling of min/max style operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NanMode {
    /// Any NaN operand makes the result NaN.
    Propagate,
    /// NaN operands are ignored unless every operand is NaN.
    Ignore,
    /// Missing or unrecognised in the graph; rejected by operators that use it.
    Unknown,
}

impl NanMode {
    fn from_fb(m: generated::NanPropagationMode) -> Self {
        match m {
            generated::NanPropagationMode::PROPAGATE => NanMode::Propagate,
            generated::NanPropagationMode::IGNORE => NanMode::Ignore,
            _ => NanMode::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    // Data nodes
    Const,
    ConstShape,
    Identity,
    Variable,
    VariableRead,
    VariableWrite,
    // Element-wise binary
    Add,
    Sub,
    Mul,
    Maximum,
    Minimum,
    Pow,
    ArithmeticRightShift,
    Intdiv,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
    LogicalLeftShift,
    LogicalRightShift,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    // Element-wise unary
    Abs,
    Negate,
    Ceil,
    Floor,
    Exp,
    Log,
    Sin,
    Cos,
    Reciprocal,
    Rsqrt,
    Erf,
    Sigmoid,
    Tanh,
    Clz,
    BitwiseNot,
    LogicalNot,
    Clamp,
    // Comparison
    Equal,
    Greater,
    GreaterEqual,
    // Ternary
    Select,
    // Reduction
    ReduceAll,
    ReduceAny,
    ReduceMax,
    ReduceMin,
    ReduceProduct,
    ReduceSum,
    // Data layout
    Reshape,
    Transpose,
    Concat,
    Slice,
    Pad,
    Tile,
    Reverse,
    Gather,
    Scatter,
    // Tensor compute
    Conv2d,
    Conv3d,
    DepthwiseConv2d,
    TransposeConv2d,
    MatMul,
    AvgPool2d,
    MaxPool2d,
    ArgMax,
    // Type conversion
    Cast,
    Rescale,
    // Table
    Table,
    // Control flow
    CondIf,
    WhileLoop,
    // Resize
    Resize,
    // Shape ops
    AddShape,
    SubShape,
    MulShape,
    ConcatShape,
    SliceShape,
    Exp2Shape,
    Log2CeilShape,
    Log2FloorShape,
    MaxShape,
    MinShape,
    ModShape,
    DivCeilShape,
    DivFloorShape,
    Dim,
    AssertEqualShape,
    // Block-scaled ops
    MatMulTBlockScaled,
    CastFromBlockScaled,
    CastToBlockScaled,
    Conv2dBlockScaled,
    ReshapeBlockScaled,
    RowGatherBlockScaled,
    // Adaptive pooling
    AvgPool2dAdaptive,
    MaxPool2dAdaptive,
    // FFT
    Fft2d,
    Rfft2d,
    // Custom
    Custom,
    // TOSA 1.1 additions
    RowGather,
    MatMulT,
    ArgMin,
    // Unknown/unsupported
    Unknown(u32),
}

impl Op {
    pub fn from_fb(op: generated::Op) -> Self {
        match op {
            generated::Op::CONST => Op::Const,
            generated::Op::CONST_SHAPE => Op::ConstShape,
            generated::Op::IDENTITY => Op::Identity,
            generated::Op::VARIABLE => Op::Variable,
            generated::Op::VARIABLE_READ => Op::VariableRead,
            generated::Op::VARIABLE_WRITE => Op::VariableWrite,
            generated::Op::ADD => Op::Add,
            generated::Op::SUB => Op::Sub,
            generated::Op::MUL => Op::Mul,
            generated::Op::MAXIMUM => Op::Maximum,
            generated::Op::MINIMUM => Op::Minimum,
            generated::Op::POW => Op::Pow,
            generated::Op::ARITHMETIC_RIGHT_SHIFT => Op::ArithmeticRightShift,
            generated::Op::INTDIV => Op::Intdiv,
            generated::Op::LOGICAL_AND => Op::LogicalAnd,
            generated::Op::LOGICAL_OR => Op::LogicalOr,
            generated::Op::LOGICAL_XOR => Op::LogicalXor,
            generated::Op::LOGICAL_LEFT_SHIFT => Op::LogicalLeftShift,
            generated::Op::LOGICAL_RIGHT_SHIFT => Op::LogicalRightShift,
            generated::Op::BITWISE_AND => Op::BitwiseAnd,
            generated::Op::BITWISE_OR => Op::BitwiseOr,
            generated::Op::BITWISE_XOR => Op::BitwiseXor,
            generated::Op::ABS => Op::Abs,
            generated::Op::NEGATE => Op::Negate,
            generated::Op::CEIL => Op::Ceil,
            generated::Op::FLOOR => Op::Floor,
            generated::Op::EXP => Op::Exp,
            generated::Op::LOG => Op::Log,
            generated::Op::SIN => Op::Sin,
            generated::Op::COS => Op::Cos,
            generated::Op::RECIPROCAL => Op::Reciprocal,
            generated::Op::RSQRT => Op::Rsqrt,
            generated::Op::ERF => Op::Erf,
            generated::Op::SIGMOID => Op::Sigmoid,
            generated::Op::TANH => Op::Tanh,
            generated::Op::CLZ => Op::Clz,
            generated::Op::BITWISE_NOT => Op::BitwiseNot,
            generated::Op::LOGICAL_NOT => Op::LogicalNot,
            generated::Op::CLAMP => Op::Clamp,
            generated::Op::EQUAL => Op::Equal,
            generated::Op::GREATER => Op::Greater,
            generated::Op::GREATER_EQUAL => Op::GreaterEqual,
            generated::Op::SELECT => Op::Select,
            generated::Op::REDUCE_ALL => Op::ReduceAll,
            generated::Op::REDUCE_ANY => Op::ReduceAny,
            generated::Op::REDUCE_MAX => Op::ReduceMax,
            generated::Op::REDUCE_MIN => Op::ReduceMin,
            generated::Op::REDUCE_PRODUCT => Op::ReduceProduct,
            generated::Op::REDUCE_SUM => Op::ReduceSum,
            generated::Op::RESHAPE => Op::Reshape,
            generated::Op::TRANSPOSE => Op::Transpose,
            generated::Op::CONCAT => Op::Concat,
            generated::Op::SLICE => Op::Slice,
            generated::Op::PAD => Op::Pad,
            generated::Op::TILE => Op::Tile,
            generated::Op::REVERSE => Op::Reverse,
            generated::Op::GATHER => Op::Gather,
            generated::Op::SCATTER => Op::Scatter,
            generated::Op::CONV2D => Op::Conv2d,
            generated::Op::CONV3D => Op::Conv3d,
            generated::Op::DEPTHWISE_CONV2D => Op::DepthwiseConv2d,
            generated::Op::TRANSPOSE_CONV2D => Op::TransposeConv2d,
            generated::Op::MATMUL => Op::MatMul,
            generated::Op::AVG_POOL2D => Op::AvgPool2d,
            generated::Op::MAX_POOL2D => Op::MaxPool2d,
            generated::Op::ARGMAX => Op::ArgMax,
            generated::Op::CAST => Op::Cast,
            generated::Op::RESCALE => Op::Rescale,
            generated::Op::TABLE => Op::Table,
            generated::Op::COND_IF => Op::CondIf,
            generated::Op::WHILE_LOOP => Op::WhileLoop,
            generated::Op::RESIZE => Op::Resize,
            generated::Op::ADD_SHAPE => Op::AddShape,
            generated::Op::SUB_SHAPE => Op::SubShape,
            generated::Op::MUL_SHAPE => Op::MulShape,
            generated::Op::CONCAT_SHAPE => Op::ConcatShape,
            generated::Op::SLICE_SHAPE => Op::SliceShape,
            generated::Op::EXP2_SHAPE => Op::Exp2Shape,
            generated::Op::LOG2_CEIL_SHAPE => Op::Log2CeilShape,
            generated::Op::LOG2_FLOOR_SHAPE => Op::Log2FloorShape,
            generated::Op::MAX_SHAPE => Op::MaxShape,
            generated::Op::MIN_SHAPE => Op::MinShape,
            generated::Op::MOD_SHAPE => Op::ModShape,
            generated::Op::DIV_CEIL_SHAPE => Op::DivCeilShape,
            generated::Op::DIV_FLOOR_SHAPE => Op::DivFloorShape,
            generated::Op::DIM => Op::Dim,
            generated::Op::ASSERT_EQUAL_SHAPE => Op::AssertEqualShape,
            generated::Op::MATMUL_T_BLOCK_SCALED => Op::MatMulTBlockScaled,
            generated::Op::CAST_FROM_BLOCK_SCALED => Op::CastFromBlockScaled,
            generated::Op::CAST_TO_BLOCK_SCALED => Op::CastToBlockScaled,
            generated::Op::CONV2D_BLOCK_SCALED => Op::Conv2dBlockScaled,
            generated::Op::RESHAPE_BLOCK_SCALED => Op::ReshapeBlockScaled,
            generated::Op::ROW_GATHER_BLOCK_SCALED => Op::RowGatherBlockScaled,
            generated::Op::AVG_POOL2D_ADAPTIVE => Op::AvgPool2dAdaptive,
            generated::Op::MAX_POOL2D_ADAPTIVE => Op::MaxPool2dAdaptive,
            generated::Op::FFT2D => Op::Fft2d,
            generated::Op::RFFT2D => Op::Rfft2d,
            generated::Op::CUSTOM => Op::Custom,
            generated::Op::ROW_GATHER => Op::RowGather,
            generated::Op::MATMUL_T => Op::MatMulT,
            generated::Op::ARGMIN => Op::ArgMin,
            other => Op::Unknown(other.0),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Op::Const => "CONST",
            Op::ConstShape => "CONST_SHAPE",
            Op::Identity => "IDENTITY",
            Op::Variable => "VARIABLE",
            Op::VariableRead => "VARIABLE_READ",
            Op::VariableWrite => "VARIABLE_WRITE",
            Op::Add => "ADD",
            Op::Sub => "SUB",
            Op::Mul => "MUL",
            Op::Maximum => "MAXIMUM",
            Op::Minimum => "MINIMUM",
            Op::Pow => "POW",
            Op::ArithmeticRightShift => "ARITHMETIC_RIGHT_SHIFT",
            Op::Intdiv => "INTDIV",
            Op::LogicalAnd => "LOGICAL_AND",
            Op::LogicalOr => "LOGICAL_OR",
            Op::LogicalXor => "LOGICAL_XOR",
            Op::LogicalLeftShift => "LOGICAL_LEFT_SHIFT",
            Op::LogicalRightShift => "LOGICAL_RIGHT_SHIFT",
            Op::BitwiseAnd => "BITWISE_AND",
            Op::BitwiseOr => "BITWISE_OR",
            Op::BitwiseXor => "BITWISE_XOR",
            Op::Abs => "ABS",
            Op::Negate => "NEGATE",
            Op::Ceil => "CEIL",
            Op::Floor => "FLOOR",
            Op::Exp => "EXP",
            Op::Log => "LOG",
            Op::Sin => "SIN",
            Op::Cos => "COS",
            Op::Reciprocal => "RECIPROCAL",
            Op::Rsqrt => "RSQRT",
            Op::Erf => "ERF",
            Op::Sigmoid => "SIGMOID",
            Op::Tanh => "TANH",
            Op::Clz => "CLZ",
            Op::BitwiseNot => "BITWISE_NOT",
            Op::LogicalNot => "LOGICAL_NOT",
            Op::Clamp => "CLAMP",
            Op::Equal => "EQUAL",
            Op::Greater => "GREATER",
            Op::GreaterEqual => "GREATER_EQUAL",
            Op::Select => "SELECT",
            Op::ReduceAll => "REDUCE_ALL",
            Op::ReduceAny => "REDUCE_ANY",
            Op::ReduceMax => "REDUCE_MAX",
            Op::ReduceMin => "REDUCE_MIN",
            Op::ReduceProduct => "REDUCE_PRODUCT",
            Op::ReduceSum => "REDUCE_SUM",
            Op::Reshape => "RESHAPE",
            Op::Transpose => "TRANSPOSE",
            Op::Concat => "CONCAT",
            Op::Slice => "SLICE",
            Op::Pad => "PAD",
            Op::Tile => "TILE",
            Op::Reverse => "REVERSE",
            Op::Gather => "GATHER",
            Op::Scatter => "SCATTER",
            Op::Conv2d => "CONV2D",
            Op::Conv3d => "CONV3D",
            Op::DepthwiseConv2d => "DEPTHWISE_CONV2D",
            Op::TransposeConv2d => "TRANSPOSE_CONV2D",
            Op::MatMul => "MATMUL",
            Op::AvgPool2d => "AVG_POOL2D",
            Op::MaxPool2d => "MAX_POOL2D",
            Op::ArgMax => "ARGMAX",
            Op::Cast => "CAST",
            Op::Rescale => "RESCALE",
            Op::Table => "TABLE",
            Op::CondIf => "COND_IF",
            Op::WhileLoop => "WHILE_LOOP",
            Op::Resize => "RESIZE",
            Op::AddShape => "ADD_SHAPE",
            Op::SubShape => "SUB_SHAPE",
            Op::MulShape => "MUL_SHAPE",
            Op::ConcatShape => "CONCAT_SHAPE",
            Op::SliceShape => "SLICE_SHAPE",
            Op::Exp2Shape => "EXP2_SHAPE",
            Op::Log2CeilShape => "LOG2_CEIL_SHAPE",
            Op::Log2FloorShape => "LOG2_FLOOR_SHAPE",
            Op::MaxShape => "MAX_SHAPE",
            Op::MinShape => "MIN_SHAPE",
            Op::ModShape => "MOD_SHAPE",
            Op::DivCeilShape => "DIV_CEIL_SHAPE",
            Op::DivFloorShape => "DIV_FLOOR_SHAPE",
            Op::Dim => "DIM",
            Op::AssertEqualShape => "ASSERT_EQUAL_SHAPE",
            Op::MatMulTBlockScaled => "MATMUL_T_BLOCK_SCALED",
            Op::CastFromBlockScaled => "CAST_FROM_BLOCK_SCALED",
            Op::CastToBlockScaled => "CAST_TO_BLOCK_SCALED",
            Op::Conv2dBlockScaled => "CONV2D_BLOCK_SCALED",
            Op::ReshapeBlockScaled => "RESHAPE_BLOCK_SCALED",
            Op::RowGatherBlockScaled => "ROW_GATHER_BLOCK_SCALED",
            Op::AvgPool2dAdaptive => "AVG_POOL2D_ADAPTIVE",
            Op::MaxPool2dAdaptive => "MAX_POOL2D_ADAPTIVE",
            Op::Fft2d => "FFT2D",
            Op::Rfft2d => "RFFT2D",
            Op::Custom => "CUSTOM",
            Op::RowGather => "ROW_GATHER",
            Op::MatMulT => "MATMUL_T",
            Op::ArgMin => "ARGMIN",
            Op::Unknown(_) => "UNKNOWN",
        }
    }
}

#[derive(Debug, Clone)]
pub enum OpAttribute {
    None,
    CondIf {
        then_graph: String,
        else_graph: String,
    },
    WhileLoop {
        cond_graph: String,
        body_graph: String,
    },
    Conv2d {
        pad: Vec<i32>,
        stride: Vec<i32>,
        dilation: Vec<i32>,
        acc_type: DType,
    },
    Conv3d {
        pad: Vec<i32>,
        stride: Vec<i32>,
        dilation: Vec<i32>,
        acc_type: DType,
    },
    DepthwiseConv2d {
        pad: Vec<i32>,
        stride: Vec<i32>,
        dilation: Vec<i32>,
        acc_type: DType,
    },
    TransposeConv2d {
        out_pad: Vec<i32>,
        stride: Vec<i32>,
        acc_type: DType,
    },
    Pool2d {
        kernel: Vec<i32>,
        stride: Vec<i32>,
        pad: Vec<i32>,
        /// AVG_POOL2D only (`Unknown` for MAX_POOL2D).
        acc_type: DType,
        /// MAX_POOL2D only (`Propagate` for AVG_POOL2D).
        nan_mode: NanMode,
    },
    ArgMax {
        axis: i32,
        nan_mode: NanMode,
    },
    Reduce {
        axis: i32,
        /// REDUCE_MAX / REDUCE_MIN only (`Propagate` otherwise).
        nan_mode: NanMode,
    },
    Dim {
        axis: i32,
    },
    Fft2d {
        inverse: bool,
    },
    AssertEqualShape {
        allow_broadcast: bool,
    },
    /// MAXIMUM / MINIMUM.
    NanPropagation {
        nan_mode: NanMode,
    },
    Concat {
        axis: i32,
    },
    Transpose {
        perms: Vec<i32>,
    },
    Clamp {
        min_val: Vec<u8>,
        max_val: Vec<u8>,
        nan_mode: NanMode,
    },
    ArithmeticRightShift {
        round: bool,
    },
    Rescale {
        scale32: bool,
        rounding_mode: RoundingMode,
        per_channel: bool,
        input_unsigned: bool,
        output_unsigned: bool,
    },
    Reverse {
        axis: i32,
    },
    Resize {
        mode: u32,
    },
}

#[derive(Debug, Clone)]
pub struct TensorDef {
    pub name: String,
    pub shape: Vec<i32>,
    pub dtype: DType,
    pub data: Option<Vec<u8>>,
    pub is_variable: bool,
}

#[derive(Debug, Clone)]
pub struct Operator {
    pub op: Op,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attribute: OpAttribute,
}

#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub name: String,
    pub operators: Vec<Operator>,
    pub tensors: Vec<TensorDef>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Region {
    pub name: String,
    pub blocks: Vec<BasicBlock>,
}

#[derive(Debug, Clone)]
pub struct Graph {
    pub regions: Vec<Region>,
}

impl Graph {
    pub fn from_flatbuffer(data: &[u8]) -> Result<Self> {
        let fb_graph =
            flatbuffers::root::<generated::TosaGraph>(data).map_err(|e| Error::Deserialize(format!("{e}")))?;

        let mut regions = Vec::new();

        if let Some(fb_regions) = fb_graph.regions() {
            for fb_region in fb_regions.iter() {
                let region_name = fb_region.name().unwrap_or("").to_string();
                let mut blocks = Vec::new();

                if let Some(fb_blocks) = fb_region.blocks() {
                    for fb_block in fb_blocks.iter() {
                        blocks.push(Self::parse_block(&fb_block)?);
                    }
                }

                regions.push(Region { name: region_name, blocks });
            }
        }

        Ok(Graph { regions })
    }

    fn parse_block(fb_block: &generated::TosaBasicBlock<'_>) -> Result<BasicBlock> {
        let name = fb_block.name().unwrap_or("").to_string();

        let mut tensors = Vec::new();
        if let Some(fb_tensors) = fb_block.tensors() {
            for fb_tensor in fb_tensors.iter() {
                let tensor_name = fb_tensor.name().unwrap_or("").to_string();
                let shape: Vec<i32> = fb_tensor.shape().map(|s| s.iter().collect()).unwrap_or_default();
                let dtype = DType::from_fb(fb_tensor.type_());
                let data = fb_tensor.data().map(|d| d.bytes().to_vec());
                let is_variable = fb_tensor.variable();

                tensors.push(TensorDef { name: tensor_name, shape, dtype, data, is_variable });
            }
        }

        // Also parse shapes and add them as tensor defs (SHAPE dtype)
        if let Some(fb_shapes) = fb_block.shapes() {
            for fb_shape in fb_shapes.iter() {
                let shape_name = fb_shape.name().unwrap_or("").to_string();
                let rank = fb_shape.rank() as i32;
                let data = fb_shape.data().map(|d| d.bytes().to_vec());

                // A SHAPE value of rank r is a list of r dimensions.
                tensors.push(TensorDef {
                    name: shape_name,
                    shape: vec![rank],
                    dtype: DType::Shape,
                    data,
                    is_variable: false,
                });
            }
        }

        let mut operators = Vec::new();
        if let Some(fb_ops) = fb_block.operators() {
            for fb_op in fb_ops.iter() {
                let op = Op::from_fb(fb_op.op());
                let inputs: Vec<String> =
                    fb_op.inputs().map(|v| v.iter().map(|s| s.to_string()).collect()).unwrap_or_default();
                let outputs: Vec<String> =
                    fb_op.outputs().map(|v| v.iter().map(|s| s.to_string()).collect()).unwrap_or_default();

                let attribute = Self::parse_attribute(&fb_op);

                operators.push(Operator { op, inputs, outputs, attribute });
            }
        }

        let block_inputs: Vec<String> =
            fb_block.inputs().map(|v| v.iter().map(|s| s.to_string()).collect()).unwrap_or_default();
        let block_outputs: Vec<String> =
            fb_block.outputs().map(|v| v.iter().map(|s| s.to_string()).collect()).unwrap_or_default();

        Ok(BasicBlock { name, operators, tensors, inputs: block_inputs, outputs: block_outputs })
    }

    fn parse_attribute(fb_op: &generated::TosaOperator<'_>) -> OpAttribute {
        match fb_op.attribute_type() {
            generated::Attribute::CondIfAttribute => {
                if let Some(attr) = fb_op.attribute_as_cond_if_attribute() {
                    return OpAttribute::CondIf {
                        then_graph: attr.then_graph().unwrap_or("").to_string(),
                        else_graph: attr.else_graph().unwrap_or("").to_string(),
                    };
                }
            }
            generated::Attribute::WhileLoopAttribute => {
                if let Some(attr) = fb_op.attribute_as_while_loop_attribute() {
                    return OpAttribute::WhileLoop {
                        cond_graph: attr.cond_graph().unwrap_or("").to_string(),
                        body_graph: attr.body_graph().unwrap_or("").to_string(),
                    };
                }
            }
            generated::Attribute::Conv2dAttribute => {
                if let Some(attr) = fb_op.attribute_as_conv_2d_attribute() {
                    return OpAttribute::Conv2d {
                        pad: attr.pad().map(|v| v.iter().collect()).unwrap_or_default(),
                        stride: attr.stride().map(|v| v.iter().collect()).unwrap_or_default(),
                        dilation: attr.dilation().map(|v| v.iter().collect()).unwrap_or_default(),
                        acc_type: DType::from_fb(attr.acc_type()),
                    };
                }
            }
            generated::Attribute::Conv3dAttribute => {
                if let Some(attr) = fb_op.attribute_as_conv_3d_attribute() {
                    return OpAttribute::Conv3d {
                        pad: attr.pad().map(|v| v.iter().collect()).unwrap_or_default(),
                        stride: attr.stride().map(|v| v.iter().collect()).unwrap_or_default(),
                        dilation: attr.dilation().map(|v| v.iter().collect()).unwrap_or_default(),
                        acc_type: DType::from_fb(attr.acc_type()),
                    };
                }
            }
            generated::Attribute::DepthwiseConv2dAttribute => {
                if let Some(attr) = fb_op.attribute_as_depthwise_conv_2d_attribute() {
                    return OpAttribute::DepthwiseConv2d {
                        pad: attr.pad().map(|v| v.iter().collect()).unwrap_or_default(),
                        stride: attr.stride().map(|v| v.iter().collect()).unwrap_or_default(),
                        dilation: attr.dilation().map(|v| v.iter().collect()).unwrap_or_default(),
                        acc_type: DType::from_fb(attr.acc_type()),
                    };
                }
            }
            generated::Attribute::TransposeConv2dAttribute => {
                if let Some(attr) = fb_op.attribute_as_transpose_conv_2d_attribute() {
                    return OpAttribute::TransposeConv2d {
                        out_pad: attr.out_pad().map(|v| v.iter().collect()).unwrap_or_default(),
                        stride: attr.stride().map(|v| v.iter().collect()).unwrap_or_default(),
                        acc_type: DType::from_fb(attr.acc_type()),
                    };
                }
            }
            generated::Attribute::AvgPool2dAttribute => {
                if let Some(attr) = fb_op.attribute_as_avg_pool_2d_attribute() {
                    return OpAttribute::Pool2d {
                        kernel: attr.kernel().map(|v| v.iter().collect()).unwrap_or_default(),
                        stride: attr.stride().map(|v| v.iter().collect()).unwrap_or_default(),
                        pad: attr.pad().map(|v| v.iter().collect()).unwrap_or_default(),
                        acc_type: DType::from_fb(attr.acc_type()),
                        nan_mode: NanMode::Propagate,
                    };
                }
            }
            generated::Attribute::MaxPool2dAttribute => {
                if let Some(attr) = fb_op.attribute_as_max_pool_2d_attribute() {
                    return OpAttribute::Pool2d {
                        kernel: attr.kernel().map(|v| v.iter().collect()).unwrap_or_default(),
                        stride: attr.stride().map(|v| v.iter().collect()).unwrap_or_default(),
                        pad: attr.pad().map(|v| v.iter().collect()).unwrap_or_default(),
                        acc_type: DType::Unknown,
                        nan_mode: NanMode::from_fb(attr.nan_mode()),
                    };
                }
            }
            generated::Attribute::ArgMaxAttribute => {
                if let Some(attr) = fb_op.attribute_as_arg_max_attribute() {
                    return OpAttribute::ArgMax { axis: attr.axis(), nan_mode: NanMode::from_fb(attr.nan_mode()) };
                }
            }
            generated::Attribute::ReduceAllAttribute => {
                if let Some(attr) = fb_op.attribute_as_reduce_all_attribute() {
                    return OpAttribute::Reduce { axis: attr.axis(), nan_mode: NanMode::Propagate };
                }
            }
            generated::Attribute::ReduceAnyAttribute => {
                if let Some(attr) = fb_op.attribute_as_reduce_any_attribute() {
                    return OpAttribute::Reduce { axis: attr.axis(), nan_mode: NanMode::Propagate };
                }
            }
            generated::Attribute::ReduceMaxAttribute => {
                if let Some(attr) = fb_op.attribute_as_reduce_max_attribute() {
                    return OpAttribute::Reduce { axis: attr.axis(), nan_mode: NanMode::from_fb(attr.nan_mode()) };
                }
            }
            generated::Attribute::ReduceMinAttribute => {
                if let Some(attr) = fb_op.attribute_as_reduce_min_attribute() {
                    return OpAttribute::Reduce { axis: attr.axis(), nan_mode: NanMode::from_fb(attr.nan_mode()) };
                }
            }
            generated::Attribute::ReduceProductAttribute => {
                if let Some(attr) = fb_op.attribute_as_reduce_product_attribute() {
                    return OpAttribute::Reduce { axis: attr.axis(), nan_mode: NanMode::Propagate };
                }
            }
            generated::Attribute::ReduceSumAttribute => {
                if let Some(attr) = fb_op.attribute_as_reduce_sum_attribute() {
                    return OpAttribute::Reduce { axis: attr.axis(), nan_mode: NanMode::Propagate };
                }
            }
            generated::Attribute::FFT2dAttribute => {
                if let Some(attr) = fb_op.attribute_as_fft2d_attribute() {
                    return OpAttribute::Fft2d { inverse: attr.inverse() };
                }
            }
            // Adaptive pools take kernel, stride and pad as SHAPE operands.
            generated::Attribute::AvgPool2dAdaptiveAttribute => {
                if let Some(attr) = fb_op.attribute_as_avg_pool_2d_adaptive_attribute() {
                    return OpAttribute::Pool2d {
                        kernel: vec![],
                        stride: vec![],
                        pad: vec![],
                        acc_type: DType::from_fb(attr.acc_type()),
                        nan_mode: NanMode::Propagate,
                    };
                }
            }
            generated::Attribute::MaxPool2dAdaptiveAttribute => {
                if let Some(attr) = fb_op.attribute_as_max_pool_2d_adaptive_attribute() {
                    return OpAttribute::Pool2d {
                        kernel: vec![],
                        stride: vec![],
                        pad: vec![],
                        acc_type: DType::Unknown,
                        nan_mode: NanMode::from_fb(attr.nan_mode()),
                    };
                }
            }
            generated::Attribute::DimAttribute => {
                if let Some(attr) = fb_op.attribute_as_dim_attribute() {
                    return OpAttribute::Dim { axis: attr.axis() };
                }
            }
            generated::Attribute::AssertEqualShapeAttribute => {
                if let Some(attr) = fb_op.attribute_as_assert_equal_shape_attribute() {
                    return OpAttribute::AssertEqualShape { allow_broadcast: attr.allow_broadcast() };
                }
            }
            generated::Attribute::MaximumAttribute => {
                if let Some(attr) = fb_op.attribute_as_maximum_attribute() {
                    return OpAttribute::NanPropagation { nan_mode: NanMode::from_fb(attr.nan_mode()) };
                }
            }
            generated::Attribute::MinimumAttribute => {
                if let Some(attr) = fb_op.attribute_as_minimum_attribute() {
                    return OpAttribute::NanPropagation { nan_mode: NanMode::from_fb(attr.nan_mode()) };
                }
            }
            generated::Attribute::ConcatAttribute => {
                if let Some(attr) = fb_op.attribute_as_concat_attribute() {
                    return OpAttribute::Concat { axis: attr.axis() };
                }
            }
            generated::Attribute::TransposeAttribute => {
                if let Some(attr) = fb_op.attribute_as_transpose_attribute() {
                    return OpAttribute::Transpose {
                        perms: attr.perms().map(|v| v.iter().collect()).unwrap_or_default(),
                    };
                }
            }
            generated::Attribute::ClampAttribute => {
                if let Some(attr) = fb_op.attribute_as_clamp_attribute() {
                    return OpAttribute::Clamp {
                        min_val: attr.min_val().map(|v| v.bytes().to_vec()).unwrap_or_default(),
                        max_val: attr.max_val().map(|v| v.bytes().to_vec()).unwrap_or_default(),
                        nan_mode: NanMode::from_fb(attr.nan_mode()),
                    };
                }
            }
            generated::Attribute::ArithmeticRightShiftAttribute => {
                if let Some(attr) = fb_op.attribute_as_arithmetic_right_shift_attribute() {
                    return OpAttribute::ArithmeticRightShift { round: attr.round() };
                }
            }
            generated::Attribute::RescaleAttribute => {
                if let Some(attr) = fb_op.attribute_as_rescale_attribute() {
                    return OpAttribute::Rescale {
                        scale32: attr.scale32(),
                        rounding_mode: RoundingMode::from_fb(attr.rounding_mode()),
                        per_channel: attr.per_channel(),
                        input_unsigned: attr.input_unsigned(),
                        output_unsigned: attr.output_unsigned(),
                    };
                }
            }
            generated::Attribute::ReverseAttribute => {
                if let Some(attr) = fb_op.attribute_as_reverse_attribute() {
                    return OpAttribute::Reverse { axis: attr.axis() };
                }
            }
            generated::Attribute::ResizeAttribute => {
                if let Some(attr) = fb_op.attribute_as_resize_attribute() {
                    return OpAttribute::Resize { mode: attr.mode().0 };
                }
            }
            _ => {}
        }
        OpAttribute::None
    }

    pub fn main_block(&self) -> Option<&BasicBlock> {
        self.regions.first()?.blocks.first()
    }

    pub fn find_block(&self, name: &str) -> Option<&BasicBlock> {
        // First: exact block name match
        for region in &self.regions {
            for block in &region.blocks {
                if block.name == name {
                    return Some(block);
                }
            }
        }
        // Second: region name match (TOSA 1.1 control flow uses region names)
        for region in &self.regions {
            if region.name == name {
                return region.blocks.first();
            }
        }
        None
    }
}
