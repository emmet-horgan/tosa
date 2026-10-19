pub mod accum;
pub mod binary;
pub mod common;
pub mod comparison;
pub mod control_flow;
pub mod conv2d;
pub mod data_layout;
pub mod data_nodes;
pub mod reduction;
pub mod shape_ops;
pub mod tensor_ops;
pub mod type_conv;
pub mod unary;

use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{Op, Operator};

pub fn dispatch(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    match op.op {
        // Data nodes
        Op::Const => data_nodes::eval_const(state, op),
        Op::ConstShape => data_nodes::eval_const_shape(state, op),
        Op::Identity => data_nodes::eval_identity(state, op),
        Op::Variable => data_nodes::eval_variable(state, op),
        Op::VariableRead => data_nodes::eval_variable_read(state, op),
        Op::VariableWrite => data_nodes::eval_variable_write(state, op),

        // Binary element-wise
        Op::Add => binary::eval_add(state, op),
        Op::Sub => binary::eval_sub(state, op),
        Op::Mul => binary::eval_mul(state, op),
        Op::Maximum => binary::eval_maximum(state, op),
        Op::Minimum => binary::eval_minimum(state, op),
        Op::Pow => binary::eval_pow(state, op),
        Op::ArithmeticRightShift => binary::eval_arithmetic_right_shift(state, op),
        Op::Intdiv => binary::eval_intdiv(state, op),
        Op::LogicalAnd => binary::eval_logical_and(state, op),
        Op::LogicalOr => binary::eval_logical_or(state, op),
        Op::LogicalXor => binary::eval_logical_xor(state, op),
        Op::LogicalLeftShift => binary::eval_logical_left_shift(state, op),
        Op::LogicalRightShift => binary::eval_logical_right_shift(state, op),
        Op::BitwiseAnd => binary::eval_bitwise_and(state, op),
        Op::BitwiseOr => binary::eval_bitwise_or(state, op),
        Op::BitwiseXor => binary::eval_bitwise_xor(state, op),

        // Unary element-wise
        Op::Abs => unary::eval_abs(state, op),
        Op::Negate => unary::eval_negate(state, op),
        Op::Ceil => unary::eval_ceil(state, op),
        Op::Floor => unary::eval_floor(state, op),
        Op::Exp => unary::eval_exp(state, op),
        Op::Log => unary::eval_log(state, op),
        Op::Sin => unary::eval_sin(state, op),
        Op::Cos => unary::eval_cos(state, op),
        Op::Reciprocal => unary::eval_reciprocal(state, op),
        Op::Rsqrt => unary::eval_rsqrt(state, op),
        Op::Erf => unary::eval_erf(state, op),
        Op::Sigmoid => unary::eval_sigmoid(state, op),
        Op::Tanh => unary::eval_tanh(state, op),
        Op::Clz => unary::eval_clz(state, op),
        Op::BitwiseNot => unary::eval_bitwise_not(state, op),
        Op::LogicalNot => unary::eval_logical_not(state, op),
        Op::Clamp => unary::eval_clamp(state, op),

        // Comparison
        Op::Equal => comparison::eval_equal(state, op),
        Op::Greater => comparison::eval_greater(state, op),
        Op::GreaterEqual => comparison::eval_greater_equal(state, op),
        Op::Select => comparison::eval_select(state, op),

        // Reduction
        Op::ReduceAll => reduction::eval_reduce_all(state, op),
        Op::ReduceAny => reduction::eval_reduce_any(state, op),
        Op::ReduceMax => reduction::eval_reduce_max(state, op),
        Op::ReduceMin => reduction::eval_reduce_min(state, op),
        Op::ReduceProduct => reduction::eval_reduce_product(state, op),
        Op::ReduceSum => reduction::eval_reduce_sum(state, op),

        // Data layout
        Op::Reshape => data_layout::eval_reshape(state, op),
        Op::Transpose => data_layout::eval_transpose(state, op),
        Op::Concat => data_layout::eval_concat(state, op),
        Op::Slice => data_layout::eval_slice(state, op),
        Op::Pad => data_layout::eval_pad(state, op),
        Op::Tile => data_layout::eval_tile(state, op),
        Op::Reverse => data_layout::eval_reverse(state, op),
        Op::Gather => data_layout::eval_gather(state, op),
        Op::Scatter => data_layout::eval_scatter(state, op),

        // Tensor compute
        Op::Conv2d => conv2d::eval_conv2d(state, op),
        Op::MatMul => tensor_ops::eval_matmul(state, op),
        Op::AvgPool2d => tensor_ops::eval_avg_pool2d(state, op),
        Op::MaxPool2d => tensor_ops::eval_max_pool2d(state, op),
        Op::ArgMax => tensor_ops::eval_argmax(state, op),

        // Type conversion
        Op::Cast => type_conv::eval_cast(state, op),
        Op::Rescale => type_conv::eval_rescale(state, op),

        // Control flow
        Op::CondIf => control_flow::eval_cond_if(state, op),
        Op::WhileLoop => control_flow::eval_while_loop(state, op),

        Op::DepthwiseConv2d => tensor_ops::eval_depthwise_conv2d(state, op),
        Op::TransposeConv2d => tensor_ops::eval_transpose_conv2d(state, op),
        Op::Conv3d => tensor_ops::eval_conv3d(state, op),
        Op::Table => tensor_ops::eval_table(state, op),
        Op::Resize => tensor_ops::eval_resize(state, op),

        // Shape ops
        Op::AddShape => shape_ops::eval_add_shape(state, op),
        Op::SubShape => shape_ops::eval_sub_shape(state, op),
        Op::MulShape => shape_ops::eval_mul_shape(state, op),
        Op::ConcatShape => shape_ops::eval_concat_shape(state, op),
        Op::SliceShape => shape_ops::eval_slice_shape(state, op),
        Op::Exp2Shape => shape_ops::eval_exp2_shape(state, op),
        Op::Log2CeilShape => shape_ops::eval_log2_ceil_shape(state, op),
        Op::Log2FloorShape => shape_ops::eval_log2_floor_shape(state, op),
        Op::MaxShape => shape_ops::eval_max_shape(state, op),
        Op::MinShape => shape_ops::eval_min_shape(state, op),
        Op::ModShape => shape_ops::eval_mod_shape(state, op),
        Op::DivCeilShape => shape_ops::eval_div_ceil_shape(state, op),
        Op::DivFloorShape => shape_ops::eval_div_floor_shape(state, op),
        Op::Dim => shape_ops::eval_dim(state, op),
        Op::AssertEqualShape => shape_ops::eval_assert_equal_shape(state, op),

        // Adaptive pooling
        Op::AvgPool2dAdaptive => tensor_ops::eval_avg_pool2d(state, op),
        Op::MaxPool2dAdaptive => tensor_ops::eval_max_pool2d(state, op),

        // Block-scaled, FFT, Custom — not yet fully implemented
        Op::MatMulTBlockScaled
        | Op::CastFromBlockScaled
        | Op::CastToBlockScaled
        | Op::Conv2dBlockScaled
        | Op::ReshapeBlockScaled
        | Op::RowGatherBlockScaled
        | Op::Fft2d
        | Op::Rfft2d
        | Op::Custom
        | Op::RowGather
        | Op::MatMulT
        | Op::ArgMin => Err(Error::UnsupportedOp(format!("{}", op.op.name()))),

        Op::Unknown(v) => Err(Error::UnsupportedOp(format!("op code {v}"))),
    }
}
