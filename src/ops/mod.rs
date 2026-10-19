pub mod accum;
pub mod common;
pub mod control_flow;
pub mod conv;
pub mod data_layout;
pub mod data_nodes;
pub mod elementwise;
pub mod fft;
pub mod math;
pub mod matmul;
pub mod pool;
pub mod quant;
pub mod reduction;
pub mod resize;
pub mod shape_ops;
pub mod type_conv;

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
        Op::Add => elementwise::eval_add(state, op),
        Op::Sub => elementwise::eval_sub(state, op),
        Op::Mul => elementwise::eval_mul(state, op),
        Op::Maximum => elementwise::eval_maximum(state, op),
        Op::Minimum => elementwise::eval_minimum(state, op),
        Op::Pow => elementwise::eval_pow(state, op),
        Op::ArithmeticRightShift => elementwise::eval_arithmetic_right_shift(state, op),
        Op::Intdiv => elementwise::eval_intdiv(state, op),
        Op::LogicalAnd => elementwise::eval_logical_and(state, op),
        Op::LogicalOr => elementwise::eval_logical_or(state, op),
        Op::LogicalXor => elementwise::eval_logical_xor(state, op),
        Op::LogicalLeftShift => elementwise::eval_logical_left_shift(state, op),
        Op::LogicalRightShift => elementwise::eval_logical_right_shift(state, op),
        Op::BitwiseAnd => elementwise::eval_bitwise_and(state, op),
        Op::BitwiseOr => elementwise::eval_bitwise_or(state, op),
        Op::BitwiseXor => elementwise::eval_bitwise_xor(state, op),

        // Unary element-wise
        Op::Abs => elementwise::eval_abs(state, op),
        Op::Negate => elementwise::eval_negate(state, op),
        Op::Ceil => elementwise::eval_ceil(state, op),
        Op::Floor => elementwise::eval_floor(state, op),
        Op::Exp => elementwise::eval_exp(state, op),
        Op::Log => elementwise::eval_log(state, op),
        Op::Sin => elementwise::eval_sin(state, op),
        Op::Cos => elementwise::eval_cos(state, op),
        Op::Reciprocal => elementwise::eval_reciprocal(state, op),
        Op::Rsqrt => elementwise::eval_rsqrt(state, op),
        Op::Erf => elementwise::eval_erf(state, op),
        Op::Sigmoid => elementwise::eval_sigmoid(state, op),
        Op::Tanh => elementwise::eval_tanh(state, op),
        Op::Clz => elementwise::eval_clz(state, op),
        Op::BitwiseNot => elementwise::eval_bitwise_not(state, op),
        Op::LogicalNot => elementwise::eval_logical_not(state, op),
        Op::Clamp => elementwise::eval_clamp(state, op),

        // Comparison
        Op::Equal => elementwise::eval_equal(state, op),
        Op::Greater => elementwise::eval_greater(state, op),
        Op::GreaterEqual => elementwise::eval_greater_equal(state, op),
        Op::Select => elementwise::eval_select(state, op),

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
        Op::Conv2d => conv::eval_conv2d(state, op),
        Op::MatMul => matmul::eval_matmul(state, op),
        Op::AvgPool2d => pool::eval_avg_pool2d(state, op),
        Op::MaxPool2d => pool::eval_max_pool2d(state, op),
        Op::ArgMax => pool::eval_argmax(state, op),
        Op::AvgPool2dAdaptive => pool::eval_avg_pool2d_adaptive(state, op),
        Op::MaxPool2dAdaptive => pool::eval_max_pool2d_adaptive(state, op),
        Op::MatMulT => matmul::eval_matmul_t(state, op),
        Op::RowGather => data_layout::eval_row_gather(state, op),
        Op::Fft2d => fft::eval_fft2d(state, op),
        Op::Rfft2d => fft::eval_rfft2d(state, op),

        // Type conversion
        Op::Cast => type_conv::eval_cast(state, op),
        Op::Rescale => type_conv::eval_rescale(state, op),

        // Control flow
        Op::CondIf => control_flow::eval_cond_if(state, op),
        Op::WhileLoop => control_flow::eval_while_loop(state, op),

        Op::DepthwiseConv2d => conv::eval_depthwise_conv2d(state, op),
        Op::TransposeConv2d => conv::eval_transpose_conv2d(state, op),
        Op::Conv3d => conv::eval_conv3d(state, op),
        Op::Table => elementwise::eval_table(state, op),
        Op::Resize => resize::eval_resize(state, op),

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

        // MX block-scaled operators (tosa-ext-mxfp), CUSTOM, and ARGMIN (not yet in the reference model)
        Op::MatMulTBlockScaled
        | Op::CastFromBlockScaled
        | Op::CastToBlockScaled
        | Op::Conv2dBlockScaled
        | Op::ReshapeBlockScaled
        | Op::RowGatherBlockScaled
        | Op::Custom
        | Op::ArgMin => Err(Error::UnsupportedOp(format!("{}", op.op.name()))),

        Op::Unknown(v) => Err(Error::UnsupportedOp(format!("op code {v}"))),
    }
}
