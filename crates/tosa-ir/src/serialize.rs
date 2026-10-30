//! Writing a [`Graph`] in the TOSA flatbuffer format, the inverse of
//! [`Graph::from_flatbuffer`].
//!
//! Every operator is written with the attribute table the schema defines for
//! it, even when the IR holds [`OpAttribute::None`] (the table is then
//! empty), as the reference model expects the attribute type to match the
//! operator. SHAPE tensors are written to the block's `shapes` list.

use flatbuffers::{FlatBufferBuilder, UnionWIPOffset, WIPOffset};

use crate::error::{Error, Result};
use crate::generated as fb;
use crate::ir::{BasicBlock, DType, Graph, Op, OpAttribute, Operator, Region, TensorDef};

type Attr = (fb::Attribute, Option<WIPOffset<UnionWIPOffset>>);

impl Graph {
    /// Serialize the graph as a TOSA flatbuffer.
    pub fn to_flatbuffer(&self) -> Result<Vec<u8>> {
        let mut b = FlatBufferBuilder::with_capacity(4096);
        let regions = self.regions.iter().map(|r| write_region(&mut b, r)).collect::<Result<Vec<_>>>()?;
        let regions = b.create_vector(&regions);
        let version = fb::Version::create(
            &mut b,
            &fb::VersionArgs {
                _major: self.version.major,
                _minor: self.version.minor,
                _patch: self.version.patch,
                _draft: self.version.draft,
            },
        );
        let graph = fb::TosaGraph::create(
            &mut b,
            &fb::TosaGraphArgs {
                version: Some(version),
                regions: Some(regions),
                software_version: None,
                compatibility: None,
            },
        );
        fb::finish_tosa_graph_buffer(&mut b, graph);
        Ok(b.finished_data().to_vec())
    }
}

fn write_region<'a>(b: &mut FlatBufferBuilder<'a>, region: &Region) -> Result<WIPOffset<fb::TosaRegion<'a>>> {
    let blocks = region.blocks.iter().map(|blk| write_block(b, blk)).collect::<Result<Vec<_>>>()?;
    let blocks = b.create_vector(&blocks);
    let name = b.create_string(&region.name);
    Ok(fb::TosaRegion::create(b, &fb::TosaRegionArgs { name: Some(name), blocks: Some(blocks) }))
}

fn write_block<'a>(b: &mut FlatBufferBuilder<'a>, block: &BasicBlock) -> Result<WIPOffset<fb::TosaBasicBlock<'a>>> {
    let operators = block.operators.iter().map(|op| write_operator(b, op)).collect::<Result<Vec<_>>>()?;
    let operators = b.create_vector(&operators);
    let tensors: Vec<_> =
        block.tensors.iter().filter(|t| t.dtype != DType::Shape).map(|t| write_tensor(b, t)).collect();
    let tensors = b.create_vector(&tensors);
    let shapes: Vec<_> = block.tensors.iter().filter(|t| t.dtype == DType::Shape).map(|t| write_shape(b, t)).collect();
    let shapes = b.create_vector(&shapes);
    let inputs = strings(b, &block.inputs);
    let outputs = strings(b, &block.outputs);
    let name = b.create_string(&block.name);
    Ok(fb::TosaBasicBlock::create(
        b,
        &fb::TosaBasicBlockArgs {
            name: Some(name),
            operators: Some(operators),
            tensors: Some(tensors),
            inputs: Some(inputs),
            outputs: Some(outputs),
            shapes: Some(shapes),
        },
    ))
}

fn strings<'a>(
    b: &mut FlatBufferBuilder<'a>,
    names: &[String],
) -> WIPOffset<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<&'a str>>> {
    let offsets: Vec<WIPOffset<&str>> = names.iter().map(|n| b.create_string(n)).collect();
    b.create_vector(&offsets)
}

fn write_tensor<'a>(b: &mut FlatBufferBuilder<'a>, t: &TensorDef) -> WIPOffset<fb::TosaTensor<'a>> {
    let name = b.create_string(&t.name);
    let shape = b.create_vector(&t.shape);
    let data = t.data.as_deref().map(|d| b.create_vector(d));
    let variable_name = t.variable_name.as_deref().map(|n| b.create_string(n));
    fb::TosaTensor::create(
        b,
        &fb::TosaTensorArgs {
            name: Some(name),
            shape: Some(shape),
            type_: t.dtype.to_fb(),
            data,
            variable: t.is_variable,
            is_unranked: t.is_unranked,
            variable_name,
            ..Default::default()
        },
    )
}

fn write_shape<'a>(b: &mut FlatBufferBuilder<'a>, t: &TensorDef) -> WIPOffset<fb::TosaShape<'a>> {
    let name = b.create_string(&t.name);
    let data = t.data.as_deref().map(|d| b.create_vector(d));
    let rank = t.shape.first().copied().unwrap_or(0).max(0) as u32;
    fb::TosaShape::create(b, &fb::TosaShapeArgs { name: Some(name), rank, data })
}

fn write_operator<'a>(b: &mut FlatBufferBuilder<'a>, op: &Operator) -> Result<WIPOffset<fb::TosaOperator<'a>>> {
    let (attribute_type, attribute) = write_attribute(b, op.op, &op.attribute)?;
    let inputs = strings(b, &op.inputs);
    let outputs = strings(b, &op.outputs);
    let location = op.location.as_deref().map(|text| {
        let text = b.create_string(text);
        fb::OpLocation::create(b, &fb::OpLocationArgs { text: Some(text) })
    });
    Ok(fb::TosaOperator::create(
        b,
        &fb::TosaOperatorArgs {
            op: op.op.to_fb(),
            attribute_type,
            attribute,
            inputs: Some(inputs),
            outputs: Some(outputs),
            location,
        },
    ))
}

/// An attribute table with default (empty) contents.
macro_rules! empty {
    ($b:expr, $table:ident) => {
        (fb::Attribute::$table, Some(fb::$table::create($b, &Default::default()).as_union_value()))
    };
}

/// An attribute table built from `$args`.
macro_rules! table {
    ($b:expr, $table:ident, $args:expr) => {
        (fb::Attribute::$table, Some(fb::$table::create($b, &$args).as_union_value()))
    };
}

fn mismatch(op: Op, attr: &OpAttribute) -> Error {
    Error::GraphError(format!("attribute {attr:?} does not belong to operator {}", op.name()))
}

fn write_attribute(b: &mut FlatBufferBuilder<'_>, op: Op, attr: &OpAttribute) -> Result<Attr> {
    Ok(match attr {
        OpAttribute::None => empty_attribute(b, op),
        OpAttribute::CondIf { then_graph, else_graph } => {
            let then_graph = b.create_string(then_graph);
            let else_graph = b.create_string(else_graph);
            table!(
                b,
                CondIfAttribute,
                fb::CondIfAttributeArgs { then_graph: Some(then_graph), else_graph: Some(else_graph) }
            )
        }
        OpAttribute::WhileLoop { cond_graph, body_graph } => {
            let cond_graph = b.create_string(cond_graph);
            let body_graph = b.create_string(body_graph);
            table!(
                b,
                WhileLoopAttribute,
                fb::WhileLoopAttributeArgs { cond_graph: Some(cond_graph), body_graph: Some(body_graph) }
            )
        }
        OpAttribute::Conv2d { pad, stride, dilation, local_bound, acc_type } => {
            let (pad, stride, dilation) = (b.create_vector(pad), b.create_vector(stride), b.create_vector(dilation));
            table!(
                b,
                Conv2dAttribute,
                fb::Conv2dAttributeArgs {
                    pad: Some(pad),
                    stride: Some(stride),
                    dilation: Some(dilation),
                    local_bound: *local_bound,
                    acc_type: acc_type.to_fb()
                }
            )
        }
        OpAttribute::Conv3d { pad, stride, dilation, local_bound, acc_type } => {
            let (pad, stride, dilation) = (b.create_vector(pad), b.create_vector(stride), b.create_vector(dilation));
            table!(
                b,
                Conv3dAttribute,
                fb::Conv3dAttributeArgs {
                    pad: Some(pad),
                    stride: Some(stride),
                    dilation: Some(dilation),
                    local_bound: *local_bound,
                    acc_type: acc_type.to_fb()
                }
            )
        }
        OpAttribute::DepthwiseConv2d { pad, stride, dilation, local_bound, acc_type } => {
            let (pad, stride, dilation) = (b.create_vector(pad), b.create_vector(stride), b.create_vector(dilation));
            table!(
                b,
                DepthwiseConv2dAttribute,
                fb::DepthwiseConv2dAttributeArgs {
                    pad: Some(pad),
                    stride: Some(stride),
                    dilation: Some(dilation),
                    local_bound: *local_bound,
                    acc_type: acc_type.to_fb()
                }
            )
        }
        OpAttribute::TransposeConv2d { out_pad, stride, local_bound, acc_type } => {
            let (out_pad, stride) = (b.create_vector(out_pad), b.create_vector(stride));
            table!(
                b,
                TransposeConv2dAttribute,
                fb::TransposeConv2dAttributeArgs {
                    out_pad: Some(out_pad),
                    stride: Some(stride),
                    local_bound: *local_bound,
                    acc_type: acc_type.to_fb()
                }
            )
        }
        OpAttribute::AvgPool2d { kernel, stride, pad, acc_type } => {
            let (kernel, stride, pad) = (b.create_vector(kernel), b.create_vector(stride), b.create_vector(pad));
            table!(
                b,
                AvgPool2dAttribute,
                fb::AvgPool2dAttributeArgs {
                    kernel: Some(kernel),
                    stride: Some(stride),
                    pad: Some(pad),
                    acc_type: acc_type.to_fb()
                }
            )
        }
        OpAttribute::MaxPool2d { kernel, stride, pad, nan_mode } => {
            let (kernel, stride, pad) = (b.create_vector(kernel), b.create_vector(stride), b.create_vector(pad));
            table!(
                b,
                MaxPool2dAttribute,
                fb::MaxPool2dAttributeArgs {
                    kernel: Some(kernel),
                    stride: Some(stride),
                    pad: Some(pad),
                    nan_mode: nan_mode.to_fb()
                }
            )
        }
        OpAttribute::AvgPool2dAdaptive { acc_type } => {
            table!(b, AvgPool2dAdaptiveAttribute, fb::AvgPool2dAdaptiveAttributeArgs { acc_type: acc_type.to_fb() })
        }
        OpAttribute::MaxPool2dAdaptive { nan_mode } => {
            table!(b, MaxPool2dAdaptiveAttribute, fb::MaxPool2dAdaptiveAttributeArgs { nan_mode: nan_mode.to_fb() })
        }
        OpAttribute::ArgMax { axis, nan_mode } => {
            table!(b, ArgMaxAttribute, fb::ArgMaxAttributeArgs { axis: *axis, nan_mode: nan_mode.to_fb() })
        }
        OpAttribute::ArgMin { axis, nan_mode } => {
            table!(b, ArgMinAttribute, fb::ArgMinAttributeArgs { axis: *axis, nan_mode: nan_mode.to_fb() })
        }
        OpAttribute::Reduce { axis, nan_mode } => match op {
            Op::ReduceAll => table!(b, ReduceAllAttribute, fb::ReduceAllAttributeArgs { axis: *axis }),
            Op::ReduceAny => table!(b, ReduceAnyAttribute, fb::ReduceAnyAttributeArgs { axis: *axis }),
            Op::ReduceProduct => table!(b, ReduceProductAttribute, fb::ReduceProductAttributeArgs { axis: *axis }),
            Op::ReduceSum => table!(b, ReduceSumAttribute, fb::ReduceSumAttributeArgs { axis: *axis }),
            Op::ReduceMax => {
                table!(b, ReduceMaxAttribute, fb::ReduceMaxAttributeArgs { axis: *axis, nan_mode: nan_mode.to_fb() })
            }
            Op::ReduceMin => {
                table!(b, ReduceMinAttribute, fb::ReduceMinAttributeArgs { axis: *axis, nan_mode: nan_mode.to_fb() })
            }
            _ => return Err(mismatch(op, attr)),
        },
        OpAttribute::Dim { axis } => table!(b, DimAttribute, fb::DimAttributeArgs { axis: *axis }),
        OpAttribute::Fft2d { inverse, local_bound } => {
            table!(b, FFT2dAttribute, fb::FFT2dAttributeArgs { inverse: *inverse, local_bound: *local_bound })
        }
        OpAttribute::Rfft2d { local_bound } => {
            table!(b, RFFT2dAttribute, fb::RFFT2dAttributeArgs { local_bound: *local_bound })
        }
        OpAttribute::AssertEqualShape { allow_broadcast } => {
            table!(
                b,
                AssertEqualShapeAttribute,
                fb::AssertEqualShapeAttributeArgs { allow_broadcast: *allow_broadcast }
            )
        }
        OpAttribute::NanPropagation { nan_mode } => match op {
            Op::Maximum => table!(b, MaximumAttribute, fb::MaximumAttributeArgs { nan_mode: nan_mode.to_fb() }),
            Op::Minimum => table!(b, MinimumAttribute, fb::MinimumAttributeArgs { nan_mode: nan_mode.to_fb() }),
            _ => return Err(mismatch(op, attr)),
        },
        OpAttribute::Concat { axis } => table!(b, ConcatAttribute, fb::ConcatAttributeArgs { axis: *axis }),
        OpAttribute::Transpose { perms } => {
            let perms = b.create_vector(perms);
            table!(b, TransposeAttribute, fb::TransposeAttributeArgs { perms: Some(perms) })
        }
        OpAttribute::Clamp { min_val, max_val, nan_mode } => {
            let (min_val, max_val) = (b.create_vector(min_val), b.create_vector(max_val));
            table!(
                b,
                ClampAttribute,
                fb::ClampAttributeArgs { min_val: Some(min_val), max_val: Some(max_val), nan_mode: nan_mode.to_fb() }
            )
        }
        OpAttribute::ArithmeticRightShift { round } => {
            table!(b, ArithmeticRightShiftAttribute, fb::ArithmeticRightShiftAttributeArgs { round: *round })
        }
        OpAttribute::Rescale { scale32, rounding_mode, per_channel, input_unsigned, output_unsigned } => {
            table!(
                b,
                RescaleAttribute,
                fb::RescaleAttributeArgs {
                    scale32: *scale32,
                    rounding_mode: rounding_mode.to_fb(),
                    per_channel: *per_channel,
                    input_unsigned: *input_unsigned,
                    output_unsigned: *output_unsigned
                }
            )
        }
        OpAttribute::Reverse { axis } => table!(b, ReverseAttribute, fb::ReverseAttributeArgs { axis: *axis }),
        OpAttribute::Resize { mode } => table!(b, ResizeAttribute, fb::ResizeAttributeArgs { mode: mode.to_fb() }),
        OpAttribute::Cast { input_unsigned } => {
            table!(b, CastAttribute, fb::CastAttributeArgs { input_unsigned: *input_unsigned })
        }
        OpAttribute::Custom { operator_name, domain_name, implementation_attrs } => {
            let operator_name = b.create_string(operator_name);
            let domain_name = b.create_string(domain_name);
            let implementation_attrs = b.create_vector(implementation_attrs);
            table!(
                b,
                CustomAttribute,
                fb::CustomAttributeArgs {
                    operator_name: Some(operator_name),
                    domain_name: Some(domain_name),
                    implementation_attrs: Some(implementation_attrs)
                }
            )
        }
        OpAttribute::BlockScaled { block_size } => {
            let size = fb::BlockSize(*block_size);
            match op {
                Op::MatMulTBlockScaled => {
                    table!(b, MatMulTBlockScaledAttribute, fb::MatMulTBlockScaledAttributeArgs { block_size: size })
                }
                Op::CastFromBlockScaled => {
                    table!(b, CastFromBlockScaledAttribute, fb::CastFromBlockScaledAttributeArgs { block_size: size })
                }
                Op::CastToBlockScaled => {
                    table!(b, CastToBlockScaledAttribute, fb::CastToBlockScaledAttributeArgs { block_size: size })
                }
                Op::Conv2dBlockScaled => {
                    table!(b, Conv2dBlockScaledAttribute, fb::Conv2dBlockScaledAttributeArgs { block_size: size })
                }
                Op::ReshapeBlockScaled => {
                    table!(b, ReshapeBlockScaledAttribute, fb::ReshapeBlockScaledAttributeArgs { block_size: size })
                }
                Op::RowGatherBlockScaled => {
                    table!(b, RowGatherBlockScaledAttribute, fb::RowGatherBlockScaledAttributeArgs { block_size: size })
                }
                _ => return Err(mismatch(op, attr)),
            }
        }
    })
}

/// The (empty) attribute table the schema assigns to `op`.
fn empty_attribute(b: &mut FlatBufferBuilder<'_>, op: Op) -> Attr {
    match op {
        Op::Const => empty!(b, ConstAttribute),
        Op::ConstShape => empty!(b, ConstShapeAttribute),
        Op::Identity => empty!(b, IdentityAttribute),
        Op::Variable => empty!(b, VariableAttribute),
        Op::VariableRead => empty!(b, VariableReadAttribute),
        Op::VariableWrite => empty!(b, VariableWriteAttribute),
        Op::Add => empty!(b, AddAttribute),
        Op::Sub => empty!(b, SubAttribute),
        Op::Mul => empty!(b, MulAttribute),
        Op::Maximum => empty!(b, MaximumAttribute),
        Op::Minimum => empty!(b, MinimumAttribute),
        Op::Pow => empty!(b, PowAttribute),
        Op::ArithmeticRightShift => empty!(b, ArithmeticRightShiftAttribute),
        Op::Intdiv => empty!(b, IntDivAttribute),
        Op::LogicalAnd => empty!(b, LogicalAndAttribute),
        Op::LogicalOr => empty!(b, LogicalOrAttribute),
        Op::LogicalXor => empty!(b, LogicalXorAttribute),
        Op::LogicalLeftShift => empty!(b, LogicalLeftShiftAttribute),
        Op::LogicalRightShift => empty!(b, LogicalRightShiftAttribute),
        Op::BitwiseAnd => empty!(b, BitwiseAndAttribute),
        Op::BitwiseOr => empty!(b, BitwiseOrAttribute),
        Op::BitwiseXor => empty!(b, BitwiseXorAttribute),
        Op::Abs => empty!(b, AbsAttribute),
        Op::Negate => empty!(b, NegateAttribute),
        Op::Ceil => empty!(b, CeilAttribute),
        Op::Floor => empty!(b, FloorAttribute),
        Op::Exp => empty!(b, ExpAttribute),
        Op::Log => empty!(b, LogAttribute),
        Op::Sin => empty!(b, SinAttribute),
        Op::Cos => empty!(b, CosAttribute),
        Op::Reciprocal => empty!(b, ReciprocalAttribute),
        Op::Rsqrt => empty!(b, RsqrtAttribute),
        Op::Erf => empty!(b, ErfAttribute),
        Op::Sigmoid => empty!(b, SigmoidAttribute),
        Op::Tanh => empty!(b, TanhAttribute),
        Op::Clz => empty!(b, ClzAttribute),
        Op::BitwiseNot => empty!(b, BitwiseNotAttribute),
        Op::LogicalNot => empty!(b, LogicalNotAttribute),
        Op::Clamp => empty!(b, ClampAttribute),
        Op::Equal => empty!(b, EqualAttribute),
        Op::Greater => empty!(b, GreaterAttribute),
        Op::GreaterEqual => empty!(b, GreaterEqualAttribute),
        Op::Select => empty!(b, SelectAttribute),
        Op::ReduceAll => empty!(b, ReduceAllAttribute),
        Op::ReduceAny => empty!(b, ReduceAnyAttribute),
        Op::ReduceMax => empty!(b, ReduceMaxAttribute),
        Op::ReduceMin => empty!(b, ReduceMinAttribute),
        Op::ReduceProduct => empty!(b, ReduceProductAttribute),
        Op::ReduceSum => empty!(b, ReduceSumAttribute),
        Op::Reshape => empty!(b, ReshapeAttribute),
        Op::Transpose => empty!(b, TransposeAttribute),
        Op::Concat => empty!(b, ConcatAttribute),
        Op::Slice => empty!(b, SliceAttribute),
        Op::Pad => empty!(b, PadAttribute),
        Op::Tile => empty!(b, TileAttribute),
        Op::Reverse => empty!(b, ReverseAttribute),
        Op::Gather => empty!(b, GatherAttribute),
        Op::Scatter => empty!(b, ScatterAttribute),
        Op::Conv2d => empty!(b, Conv2dAttribute),
        Op::Conv3d => empty!(b, Conv3dAttribute),
        Op::DepthwiseConv2d => empty!(b, DepthwiseConv2dAttribute),
        Op::TransposeConv2d => empty!(b, TransposeConv2dAttribute),
        Op::MatMul => empty!(b, MatMulAttribute),
        Op::AvgPool2d => empty!(b, AvgPool2dAttribute),
        Op::MaxPool2d => empty!(b, MaxPool2dAttribute),
        Op::ArgMax => empty!(b, ArgMaxAttribute),
        Op::Cast => empty!(b, CastAttribute),
        Op::Rescale => empty!(b, RescaleAttribute),
        Op::Table => empty!(b, TableAttribute),
        Op::CondIf => empty!(b, CondIfAttribute),
        Op::WhileLoop => empty!(b, WhileLoopAttribute),
        Op::Resize => empty!(b, ResizeAttribute),
        Op::AddShape => empty!(b, AddShapeAttribute),
        Op::SubShape => empty!(b, SubShapeAttribute),
        Op::MulShape => empty!(b, MulShapeAttribute),
        Op::ConcatShape => empty!(b, ConcatShapeAttribute),
        Op::SliceShape => empty!(b, SliceShapeAttribute),
        Op::Exp2Shape => empty!(b, Exp2ShapeAttribute),
        Op::Log2CeilShape => empty!(b, Log2CeilShapeAttribute),
        Op::Log2FloorShape => empty!(b, Log2FloorShapeAttribute),
        Op::MaxShape => empty!(b, MaxShapeAttribute),
        Op::MinShape => empty!(b, MinShapeAttribute),
        Op::ModShape => empty!(b, ModShapeAttribute),
        Op::DivCeilShape => empty!(b, DivCeilShapeAttribute),
        Op::DivFloorShape => empty!(b, DivFloorShapeAttribute),
        Op::Dim => empty!(b, DimAttribute),
        Op::AssertEqualShape => empty!(b, AssertEqualShapeAttribute),
        Op::MatMulTBlockScaled => empty!(b, MatMulTBlockScaledAttribute),
        Op::CastFromBlockScaled => empty!(b, CastFromBlockScaledAttribute),
        Op::CastToBlockScaled => empty!(b, CastToBlockScaledAttribute),
        Op::Conv2dBlockScaled => empty!(b, Conv2dBlockScaledAttribute),
        Op::ReshapeBlockScaled => empty!(b, ReshapeBlockScaledAttribute),
        Op::RowGatherBlockScaled => empty!(b, RowGatherBlockScaledAttribute),
        Op::AvgPool2dAdaptive => empty!(b, AvgPool2dAdaptiveAttribute),
        Op::MaxPool2dAdaptive => empty!(b, MaxPool2dAdaptiveAttribute),
        Op::Fft2d => empty!(b, FFT2dAttribute),
        Op::Rfft2d => empty!(b, RFFT2dAttribute),
        Op::Custom => empty!(b, CustomAttribute),
        Op::RowGather => empty!(b, RowGatherAttribute),
        Op::MatMulT => empty!(b, MatMulTAttribute),
        Op::ArgMin => empty!(b, ArgMinAttribute),
        Op::Unknown(_) => (fb::Attribute::NONE, None),
    }
}
