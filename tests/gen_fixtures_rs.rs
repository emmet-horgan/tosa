//! Generates test fixture .tosa files using the FlatBuffers builder.
//! Run with: cargo test --test gen_fixtures_rs -- --ignored --nocapture

use std::fs;
use std::path::{Path, PathBuf};

use flatbuffers::{FlatBufferBuilder, WIPOffset, ForwardsUOffset};

#[path = "../src/generated.rs"]
mod generated;
use generated::tosa::tosa;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures")
}

fn create_string_vector<'a>(
    builder: &mut FlatBufferBuilder<'a>,
    strings: &[&str],
) -> WIPOffset<flatbuffers::Vector<'a, ForwardsUOffset<&'a str>>> {
    let offsets: Vec<_> = strings.iter().map(|s| builder.create_string(s)).collect();
    builder.create_vector(&offsets)
}

fn write_npy_f32(path: &Path, data: &[f32], shape: &[usize]) {
    use std::io::Write;
    let mut f = fs::File::create(path).unwrap();
    let descr = "'<f4'";
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!("({})", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "))
    };
    let header_dict = format!(
        "{{'descr': {}, 'fortran_order': False, 'shape': {}, }}",
        descr, shape_str
    );
    let prefix_len = 10;
    let total_unpadded = prefix_len + header_dict.len() + 1;
    let pad_len = (64 - (total_unpadded % 64)) % 64;
    let padded_header = format!("{}{}\n", header_dict, " ".repeat(pad_len));
    let header_len = padded_header.len() as u16;

    f.write_all(&[0x93]).unwrap();
    f.write_all(b"NUMPY").unwrap();
    f.write_all(&[1, 0]).unwrap();
    f.write_all(&header_len.to_le_bytes()).unwrap();
    f.write_all(padded_header.as_bytes()).unwrap();
    for &val in data {
        f.write_all(&val.to_le_bytes()).unwrap();
    }
}

fn write_npy_i32(path: &Path, data: &[i32], shape: &[usize]) {
    use std::io::Write;
    let mut f = fs::File::create(path).unwrap();
    let descr = "'<i4'";
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!("({})", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "))
    };
    let header_dict = format!(
        "{{'descr': {}, 'fortran_order': False, 'shape': {}, }}",
        descr, shape_str
    );
    let prefix_len = 10;
    let total_unpadded = prefix_len + header_dict.len() + 1;
    let pad_len = (64 - (total_unpadded % 64)) % 64;
    let padded_header = format!("{}{}\n", header_dict, " ".repeat(pad_len));
    let header_len = padded_header.len() as u16;

    f.write_all(&[0x93]).unwrap();
    f.write_all(b"NUMPY").unwrap();
    f.write_all(&[1, 0]).unwrap();
    f.write_all(&header_len.to_le_bytes()).unwrap();
    f.write_all(padded_header.as_bytes()).unwrap();
    for &val in data {
        f.write_all(&val.to_le_bytes()).unwrap();
    }
}

fn f32_to_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i32_to_bytes(vals: &[i32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn build_graph_with_ops(ops_builder: impl FnOnce(&mut FlatBufferBuilder) -> Vec<u8>) -> Vec<u8> {
    let mut builder = FlatBufferBuilder::new();
    ops_builder(&mut builder)
}

#[test]
#[ignore]
fn generate_all_fixtures() {
    let dir = fixtures_dir();
    fs::create_dir_all(&dir).unwrap();

    gen_const_add_f32(&dir);
    gen_sub_i32(&dir);
    gen_broadcast_add_f32(&dir);
    gen_input_add_f32(&dir);
    gen_abs_negate_f32(&dir);
    gen_reshape_f32(&dir);
    gen_mul_f32(&dir);

    println!("All fixtures generated in {:?}", dir);
}

fn gen_const_add_f32(dir: &Path) {
    let fixture_dir = dir.join("const_add_f32");
    fs::create_dir_all(&fixture_dir).unwrap();

    let a_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_vals: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
    let expected: Vec<f32> = a_vals.iter().zip(&b_vals).map(|(a, b)| a + b).collect();

    let mut builder = FlatBufferBuilder::new();

    // Tensors
    let a_name = builder.create_string("const_a");
    let a_shape = builder.create_vector(&[4i32]);
    let a_data = builder.create_vector(&f32_to_bytes(&a_vals));
    let tensor_a = tosa::TosaTensor::create(&mut builder, &tosa::TosaTensorArgs {
        name: Some(a_name), shape: Some(a_shape), type_: tosa::DType::FP32,
        data: Some(a_data), ..Default::default()
    });

    let b_name = builder.create_string("const_b");
    let b_shape = builder.create_vector(&[4i32]);
    let b_data = builder.create_vector(&f32_to_bytes(&b_vals));
    let tensor_b = tosa::TosaTensor::create(&mut builder, &tosa::TosaTensorArgs {
        name: Some(b_name), shape: Some(b_shape), type_: tosa::DType::FP32,
        data: Some(b_data), ..Default::default()
    });

    let out_name = builder.create_string("TosaOutput_0");
    let out_shape = builder.create_vector(&[4i32]);
    let tensor_out = tosa::TosaTensor::create(&mut builder, &tosa::TosaTensorArgs {
        name: Some(out_name), shape: Some(out_shape), type_: tosa::DType::FP32,
        ..Default::default()
    });

    // Ops
    let const_a_out = create_string_vector(&mut builder, &["const_a"]);
    let attr_ca = tosa::ConstAttribute::create(&mut builder, &tosa::ConstAttributeArgs {});
    let op_ca = tosa::TosaOperator::create(&mut builder, &tosa::TosaOperatorArgs {
        op: tosa::Op::CONST, outputs: Some(const_a_out),
        attribute_type: tosa::Attribute::ConstAttribute, attribute: Some(attr_ca.as_union_value()),
        ..Default::default()
    });

    let const_b_out = create_string_vector(&mut builder, &["const_b"]);
    let attr_cb = tosa::ConstAttribute::create(&mut builder, &tosa::ConstAttributeArgs {});
    let op_cb = tosa::TosaOperator::create(&mut builder, &tosa::TosaOperatorArgs {
        op: tosa::Op::CONST, outputs: Some(const_b_out),
        attribute_type: tosa::Attribute::ConstAttribute, attribute: Some(attr_cb.as_union_value()),
        ..Default::default()
    });

    let add_in = create_string_vector(&mut builder, &["const_a", "const_b"]);
    let add_out = create_string_vector(&mut builder, &["TosaOutput_0"]);
    let add_attr = tosa::AddAttribute::create(&mut builder, &tosa::AddAttributeArgs {});
    let op_add = tosa::TosaOperator::create(&mut builder, &tosa::TosaOperatorArgs {
        op: tosa::Op::ADD, inputs: Some(add_in), outputs: Some(add_out),
        attribute_type: tosa::Attribute::AddAttribute, attribute: Some(add_attr.as_union_value()),
        ..Default::default()
    });

    // Block
    let operators = builder.create_vector(&[op_ca, op_cb, op_add]);
    let tensors = builder.create_vector(&[tensor_a, tensor_b, tensor_out]);
    let block_outputs = create_string_vector(&mut builder, &["TosaOutput_0"]);
    let block_name = builder.create_string("main");
    let block = tosa::TosaBasicBlock::create(&mut builder, &tosa::TosaBasicBlockArgs {
        name: Some(block_name), operators: Some(operators), tensors: Some(tensors),
        outputs: Some(block_outputs), ..Default::default()
    });

    // Region + Graph
    let blocks = builder.create_vector(&[block]);
    let region_name = builder.create_string("main");
    let region = tosa::TosaRegion::create(&mut builder, &tosa::TosaRegionArgs {
        name: Some(region_name), blocks: Some(blocks),
    });
    let version = tosa::Version::create(&mut builder, &tosa::VersionArgs {
        _major: 1, _minor: 1, _patch: 0, _draft: true,
    });
    let regions = builder.create_vector(&[region]);
    let graph = tosa::TosaGraph::create(&mut builder, &tosa::TosaGraphArgs {
        version: Some(version), regions: Some(regions), ..Default::default()
    });
    builder.finish(graph, Some("TOSA"));

    fs::write(fixture_dir.join("graph.tosa"), builder.finished_data()).unwrap();
    write_npy_f32(&fixture_dir.join("expected_TosaOutput_0.npy"), &expected, &[4]);
}

fn gen_sub_i32(dir: &Path) {
    let fixture_dir = dir.join("sub_i32");
    fs::create_dir_all(&fixture_dir).unwrap();

    let a_vals: Vec<i32> = vec![100, 200, 300, 400];
    let b_vals: Vec<i32> = vec![10, 20, 30, 40];
    let expected: Vec<i32> = a_vals.iter().zip(&b_vals).map(|(a, b)| a - b).collect();

    let mut builder = FlatBufferBuilder::new();

    let a_name = builder.create_string("const_a");
    let a_shape = builder.create_vector(&[4i32]);
    let a_data = builder.create_vector(&i32_to_bytes(&a_vals));
    let tensor_a = tosa::TosaTensor::create(&mut builder, &tosa::TosaTensorArgs {
        name: Some(a_name), shape: Some(a_shape), type_: tosa::DType::INT32,
        data: Some(a_data), ..Default::default()
    });

    let b_name = builder.create_string("const_b");
    let b_shape = builder.create_vector(&[4i32]);
    let b_data = builder.create_vector(&i32_to_bytes(&b_vals));
    let tensor_b = tosa::TosaTensor::create(&mut builder, &tosa::TosaTensorArgs {
        name: Some(b_name), shape: Some(b_shape), type_: tosa::DType::INT32,
        data: Some(b_data), ..Default::default()
    });

    let out_name = builder.create_string("TosaOutput_0");
    let out_shape = builder.create_vector(&[4i32]);
    let tensor_out = tosa::TosaTensor::create(&mut builder, &tosa::TosaTensorArgs {
        name: Some(out_name), shape: Some(out_shape), type_: tosa::DType::INT32,
        ..Default::default()
    });

    let const_a_out = create_string_vector(&mut builder, &["const_a"]);
    let attr_ca = tosa::ConstAttribute::create(&mut builder, &tosa::ConstAttributeArgs {});
    let op_ca = tosa::TosaOperator::create(&mut builder, &tosa::TosaOperatorArgs {
        op: tosa::Op::CONST, outputs: Some(const_a_out),
        attribute_type: tosa::Attribute::ConstAttribute, attribute: Some(attr_ca.as_union_value()),
        ..Default::default()
    });

    let const_b_out = create_string_vector(&mut builder, &["const_b"]);
    let attr_cb = tosa::ConstAttribute::create(&mut builder, &tosa::ConstAttributeArgs {});
    let op_cb = tosa::TosaOperator::create(&mut builder, &tosa::TosaOperatorArgs {
        op: tosa::Op::CONST, outputs: Some(const_b_out),
        attribute_type: tosa::Attribute::ConstAttribute, attribute: Some(attr_cb.as_union_value()),
        ..Default::default()
    });

    let sub_in = create_string_vector(&mut builder, &["const_a", "const_b"]);
    let sub_out = create_string_vector(&mut builder, &["TosaOutput_0"]);
    let sub_attr = tosa::SubAttribute::create(&mut builder, &tosa::SubAttributeArgs {});
    let op_sub = tosa::TosaOperator::create(&mut builder, &tosa::TosaOperatorArgs {
        op: tosa::Op::SUB, inputs: Some(sub_in), outputs: Some(sub_out),
        attribute_type: tosa::Attribute::SubAttribute, attribute: Some(sub_attr.as_union_value()),
        ..Default::default()
    });

    let operators = builder.create_vector(&[op_ca, op_cb, op_sub]);
    let tensors = builder.create_vector(&[tensor_a, tensor_b, tensor_out]);
    let block_outputs = create_string_vector(&mut builder, &["TosaOutput_0"]);
    let block_name = builder.create_string("main");
    let block = tosa::TosaBasicBlock::create(&mut builder, &tosa::TosaBasicBlockArgs {
        name: Some(block_name), operators: Some(operators), tensors: Some(tensors),
        outputs: Some(block_outputs), ..Default::default()
    });

    let blocks = builder.create_vector(&[block]);
    let region_name = builder.create_string("main");
    let region = tosa::TosaRegion::create(&mut builder, &tosa::TosaRegionArgs {
        name: Some(region_name), blocks: Some(blocks),
    });
    let version = tosa::Version::create(&mut builder, &tosa::VersionArgs {
        _major: 1, _minor: 1, _patch: 0, _draft: true,
    });
    let regions = builder.create_vector(&[region]);
    let graph = tosa::TosaGraph::create(&mut builder, &tosa::TosaGraphArgs {
        version: Some(version), regions: Some(regions), ..Default::default()
    });
    builder.finish(graph, Some("TOSA"));

    fs::write(fixture_dir.join("graph.tosa"), builder.finished_data()).unwrap();
    write_npy_i32(&fixture_dir.join("expected_TosaOutput_0.npy"), &expected, &[4]);
}

fn gen_broadcast_add_f32(dir: &Path) {
    let fixture_dir = dir.join("broadcast_add_f32");
    fs::create_dir_all(&fixture_dir).unwrap();

    let a_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_vals: Vec<f32> = vec![10.0, 20.0, 30.0];
    let expected: Vec<f32> = vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0];

    let mut builder = FlatBufferBuilder::new();

    let a_name = builder.create_string("const_a");
    let a_shape = builder.create_vector(&[2i32, 3]);
    let a_data = builder.create_vector(&f32_to_bytes(&a_vals));
    let tensor_a = tosa::TosaTensor::create(&mut builder, &tosa::TosaTensorArgs {
        name: Some(a_name), shape: Some(a_shape), type_: tosa::DType::FP32,
        data: Some(a_data), ..Default::default()
    });

    let b_name = builder.create_string("const_b");
    let b_shape = builder.create_vector(&[1i32, 3]);
    let b_data = builder.create_vector(&f32_to_bytes(&b_vals));
    let tensor_b = tosa::TosaTensor::create(&mut builder, &tosa::TosaTensorArgs {
        name: Some(b_name), shape: Some(b_shape), type_: tosa::DType::FP32,
        data: Some(b_data), ..Default::default()
    });

    let out_name = builder.create_string("TosaOutput_0");
    let out_shape = builder.create_vector(&[2i32, 3]);
    let tensor_out = tosa::TosaTensor::create(&mut builder, &tosa::TosaTensorArgs {
        name: Some(out_name), shape: Some(out_shape), type_: tosa::DType::FP32,
        ..Default::default()
    });

    let const_a_out = create_string_vector(&mut builder, &["const_a"]);
    let attr_ca = tosa::ConstAttribute::create(&mut builder, &tosa::ConstAttributeArgs {});
    let op_ca = tosa::TosaOperator::create(&mut builder, &tosa::TosaOperatorArgs {
        op: tosa::Op::CONST, outputs: Some(const_a_out),
        attribute_type: tosa::Attribute::ConstAttribute, attribute: Some(attr_ca.as_union_value()),
        ..Default::default()
    });

    let const_b_out = create_string_vector(&mut builder, &["const_b"]);
    let attr_cb = tosa::ConstAttribute::create(&mut builder, &tosa::ConstAttributeArgs {});
    let op_cb = tosa::TosaOperator::create(&mut builder, &tosa::TosaOperatorArgs {
        op: tosa::Op::CONST, outputs: Some(const_b_out),
        attribute_type: tosa::Attribute::ConstAttribute, attribute: Some(attr_cb.as_union_value()),
        ..Default::default()
    });

    let add_in = create_string_vector(&mut builder, &["const_a", "const_b"]);
    let add_out = create_string_vector(&mut builder, &["TosaOutput_0"]);
    let add_attr = tosa::AddAttribute::create(&mut builder, &tosa::AddAttributeArgs {});
    let op_add = tosa::TosaOperator::create(&mut builder, &tosa::TosaOperatorArgs {
        op: tosa::Op::ADD, inputs: Some(add_in), outputs: Some(add_out),
        attribute_type: tosa::Attribute::AddAttribute, attribute: Some(add_attr.as_union_value()),
        ..Default::default()
    });

    let operators = builder.create_vector(&[op_ca, op_cb, op_add]);
    let tensors = builder.create_vector(&[tensor_a, tensor_b, tensor_out]);
    let block_outputs = create_string_vector(&mut builder, &["TosaOutput_0"]);
    let block_name = builder.create_string("main");
    let block = tosa::TosaBasicBlock::create(&mut builder, &tosa::TosaBasicBlockArgs {
        name: Some(block_name), operators: Some(operators), tensors: Some(tensors),
        outputs: Some(block_outputs), ..Default::default()
    });

    let blocks = builder.create_vector(&[block]);
    let region_name = builder.create_string("main");
    let region = tosa::TosaRegion::create(&mut builder, &tosa::TosaRegionArgs {
        name: Some(region_name), blocks: Some(blocks),
    });
    let version = tosa::Version::create(&mut builder, &tosa::VersionArgs {
        _major: 1, _minor: 1, _patch: 0, _draft: true,
    });
    let regions = builder.create_vector(&[region]);
    let graph = tosa::TosaGraph::create(&mut builder, &tosa::TosaGraphArgs {
        version: Some(version), regions: Some(regions), ..Default::default()
    });
    builder.finish(graph, Some("TOSA"));

    fs::write(fixture_dir.join("graph.tosa"), builder.finished_data()).unwrap();
    write_npy_f32(&fixture_dir.join("expected_TosaOutput_0.npy"), &expected, &[2, 3]);
}

fn gen_input_add_f32(dir: &Path) {
    let fixture_dir = dir.join("input_add_f32");
    fs::create_dir_all(&fixture_dir).unwrap();

    let a_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_vals: Vec<f32> = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5];
    let expected: Vec<f32> = a_vals.iter().zip(&b_vals).map(|(a, b)| a + b).collect();

    let mut builder = FlatBufferBuilder::new();

    let a_name = builder.create_string("TosaInput_0");
    let a_shape = builder.create_vector(&[2i32, 3]);
    let tensor_a = tosa::TosaTensor::create(&mut builder, &tosa::TosaTensorArgs {
        name: Some(a_name), shape: Some(a_shape), type_: tosa::DType::FP32,
        ..Default::default()
    });

    let b_name = builder.create_string("TosaInput_1");
    let b_shape = builder.create_vector(&[2i32, 3]);
    let tensor_b = tosa::TosaTensor::create(&mut builder, &tosa::TosaTensorArgs {
        name: Some(b_name), shape: Some(b_shape), type_: tosa::DType::FP32,
        ..Default::default()
    });

    let out_name = builder.create_string("TosaOutput_0");
    let out_shape = builder.create_vector(&[2i32, 3]);
    let tensor_out = tosa::TosaTensor::create(&mut builder, &tosa::TosaTensorArgs {
        name: Some(out_name), shape: Some(out_shape), type_: tosa::DType::FP32,
        ..Default::default()
    });

    let add_in = create_string_vector(&mut builder, &["TosaInput_0", "TosaInput_1"]);
    let add_out = create_string_vector(&mut builder, &["TosaOutput_0"]);
    let add_attr = tosa::AddAttribute::create(&mut builder, &tosa::AddAttributeArgs {});
    let op_add = tosa::TosaOperator::create(&mut builder, &tosa::TosaOperatorArgs {
        op: tosa::Op::ADD, inputs: Some(add_in), outputs: Some(add_out),
        attribute_type: tosa::Attribute::AddAttribute, attribute: Some(add_attr.as_union_value()),
        ..Default::default()
    });

    let operators = builder.create_vector(&[op_add]);
    let tensors = builder.create_vector(&[tensor_a, tensor_b, tensor_out]);
    let block_inputs = create_string_vector(&mut builder, &["TosaInput_0", "TosaInput_1"]);
    let block_outputs = create_string_vector(&mut builder, &["TosaOutput_0"]);
    let block_name = builder.create_string("main");
    let block = tosa::TosaBasicBlock::create(&mut builder, &tosa::TosaBasicBlockArgs {
        name: Some(block_name), operators: Some(operators), tensors: Some(tensors),
        inputs: Some(block_inputs), outputs: Some(block_outputs), ..Default::default()
    });

    let blocks = builder.create_vector(&[block]);
    let region_name = builder.create_string("main");
    let region = tosa::TosaRegion::create(&mut builder, &tosa::TosaRegionArgs {
        name: Some(region_name), blocks: Some(blocks),
    });
    let version = tosa::Version::create(&mut builder, &tosa::VersionArgs {
        _major: 1, _minor: 1, _patch: 0, _draft: true,
    });
    let regions = builder.create_vector(&[region]);
    let graph = tosa::TosaGraph::create(&mut builder, &tosa::TosaGraphArgs {
        version: Some(version), regions: Some(regions), ..Default::default()
    });
    builder.finish(graph, Some("TOSA"));

    fs::write(fixture_dir.join("graph.tosa"), builder.finished_data()).unwrap();
    write_npy_f32(&fixture_dir.join("input_TosaInput_0.npy"), &a_vals, &[2, 3]);
    write_npy_f32(&fixture_dir.join("input_TosaInput_1.npy"), &b_vals, &[2, 3]);
    write_npy_f32(&fixture_dir.join("expected_TosaOutput_0.npy"), &expected, &[2, 3]);
}

fn gen_abs_negate_f32(dir: &Path) {
    let fixture_dir = dir.join("abs_negate_f32");
    fs::create_dir_all(&fixture_dir).unwrap();

    let a_vals: Vec<f32> = vec![-1.0, 2.0, -3.0, 4.0];
    let expected: Vec<f32> = a_vals.iter().map(|&x| (-x).abs()).collect();

    let mut builder = FlatBufferBuilder::new();

    let a_name = builder.create_string("const_a");
    let a_shape = builder.create_vector(&[4i32]);
    let a_data = builder.create_vector(&f32_to_bytes(&a_vals));
    let tensor_a = tosa::TosaTensor::create(&mut builder, &tosa::TosaTensorArgs {
        name: Some(a_name), shape: Some(a_shape), type_: tosa::DType::FP32,
        data: Some(a_data), ..Default::default()
    });

    let neg_name = builder.create_string("neg_out");
    let neg_shape = builder.create_vector(&[4i32]);
    let tensor_neg = tosa::TosaTensor::create(&mut builder, &tosa::TosaTensorArgs {
        name: Some(neg_name), shape: Some(neg_shape), type_: tosa::DType::FP32,
        ..Default::default()
    });

    let out_name = builder.create_string("TosaOutput_0");
    let out_shape = builder.create_vector(&[4i32]);
    let tensor_out = tosa::TosaTensor::create(&mut builder, &tosa::TosaTensorArgs {
        name: Some(out_name), shape: Some(out_shape), type_: tosa::DType::FP32,
        ..Default::default()
    });

    // CONST
    let const_out = create_string_vector(&mut builder, &["const_a"]);
    let attr_c = tosa::ConstAttribute::create(&mut builder, &tosa::ConstAttributeArgs {});
    let op_const = tosa::TosaOperator::create(&mut builder, &tosa::TosaOperatorArgs {
        op: tosa::Op::CONST, outputs: Some(const_out),
        attribute_type: tosa::Attribute::ConstAttribute, attribute: Some(attr_c.as_union_value()),
        ..Default::default()
    });

    // NEGATE
    let neg_in = create_string_vector(&mut builder, &["const_a"]);
    let neg_out = create_string_vector(&mut builder, &["neg_out"]);
    let neg_attr = tosa::NegateAttribute::create(&mut builder, &tosa::NegateAttributeArgs {});
    let op_neg = tosa::TosaOperator::create(&mut builder, &tosa::TosaOperatorArgs {
        op: tosa::Op::NEGATE, inputs: Some(neg_in), outputs: Some(neg_out),
        attribute_type: tosa::Attribute::NegateAttribute, attribute: Some(neg_attr.as_union_value()),
        ..Default::default()
    });

    // ABS
    let abs_in = create_string_vector(&mut builder, &["neg_out"]);
    let abs_out = create_string_vector(&mut builder, &["TosaOutput_0"]);
    let abs_attr = tosa::AbsAttribute::create(&mut builder, &tosa::AbsAttributeArgs {});
    let op_abs = tosa::TosaOperator::create(&mut builder, &tosa::TosaOperatorArgs {
        op: tosa::Op::ABS, inputs: Some(abs_in), outputs: Some(abs_out),
        attribute_type: tosa::Attribute::AbsAttribute, attribute: Some(abs_attr.as_union_value()),
        ..Default::default()
    });

    let operators = builder.create_vector(&[op_const, op_neg, op_abs]);
    let tensors = builder.create_vector(&[tensor_a, tensor_neg, tensor_out]);
    let block_outputs = create_string_vector(&mut builder, &["TosaOutput_0"]);
    let block_name = builder.create_string("main");
    let block = tosa::TosaBasicBlock::create(&mut builder, &tosa::TosaBasicBlockArgs {
        name: Some(block_name), operators: Some(operators), tensors: Some(tensors),
        outputs: Some(block_outputs), ..Default::default()
    });

    let blocks = builder.create_vector(&[block]);
    let region_name = builder.create_string("main");
    let region = tosa::TosaRegion::create(&mut builder, &tosa::TosaRegionArgs {
        name: Some(region_name), blocks: Some(blocks),
    });
    let version = tosa::Version::create(&mut builder, &tosa::VersionArgs {
        _major: 1, _minor: 1, _patch: 0, _draft: true,
    });
    let regions = builder.create_vector(&[region]);
    let graph = tosa::TosaGraph::create(&mut builder, &tosa::TosaGraphArgs {
        version: Some(version), regions: Some(regions), ..Default::default()
    });
    builder.finish(graph, Some("TOSA"));

    fs::write(fixture_dir.join("graph.tosa"), builder.finished_data()).unwrap();
    write_npy_f32(&fixture_dir.join("expected_TosaOutput_0.npy"), &expected, &[4]);
}

fn gen_reshape_f32(dir: &Path) {
    let fixture_dir = dir.join("reshape_f32");
    fs::create_dir_all(&fixture_dir).unwrap();

    let a_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let expected = a_vals.clone(); // reshape doesn't change data order

    let mut builder = FlatBufferBuilder::new();

    let a_name = builder.create_string("const_a");
    let a_shape = builder.create_vector(&[2i32, 3]);
    let a_data = builder.create_vector(&f32_to_bytes(&a_vals));
    let tensor_a = tosa::TosaTensor::create(&mut builder, &tosa::TosaTensorArgs {
        name: Some(a_name), shape: Some(a_shape), type_: tosa::DType::FP32,
        data: Some(a_data), ..Default::default()
    });

    let out_name = builder.create_string("TosaOutput_0");
    let out_shape = builder.create_vector(&[3i32, 2]);
    let tensor_out = tosa::TosaTensor::create(&mut builder, &tosa::TosaTensorArgs {
        name: Some(out_name), shape: Some(out_shape), type_: tosa::DType::FP32,
        ..Default::default()
    });

    let const_out = create_string_vector(&mut builder, &["const_a"]);
    let attr_c = tosa::ConstAttribute::create(&mut builder, &tosa::ConstAttributeArgs {});
    let op_const = tosa::TosaOperator::create(&mut builder, &tosa::TosaOperatorArgs {
        op: tosa::Op::CONST, outputs: Some(const_out),
        attribute_type: tosa::Attribute::ConstAttribute, attribute: Some(attr_c.as_union_value()),
        ..Default::default()
    });

    let reshape_in = create_string_vector(&mut builder, &["const_a"]);
    let reshape_out = create_string_vector(&mut builder, &["TosaOutput_0"]);
    let reshape_attr = tosa::ReshapeAttribute::create(&mut builder, &tosa::ReshapeAttributeArgs {});
    let op_reshape = tosa::TosaOperator::create(&mut builder, &tosa::TosaOperatorArgs {
        op: tosa::Op::RESHAPE, inputs: Some(reshape_in), outputs: Some(reshape_out),
        attribute_type: tosa::Attribute::ReshapeAttribute, attribute: Some(reshape_attr.as_union_value()),
        ..Default::default()
    });

    let operators = builder.create_vector(&[op_const, op_reshape]);
    let tensors = builder.create_vector(&[tensor_a, tensor_out]);
    let block_outputs = create_string_vector(&mut builder, &["TosaOutput_0"]);
    let block_name = builder.create_string("main");
    let block = tosa::TosaBasicBlock::create(&mut builder, &tosa::TosaBasicBlockArgs {
        name: Some(block_name), operators: Some(operators), tensors: Some(tensors),
        outputs: Some(block_outputs), ..Default::default()
    });

    let blocks = builder.create_vector(&[block]);
    let region_name = builder.create_string("main");
    let region = tosa::TosaRegion::create(&mut builder, &tosa::TosaRegionArgs {
        name: Some(region_name), blocks: Some(blocks),
    });
    let version = tosa::Version::create(&mut builder, &tosa::VersionArgs {
        _major: 1, _minor: 1, _patch: 0, _draft: true,
    });
    let regions = builder.create_vector(&[region]);
    let graph = tosa::TosaGraph::create(&mut builder, &tosa::TosaGraphArgs {
        version: Some(version), regions: Some(regions), ..Default::default()
    });
    builder.finish(graph, Some("TOSA"));

    fs::write(fixture_dir.join("graph.tosa"), builder.finished_data()).unwrap();
    write_npy_f32(&fixture_dir.join("expected_TosaOutput_0.npy"), &expected, &[3, 2]);
}

fn gen_mul_f32(dir: &Path) {
    let fixture_dir = dir.join("mul_f32");
    fs::create_dir_all(&fixture_dir).unwrap();

    let a_vals: Vec<f32> = vec![2.0, 3.0, 4.0, 5.0];
    let b_vals: Vec<f32> = vec![0.5, 1.5, 2.5, 3.5];
    let expected: Vec<f32> = a_vals.iter().zip(&b_vals).map(|(a, b)| a * b).collect();

    let mut builder = FlatBufferBuilder::new();

    let a_name = builder.create_string("const_a");
    let a_shape = builder.create_vector(&[4i32]);
    let a_data = builder.create_vector(&f32_to_bytes(&a_vals));
    let tensor_a = tosa::TosaTensor::create(&mut builder, &tosa::TosaTensorArgs {
        name: Some(a_name), shape: Some(a_shape), type_: tosa::DType::FP32,
        data: Some(a_data), ..Default::default()
    });

    let b_name = builder.create_string("const_b");
    let b_shape = builder.create_vector(&[4i32]);
    let b_data = builder.create_vector(&f32_to_bytes(&b_vals));
    let tensor_b = tosa::TosaTensor::create(&mut builder, &tosa::TosaTensorArgs {
        name: Some(b_name), shape: Some(b_shape), type_: tosa::DType::FP32,
        data: Some(b_data), ..Default::default()
    });

    // Shift tensor (i8, value 0 for float mul)
    let shift_name = builder.create_string("shift");
    let shift_shape = builder.create_vector(&[1i32]);
    let shift_data = builder.create_vector(&[0u8]);
    let tensor_shift = tosa::TosaTensor::create(&mut builder, &tosa::TosaTensorArgs {
        name: Some(shift_name), shape: Some(shift_shape), type_: tosa::DType::INT8,
        data: Some(shift_data), ..Default::default()
    });

    let out_name = builder.create_string("TosaOutput_0");
    let out_shape = builder.create_vector(&[4i32]);
    let tensor_out = tosa::TosaTensor::create(&mut builder, &tosa::TosaTensorArgs {
        name: Some(out_name), shape: Some(out_shape), type_: tosa::DType::FP32,
        ..Default::default()
    });

    // CONST ops
    let const_a_out = create_string_vector(&mut builder, &["const_a"]);
    let attr_ca = tosa::ConstAttribute::create(&mut builder, &tosa::ConstAttributeArgs {});
    let op_ca = tosa::TosaOperator::create(&mut builder, &tosa::TosaOperatorArgs {
        op: tosa::Op::CONST, outputs: Some(const_a_out),
        attribute_type: tosa::Attribute::ConstAttribute, attribute: Some(attr_ca.as_union_value()),
        ..Default::default()
    });

    let const_b_out = create_string_vector(&mut builder, &["const_b"]);
    let attr_cb = tosa::ConstAttribute::create(&mut builder, &tosa::ConstAttributeArgs {});
    let op_cb = tosa::TosaOperator::create(&mut builder, &tosa::TosaOperatorArgs {
        op: tosa::Op::CONST, outputs: Some(const_b_out),
        attribute_type: tosa::Attribute::ConstAttribute, attribute: Some(attr_cb.as_union_value()),
        ..Default::default()
    });

    let const_s_out = create_string_vector(&mut builder, &["shift"]);
    let attr_cs = tosa::ConstAttribute::create(&mut builder, &tosa::ConstAttributeArgs {});
    let op_cs = tosa::TosaOperator::create(&mut builder, &tosa::TosaOperatorArgs {
        op: tosa::Op::CONST, outputs: Some(const_s_out),
        attribute_type: tosa::Attribute::ConstAttribute, attribute: Some(attr_cs.as_union_value()),
        ..Default::default()
    });

    // MUL
    let mul_in = create_string_vector(&mut builder, &["const_a", "const_b", "shift"]);
    let mul_out = create_string_vector(&mut builder, &["TosaOutput_0"]);
    let mul_attr = tosa::MulAttribute::create(&mut builder, &tosa::MulAttributeArgs {});
    let op_mul = tosa::TosaOperator::create(&mut builder, &tosa::TosaOperatorArgs {
        op: tosa::Op::MUL, inputs: Some(mul_in), outputs: Some(mul_out),
        attribute_type: tosa::Attribute::MulAttribute, attribute: Some(mul_attr.as_union_value()),
        ..Default::default()
    });

    let operators = builder.create_vector(&[op_ca, op_cb, op_cs, op_mul]);
    let tensors = builder.create_vector(&[tensor_a, tensor_b, tensor_shift, tensor_out]);
    let block_outputs = create_string_vector(&mut builder, &["TosaOutput_0"]);
    let block_name = builder.create_string("main");
    let block = tosa::TosaBasicBlock::create(&mut builder, &tosa::TosaBasicBlockArgs {
        name: Some(block_name), operators: Some(operators), tensors: Some(tensors),
        outputs: Some(block_outputs), ..Default::default()
    });

    let blocks = builder.create_vector(&[block]);
    let region_name = builder.create_string("main");
    let region = tosa::TosaRegion::create(&mut builder, &tosa::TosaRegionArgs {
        name: Some(region_name), blocks: Some(blocks),
    });
    let version = tosa::Version::create(&mut builder, &tosa::VersionArgs {
        _major: 1, _minor: 1, _patch: 0, _draft: true,
    });
    let regions = builder.create_vector(&[region]);
    let graph = tosa::TosaGraph::create(&mut builder, &tosa::TosaGraphArgs {
        version: Some(version), regions: Some(regions), ..Default::default()
    });
    builder.finish(graph, Some("TOSA"));

    fs::write(fixture_dir.join("graph.tosa"), builder.finished_data()).unwrap();
    write_npy_f32(&fixture_dir.join("expected_TosaOutput_0.npy"), &expected, &[4]);
}
