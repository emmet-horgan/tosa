//! A minimal protobuf wire-format reader and the ONNX messages the frontend
//! needs (`onnx.proto3`: ModelProto, GraphProto, NodeProto, AttributeProto,
//! TensorProto, ValueInfoProto, TypeProto, TensorShapeProto).
//!
//! Written by hand rather than generated so that no protobuf toolchain is
//! required to build; the ONNX schema is small and stable, and unknown
//! fields are skipped.

use crate::error::{ConvertError, Result};

fn err(reason: impl Into<String>) -> ConvertError {
    ConvertError::Frontend { format: "onnx".into(), reason: reason.into() }
}

/// Wire types of protobuf fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Wire {
    Varint,
    Fixed64,
    Bytes,
    Fixed32,
}

/// A cursor over a protobuf message.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        let mut shift = 0;
        loop {
            let byte = *self.buf.get(self.pos).ok_or_else(|| err("truncated varint"))?;
            self.pos += 1;
            if shift < 64 {
                value |= ((byte & 0x7f) as u64) << shift;
            }
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
            if shift > 70 {
                return Err(err("varint too long"));
            }
        }
    }

    fn fixed32(&mut self) -> Result<u32> {
        let b = self.buf.get(self.pos..self.pos + 4).ok_or_else(|| err("truncated fixed32"))?;
        self.pos += 4;
        Ok(u32::from_le_bytes(b.try_into().unwrap()))
    }

    fn fixed64(&mut self) -> Result<u64> {
        let b = self.buf.get(self.pos..self.pos + 8).ok_or_else(|| err("truncated fixed64"))?;
        self.pos += 8;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }

    fn bytes(&mut self) -> Result<&'a [u8]> {
        let len = self.varint()? as usize;
        let b = self.buf.get(self.pos..self.pos + len).ok_or_else(|| err("truncated length-delimited field"))?;
        self.pos += len;
        Ok(b)
    }

    /// The next field's number and wire type.
    fn tag(&mut self) -> Result<(u32, Wire)> {
        let key = self.varint()?;
        let wire = match key & 7 {
            0 => Wire::Varint,
            1 => Wire::Fixed64,
            2 => Wire::Bytes,
            5 => Wire::Fixed32,
            w => return Err(err(format!("unsupported wire type {w}"))),
        };
        Ok(((key >> 3) as u32, wire))
    }

    fn skip(&mut self, wire: Wire) -> Result<()> {
        match wire {
            Wire::Varint => self.varint().map(|_| ()),
            Wire::Fixed64 => self.fixed64().map(|_| ()),
            Wire::Bytes => self.bytes().map(|_| ()),
            Wire::Fixed32 => self.fixed32().map(|_| ()),
        }
    }
}

fn string(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

/// Packed or unpacked repeated varints.
fn varints(r: &mut Reader<'_>, wire: Wire, out: &mut Vec<i64>) -> Result<()> {
    match wire {
        Wire::Bytes => {
            let mut inner = Reader::new(r.bytes()?);
            while !inner.done() {
                out.push(inner.varint()? as i64);
            }
        }
        _ => out.push(r.varint()? as i64),
    }
    Ok(())
}

fn floats(r: &mut Reader<'_>, wire: Wire, out: &mut Vec<f32>) -> Result<()> {
    match wire {
        Wire::Bytes => {
            let mut inner = Reader::new(r.bytes()?);
            while !inner.done() {
                out.push(f32::from_bits(inner.fixed32()?));
            }
        }
        _ => out.push(f32::from_bits(r.fixed32()?)),
    }
    Ok(())
}

fn doubles(r: &mut Reader<'_>, wire: Wire, out: &mut Vec<f64>) -> Result<()> {
    match wire {
        Wire::Bytes => {
            let mut inner = Reader::new(r.bytes()?);
            while !inner.done() {
                out.push(f64::from_bits(inner.fixed64()?));
            }
        }
        _ => out.push(f64::from_bits(r.fixed64()?)),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
pub struct Model {
    pub ir_version: i64,
    pub producer_name: String,
    pub graph: Graph,
    /// (domain, version); the default domain is `""` or `"ai.onnx"`.
    pub opset_import: Vec<(String, i64)>,
}

#[derive(Debug, Default, Clone)]
pub struct Graph {
    pub name: String,
    pub node: Vec<Node>,
    pub initializer: Vec<Tensor>,
    pub input: Vec<ValueInfo>,
    pub output: Vec<ValueInfo>,
    pub value_info: Vec<ValueInfo>,
}

#[derive(Debug, Default, Clone)]
pub struct Node {
    pub input: Vec<String>,
    pub output: Vec<String>,
    pub name: String,
    pub op_type: String,
    pub domain: String,
    pub attribute: Vec<Attribute>,
}

#[derive(Debug, Clone)]
pub enum AttributeValue {
    Float(f32),
    Int(i64),
    String(Vec<u8>),
    Tensor(Box<Tensor>),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<Vec<u8>>),
    /// GRAPH, TENSORS, GRAPHS, SPARSE_TENSOR: recorded but not decoded.
    Unsupported(i64),
}

#[derive(Debug, Clone)]
pub struct Attribute {
    pub name: String,
    pub value: AttributeValue,
}

/// `TensorProto.DataType` values.
pub mod data_type {
    pub const FLOAT: i64 = 1;
    pub const UINT8: i64 = 2;
    pub const INT8: i64 = 3;
    pub const UINT16: i64 = 4;
    pub const INT16: i64 = 5;
    pub const INT32: i64 = 6;
    pub const INT64: i64 = 7;
    pub const STRING: i64 = 8;
    pub const BOOL: i64 = 9;
    pub const FLOAT16: i64 = 10;
    pub const DOUBLE: i64 = 11;
    pub const UINT32: i64 = 12;
    pub const UINT64: i64 = 13;
    pub const BFLOAT16: i64 = 16;
    pub const INT4: i64 = 22;
}

#[derive(Debug, Default, Clone)]
pub struct Tensor {
    pub dims: Vec<i64>,
    pub data_type: i64,
    pub name: String,
    pub raw_data: Vec<u8>,
    pub float_data: Vec<f32>,
    pub int32_data: Vec<i64>,
    pub int64_data: Vec<i64>,
    pub double_data: Vec<f64>,
    pub uint64_data: Vec<i64>,
    pub string_data: Vec<Vec<u8>>,
    /// `data_location`: 1 means the data lives in an external file.
    pub external: bool,
}

#[derive(Debug, Default, Clone)]
pub struct ValueInfo {
    pub name: String,
    /// `TypeProto.tensor_type.elem_type`, if the value is a tensor.
    pub elem_type: Option<i64>,
    /// Dimensions: `Some(n)` for a fixed size, `None` for symbolic.
    pub shape: Option<Vec<Option<i64>>>,
}

fn parse_model(buf: &[u8]) -> Result<Model> {
    let mut r = Reader::new(buf);
    let mut m = Model::default();
    while !r.done() {
        let (field, wire) = r.tag()?;
        match field {
            1 => m.ir_version = r.varint()? as i64,
            2 => m.producer_name = string(r.bytes()?),
            7 => m.graph = parse_graph(r.bytes()?)?,
            8 => {
                let mut inner = Reader::new(r.bytes()?);
                let (mut domain, mut version) = (String::new(), 0i64);
                while !inner.done() {
                    let (f, w) = inner.tag()?;
                    match f {
                        1 => domain = string(inner.bytes()?),
                        2 => version = inner.varint()? as i64,
                        _ => inner.skip(w)?,
                    }
                }
                m.opset_import.push((domain, version));
            }
            _ => r.skip(wire)?,
        }
    }
    Ok(m)
}

fn parse_graph(buf: &[u8]) -> Result<Graph> {
    let mut r = Reader::new(buf);
    let mut g = Graph::default();
    while !r.done() {
        let (field, wire) = r.tag()?;
        match field {
            1 => g.node.push(parse_node(r.bytes()?)?),
            2 => g.name = string(r.bytes()?),
            5 => g.initializer.push(parse_tensor(r.bytes()?)?),
            11 => g.input.push(parse_value_info(r.bytes()?)?),
            12 => g.output.push(parse_value_info(r.bytes()?)?),
            13 => g.value_info.push(parse_value_info(r.bytes()?)?),
            _ => r.skip(wire)?,
        }
    }
    Ok(g)
}

fn parse_node(buf: &[u8]) -> Result<Node> {
    let mut r = Reader::new(buf);
    let mut n = Node::default();
    while !r.done() {
        let (field, wire) = r.tag()?;
        match field {
            1 => n.input.push(string(r.bytes()?)),
            2 => n.output.push(string(r.bytes()?)),
            3 => n.name = string(r.bytes()?),
            4 => n.op_type = string(r.bytes()?),
            5 => n.attribute.push(parse_attribute(r.bytes()?)?),
            7 => n.domain = string(r.bytes()?),
            _ => r.skip(wire)?,
        }
    }
    Ok(n)
}

fn parse_attribute(buf: &[u8]) -> Result<Attribute> {
    let mut r = Reader::new(buf);
    let mut name = String::new();
    let mut kind = 0i64;
    let (mut f, mut i, mut s, mut t) = (None, None, None, None);
    let (mut floats_v, mut ints_v, mut strings_v) = (Vec::new(), Vec::new(), Vec::new());
    while !r.done() {
        let (field, wire) = r.tag()?;
        match field {
            1 => name = string(r.bytes()?),
            2 => f = Some(f32::from_bits(r.fixed32()?)),
            3 => i = Some(r.varint()? as i64),
            4 => s = Some(r.bytes()?.to_vec()),
            5 => t = Some(Box::new(parse_tensor(r.bytes()?)?)),
            7 => floats(&mut r, wire, &mut floats_v)?,
            8 => varints(&mut r, wire, &mut ints_v)?,
            9 => strings_v.push(r.bytes()?.to_vec()),
            20 => kind = r.varint()? as i64,
            _ => r.skip(wire)?,
        }
    }
    // The type field is authoritative when present; otherwise infer from
    // which value fields were set.
    let value = match kind {
        1 => AttributeValue::Float(f.unwrap_or(0.0)),
        2 => AttributeValue::Int(i.unwrap_or(0)),
        3 => AttributeValue::String(s.unwrap_or_default()),
        4 => AttributeValue::Tensor(t.unwrap_or_default()),
        6 => AttributeValue::Floats(floats_v),
        7 => AttributeValue::Ints(ints_v),
        8 => AttributeValue::Strings(strings_v),
        0 => {
            if let Some(v) = f {
                AttributeValue::Float(v)
            } else if let Some(v) = i {
                AttributeValue::Int(v)
            } else if let Some(v) = s {
                AttributeValue::String(v)
            } else if let Some(v) = t {
                AttributeValue::Tensor(v)
            } else if !floats_v.is_empty() {
                AttributeValue::Floats(floats_v)
            } else if !ints_v.is_empty() {
                AttributeValue::Ints(ints_v)
            } else if !strings_v.is_empty() {
                AttributeValue::Strings(strings_v)
            } else {
                AttributeValue::Unsupported(0)
            }
        }
        other => AttributeValue::Unsupported(other),
    };
    Ok(Attribute { name, value })
}

fn parse_tensor(buf: &[u8]) -> Result<Tensor> {
    let mut r = Reader::new(buf);
    let mut t = Tensor::default();
    while !r.done() {
        let (field, wire) = r.tag()?;
        match field {
            1 => varints(&mut r, wire, &mut t.dims)?,
            2 => t.data_type = r.varint()? as i64,
            4 => floats(&mut r, wire, &mut t.float_data)?,
            5 => varints(&mut r, wire, &mut t.int32_data)?,
            6 => t.string_data.push(r.bytes()?.to_vec()),
            7 => varints(&mut r, wire, &mut t.int64_data)?,
            8 => t.name = string(r.bytes()?),
            9 => t.raw_data = r.bytes()?.to_vec(),
            10 => doubles(&mut r, wire, &mut t.double_data)?,
            11 => varints(&mut r, wire, &mut t.uint64_data)?,
            14 => t.external = r.varint()? == 1,
            _ => r.skip(wire)?,
        }
    }
    Ok(t)
}

fn parse_value_info(buf: &[u8]) -> Result<ValueInfo> {
    let mut r = Reader::new(buf);
    let mut v = ValueInfo::default();
    while !r.done() {
        let (field, wire) = r.tag()?;
        match field {
            1 => v.name = string(r.bytes()?),
            2 => {
                // TypeProto
                let mut tr = Reader::new(r.bytes()?);
                while !tr.done() {
                    let (f, w) = tr.tag()?;
                    if f != 1 {
                        tr.skip(w)?;
                        continue;
                    }
                    // TypeProto.Tensor
                    let mut tt = Reader::new(tr.bytes()?);
                    while !tt.done() {
                        let (f2, w2) = tt.tag()?;
                        match f2 {
                            1 => v.elem_type = Some(tt.varint()? as i64),
                            2 => {
                                // TensorShapeProto
                                let mut sr = Reader::new(tt.bytes()?);
                                let mut dims = Vec::new();
                                while !sr.done() {
                                    let (f3, w3) = sr.tag()?;
                                    if f3 != 1 {
                                        sr.skip(w3)?;
                                        continue;
                                    }
                                    // Dimension
                                    let mut dr = Reader::new(sr.bytes()?);
                                    let mut dim = None;
                                    while !dr.done() {
                                        let (f4, w4) = dr.tag()?;
                                        match f4 {
                                            1 => dim = Some(dr.varint()? as i64),
                                            _ => dr.skip(w4)?,
                                        }
                                    }
                                    dims.push(dim);
                                }
                                v.shape = Some(dims);
                            }
                            _ => tt.skip(w2)?,
                        }
                    }
                }
            }
            _ => r.skip(wire)?,
        }
    }
    Ok(v)
}

/// Parse a serialized `ModelProto`.
pub fn parse(bytes: &[u8]) -> Result<Model> {
    if bytes.is_empty() {
        return Err(err("empty file"));
    }
    parse_model(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode a field for tests.
    fn varint(mut v: u64, out: &mut Vec<u8>) {
        loop {
            let byte = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(byte);
                return;
            }
            out.push(byte | 0x80);
        }
    }
    fn field_varint(n: u32, v: u64, out: &mut Vec<u8>) {
        varint((n as u64) << 3, out);
        varint(v, out);
    }
    fn field_bytes(n: u32, b: &[u8], out: &mut Vec<u8>) {
        varint(((n as u64) << 3) | 2, out);
        varint(b.len() as u64, out);
        out.extend_from_slice(b);
    }

    #[test]
    fn parses_a_small_model() {
        // initializer "w" FLOAT [2] raw, node Relu(x) -> y, input x, output y.
        let mut tensor = Vec::new();
        field_varint(1, 2, &mut tensor);
        field_varint(2, data_type::FLOAT as u64, &mut tensor);
        field_bytes(8, b"w", &mut tensor);
        field_bytes(9, &[1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat(), &mut tensor);

        let mut attr = Vec::new();
        field_bytes(1, b"alpha", &mut attr);
        varint((2 << 3) | 5, &mut attr);
        attr.extend_from_slice(&0.5f32.to_le_bytes());
        field_varint(20, 1, &mut attr);

        let mut node = Vec::new();
        field_bytes(1, b"x", &mut node);
        field_bytes(2, b"y", &mut node);
        field_bytes(4, b"LeakyRelu", &mut node);
        field_bytes(5, &attr, &mut node);

        let mut dim = Vec::new();
        field_varint(1, 3, &mut dim);
        let mut shape = Vec::new();
        field_bytes(1, &dim, &mut shape);
        let mut tensor_type = Vec::new();
        field_varint(1, data_type::FLOAT as u64, &mut tensor_type);
        field_bytes(2, &shape, &mut tensor_type);
        let mut ty = Vec::new();
        field_bytes(1, &tensor_type, &mut ty);
        let mut vi = Vec::new();
        field_bytes(1, b"x", &mut vi);
        field_bytes(2, &ty, &mut vi);

        let mut graph = Vec::new();
        field_bytes(1, &node, &mut graph);
        field_bytes(2, b"g", &mut graph);
        field_bytes(5, &tensor, &mut graph);
        field_bytes(11, &vi, &mut graph);
        field_bytes(12, &vi, &mut graph);

        let mut opset = Vec::new();
        field_bytes(1, b"", &mut opset);
        field_varint(2, 13, &mut opset);

        let mut model = Vec::new();
        field_varint(1, 8, &mut model);
        field_bytes(7, &graph, &mut model);
        field_bytes(8, &opset, &mut model);
        field_varint(99, 5, &mut model); // unknown field, skipped

        let m = parse(&model).unwrap();
        assert_eq!(m.ir_version, 8);
        assert_eq!(m.opset_import, vec![(String::new(), 13)]);
        assert_eq!(m.graph.name, "g");
        assert_eq!(m.graph.initializer[0].name, "w");
        assert_eq!(m.graph.initializer[0].dims, vec![2]);
        assert_eq!(m.graph.initializer[0].raw_data.len(), 8);
        let n = &m.graph.node[0];
        assert_eq!(n.op_type, "LeakyRelu");
        assert!(matches!(n.attribute[0].value, AttributeValue::Float(v) if v == 0.5));
        assert_eq!(m.graph.input[0].elem_type, Some(data_type::FLOAT));
        assert_eq!(m.graph.input[0].shape, Some(vec![Some(3)]));
        assert!(parse(b"\xff\xff\xff").is_err());
    }
}
