//! Minimal NumPy `.npy` reader/writer that understands the encodings used by
//! the TOSA reference model for every TOSA element type.
//!
//! NumPy has no native BF16 or FP8 types, so the reference model stores them
//! as raw bit patterns: BF16 as `<V2`, FP8E4M3 as `<V1`, FP8E5M2 as `|u1`.
//! INT48 is stored as `<i8` and INT4 as `|i1`. Reading therefore needs the
//! target TOSA dtype as well as the file.

use crate::error::{Error, Result};
use crate::float8::{Fp8E4M3, Fp8E5M2};
use crate::ir::DType;
use crate::tensor::TensorData;

const MAGIC: &[u8] = b"\x93NUMPY";

/// Element encoding parsed from an npy `descr` string such as `'<f4'`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NpyDescr {
    /// NumPy kind character: `b`, `i`, `u`, `f` or `V`.
    pub kind: char,
    pub size: usize,
}

#[derive(Debug, Clone)]
pub struct NpyArray {
    pub descr: NpyDescr,
    pub shape: Vec<usize>,
    /// Raw little-endian element data, `shape.product() * descr.size` bytes.
    pub data: Vec<u8>,
}

fn err(msg: impl std::fmt::Display) -> Error {
    Error::Deserialize(format!("npy: {msg}"))
}

fn header_value<'a>(header: &'a str, key: &str) -> Result<&'a str> {
    let pat = format!("'{key}':");
    let start = header.find(&pat).ok_or_else(|| err(format!("missing '{key}' in header")))? + pat.len();
    Ok(header[start..].trim_start())
}

fn parse_descr(header: &str) -> Result<NpyDescr> {
    let v = header_value(header, "descr")?;
    let quote = v.chars().next().ok_or_else(|| err("empty descr"))?;
    if quote != '\'' && quote != '"' {
        return Err(err("structured dtypes are not supported"));
    }
    let end = v[1..].find(quote).ok_or_else(|| err("unterminated descr"))? + 1;
    let s = &v[1..end];
    let mut chars = s.chars();
    let order = chars.next().ok_or_else(|| err("empty descr"))?;
    if order == '>' {
        return Err(err(format!("big-endian data is not supported ({s})")));
    }
    let (kind, size) = if matches!(order, '<' | '|' | '=') {
        let kind = chars.next().ok_or_else(|| err(format!("bad descr {s}")))?;
        (kind, chars.as_str())
    } else {
        (order, chars.as_str())
    };
    let size = size.parse::<usize>().map_err(|_| err(format!("bad descr {s}")))?;
    if !matches!(kind, 'b' | 'i' | 'u' | 'f' | 'V') {
        return Err(err(format!("unsupported descr {s}")));
    }
    Ok(NpyDescr { kind, size })
}

fn parse_shape(header: &str) -> Result<Vec<usize>> {
    let v = header_value(header, "shape")?;
    let v = v.strip_prefix('(').ok_or_else(|| err("bad shape"))?;
    let end = v.find(')').ok_or_else(|| err("bad shape"))?;
    v[..end]
        .split(',')
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .map(|d| d.parse::<usize>().map_err(|_| err(format!("bad shape dimension {d}"))))
        .collect()
}

pub fn parse(bytes: &[u8]) -> Result<NpyArray> {
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err(err("not an npy file"));
    }
    let (header_len, header_start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err(err("truncated header"));
            }
            (u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize, 12)
        }
        v => return Err(err(format!("unsupported version {v}"))),
    };
    let data_start = header_start + header_len;
    let header = bytes
        .get(header_start..data_start)
        .ok_or_else(|| err("truncated header"))?;
    let header = std::str::from_utf8(header).map_err(|_| err("header is not UTF-8"))?;
    if header.contains("'fortran_order': True") {
        return Err(err("fortran-ordered arrays are not supported"));
    }
    let descr = parse_descr(header)?;
    let shape = parse_shape(header)?;
    let len = shape.iter().product::<usize>() * descr.size;
    let data = bytes
        .get(data_start..data_start + len)
        .ok_or_else(|| err(format!("expected {len} bytes of data")))?
        .to_vec();
    Ok(NpyArray { descr, shape, data })
}

pub fn read(path: &std::path::Path) -> Result<NpyArray> {
    parse(&std::fs::read(path)?)
}

impl NpyArray {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn elements(&self) -> impl Iterator<Item = &[u8]> {
        self.data.chunks_exact(self.descr.size)
    }

    fn uint_bits(&self) -> impl Iterator<Item = u64> + '_ {
        self.elements().map(|c| {
            let mut b = [0u8; 8];
            b[..c.len()].copy_from_slice(c);
            u64::from_le_bytes(b)
        })
    }

    fn ints(&self) -> Result<Vec<i64>> {
        let size = self.descr.size;
        match self.descr.kind {
            'i' if size <= 8 => Ok(self
                .uint_bits()
                .map(|u| {
                    let shift = 64 - 8 * size as u32;
                    ((u << shift) as i64) >> shift
                })
                .collect()),
            'u' | 'b' if size <= 8 => Ok(self.uint_bits().map(|u| u as i64).collect()),
            _ => Err(self.mismatch("an integer")),
        }
    }

    fn floats(&self) -> Result<Vec<f64>> {
        match (self.descr.kind, self.descr.size) {
            ('f', 2) => Ok(self.uint_bits().map(|u| half::f16::from_bits(u as u16).to_f64()).collect()),
            ('f', 4) => Ok(self.uint_bits().map(|u| f32::from_bits(u as u32) as f64).collect()),
            ('f', 8) => Ok(self.uint_bits().map(f64::from_bits).collect()),
            _ => Err(self.mismatch("a float")),
        }
    }

    /// Raw bit patterns of `size`-byte elements stored as `V`, `u` or `i`.
    fn bits(&self, size: usize) -> Result<Vec<u64>> {
        if matches!(self.descr.kind, 'V' | 'u' | 'i') && self.descr.size == size {
            Ok(self.uint_bits().collect())
        } else {
            Err(self.mismatch(&format!("{size}-byte raw bits")))
        }
    }

    fn mismatch(&self, wanted: &str) -> Error {
        Error::DTypeMismatch(format!(
            "npy: expected {wanted}, file holds '{}{}'",
            self.descr.kind, self.descr.size
        ))
    }

    /// Decode the array as elements of TOSA type `dtype`.
    pub fn to_tensor_data(&self, dtype: DType) -> Result<TensorData> {
        Ok(match dtype {
            DType::Bool => TensorData::Bool(self.ints()?.into_iter().map(|v| v != 0).collect()),
            DType::I4 => TensorData::I4(self.ints()?.into_iter().map(|v| v as i8).collect()),
            DType::I8 => TensorData::I8(self.ints()?.into_iter().map(|v| v as i8).collect()),
            DType::I16 => TensorData::I16(self.ints()?.into_iter().map(|v| v as i16).collect()),
            DType::I32 => TensorData::I32(self.ints()?.into_iter().map(|v| v as i32).collect()),
            DType::I48 => TensorData::I48(self.ints()?),
            DType::I64 | DType::Shape => TensorData::I64(self.ints()?),
            DType::F32 => TensorData::F32(self.floats()?.into_iter().map(|v| v as f32).collect()),
            DType::F64 => TensorData::F64(self.floats()?),
            DType::F16 => match self.descr.kind {
                'f' if self.descr.size == 2 => TensorData::F16(
                    self.uint_bits().map(|u| half::f16::from_bits(u as u16)).collect(),
                ),
                _ => TensorData::F16(self.bits(2)?.into_iter().map(|u| half::f16::from_bits(u as u16)).collect()),
            },
            DType::Bf16 => TensorData::Bf16(
                self.bits(2)?.into_iter().map(|u| half::bf16::from_bits(u as u16)).collect(),
            ),
            DType::Fp8E4M3 => {
                TensorData::Fp8E4M3(self.bits(1)?.into_iter().map(|u| Fp8E4M3(u as u8)).collect())
            }
            DType::Fp8E5M2 => {
                TensorData::Fp8E5M2(self.bits(1)?.into_iter().map(|u| Fp8E5M2(u as u8)).collect())
            }
            other => {
                return Err(Error::DTypeMismatch(format!("npy: cannot decode {other:?} tensors")));
            }
        })
    }
}

/// Serialize tensor data using the same encodings the reference model writes.
pub fn write(shape: &[usize], data: &TensorData) -> Vec<u8> {
    let (descr, body): (&str, Vec<u8>) = match data {
        TensorData::Bool(v) => ("|b1", v.iter().map(|&b| b as u8).collect()),
        TensorData::I4(v) | TensorData::I8(v) => ("|i1", v.iter().map(|&x| x as u8).collect()),
        TensorData::I16(v) => ("<i2", v.iter().flat_map(|x| x.to_le_bytes()).collect()),
        TensorData::I32(v) => ("<i4", v.iter().flat_map(|x| x.to_le_bytes()).collect()),
        TensorData::I48(v) | TensorData::I64(v) => ("<i8", v.iter().flat_map(|x| x.to_le_bytes()).collect()),
        TensorData::Fp8E4M3(v) => ("<V1", v.iter().map(|x| x.0).collect()),
        TensorData::Fp8E5M2(v) => ("|u1", v.iter().map(|x| x.0).collect()),
        TensorData::F16(v) => ("<f2", v.iter().flat_map(|x| x.to_le_bytes()).collect()),
        TensorData::Bf16(v) => ("<V2", v.iter().flat_map(|x| x.to_le_bytes()).collect()),
        TensorData::F32(v) => ("<f4", v.iter().flat_map(|x| x.to_le_bytes()).collect()),
        TensorData::F64(v) => ("<f8", v.iter().flat_map(|x| x.to_le_bytes()).collect()),
    };
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let shape_str = match dims.len() {
        1 => format!("({},)", dims[0]),
        _ => format!("({})", dims.join(", ")),
    };
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}");
    // Pad so that the data starts on a 64-byte boundary, ending with '\n'.
    let unpadded = MAGIC.len() + 4 + header.len() + 1;
    header.push_str(&" ".repeat((64 - unpadded % 64) % 64));
    header.push('\n');

    let mut out = Vec::with_capacity(MAGIC.len() + 4 + header.len() + body.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&body);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_every_dtype() {
        let cases = vec![
            (DType::Bool, TensorData::Bool(vec![true, false, true])),
            (DType::I4, TensorData::I4(vec![-7, 0, 7])),
            (DType::I8, TensorData::I8(vec![-128, 0, 127])),
            (DType::I16, TensorData::I16(vec![-32768, 1, 32767])),
            (DType::I32, TensorData::I32(vec![i32::MIN, 2, i32::MAX])),
            (DType::I48, TensorData::I48(vec![-(1 << 47), 3, (1 << 47) - 1])),
            (DType::Fp8E4M3, TensorData::Fp8E4M3(vec![Fp8E4M3(0x7e), Fp8E4M3(1), Fp8E4M3(0x80)])),
            (DType::Fp8E5M2, TensorData::Fp8E5M2(vec![Fp8E5M2(0x7c), Fp8E5M2(1), Fp8E5M2(0xfd)])),
            (DType::F16, TensorData::F16(vec![half::f16::from_f32(1.5); 3])),
            (DType::Bf16, TensorData::Bf16(vec![half::bf16::from_f32(-2.25); 3])),
            (DType::F32, TensorData::F32(vec![1.0, f32::NEG_INFINITY, -0.0])),
        ];
        for (dtype, data) in cases {
            let bytes = write(&[3], &data);
            let arr = parse(&bytes).unwrap();
            assert_eq!(arr.shape, vec![3]);
            let back = arr.to_tensor_data(dtype).unwrap();
            assert_eq!(format!("{back:?}"), format!("{data:?}"), "{dtype:?}");
        }
    }

    #[test]
    fn scalar_and_multi_dim_shapes() {
        let arr = parse(&write(&[], &TensorData::I32(vec![5]))).unwrap();
        assert!(arr.shape.is_empty());
        assert_eq!(arr.len(), 1);
        let arr = parse(&write(&[2, 3], &TensorData::I8(vec![0; 6]))).unwrap();
        assert_eq!(arr.shape, vec![2, 3]);
    }
}
