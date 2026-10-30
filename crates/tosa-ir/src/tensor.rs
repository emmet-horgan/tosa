use crate::error::{Error, Result};
use crate::float8::{Fp8E4M3, Fp8E5M2};
use crate::ir::DType;

#[derive(Debug, Clone, PartialEq)]
pub enum TensorData {
    Bool(Vec<bool>),
    /// INT4 values in [-8, 7], held one per byte.
    I4(Vec<i8>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I48(Vec<i64>),
    I64(Vec<i64>),
    Fp8E4M3(Vec<Fp8E4M3>),
    Fp8E5M2(Vec<Fp8E5M2>),
    F16(Vec<half::f16>),
    Bf16(Vec<half::bf16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl TensorData {
    pub fn dtype(&self) -> DType {
        match self {
            TensorData::Bool(_) => DType::Bool,
            TensorData::I4(_) => DType::I4,
            TensorData::I8(_) => DType::I8,
            TensorData::I16(_) => DType::I16,
            TensorData::I32(_) => DType::I32,
            TensorData::I48(_) => DType::I48,
            TensorData::I64(_) => DType::I64,
            TensorData::Fp8E4M3(_) => DType::Fp8E4M3,
            TensorData::Fp8E5M2(_) => DType::Fp8E5M2,
            TensorData::F16(_) => DType::F16,
            TensorData::Bf16(_) => DType::Bf16,
            TensorData::F32(_) => DType::F32,
            TensorData::F64(_) => DType::F64,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            TensorData::Bool(v) => v.len(),
            TensorData::I4(v) | TensorData::I8(v) => v.len(),
            TensorData::I16(v) => v.len(),
            TensorData::I32(v) => v.len(),
            TensorData::I48(v) => v.len(),
            TensorData::I64(v) => v.len(),
            TensorData::Fp8E4M3(v) => v.len(),
            TensorData::Fp8E5M2(v) => v.len(),
            TensorData::F16(v) => v.len(),
            TensorData::Bf16(v) => v.len(),
            TensorData::F32(v) => v.len(),
            TensorData::F64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn zeros(dtype: DType, count: usize) -> Result<Self> {
        Ok(match dtype {
            DType::Bool => TensorData::Bool(vec![false; count]),
            DType::I4 => TensorData::I4(vec![0; count]),
            DType::I8 => TensorData::I8(vec![0; count]),
            DType::I16 => TensorData::I16(vec![0; count]),
            DType::I32 => TensorData::I32(vec![0; count]),
            DType::I48 => TensorData::I48(vec![0; count]),
            DType::I64 | DType::Shape => TensorData::I64(vec![0; count]),
            DType::Fp8E4M3 => TensorData::Fp8E4M3(vec![Fp8E4M3::ZERO; count]),
            DType::Fp8E5M2 => TensorData::Fp8E5M2(vec![Fp8E5M2::ZERO; count]),
            DType::F16 => TensorData::F16(vec![half::f16::ZERO; count]),
            DType::Bf16 => TensorData::Bf16(vec![half::bf16::ZERO; count]),
            DType::F32 => TensorData::F32(vec![0.0; count]),
            DType::F64 => TensorData::F64(vec![0.0; count]),
            _ => return Err(Error::DTypeMismatch(format!("cannot create zeros for {:?}", dtype))),
        })
    }

    /// Decode serialized (little-endian) tensor data holding `count` elements.
    ///
    /// INT4 is packed two per byte, low nibble first; INT48 uses 6 bytes per
    /// element. Trailing alignment padding in `bytes` is ignored.
    pub fn from_raw_bytes(dtype: DType, bytes: &[u8], count: usize) -> Result<Self> {
        let needed = match dtype {
            DType::I4 => count.div_ceil(2),
            DType::I48 => count * 6,
            _ => count * dtype.element_size_bytes(),
        };
        if bytes.len() < needed {
            return Err(Error::Deserialize(format!(
                "{dtype:?} tensor of {count} elements needs {needed} bytes, got {}",
                bytes.len()
            )));
        }
        let bytes = &bytes[..needed];
        Ok(match dtype {
            DType::Bool => TensorData::Bool(bytes.iter().map(|&b| b != 0).collect()),
            DType::I4 => TensorData::I4(
                (0..count)
                    .map(|i| {
                        let nibble = (bytes[i / 2] >> (4 * (i % 2))) & 0xF;
                        ((nibble << 4) as i8) >> 4
                    })
                    .collect(),
            ),
            DType::I8 => TensorData::I8(bytes.iter().map(|&b| b as i8).collect()),
            DType::Fp8E4M3 => TensorData::Fp8E4M3(bytes.iter().map(|&b| Fp8E4M3(b)).collect()),
            DType::Fp8E5M2 => TensorData::Fp8E5M2(bytes.iter().map(|&b| Fp8E5M2(b)).collect()),
            DType::I16 => {
                let data: Vec<i16> = bytes.as_chunks::<2>().0.iter().map(|c| i16::from_le_bytes(*c)).collect();
                TensorData::I16(data)
            }
            DType::I32 => {
                let data: Vec<i32> = bytes.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect();
                TensorData::I32(data)
            }
            DType::Shape => {
                // SHAPE dtype stores i64 values (8 bytes each)
                let data: Vec<i64> = bytes.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c)).collect();
                TensorData::I64(data)
            }
            DType::I48 => {
                let data: Vec<i64> = bytes
                    .as_chunks::<6>()
                    .0
                    .iter()
                    .map(|c| {
                        let mut buf = [0u8; 8];
                        buf[..6].copy_from_slice(c);
                        // sign extend from 48 bits
                        if buf[5] & 0x80 != 0 {
                            buf[6] = 0xFF;
                            buf[7] = 0xFF;
                        }
                        i64::from_le_bytes(buf)
                    })
                    .collect();
                TensorData::I48(data)
            }
            DType::I64 => {
                let data: Vec<i64> = bytes.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c)).collect();
                TensorData::I64(data)
            }
            DType::F32 => {
                let data: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
                TensorData::F32(data)
            }
            DType::F16 => {
                let data: Vec<half::f16> =
                    bytes.as_chunks::<2>().0.iter().map(|c| half::f16::from_le_bytes(*c)).collect();
                TensorData::F16(data)
            }
            DType::Bf16 => {
                let data: Vec<half::bf16> =
                    bytes.as_chunks::<2>().0.iter().map(|c| half::bf16::from_le_bytes(*c)).collect();
                TensorData::Bf16(data)
            }
            DType::F64 => {
                let data: Vec<f64> = bytes.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c)).collect();
                TensorData::F64(data)
            }
            _ => return Err(Error::DTypeMismatch(format!("cannot deserialize raw bytes for {:?}", dtype))),
        })
    }

    /// Encode the elements as serialized (little-endian) bytes, the inverse
    /// of [`TensorData::from_raw_bytes`]: INT4 packed two per byte, low
    /// nibble first; INT48 as 6 bytes per element.
    pub fn to_raw_bytes(&self) -> Vec<u8> {
        match self {
            TensorData::Bool(v) => v.iter().map(|&b| b as u8).collect(),
            TensorData::I4(v) => v
                .chunks(2)
                .map(|pair| {
                    let lo = (pair[0] as u8) & 0xF;
                    let hi = pair.get(1).map_or(0, |&h| (h as u8) & 0xF);
                    lo | (hi << 4)
                })
                .collect(),
            TensorData::I8(v) => v.iter().map(|&x| x as u8).collect(),
            TensorData::I16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorData::I32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorData::I48(v) => v.iter().flat_map(|x| x.to_le_bytes()[..6].to_vec()).collect(),
            TensorData::I64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorData::Fp8E4M3(v) => v.iter().map(|x| x.0).collect(),
            TensorData::Fp8E5M2(v) => v.iter().map(|x| x.0).collect(),
            TensorData::F16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorData::Bf16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorData::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorData::F64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }

    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            TensorData::F32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_i32_slice(&self) -> Option<&[i32]> {
        match self {
            TensorData::I32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_bool_slice(&self) -> Option<&[bool]> {
        match self {
            TensorData::Bool(v) => Some(v),
            _ => None,
        }
    }

    /// Elements `self[indices[0]], self[indices[1]], ...`, for any dtype.
    pub fn gather(&self, indices: &[usize]) -> TensorData {
        macro_rules! pick {
            ($($variant:ident),*) => {
                match self {
                    $(TensorData::$variant(v) => TensorData::$variant(indices.iter().map(|&i| v[i]).collect()),)*
                }
            };
        }
        pick!(Bool, I4, I8, I16, I32, I48, I64, Fp8E4M3, Fp8E5M2, F16, Bf16, F32, F64)
    }

    /// The elements of `parts`, which must share a dtype, one after another.
    pub fn concat(parts: &[&TensorData]) -> Result<TensorData> {
        let first = parts.first().ok_or_else(|| Error::GraphError("concat of no tensors".into()))?;
        if let Some(p) = parts.iter().find(|p| p.dtype() != first.dtype()) {
            return Err(Error::DTypeMismatch(format!("cannot concatenate {:?} with {:?}", first.dtype(), p.dtype())));
        }
        macro_rules! join {
            ($($variant:ident),*) => {
                match first {
                    $(TensorData::$variant(_) => TensorData::$variant(
                        parts
                            .iter()
                            .flat_map(|p| match p {
                                TensorData::$variant(v) => v.iter().copied(),
                                _ => unreachable!("dtypes checked above"),
                            })
                            .collect(),
                    ),)*
                }
            };
        }
        Ok(join!(Bool, I4, I8, I16, I32, I48, I64, Fp8E4M3, Fp8E5M2, F16, Bf16, F32, F64))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: TensorData,
}

impl Tensor {
    pub fn new(name: String, shape: Vec<usize>, data: TensorData) -> Self {
        Tensor { name, shape, data }
    }

    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }

    /// Number of elements (0 for a tensor with a zero-sized dimension).
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size_bytes(&self) -> usize {
        self.element_count() * self.dtype().element_size_bytes()
    }
}

/// The element count of `shape`, or an error if it overflows `usize`.
pub fn checked_element_count(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| Error::TensorTooLarge(format!("element count of shape {shape:?} overflows")))
}

pub fn shape_from_ir(ir_shape: &[i32]) -> Vec<usize> {
    ir_shape.iter().map(|&d| d.max(0) as usize).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float8::{Fp8E4M3, Fp8E5M2};

    #[test]
    fn raw_bytes_round_trip_every_dtype() {
        let cases = vec![
            (DType::Bool, TensorData::Bool(vec![true, false, true])),
            (DType::I4, TensorData::I4(vec![-8, 7, 3])),
            (DType::I8, TensorData::I8(vec![-128, 0, 127])),
            (DType::I16, TensorData::I16(vec![-32768, 1, 32767])),
            (DType::I32, TensorData::I32(vec![i32::MIN, 2, i32::MAX])),
            (DType::I48, TensorData::I48(vec![-(1 << 47), 3, (1 << 47) - 1])),
            (DType::I64, TensorData::I64(vec![i64::MIN, 4, i64::MAX])),
            (DType::Shape, TensorData::I64(vec![1, 2, 3])),
            (DType::Fp8E4M3, TensorData::Fp8E4M3(vec![Fp8E4M3(0x7e), Fp8E4M3(1), Fp8E4M3(0x80)])),
            (DType::Fp8E5M2, TensorData::Fp8E5M2(vec![Fp8E5M2(0x7c), Fp8E5M2(1), Fp8E5M2(0xfd)])),
            (DType::F16, TensorData::F16(vec![half::f16::from_f32(1.5); 3])),
            (DType::Bf16, TensorData::Bf16(vec![half::bf16::from_f32(-2.25); 3])),
            (DType::F32, TensorData::F32(vec![1.0, f32::NEG_INFINITY, -0.0])),
            (DType::F64, TensorData::F64(vec![1.0, 2.0, 3.0])),
        ];
        for (dtype, data) in cases {
            let bytes = data.to_raw_bytes();
            let back = TensorData::from_raw_bytes(dtype, &bytes, 3).unwrap();
            assert_eq!(format!("{back:?}"), format!("{data:?}"), "{dtype:?}");
        }
    }
}
