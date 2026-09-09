use crate::error::{Error, Result};
use crate::ir::DType;

#[derive(Debug, Clone)]
pub enum TensorData {
    Bool(Vec<bool>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I48(Vec<i64>),
    I64(Vec<i64>),
    F16(Vec<half::f16>),
    Bf16(Vec<half::bf16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl TensorData {
    pub fn dtype(&self) -> DType {
        match self {
            TensorData::Bool(_) => DType::Bool,
            TensorData::I8(_) => DType::I8,
            TensorData::I16(_) => DType::I16,
            TensorData::I32(_) => DType::I32,
            TensorData::I48(_) => DType::I48,
            TensorData::I64(_) => DType::I64,
            TensorData::F16(_) => DType::F16,
            TensorData::Bf16(_) => DType::Bf16,
            TensorData::F32(_) => DType::F32,
            TensorData::F64(_) => DType::F64,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            TensorData::Bool(v) => v.len(),
            TensorData::I8(v) => v.len(),
            TensorData::I16(v) => v.len(),
            TensorData::I32(v) => v.len(),
            TensorData::I48(v) => v.len(),
            TensorData::I64(v) => v.len(),
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
            DType::I8 => TensorData::I8(vec![0; count]),
            DType::I16 => TensorData::I16(vec![0; count]),
            DType::I32 => TensorData::I32(vec![0; count]),
            DType::I48 => TensorData::I48(vec![0; count]),
            DType::I64 => TensorData::I64(vec![0; count]),
            DType::F16 => TensorData::F16(vec![half::f16::ZERO; count]),
            DType::Bf16 => TensorData::Bf16(vec![half::bf16::ZERO; count]),
            DType::F32 => TensorData::F32(vec![0.0; count]),
            DType::F64 => TensorData::F64(vec![0.0; count]),
            _ => return Err(Error::DTypeMismatch(format!("cannot create zeros for {:?}", dtype))),
        })
    }

    pub fn from_raw_bytes(dtype: DType, bytes: &[u8]) -> Result<Self> {
        Ok(match dtype {
            DType::Bool => {
                TensorData::Bool(bytes.iter().map(|&b| b != 0).collect())
            }
            DType::I8 => {
                TensorData::I8(bytes.iter().map(|&b| b as i8).collect())
            }
            DType::I16 => {
                let data: Vec<i16> = bytes
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]))
                    .collect();
                TensorData::I16(data)
            }
            DType::I32 => {
                let data: Vec<i32> = bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                TensorData::I32(data)
            }
            DType::Shape => {
                // SHAPE dtype stores i64 values (8 bytes each)
                let data: Vec<i64> = bytes
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                TensorData::I64(data)
            }
            DType::I48 => {
                let data: Vec<i64> = bytes
                    .chunks_exact(6)
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
                let data: Vec<i64> = bytes
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                TensorData::I64(data)
            }
            DType::F32 => {
                let data: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                TensorData::F32(data)
            }
            DType::F16 => {
                let data: Vec<half::f16> = bytes
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
                    .collect();
                TensorData::F16(data)
            }
            DType::Bf16 => {
                let data: Vec<half::bf16> = bytes
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_le_bytes([c[0], c[1]]))
                    .collect();
                TensorData::Bf16(data)
            }
            DType::F64 => {
                let data: Vec<f64> = bytes
                    .chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                TensorData::F64(data)
            }
            _ => {
                return Err(Error::DTypeMismatch(format!(
                    "cannot deserialize raw bytes for {:?}",
                    dtype
                )))
            }
        })
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
}

#[derive(Debug, Clone)]
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

    pub fn element_count(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }

    pub fn size_bytes(&self) -> usize {
        self.element_count() * self.dtype().element_size_bytes()
    }
}

pub fn shape_from_ir(ir_shape: &[i32]) -> Vec<usize> {
    ir_shape.iter().map(|&d| d.max(0) as usize).collect()
}
