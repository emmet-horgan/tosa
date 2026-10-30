//! Accumulator arithmetic for dot-product style operators (convolutions,
//! matmul), emulating the reference model bit for bit.
//!
//! The reference model accumulates in the C++ type selected by `acc_type`:
//! - INT32: `int32_t`, wrapping on overflow.
//! - INT48: `int64_t` (the result is clamped to 48 bits by the caller).
//! - FP32: `float`.
//! - FP16 / BF16: a software float (`ct::cfloat_advanced`) with no arithmetic
//!   operators of its own: values are combined in f32 and rounded back to the
//!   accumulator type on assignment. `acc += a * b` (convolutions) therefore
//!   rounds the product before adding it, while `acc = acc + a * b` (MATMUL)
//!   does not.
//!
//! Operands are converted to the accumulator type before multiplying. For
//! every legal type combination that conversion is exact.

use crate::ir::DType;
use crate::ops::common::round_to;

pub trait Accumulator: Copy {
    /// Operand type after conversion into the accumulator domain.
    type Operand: Copy;

    fn zero() -> Self;
    fn mac(self, a: Self::Operand, b: Self::Operand) -> Self;
}

#[derive(Clone, Copy)]
pub struct AccI32(pub i32);

impl Accumulator for AccI32 {
    type Operand = i32;

    fn zero() -> Self {
        AccI32(0)
    }

    fn mac(self, a: i32, b: i32) -> Self {
        AccI32(self.0.wrapping_add(a.wrapping_mul(b)))
    }
}

#[derive(Clone, Copy)]
pub struct AccI64(pub i64);

impl Accumulator for AccI64 {
    type Operand = i64;

    fn zero() -> Self {
        AccI64(0)
    }

    fn mac(self, a: i64, b: i64) -> Self {
        AccI64(self.0.wrapping_add(a.wrapping_mul(b)))
    }
}

#[derive(Clone, Copy)]
pub struct AccF32(pub f32);

impl Accumulator for AccF32 {
    type Operand = f32;

    fn zero() -> Self {
        AccF32(0.0)
    }

    fn mac(self, a: f32, b: f32) -> Self {
        AccF32(self.0 + a * b)
    }
}

/// FP16 accumulator, held as an f32 that is always exactly representable in f16.
#[derive(Clone, Copy)]
pub struct AccF16(pub f32);

impl Accumulator for AccF16 {
    type Operand = f32;

    fn zero() -> Self {
        AccF16(0.0)
    }

    fn mac(self, a: f32, b: f32) -> Self {
        let product = round_to(DType::F16, a * b);
        AccF16(round_to(DType::F16, self.0 + product))
    }
}

/// FP16 accumulator for `acc = acc + a * b` (as opposed to `acc += a * b`):
/// the product stays in f32 and only the sum is rounded to f16. The
/// reference model's MATMUL accumulates this way.
#[derive(Clone, Copy)]
pub struct AccF16UnroundedProduct(pub f32);

impl Accumulator for AccF16UnroundedProduct {
    type Operand = f32;

    fn zero() -> Self {
        AccF16UnroundedProduct(0.0)
    }

    fn mac(self, a: f32, b: f32) -> Self {
        AccF16UnroundedProduct(round_to(DType::F16, self.0 + a * b))
    }
}

/// BF16 counterpart of [`AccF16UnroundedProduct`] (MATMUL_T with BF16 output).
#[derive(Clone, Copy)]
pub struct AccBf16UnroundedProduct(pub f32);

impl Accumulator for AccBf16UnroundedProduct {
    type Operand = f32;

    fn zero() -> Self {
        AccBf16UnroundedProduct(0.0)
    }

    fn mac(self, a: f32, b: f32) -> Self {
        AccBf16UnroundedProduct(round_to(DType::Bf16, self.0 + a * b))
    }
}

/// BF16 accumulator, held as an f32 that is always exactly representable in bf16.
#[derive(Clone, Copy)]
pub struct AccBf16(pub f32);

impl Accumulator for AccBf16 {
    type Operand = f32;

    fn zero() -> Self {
        AccBf16(0.0)
    }

    fn mac(self, a: f32, b: f32) -> Self {
        let product = round_to(DType::Bf16, a * b);
        AccBf16(round_to(DType::Bf16, self.0 + product))
    }
}
