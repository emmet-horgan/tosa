//! 8-bit floating point formats used by TOSA.
//!
//! - `Fp8E4M3`: 1 sign, 4 exponent, 3 mantissa bits, bias 7. No infinity;
//!   NaN is `S.1111.111`. Largest finite value is 448. (OCP "E4M3FN".)
//! - `Fp8E5M2`: 1 sign, 5 exponent, 2 mantissa bits, bias 15. IEEE-754 style
//!   with infinities and NaNs.
//!
//! Both support subnormals. Conversion from f32 rounds to nearest, ties to
//! even. Values that overflow after rounding become infinity (E5M2) or NaN
//! (E4M3), matching the reference model's default `cfloat_cast` behaviour.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Fp8E4M3(pub u8);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Fp8E5M2(pub u8);

struct Format {
    exp_bits: u32,
    man_bits: u32,
    bias: i32,
    has_inf: bool,
}

const E4M3: Format = Format { exp_bits: 4, man_bits: 3, bias: 7, has_inf: false };
const E5M2: Format = Format { exp_bits: 5, man_bits: 2, bias: 15, has_inf: true };

impl Format {
    fn max_exp_field(&self) -> u32 {
        (1 << self.exp_bits) - 1
    }

    fn man_mask(&self) -> u32 {
        (1 << self.man_bits) - 1
    }

    fn is_nan(&self, bits: u8) -> bool {
        let exp = (bits as u32 >> self.man_bits) & self.max_exp_field();
        let man = bits as u32 & self.man_mask();
        if self.has_inf {
            exp == self.max_exp_field() && man != 0
        } else {
            exp == self.max_exp_field() && man == self.man_mask()
        }
    }

    fn nan(&self) -> u8 {
        // Positive canonical quiet NaN.
        ((self.max_exp_field() << self.man_bits) | self.man_mask()) as u8
    }

    fn decode(&self, bits: u8) -> f32 {
        let sign = if bits & 0x80 != 0 { -1.0f64 } else { 1.0f64 };
        let exp = (bits as u32 >> self.man_bits) & self.max_exp_field();
        let man = (bits as u32 & self.man_mask()) as f64;
        if self.is_nan(bits) {
            return f32::NAN;
        }
        if self.has_inf && exp == self.max_exp_field() {
            return (sign * f64::INFINITY) as f32;
        }
        let scale = (1u64 << self.man_bits) as f64;
        let mag = if exp == 0 {
            man / scale * 2f64.powi(1 - self.bias)
        } else {
            (1.0 + man / scale) * 2f64.powi(exp as i32 - self.bias)
        };
        (sign * mag) as f32
    }

    fn encode(&self, x: f32) -> u8 {
        if x.is_nan() {
            return self.nan();
        }
        let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0 };
        let overflow = if self.has_inf {
            sign | (self.max_exp_field() << self.man_bits) as u8
        } else {
            sign | self.nan()
        };
        if x.is_infinite() {
            return overflow;
        }
        let a = (x as f64).abs();
        if a == 0.0 {
            return sign;
        }

        let man_scale = (1u64 << self.man_bits) as f64;
        let min_normal_exp = 1 - self.bias;

        // Unbiased exponent of `a` (exact: f64 holds every f32 exactly).
        let e = ((a.to_bits() >> 52) & 0x7ff) as i32 - 1023;
        let (exp_field, man) = if e < min_normal_exp {
            // Subnormal range: quantise to multiples of the smallest subnormal.
            // A carry into 2^man_bits yields the smallest normal, which the
            // bit layout encodes correctly as exp_field=1, man=0.
            let q = (a / 2f64.powi(min_normal_exp) * man_scale).round_ties_even() as u32;
            (q >> self.man_bits, q & self.man_mask())
        } else {
            let mut e = e;
            let mut q = (a / 2f64.powi(e) * man_scale).round_ties_even() as u32;
            if q == 2 << self.man_bits {
                e += 1;
                q = 1 << self.man_bits;
            }
            ((e + self.bias) as u32, q & self.man_mask())
        };

        let finite_limit_exceeded = if self.has_inf {
            exp_field >= self.max_exp_field()
        } else {
            exp_field > self.max_exp_field()
                || (exp_field == self.max_exp_field() && man == self.man_mask())
        };
        if finite_limit_exceeded {
            return overflow;
        }
        sign | ((exp_field << self.man_bits) | man) as u8
    }
}

macro_rules! impl_fp8 {
    ($ty:ident, $fmt:expr) => {
        impl $ty {
            pub const ZERO: $ty = $ty(0);

            pub fn from_bits(bits: u8) -> Self {
                $ty(bits)
            }

            pub fn to_bits(self) -> u8 {
                self.0
            }

            pub fn to_f32(self) -> f32 {
                $fmt.decode(self.0)
            }

            pub fn from_f32(x: f32) -> Self {
                $ty($fmt.encode(x))
            }

            pub fn is_nan(self) -> bool {
                $fmt.is_nan(self.0)
            }
        }
    };
}

impl_fp8!(Fp8E4M3, E4M3);
impl_fp8!(Fp8E5M2, E5M2);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_known_values() {
        assert_eq!(Fp8E4M3(0x7e).to_f32(), 448.0);
        assert_eq!(Fp8E4M3(0x01).to_f32(), 2f32.powi(-9));
        assert_eq!(Fp8E4M3(0x08).to_f32(), 2f32.powi(-6));
        assert_eq!(Fp8E4M3(0x38).to_f32(), 1.0);
        assert!(Fp8E4M3(0x7f).to_f32().is_nan());
        assert!(Fp8E4M3(0xff).to_f32().is_nan());
        assert_eq!(Fp8E4M3(0x78).to_f32(), 256.0);
        assert!(Fp8E4M3::from_f32(f32::INFINITY).is_nan());
        assert!(Fp8E4M3::from_f32(480.0).is_nan());
        assert_eq!(Fp8E4M3::from_f32(464.0), Fp8E4M3(0x7e)); // tie -> even (448)
    }

    #[test]
    fn e5m2_known_values() {
        assert_eq!(Fp8E5M2(0x7b).to_f32(), 57344.0);
        assert_eq!(Fp8E5M2(0x7c).to_f32(), f32::INFINITY);
        assert_eq!(Fp8E5M2(0xfc).to_f32(), f32::NEG_INFINITY);
        assert!(Fp8E5M2(0x7d).to_f32().is_nan());
        assert_eq!(Fp8E5M2(0x01).to_f32(), 2f32.powi(-16));
        assert_eq!(Fp8E5M2(0x3c).to_f32(), 1.0);
        assert_eq!(Fp8E5M2::from_f32(61440.0), Fp8E5M2(0x7c)); // tie rounds up to inf
        assert_eq!(Fp8E5M2::from_f32(60000.0), Fp8E5M2(0x7b));
    }

    fn check_all<F: Fn(u8) -> f32, G: Fn(f32) -> u8>(decode: F, encode: G, is_nan: fn(u8) -> bool) {
        // Every non-NaN pattern round-trips exactly.
        for b in 0..=255u8 {
            if is_nan(b) {
                assert!(decode(b).is_nan());
                continue;
            }
            assert_eq!(encode(decode(b)), b, "roundtrip of {b:#04x}");
        }
        // Midpoints between adjacent positive finite values round to the
        // value with an even mantissa; points just either side round to
        // the nearer neighbour.
        for b in 0..0x7fu8 {
            let (lo, hi) = (decode(b), decode(b + 1));
            if is_nan(b + 1) || !hi.is_finite() {
                continue;
            }
            let mid = ((lo as f64 + hi as f64) / 2.0) as f32;
            let even = if b & 1 == 0 { b } else { b + 1 };
            assert_eq!(encode(mid), even, "tie between {b:#04x} and {:#04x}", b + 1);
            assert_eq!(encode(-mid), even | 0x80);
            let below = f32::from_bits(mid.to_bits() - 1);
            let above = f32::from_bits(mid.to_bits() + 1);
            assert_eq!(encode(below), b);
            assert_eq!(encode(above), b + 1);
        }
    }

    #[test]
    fn e4m3_exhaustive() {
        check_all(|b| Fp8E4M3(b).to_f32(), |x| Fp8E4M3::from_f32(x).0, |b| Fp8E4M3(b).is_nan());
    }

    #[test]
    fn e5m2_exhaustive() {
        check_all(|b| Fp8E5M2(b).to_f32(), |x| Fp8E5M2::from_f32(x).0, |b| Fp8E5M2(b).is_nan());
    }
}
