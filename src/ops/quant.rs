//! Fixed-point scaling helpers, matching the reference model's `QuantUtil`.

use crate::error::{Error, Result};

/// Multiplier and shift approximating `1 / value` (AVG_POOL2D).
pub fn reciprocal_scale(value: i32) -> (i32, i32) {
    debug_assert!(value > 0);
    let k = 32 - ((value as u32) - 1).leading_zeros() as i32; // (1<<k)/2 < value <= (1<<k)
    let numerator = ((1i64 << 30) + 1) << k;
    ((numerator / value as i64) as i32, 30 + k)
}

/// `round(value * multiplier / 2^shift)`; with `double_round`, a second
/// rounding step is applied for shifts above 31, as TOSA's double rounding.
///
/// Errors where the reference model reports a REQUIRE failure.
pub fn apply_scale_32(value: i32, multiplier: i32, shift: i32, double_round: bool) -> Result<i32> {
    if multiplier < 0 || !(2..=62).contains(&shift) {
        return Err(Error::GraphError(format!(
            "apply_scale_32: invalid multiplier {multiplier} or shift {shift}"
        )));
    }
    let (low, high) = (-1i64 << (shift - 1), 1i64 << (shift - 1));
    let value = value as i64;
    if value < low || value >= high {
        return Err(Error::GraphError(format!(
            "apply_scale_32: value {value} out of range for shift {shift}"
        )));
    }
    let mut round = 1i64 << (shift - 1);
    if double_round && shift > 31 {
        round += if value >= 0 { 1 << 30 } else { -(1 << 30) };
    }
    let result = (value * multiplier as i64 + round) >> shift;
    i32::try_from(result).map_err(|_| Error::GraphError("apply_scale_32: result exceeds int32".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reciprocal_scale_divides() {
        for count in 1..=64 {
            let (m, s) = reciprocal_scale(count);
            for sum in [-1000, -7, 0, 5, 999] {
                let q = apply_scale_32(sum, m, s, false).unwrap();
                let exact = sum as f64 / count as f64;
                assert!((q as f64 - exact).abs() <= 0.5 + 1e-9, "{sum}/{count}: {q}");
            }
        }
    }
}
