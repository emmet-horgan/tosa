//! Bit-exact ports of the math routines the reference model bundles, so that
//! results do not depend on the platform's libm.

/// Single-precision e^x: a port of the musl / Arm optimized-routines `expf`
/// the reference model uses for EXP (`reference_model/src/ops/math/expf.c`).
///
/// Copyright (c) 2017-2018, 2025-2026 Arm Limited. SPDX-License-Identifier: MIT
pub fn expf(x: f32) -> f32 {
    const TABLE_BITS: u32 = 5;
    const N: u64 = 1 << TABLE_BITS;
    // tab[i] = uint(2^(i/N)) - (i << 52-BITS)
    const TAB: [u64; N as usize] = [
        0x3ff0000000000000,
        0x3fefd9b0d3158574,
        0x3fefb5586cf9890f,
        0x3fef9301d0125b51,
        0x3fef72b83c7d517b,
        0x3fef54873168b9aa,
        0x3fef387a6e756238,
        0x3fef1e9df51fdee1,
        0x3fef06fe0a31b715,
        0x3feef1a7373aa9cb,
        0x3feedea64c123422,
        0x3feece086061892d,
        0x3feebfdad5362a27,
        0x3feeb42b569d4f82,
        0x3feeab07dd485429,
        0x3feea47eb03a5585,
        0x3feea09e667f3bcd,
        0x3fee9f75e8ec5f74,
        0x3feea11473eb0187,
        0x3feea589994cce13,
        0x3feeace5422aa0db,
        0x3feeb737b0cdc5e5,
        0x3feec49182a3f090,
        0x3feed503b23e255d,
        0x3feee89f995ad3ad,
        0x3feeff76f2fb5e47,
        0x3fef199bdd85529c,
        0x3fef3720dcef9069,
        0x3fef5818dcfba487,
        0x3fef7c97337b9b5f,
        0x3fefa4afa2a490da,
        0x3fefd0765b6e4540,
    ];
    let shift = f64::from_bits(0x4338000000000000); // 0x1.8p+52
    let inv_ln2_n = f64::from_bits(0x40471547652b82fe); // 0x1.71547652b82fep+0 * N
    let c = [
        f64::from_bits(0x3ebc6af84b912394), // 0x1.c6af84b912394p-5 / N^3
        f64::from_bits(0x3f2ebfce50fac4f3), // 0x1.ebfce50fac4f3p-3 / N^2
        f64::from_bits(0x3f962e42ff0c52d6), // 0x1.62e42ff0c52d6p-1 / N
    ];

    let top12 = |v: f32| v.to_bits() >> 20;
    let abstop = top12(x) & 0x7ff;
    if abstop >= top12(88.0) {
        // |x| >= 88 or x is NaN.
        if x == f32::NEG_INFINITY {
            return 0.0;
        }
        if abstop >= top12(f32::INFINITY) {
            return x + x;
        }
        if x > f32::from_bits(0x42b17217) {
            return f32::INFINITY; // overflow: x > log(0x1p128)
        }
        if x < f32::from_bits(0xc2cff1b4) {
            return 0.0; // underflow: x < log(0x1p-150)
        }
    }

    // x*N/Ln2 = k + r with r in [-1/2, 1/2] and integer k.
    let z = inv_ln2_n * x as f64;
    let mut kd = z + shift;
    let ki = kd.to_bits();
    kd -= shift;
    let r = z - kd;

    // exp(x) = 2^(k/N) * 2^(r/N) ~= s * (C0*r^3 + C1*r^2 + C2*r + 1)
    let t = TAB[(ki % N) as usize].wrapping_add(ki << (52 - TABLE_BITS));
    let s = f64::from_bits(t);
    let z = c[0] * r + c[1];
    let r2 = r * r;
    let y = c[2] * r + 1.0;
    let y = z * r2 + y;
    (y * s) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expf_matches_std_closely() {
        for i in -2000..2000 {
            let x = i as f32 * 0.05;
            let (a, b) = (expf(x), (x as f64).exp() as f32);
            let ulps = (a.to_bits() as i64 - b.to_bits() as i64).abs();
            assert!(ulps <= 1, "expf({x}) = {a}, expected ~{b}");
        }
        assert_eq!(expf(0.0), 1.0);
        assert_eq!(expf(f32::NEG_INFINITY), 0.0);
        assert_eq!(expf(f32::INFINITY), f32::INFINITY);
        assert!(expf(f32::NAN).is_nan());
        assert_eq!(expf(100.0), f32::INFINITY);
        assert_eq!(expf(-110.0), 0.0);
    }
}
