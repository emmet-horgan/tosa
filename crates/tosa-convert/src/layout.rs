//! Tensor layout helpers. TOSA is channels-last (NHWC for 4-D activations,
//! `[OC, KH, KW, IC]` for CONV2D weights, `[KH, KW, C, M]` for depthwise
//! weights); frontends whose models are channels-first insert TRANSPOSE
//! operators, which the cleanup passes cancel or fold where possible.

use tosa_ir::{Op, OpAttribute, TensorRef};

use crate::lower::LowerCtx;

/// Activations: NCHW -> NHWC.
pub const NCHW_TO_NHWC: [i32; 4] = [0, 2, 3, 1];
/// Activations: NHWC -> NCHW.
pub const NHWC_TO_NCHW: [i32; 4] = [0, 3, 1, 2];
/// Convolution weights: `[O, I, H, W]` -> `[O, H, W, I]`.
pub const OIHW_TO_OHWI: [i32; 4] = [0, 2, 3, 1];
/// Convolution weights: `[H, W, I, O]` -> `[O, H, W, I]`.
pub const HWIO_TO_OHWI: [i32; 4] = [3, 0, 1, 2];
// Depthwise weights as TFLite stores them, `[1, H, W, C*M]`, become TOSA's
// `[H, W, C, M]` by RESHAPE, not transpose; see the frontends.

/// `shape` permuted by `perms`: `out[i] = shape[perms[i]]`.
pub fn permute_shape(shape: &[i32], perms: &[i32]) -> Vec<i32> {
    perms.iter().map(|&p| shape[p as usize]).collect()
}

/// The permutation that undoes `perms`.
pub fn inverse_perms(perms: &[i32]) -> Vec<i32> {
    let mut inv = vec![0; perms.len()];
    for (i, &p) in perms.iter().enumerate() {
        inv[p as usize] = i as i32;
    }
    inv
}

/// `first` followed by `second`, as one permutation.
pub fn compose_perms(first: &[i32], second: &[i32]) -> Vec<i32> {
    second.iter().map(|&p| first[p as usize]).collect()
}

pub fn is_identity(perms: &[i32]) -> bool {
    perms.iter().enumerate().all(|(i, &p)| p as usize == i)
}

/// Emit `TRANSPOSE(input, perms)`.
pub fn transpose(ctx: &mut LowerCtx<'_>, input: &TensorRef, perms: &[i32], hint: &str) -> TensorRef {
    let shape = permute_shape(&input.shape, perms);
    ctx.emit_temp(Op::Transpose, OpAttribute::Transpose { perms: perms.to_vec() }, &[input], input.dtype, &shape, hint)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutations_compose_and_invert() {
        assert_eq!(permute_shape(&[1, 3, 8, 9], &NCHW_TO_NHWC), vec![1, 8, 9, 3]);
        assert_eq!(inverse_perms(&NCHW_TO_NHWC), NHWC_TO_NCHW.to_vec());
        assert!(is_identity(&compose_perms(&NCHW_TO_NHWC, &NHWC_TO_NCHW)));
        assert_eq!(compose_perms(&[1, 0, 2], &[2, 0, 1]), vec![2, 1, 0]);
        let shape = [2, 3, 4];
        let once = permute_shape(&shape, &[1, 0, 2]);
        let twice = permute_shape(&once, &[2, 0, 1]);
        assert_eq!(twice, permute_shape(&shape, &compose_perms(&[1, 0, 2], &[2, 0, 1])));
    }
}
