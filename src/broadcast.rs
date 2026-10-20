use crate::error::{Error, Result};

pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let max_rank = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_rank);

    let a_iter = a.iter().rev();
    let b_iter = b.iter().rev();

    for (da, db) in
        a_iter.copied().chain(std::iter::repeat(1)).zip(b_iter.copied().chain(std::iter::repeat(1))).take(max_rank)
    {
        if da == db {
            result.push(da);
        } else if da == 1 {
            result.push(db);
        } else if db == 1 {
            result.push(da);
        } else {
            return Err(Error::Broadcast(format!("shapes {:?} and {:?} are not broadcast-compatible", a, b)));
        }
    }

    result.reverse();
    Ok(result)
}

pub fn broadcast_index(index: &[usize], source_shape: &[usize], out_shape: &[usize]) -> usize {
    let rank_diff = out_shape.len() - source_shape.len();
    let mut flat = 0usize;
    let mut stride = 1usize;

    for dim in (0..source_shape.len()).rev() {
        let src_dim = source_shape[dim];
        let out_idx = index[dim + rank_diff];
        let idx = if src_dim == 1 { 0 } else { out_idx };
        flat += idx * stride;
        stride *= src_dim;
    }
    flat
}

pub fn flat_to_nd(flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    let mut remaining = flat;
    for dim in (0..shape.len()).rev() {
        if shape[dim] > 0 {
            indices[dim] = remaining % shape[dim];
            remaining /= shape[dim];
        }
    }
    indices
}

pub fn nd_to_flat(indices: &[usize], shape: &[usize]) -> usize {
    let mut flat = 0;
    let mut stride = 1;
    for dim in (0..shape.len()).rev() {
        flat += indices[dim] * stride;
        stride *= shape[dim];
    }
    flat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_same_shape() {
        assert_eq!(broadcast_shapes(&[3, 4], &[3, 4]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn test_broadcast_scalar() {
        assert_eq!(broadcast_shapes(&[3, 4], &[1]).unwrap(), vec![3, 4]);
        assert_eq!(broadcast_shapes(&[1], &[3, 4]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn test_broadcast_different_ranks() {
        assert_eq!(broadcast_shapes(&[1, 3, 1], &[4, 1, 5]).unwrap(), vec![4, 3, 5]);
    }

    #[test]
    fn test_broadcast_incompatible() {
        assert!(broadcast_shapes(&[3, 4], &[3, 5]).is_err());
    }

    #[test]
    fn test_flat_to_nd_round_trip() {
        let shape = [2, 3, 4];
        for flat in 0..24 {
            let nd = flat_to_nd(flat, &shape);
            assert_eq!(nd_to_flat(&nd, &shape), flat);
        }
    }
}
