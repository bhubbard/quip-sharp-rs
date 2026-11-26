//! Fast Walsh-Hadamard Transform (FWHT) and Randomized Hadamard Transform (RHT).

use crate::{Matrix, QuipError};

/// Direction of a Randomized Hadamard Transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RhtDirection {
    /// Forward incoherence transform (`W_rot = V * W * U^T`).
    Forward,
    /// Inverse transform (`W = V^T * W_rot * U`).
    Inverse,
}

/// Applies an in-place unitary Fast Walsh-Hadamard Transform (`H_n / sqrt(n)`)
/// to a power-of-two slice.
///
/// # Errors
/// Returns [`QuipError::Dimension`] if `buf.len()` is zero or not a power of two.
pub fn fwht_unitary_inplace(buf: &mut [f64]) -> Result<(), QuipError> {
    let n = buf.len();
    if n == 0 || !n.is_power_of_two() {
        return Err(QuipError::Dimension(format!(
            "FWHT length must be a power of two, got {n}"
        )));
    }
    let mut h = 1;
    while h < n {
        for chunk in buf.chunks_exact_mut(h * 2) {
            let (left, right) = chunk.split_at_mut(h);
            for i in 0..h {
                let x = left[i];
                let y = right[i];
                left[i] = x + y;
                right[i] = x - y;
            }
        }
        h *= 2;
    }
    let scale = 1.0 / (n as f64).sqrt();
    for v in buf {
        *v *= scale;
    }
    Ok(())
}

/// Generates a deterministic Rademacher sign vector (`+1.0` or `-1.0`) from `seed`.
#[must_use]
pub fn rademacher_signs(len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if state & 1 == 0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// Applies the Randomized Hadamard Transform to a symmetric Hessian `H` (`n x n`):
/// `H_r = H_had * diag(su) * H * diag(su) * H_had^T`.
///
/// # Errors
/// Returns [`QuipError::Dimension`] if `h` is not square or `su.len() != h.rows`.
pub fn rht_hessian(h: &Matrix, su: &[f64]) -> Result<Matrix, QuipError> {
    if h.rows != h.cols || h.rows != su.len() {
        return Err(QuipError::Dimension(format!(
            "rht_hessian expected {}x{} matrix matching su len {}",
            su.len(),
            su.len(),
            su.len()
        )));
    }
    let n = h.rows;
    let mut out = h.clone();
    // Apply diag(su) and FWHT to rows
    for r in 0..n {
        let row = out.row_mut(r);
        for (c, val) in row.iter_mut().enumerate() {
            *val *= su[c];
        }
        fwht_unitary_inplace(row)?;
    }
    // Transpose, apply diag(su) and FWHT to columns, transpose back
    let mut trans = out.transpose();
    for r in 0..n {
        let row = trans.row_mut(r);
        for (c, val) in row.iter_mut().enumerate() {
            *val *= su[c];
        }
        fwht_unitary_inplace(row)?;
    }
    Ok(trans.transpose())
}

/// Applies the Randomized Hadamard Transform to weight matrix `W` (`m x n`)
/// using sign vectors `su` (length `n`) and `sv` (length `m`).
///
/// # Errors
/// Returns [`QuipError::Dimension`] if dimensions do not match or are not powers of two.
pub fn rht_weight(
    w: &Matrix,
    su: &[f64],
    sv: &[f64],
    direction: RhtDirection,
) -> Result<Matrix, QuipError> {
    if w.cols != su.len() || w.rows != sv.len() {
        return Err(QuipError::Dimension(format!(
            "rht_weight shape {}x{} incompatible with sv({}) and su({})",
            w.rows,
            w.cols,
            sv.len(),
            su.len()
        )));
    }
    match direction {
        RhtDirection::Forward => {
            let mut out = w.clone();
            for r in 0..out.rows {
                let row = out.row_mut(r);
                for (c, val) in row.iter_mut().enumerate() {
                    *val *= su[c];
                }
                fwht_unitary_inplace(row)?;
            }
            let mut trans = out.transpose();
            for r in 0..trans.rows {
                let row = trans.row_mut(r);
                for (c, val) in row.iter_mut().enumerate() {
                    *val *= sv[c];
                }
                fwht_unitary_inplace(row)?;
            }
            Ok(trans.transpose())
        }
        RhtDirection::Inverse => {
            // FWHT is symmetric & orthogonal: H_n^-1 = H_n, then undo signs.
            let mut out = w.clone();
            for r in 0..out.rows {
                let row = out.row_mut(r);
                fwht_unitary_inplace(row)?;
                for (c, val) in row.iter_mut().enumerate() {
                    *val *= su[c];
                }
            }
            let mut trans = out.transpose();
            for r in 0..trans.rows {
                let row = trans.row_mut(r);
                fwht_unitary_inplace(row)?;
                for (c, val) in row.iter_mut().enumerate() {
                    *val *= sv[c];
                }
            }
            Ok(trans.transpose())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rht_weight_round_trips_exactly() {
        let mut w = Matrix::zeros(8, 16);
        for i in 0..w.data.len() {
            w.data[i] = (i as f64 * 0.37).sin();
        }
        let su = rademacher_signs(16, 42);
        let sv = rademacher_signs(8, 99);
        let rot = rht_weight(&w, &su, &sv, RhtDirection::Forward).expect("forward RHT");
        let rec = rht_weight(&rot, &su, &sv, RhtDirection::Inverse).expect("inverse RHT");
        for (a, b) in w.data.iter().zip(rec.data.iter()) {
            assert!((a - b).abs() < 1e-11);
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    #[kani::unwind(5)]
    fn verify_rademacher_signs_strictly_unit() {
        let seed: u64 = kani::any();
        let signs = rademacher_signs(4, seed);
        assert_eq!(signs.len(), 4);
        for &s in &signs {
            assert!(s == 1.0 || s == -1.0);
        }
    }
}
