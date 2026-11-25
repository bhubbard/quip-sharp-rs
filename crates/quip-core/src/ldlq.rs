//! QuIP# LDLQ quantization and `QuantizedLinear` layer.

use crate::e8p::{E8pCodebook, E8P_OPT_SCALE};
use crate::hadamard::{rademacher_signs, rht_hessian, rht_weight, RhtDirection};
use crate::{Matrix, QuipError};

/// Configuration for QuIP# incoherence + LDLQ quantization.
#[derive(Debug, Clone, Copy)]
pub struct QuipConfig {
    /// Diagonal Hessian dampening factor (`sigma_reg`).
    pub sigma_reg: f64,
    /// Deterministic PRNG seed for Rademacher sign vectors `SU`, `SV`.
    pub seed: u64,
}

impl Default for QuipConfig {
    fn default() -> Self {
        Self {
            sigma_reg: 1e-2,
            seed: 1337,
        }
    }
}

/// 2-bit QuIP#-quantized linear layer (`W` of shape `out_features x in_features`).
#[derive(Debug, Clone)]
pub struct QuantizedLinear {
    /// Output feature dimension (`m`).
    pub out_features: usize,
    /// Input feature dimension (`n`).
    pub in_features: usize,
    /// Packed 16-bit E8P codebook indices (`out_features * (in_features / 8)`).
    pub q_idxs: Vec<u16>,
    /// Rademacher sign vector `SU` (length `in_features`).
    pub su: Vec<f64>,
    /// Rademacher sign vector `SV` (length `out_features`).
    pub sv: Vec<f64>,
    /// Scale factor applied to quantized weights.
    pub scale: f64,
}

impl QuantizedLinear {
    /// Dequantizes the 2-bit E8P weights back into an `m x n` float matrix by
    /// decoding each 8-weight block and applying the inverse RHT.
    ///
    /// # Errors
    /// Returns [`QuipError::Dimension`] if internal dimensions are invalid.
    pub fn dequantize(&self, codebook: &E8pCodebook) -> Result<Matrix, QuipError> {
        let mut rot_w = Matrix::zeros(self.out_features, self.in_features);
        let blocks_per_row = self.in_features / 8;
        for r in 0..self.out_features {
            for b in 0..blocks_per_row {
                let idx = self.q_idxs[r * blocks_per_row + b];
                let decoded = codebook.decode(idx);
                let row = rot_w.row_mut(r);
                for (k, &val) in decoded.iter().enumerate() {
                    row[b * 8 + k] = val * self.scale;
                }
            }
        }
        rht_weight(&rot_w, &self.su, &self.sv, RhtDirection::Inverse)
    }

    /// Computes `y = x @ W^T` for input batch `x` (`batch x in_features`).
    ///
    /// # Errors
    /// Returns [`QuipError::Dimension`] if `x.cols != self.in_features`.
    pub fn forward(&self, x: &Matrix, codebook: &E8pCodebook) -> Result<Matrix, QuipError> {
        let w = self.dequantize(codebook)?;
        x.matmul(&w.transpose())
    }
}

/// Quantizes weight matrix `W` (`m x n`) using Randomized Hadamard Transform
/// incoherence processing followed by Hessian-aware LDLQ over 8-wide E8P blocks.
///
/// # Errors
/// Returns [`QuipError`] if `m` or `n` are not powers of two (`>= 8`) or if
/// the Hessian is not positive-definite.
pub fn ldlq_quantize(
    w: &Matrix,
    hessian: &Matrix,
    codebook: &E8pCodebook,
    config: QuipConfig,
) -> Result<QuantizedLinear, QuipError> {
    if !w.rows.is_power_of_two() || !w.cols.is_power_of_two() || w.cols < 8 {
        return Err(QuipError::Dimension(format!(
            "ldlq_quantize requires power-of-two dimensions with in_features >= 8, got {}x{}",
            w.rows, w.cols
        )));
    }
    let m = w.rows;
    let n = w.cols;
    let su = rademacher_signs(n, config.seed);
    let sv = rademacher_signs(m, config.seed.wrapping_add(1));

    // Regularize Hessian before RHT.
    let mut h_reg = hessian.clone();
    let mean_diag = (0..n).map(|i| h_reg.get(i, i)).sum::<f64>() / (n as f64);
    let reg = mean_diag.max(1e-6) * config.sigma_reg;
    for i in 0..n {
        h_reg.set(i, i, h_reg.get(i, i) + reg);
    }

    let hr = rht_hessian(&h_reg, &su)?;
    let wr = rht_weight(w, &su, &sv, RhtDirection::Forward)?;

    // Global scale so rotated weights match E8P Gaussian variance.
    let rms = (wr.data.iter().map(|v| v * v).sum::<f64>() / (wr.data.len() as f64)).sqrt();
    let scale = (rms / E8P_OPT_SCALE).max(1e-8);

    let (l, _d) = hr.ldl()?;
    let mut hat_wr = Matrix::zeros(m, n);
    let blocks_per_row = n / 8;
    let mut q_idxs = vec![0u16; m * blocks_per_row];

    // Backward column-block LDLQ sweep (from last 8-col block to first).
    for b in (0..blocks_per_row).rev() {
        let c0 = b * 8;
        for r in 0..m {
            let mut target = [0.0; 8];
            for (k, slot) in target.iter_mut().enumerate() {
                let col = c0 + k;
                let mut feedback = 0.0;
                for j in (col + 1)..n {
                    feedback += l.get(j, col) * (wr.get(r, j) - hat_wr.get(r, j));
                }
                *slot = (wr.get(r, col) + feedback) / scale;
            }
            let (idx, q_block) = codebook.quantize_block(&target);
            q_idxs[r * blocks_per_row + b] = idx;
            for (k, &q_val) in q_block.iter().enumerate() {
                hat_wr.set(r, c0 + k, q_val * scale);
            }
        }
    }

    Ok(QuantizedLinear {
        out_features: m,
        in_features: n,
        q_idxs,
        su,
        sv,
        scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ldlq_quantizes_and_reconstructs_with_bounded_error() {
        let cb = E8pCodebook::new();
        let mut w = Matrix::zeros(8, 16);
        for i in 0..w.data.len() {
            w.data[i] = ((i as f64 * 0.73).sin()) * 0.25;
        }
        let h = Matrix::eye(16);
        let q = ldlq_quantize(&w, &h, &cb, QuipConfig::default()).expect("LDLQ succeeds");
        let w_hat = q.dequantize(&cb).expect("dequantize succeeds");

        let mut mse = 0.0;
        let mut ref_norm2 = 0.0;
        for (a, b) in w.data.iter().zip(w_hat.data.iter()) {
            mse += (a - b) * (a - b);
            ref_norm2 += a * a;
        }
        let rel_err = (mse / ref_norm2).sqrt();
        assert!(rel_err < 0.45, "relative RMSE too high: {rel_err}");
    }
}
