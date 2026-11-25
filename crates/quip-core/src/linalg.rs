//! Lightweight dense matrix and Cholesky/LDL linear algebra in pure Rust.

use crate::QuipError;

/// Row-major dense `f64` matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Row-major data buffer (`rows * cols`).
    pub data: Vec<f64>,
}

impl Matrix {
    /// Creates a zero-initialized `rows x cols` matrix.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Creates a `rows x cols` matrix from row-major data.
    ///
    /// # Errors
    /// Returns [`QuipError::Dimension`] if `data.len() != rows * cols`.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, QuipError> {
        if data.len() != rows * cols {
            return Err(QuipError::Dimension(format!(
                "expected {} elements for {rows}x{cols}, got {}",
                rows * cols,
                data.len()
            )));
        }
        Ok(Self { rows, cols, data })
    }

    /// Creates an `n x n` identity matrix.
    #[must_use]
    pub fn eye(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    /// Returns element `(r, c)`.
    #[must_use]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    /// Sets element `(r, c)`.
    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] = val;
    }

    /// Returns a slice of row `r`.
    #[must_use]
    pub fn row(&self, r: usize) -> &[f64] {
        let start = r * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Returns a mutable slice of row `r`.
    pub fn row_mut(&mut self, r: usize) -> &mut [f64] {
        let start = r * self.cols;
        &mut self.data[start..start + self.cols]
    }

    /// Returns the transpose of `self`.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut out = Self::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }

    /// Matrix multiplication `self @ rhs`.
    ///
    /// # Errors
    /// Returns [`QuipError::Dimension`] if `self.cols != rhs.rows`.
    pub fn matmul(&self, rhs: &Self) -> Result<Self, QuipError> {
        if self.cols != rhs.rows {
            return Err(QuipError::Dimension(format!(
                "cannot multiply {}x{} by {}x{}",
                self.rows, self.cols, rhs.rows, rhs.cols
            )));
        }
        let mut out = Self::zeros(self.rows, rhs.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                for j in 0..rhs.cols {
                    let cur = out.get(i, j);
                    out.set(i, j, cur + a * rhs.get(k, j));
                }
            }
        }
        Ok(out)
    }

    /// Computes the lower-triangular Cholesky factor `L` such that `L * L^T = self`.
    ///
    /// # Errors
    /// Returns [`QuipError::Dimension`] if not square, or
    /// [`QuipError::NotPositiveDefinite`] if a pivot is `<= 0`.
    pub fn cholesky(&self) -> Result<Self, QuipError> {
        if self.rows != self.cols {
            return Err(QuipError::Dimension(format!(
                "Cholesky requires square matrix, got {}x{}",
                self.rows, self.cols
            )));
        }
        let n = self.rows;
        let mut l = Self::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let mut sum = self.get(i, j);
                for k in 0..j {
                    sum -= l.get(i, k) * l.get(j, k);
                }
                if i == j {
                    if sum <= 0.0 || !sum.is_finite() {
                        return Err(QuipError::NotPositiveDefinite {
                            index: i,
                            diag: sum,
                        });
                    }
                    l.set(i, j, sum.sqrt());
                } else {
                    l.set(i, j, sum / l.get(j, j));
                }
            }
        }
        Ok(l)
    }

    /// Computes unit-lower-triangular `L` and diagonal `D` such that `L * diag(D) * L^T = self`.
    ///
    /// # Errors
    /// Returns [`QuipError`] if the matrix is not square or not positive-definite.
    pub fn ldl(&self) -> Result<(Self, Vec<f64>), QuipError> {
        let chol = self.cholesky()?;
        let n = self.rows;
        let mut l = Self::eye(n);
        let mut d = vec![0.0; n];
        for (j, d_slot) in d.iter_mut().enumerate() {
            let diag = chol.get(j, j);
            *d_slot = diag * diag;
            for i in (j + 1)..n {
                l.set(i, j, chol.get(i, j) / diag);
            }
        }
        Ok((l, d))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_and_ldl_reconstruct_spd_matrix() {
        let h = Matrix::from_vec(3, 3, vec![4.0, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 6.0])
            .expect("valid 3x3");
        let l = h.cholesky().expect("SPD");
        let recon = l.matmul(&l.transpose()).expect("matmul");
        for i in 0..9 {
            assert!((recon.data[i] - h.data[i]).abs() < 1e-10);
        }

        let (unit_l, d) = h.ldl().expect("LDL succeeds");
        for (i, &d_val) in d.iter().enumerate() {
            assert!((unit_l.get(i, i) - 1.0).abs() < 1e-12);
            assert!(d_val > 0.0);
        }
    }
}
