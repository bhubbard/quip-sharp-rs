//! Error types for `quip-core`.

use thiserror::Error;

/// Errors returned by QuIP# matrix, Hadamard, and quantization operations.
#[derive(Debug, Error)]
pub enum QuipError {
    /// Matrix or vector dimensions are incompatible or not divisible by block size.
    #[error("dimension mismatch: {0}")]
    Dimension(String),

    /// Matrix is not positive-definite during Cholesky decomposition.
    #[error("non-positive-definite Hessian at pivot {index} (diag={diag})")]
    NotPositiveDefinite {
        /// Diagonal pivot index where Cholesky failed.
        index: usize,
        /// Value encountered at the pivot.
        diag: f64,
    },
}
