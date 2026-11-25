//! Pure-Rust port of Cornell RelaxML's QuIP# 2-bit LLM quantization.
//!
//! Implements Randomized Hadamard Transform (RHT) incoherence processing,
//! Cholesky/LDL-based sequential quantization (LDLQ), and the 2-bit `E8P12`
//! lattice codebook ($2^{16}$ entries in $E_8 + \frac{1}{4}$).

pub mod e8p;
pub mod error;
pub mod hadamard;
pub mod ldlq;
pub mod linalg;

pub use e8p::E8pCodebook;
pub use error::QuipError;
pub use hadamard::{fwht_unitary_inplace, rht_hessian, rht_weight, RhtDirection};
pub use ldlq::{ldlq_quantize, QuantizedLinear, QuipConfig};
pub use linalg::Matrix;
