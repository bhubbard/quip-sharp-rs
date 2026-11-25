# quip-sharp-rs

Pure-Rust port of [Cornell-RelaxML/quip-sharp](https://github.com/Cornell-RelaxML/quip-sharp) — extreme 2-bit LLM quantization using Randomized Hadamard Transform (RHT) incoherence processing, Hessian-aware LDLQ, and the `E8P12` lattice codebook ($E_8 + \frac{1}{4}$).

Zero Python or CUDA runtime required.

## Workspace Crates

| Crate | Purpose |
| :--- | :--- |
| `quip-core` | Pure-Rust FWHT & RHT (`rht_hessian`, `rht_weight`), Cholesky/LDL decomposition, 65,536-entry `E8pCodebook`, `ldlq_quantize`, and `QuantizedLinear` |
| `quip-cli` | `quip-sharp` CLI (`verify-e8p`, `demo`) |

## Quickstart

```bash
cargo test --workspace
cargo run -p quip-cli -- verify-e8p
cargo run -p quip-cli -- demo --rows 16 --cols 32
```
