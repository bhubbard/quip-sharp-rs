# Benchmarks & Coverage

Measured on **Apple M3 Pro** (`arm64`, 11 cores, 18 GiB RAM, macOS) with `rustc 1.100.0` (`--release`) vs. `Python 3.15.2` (same host) and upstream `Cornell-RelaxML/quip-sharp`.

## 1. Code Coverage (`cargo llvm-cov --workspace --summary-only`)

| File | Region Coverage | Function Coverage | Line Coverage |
| :--- | ---: | ---: | ---: |
| `quip-core/src/e8p.rs` | 97.24% | 94.12% | 94.49% |
| `quip-core/src/hadamard.rs` | 89.31% | 100.00% | — |
| **Workspace Total** *(incl. CLI)* | **87.02%** | **88.64%** | **82.55%** |

## 2. Footprint, Startup & Microbenchmarks

| Metric / Operation | `quip-sharp-rs` (Rust) | Upstream `quip-sharp` (Python / PyTorch / CUDA) | Comparison |
| :--- | ---: | ---: | :--- |
| **Standalone Binary Size** | **1.01 MiB** | PyTorch + `quiptools_cuda` + `fast_hadamard_transform` | **>700× smaller** footprint |
| **Cold Startup (5-run median)** | **5.41 ms** | **23.61 ms** (`python3 -c ""` bare stdlib) | **4.36× faster** cold start |
| `E8pCodebook::new` (65,536 entries) | **1.12 ms** | PyTorch `cartesian_prod` + Python loop | Sub-2 ms lattice build |
| `rht_weight(16x32)` | **0.0034 ms** | Requires `fast_hadamard_transform` / CUDA | Pure-Rust FWHT |
| `ldlq_quantize(16x32)` | **8.4148 ms** | Requires PyTorch + CUDA | Pure-Rust Cholesky/LDL + E8P |
| `dequantize(16x32)` | **0.0038 ms** | Requires `quiptools_cuda` | Pure-Rust 2-bit unpack + RHT |

Reproduce locally:
```bash
cargo run --release -p quip-core --example bench
cargo llvm-cov --workspace --summary-only
```
