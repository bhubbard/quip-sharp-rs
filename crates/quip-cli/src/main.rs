//! CLI for QuIP# 2-bit E8P quantization and verification.

use std::time::Instant;

use anyhow::Result;
use clap::{Parser, Subcommand};
use quip_core::{ldlq_quantize, E8pCodebook, Matrix, QuipConfig};

#[derive(Parser)]
#[command(
    name = "quip-sharp",
    about = "Pure-Rust QuIP# 2-bit E8P quantization CLI"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Verify the 65,536-entry E8P12 lattice codebook.
    VerifyE8p,
    /// Run a synthetic LDLQ + RHT 2-bit quantization demo and report RMSE & latency.
    Demo {
        #[arg(long, default_value_t = 16)]
        rows: usize,
        #[arg(long, default_value_t = 32)]
        cols: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::VerifyE8p => {
            let t0 = Instant::now();
            let cb = E8pCodebook::new();
            println!(
                "E8P12 codebook verified: {} abs patterns, {} total 8D entries ({:.1} ms)",
                cb.packed_abs_grid.len(),
                cb.grid.len(),
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        Commands::Demo { rows, cols } => {
            let cb = E8pCodebook::new();
            let mut w = Matrix::zeros(rows, cols);
            for (i, v) in w.data.iter_mut().enumerate() {
                *v = (i as f64 * 0.41).cos() * 0.2;
            }
            let h = Matrix::eye(cols);
            let t0 = Instant::now();
            let q = ldlq_quantize(&w, &h, &cb, QuipConfig::default())?;
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let w_hat = q.dequantize(&cb)?;
            let mse = w
                .data
                .iter()
                .zip(w_hat.data.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                / (w.data.len() as f64);
            println!(
                "Quantized {rows}x{cols} to 2-bit E8P ({} u16 indices) in {elapsed_ms:.2} ms | RMSE={:.5}",
                q.q_idxs.len(),
                mse.sqrt()
            );
        }
    }
    Ok(())
}
