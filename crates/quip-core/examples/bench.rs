//! Release benchmark for `quip-core` E8P12 codebook, RHT, and LDLQ 2-bit quantization.

use std::hint::black_box;
use std::time::Instant;

use quip_core::{ldlq_quantize, rht_weight, E8pCodebook, Matrix, QuipConfig, RhtDirection};

fn bench<F: FnMut()>(name: &str, warmup: usize, iters: usize, mut f: F) {
    for _ in 0..warmup {
        f();
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    println!(
        "{name}: median={:.4} ms (min={:.4} ms)",
        times[times.len() / 2],
        times[0]
    );
}

fn main() {
    let t0 = Instant::now();
    let cb = E8pCodebook::new();
    println!(
        "E8pCodebook::new (65,536 entries): {:.2} ms",
        t0.elapsed().as_secs_f64() * 1000.0
    );

    let mut w = Matrix::zeros(16, 32);
    for (i, v) in w.data.iter_mut().enumerate() {
        *v = (i as f64 * 0.37).sin() * 0.25;
    }
    let h = Matrix::eye(32);
    let su = vec![1.0; 32];
    let sv = vec![1.0; 16];

    bench("rht_weight(16x32)", 3, 15, || {
        let _ = black_box(rht_weight(&w, &su, &sv, RhtDirection::Forward));
    });
    bench("ldlq_quantize(16x32)", 1, 5, || {
        let _ = black_box(ldlq_quantize(&w, &h, &cb, QuipConfig::default()));
    });
    let q = ldlq_quantize(&w, &h, &cb, QuipConfig::default()).expect("ldlq");
    bench("dequantize(16x32)", 3, 15, || {
        let _ = black_box(q.dequantize(&cb));
    });
}
