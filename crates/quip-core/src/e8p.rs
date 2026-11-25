//! 2-bit `E8P12` lattice codebook ($E_8 + \frac{1}{4}$) ported from QuIP#.
//!
//! Combines 227 $D_8^+$ half-integer vectors inside radius $\sqrt{10}$ with
//! 29 norm-12 vectors (256 absolute patterns), multiplied by $2^8$ sign/parity
//! bits to form a $2^{16}$-entry 8D codebook (2.0 bits per weight).

use crate::QuipError;

/// Dimension of each E8P lattice block.
pub const E8P_BLOCK_SIZE: usize = 8;

/// Optimal Gaussian scale factor from QuIP# (`E8P12_codebook.opt_scale`).
pub const E8P_OPT_SCALE: f64 = 1.03;

/// 29 norm-12 entries in $E_8 + \frac{1}{4}$ (scaled by 2, i.e., in half-units).
const NORM12_HALF_UNITS: [[i8; 8]; 29] = [
    [3, 1, 1, 1, 3, 3, 3, 3],
    [1, 3, 1, 1, 3, 3, 3, 3],
    [1, 1, 3, 1, 3, 3, 3, 3],
    [1, 1, 1, 3, 3, 3, 3, 3],
    [3, 3, 3, 1, 3, 3, 1, 1],
    [3, 3, 3, 1, 3, 1, 3, 1],
    [3, 3, 3, 1, 1, 3, 3, 1],
    [3, 3, 3, 1, 3, 1, 1, 3],
    [3, 3, 3, 1, 1, 3, 1, 3],
    [3, 3, 3, 1, 1, 1, 3, 3],
    [3, 3, 1, 3, 3, 3, 1, 1],
    [3, 3, 1, 3, 3, 1, 3, 1],
    [3, 3, 1, 3, 1, 3, 3, 1],
    [3, 3, 1, 3, 3, 1, 1, 3],
    [3, 3, 1, 3, 1, 3, 1, 3],
    [3, 3, 1, 3, 1, 1, 3, 3],
    [3, 1, 3, 3, 3, 3, 1, 1],
    [3, 1, 3, 3, 3, 1, 3, 1],
    [3, 1, 3, 3, 1, 3, 3, 1],
    [3, 1, 3, 3, 3, 1, 1, 3],
    [3, 1, 3, 3, 1, 3, 1, 3],
    [1, 3, 3, 3, 1, 1, 3, 3],
    [1, 3, 3, 3, 3, 3, 1, 1],
    [1, 3, 3, 3, 3, 1, 3, 1],
    [1, 3, 3, 3, 1, 3, 3, 1],
    [1, 3, 3, 3, 3, 1, 1, 3],
    [1, 3, 3, 3, 1, 3, 1, 3],
    [1, 1, 3, 3, 1, 3, 3, 3],
    [3, 3, 1, 1, 3, 3, 3, 1],
];

/// Full $2^{16} \times 8$ `E8P12` codebook supporting exact nearest-neighbor encoding
/// and fast 16-bit index decoding.
#[derive(Debug, Clone)]
pub struct E8pCodebook {
    /// 256 packed 4-bit absolute magnitudes (8 nibbles per `u32`).
    pub packed_abs_grid: Vec<u32>,
    /// Precomputed 65,536 x 8 float lattice points.
    pub grid: Vec<[f64; E8P_BLOCK_SIZE]>,
}

impl Default for E8pCodebook {
    fn default() -> Self {
        Self::new()
    }
}

impl E8pCodebook {
    /// Constructs the 65,536-entry `E8P12` codebook.
    #[must_use]
    pub fn new() -> Self {
        let abs_grid = build_abs_grid();
        let packed_abs_grid = pack_abs_grid(&abs_grid);
        let grid = build_full_grid(&packed_abs_grid);
        Self {
            packed_abs_grid,
            grid,
        }
    }

    /// Decodes a 16-bit index into an 8-element lattice vector.
    #[must_use]
    pub fn decode(&self, idx: u16) -> [f64; E8P_BLOCK_SIZE] {
        self.grid[idx as usize]
    }

    /// Finds the nearest 16-bit codebook entry (minimum Euclidean distance) for an
    /// 8-element vector.
    #[must_use]
    pub fn quantize_block(&self, block: &[f64; E8P_BLOCK_SIZE]) -> (u16, [f64; E8P_BLOCK_SIZE]) {
        let mut best_idx = 0u16;
        let mut best_err = f64::INFINITY;
        for (i, cand) in self.grid.iter().enumerate() {
            let mut err = 0.0;
            for k in 0..E8P_BLOCK_SIZE {
                let d = cand[k] - block[k];
                err += d * d;
            }
            if err < best_err {
                best_err = err;
                best_idx = i as u16;
            }
        }
        (best_idx, self.grid[best_idx as usize])
    }

    /// Quantizes a slice whose length is a multiple of 8, returning 16-bit indices
    /// and the reconstructed vector.
    ///
    /// # Errors
    /// Returns [`QuipError::Dimension`] if `vals.len() % 8 != 0`.
    pub fn quantize_slice(&self, vals: &[f64]) -> Result<(Vec<u16>, Vec<f64>), QuipError> {
        if !vals.len().is_multiple_of(E8P_BLOCK_SIZE) {
            return Err(QuipError::Dimension(format!(
                "E8P requires length divisible by {E8P_BLOCK_SIZE}, got {}",
                vals.len()
            )));
        }
        let mut indices = Vec::with_capacity(vals.len() / E8P_BLOCK_SIZE);
        let mut recon = Vec::with_capacity(vals.len());
        for block in vals.as_chunks::<E8P_BLOCK_SIZE>().0 {
            let (idx, q) = self.quantize_block(block);
            indices.push(idx);
            recon.extend_from_slice(&q);
        }
        Ok((indices, recon))
    }
}

fn build_abs_grid() -> Vec<[f64; 8]> {
    let mut unique_abs: Vec<[u8; 8]> = Vec::new();
    // Enumerate [-4, 3]^8 + 0.5, filter sum % 2 == 0 and norm^2 <= 10.
    // Note: in half-units (2 * x), coords are odd integers in [-7, 7], norm^2 <= 40, sum % 4 == 0.
    let mut cur = [0i32; 8];
    fn recurse(depth: usize, cur: &mut [i32; 8], sum: i32, norm2: i32, out: &mut Vec<[u8; 8]>) {
        if norm2 > 40 {
            return;
        }
        if depth == 8 {
            if sum.rem_euclid(4) == 0 {
                let abs_v: [u8; 8] = cur.map(|v| v.unsigned_abs() as u8);
                out.push(abs_v);
            }
            return;
        }
        for v in [-7, -5, -3, -1, 1, 3, 5, 7] {
            cur[depth] = v;
            recurse(depth + 1, cur, sum + v, norm2 + v * v, out);
        }
    }
    recurse(0, &mut cur, 0, 0, &mut unique_abs);
    unique_abs.sort_unstable();
    unique_abs.dedup();

    let mut cba: Vec<[f64; 8]> = unique_abs
        .into_iter()
        .map(|a| a.map(|v| f64::from(v) * 0.5))
        .collect();
    for entry in NORM12_HALF_UNITS {
        cba.push(entry.map(|v| f64::from(v) * 0.5));
    }
    cba
}

fn pack_abs_grid(abs_grid: &[[f64; 8]]) -> Vec<u32> {
    let perm = [0, 2, 4, 6, 1, 3, 5, 7];
    abs_grid
        .iter()
        .map(|row| {
            let mut p = perm.map(|idx| row[idx]);
            let sum: f64 = p.iter().sum();
            let parity = (sum.round() as i32).rem_euclid(2);
            p[7] *= 1.0 - 2.0 * f64::from(parity);
            let mut acc = 0u32;
            for (i, &val) in p.iter().enumerate() {
                let nibble = ((val * 2.0 + 8.0).round() as i32).clamp(0, 15) as u32;
                acc |= nibble << (i * 4);
            }
            acc
        })
        .collect()
}

fn build_full_grid(packed_abs_grid: &[u32]) -> Vec<[f64; 8]> {
    let shuffle_map = [0usize, 4, 1, 5, 2, 6, 3, 7];
    let mut grid = vec![[0.0; 8]; 1 << 16];
    for (c, slot) in grid.iter_mut().enumerate() {
        let mut signs = c & 255;
        let abs_idx = (c >> 8) % packed_abs_grid.len();
        let mut parity = 0usize;
        for i in 0..8 {
            parity ^= (signs >> i) & 1;
        }
        signs ^= parity;
        let abs_code = packed_abs_grid[abs_idx];
        for (i, &ii) in shuffle_map.iter().enumerate() {
            let nib = ((abs_code >> (4 * ii)) & 15) as i32;
            let mut v = f64::from(nib - 8) * 0.5;
            if ((signs >> ii) & 1) != 0 {
                v = -v;
            }
            if parity != 0 {
                v -= 0.25;
            } else {
                v += 0.25;
            }
            slot[i] = v;
        }
    }
    grid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abs_grid_has_exact_256_entries() {
        let cb = E8pCodebook::new();
        assert_eq!(cb.packed_abs_grid.len(), 256);
        assert_eq!(cb.grid.len(), 65_536);
    }

    #[test]
    fn quantize_and_decode_round_trip() {
        let cb = E8pCodebook::new();
        let input = [0.25, -0.75, 1.25, -0.25, 0.75, 0.25, -1.25, 0.25];
        let (indices, recon) = cb.quantize_slice(&input).expect("8-aligned");
        assert_eq!(indices.len(), 1);
        assert_eq!(cb.decode(indices[0]).to_vec(), recon);
    }
}
