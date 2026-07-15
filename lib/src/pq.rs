//! Product Quantization (PQ)
//! Reference: Jégou, Douze, Schmid (TPAMI 2011).
//!
//! PQ compresses high-dimensional vectors into short codes (8–64 bytes vs
//! 256–3072 for f32) while preserving approximate distance rankings:
//!
//! 1. Split each D-dim vector into M subvectors of dimension D/M.
//! 2. Train a k-means codebook of 256 centroids per subvector space.
//! 3. Encode: replace each subvector with its nearest centroid index (M bytes).
//! 4. Search: build Asymmetric Distance Computation (ADC) table (query in
//!    full precision, stored vectors quantised) — distance = sum of M table
//!    lookups, no floating-point per candidate.
//!
//! ADC is more accurate than symmetric distance (both sides quantised)
//! because quantisation error only enters once.
//!
//! For 768-dim f32: 3072 B/vector → 286 GB for 100M vectors.
//! With PQ 64-byte codes: 64 B/vector → 6.4 GB — fits in RAM.
//! The HNSW graph (~15-20 GB) stays on NVMe via mmap.

use rand::Rng;
use rayon::prelude::*;

// A trained Product Quantizer: encode, decode, build ADC tables, compute ADC distances.
pub struct ProductQuantizer {
    pub num_subvectors: usize, // M
    pub subdim: usize,         // D / M
    // codebook[subvec][centroid * subdim .. (centroid+1) * subdim]
    pub codebook: Vec<Vec<f32>>,
}

impl ProductQuantizer {
    // Train M independent k-means (256 centroids each, 20 iterations) in parallel via rayon.
    // For each subvector s, collect the s-th subvector of every training vector and cluster.
    // Training cost: ~20 × N × D × 256 f32 ops — ~30-60s for 1M×768 on modern CPU.
    // Done during compaction, not at query time.
    pub fn train(data: &[f32], dims: usize, num_subvectors: usize) -> Self {
        assert!(
            dims.is_multiple_of(num_subvectors),
            "dims must be divisible by num_subvectors"
        );
        let subdim = dims / num_subvectors;
        let n = data.len() / dims;
        let num_centroids = 256usize;

        /*
         * Train one k-means per subvector in parallel.
         * rayon::into_par_iter() distributes the M tasks across available
         * cores. Each task collects the relevant subvectors from all N
         * training vectors and runs Lloyd's algorithm.
         */
        let codebook: Vec<Vec<f32>> = (0..num_subvectors)
            .into_par_iter()
            .map(|sv| {
                let mut subvectors = Vec::with_capacity(n * subdim);
                for i in 0..n {
                    let base = i * dims + sv * subdim;
                    subvectors.extend_from_slice(&data[base..base + subdim]);
                }
                train_kmeans(&subvectors, subdim, num_centroids, 20)
            })
            .collect();

        Self {
            num_subvectors,
            subdim,
            codebook,
        }
    }

    // For each subvector, find nearest centroid → M bytes of PQ codes.
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.num_subvectors);
        for sv in 0..self.num_subvectors {
            let start = sv * self.subdim;
            let slice = &vector[start..start + self.subdim];
            let mut best_idx = 0u8;
            let mut best_dist = f32::MAX;
            // Linear scan of 256 centroids — faster than a tree at this size.
            for (ci, centroid) in self.codebook[sv].chunks_exact(self.subdim).enumerate() {
                let d = l2_sq(slice, centroid);
                if d < best_dist {
                    best_dist = d;
                    best_idx = ci as u8;
                }
            }
            codes.push(best_idx);
        }
        codes
    }

    // Decode PQ codes → approximate vector. Used for debugging and rerank.
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let mut vec = Vec::with_capacity(self.num_subvectors * self.subdim);
        for (sv, &code) in codes.iter().enumerate() {
            let start = (code as usize) * self.subdim;
            let slice = &self.codebook[sv][start..start + self.subdim];
            vec.extend_from_slice(slice);
        }
        vec
    }

    // Pre-compute L2(query_subvec[s], centroid[s][c]) for all s, c → M×256 table.
    // Query stays in full precision; stored vectors are quantised.
    pub fn build_adc_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let mut table = Vec::with_capacity(self.num_subvectors);
        for sv in 0..self.num_subvectors {
            let q_start = sv * self.subdim;
            let q_slice = &query[q_start..q_start + self.subdim];
            let mut row = Vec::with_capacity(256);
            for centroid in self.codebook[sv].chunks_exact(self.subdim) {
                row.push(l2_sq(q_slice, centroid));
            }
            table.push(row);
        }
        table
    }

    // O(M) — one table lookup per subvector. Table must match query (build_adc_table).
    pub fn adc_distance(table: &[Vec<f32>], codes: &[u8]) -> f32 {
        codes
            .iter()
            .enumerate()
            .map(|(sv, &code)| table[sv][code as usize])
            .sum()
    }
}

// Same as distance::l2_squared but local to avoid module coupling concerns.
// ponytail: swap for aliased SIMD version once distance::l2_squared gains acceleration.
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

// Lloyd's algorithm, k=256. Forgy init (random distinct vectors). If k > n,
// duplicates existing ones — degenerate but handled gracefully. max_iter=20.
// Output: k × subdim f32 values packed flat.
fn train_kmeans(data: &[f32], subdim: usize, k: usize, max_iter: usize) -> Vec<f32> {
    let n = data.len() / subdim;
    if n == 0 {
        return vec![0.0; k * subdim];
    }
    let mut rng = rand::rng();

    let mut centroids: Vec<f32> = Vec::with_capacity(k * subdim);
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..k.min(n) {
        let j = rng.random_range(i..n);
        indices.swap(i, j);
        let base = indices[i] * subdim;
        centroids.extend_from_slice(&data[base..base + subdim]);
    }
    while centroids.len() < k * subdim {
        let src = centroids[..centroids.len().min(subdim)].to_vec();
        centroids.extend_from_slice(&src);
    }

    let mut labels = vec![0usize; n];

    for _ in 0..max_iter {
        let mut changed = false;
        for (i, label) in labels.iter_mut().enumerate() {
            let base = i * subdim;
            let mut best = *label;
            let mut best_d = l2_sq(
                &data[base..base + subdim],
                &centroids[best * subdim..(best + 1) * subdim],
            );
            for c in 0..k {
                let d = l2_sq(
                    &data[base..base + subdim],
                    &centroids[c * subdim..(c + 1) * subdim],
                );
                if d < best_d {
                    best_d = d;
                    best = c;
                    changed = true;
                }
            }
            *label = best;
        }
        if !changed {
            break;
        }

        let mut sums = vec![vec![0.0f32; subdim]; k];
        let mut counts = vec![0usize; k];
        for (i, &c) in labels.iter().enumerate() {
            let base = i * subdim;
            counts[c] += 1;
            for j in 0..subdim {
                sums[c][j] += data[base + j];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for j in 0..subdim {
                    centroids[c * subdim + j] = sums[c][j] * inv;
                }
            }
        }
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    /*
     * Train PQ on random 8-dim data with M=4 (subdim=2),
     * encode the first vector, decode it, verify the code length
     * and decoded vector dimension.
     */
    #[test]
    fn test_pq_roundtrip() {
        let dims = 8;
        let n = 100;
        let mut data = Vec::with_capacity(n * dims);
        let mut rng = rand::rng();
        for _ in 0..n {
            for _ in 0..dims {
                data.push(rng.random::<f32>());
            }
        }

        let pq = ProductQuantizer::train(&data, dims, 4);
        assert_eq!(pq.num_subvectors, 4);
        assert_eq!(pq.subdim, 2);

        let codes = pq.encode(&data[..dims]);
        assert_eq!(codes.len(), 4);

        let decoded = pq.decode(&codes);
        assert_eq!(decoded.len(), dims);
    }

    /*
     * Build an ADC table, verify that adc_distance with the table
     * matches the L2 distance to the decoded vector (this confirms
     * the table + code path is internally consistent).
     */
    #[test]
    fn test_adc_table() {
        let dims = 4;
        let n = 50;
        let mut data = Vec::with_capacity(n * dims);
        let mut rng = rand::rng();
        for _ in 0..n {
            for _ in 0..dims {
                data.push(rng.random::<f32>());
            }
        }

        let pq = ProductQuantizer::train(&data, dims, 2);
        let query = vec![0.5; dims];
        let table = pq.build_adc_table(&query);
        assert_eq!(table.len(), 2);
        assert_eq!(table[0].len(), 256);

        let codes = pq.encode(&query);
        let d1 = ProductQuantizer::adc_distance(&table, &codes);
        let d2 = l2_sq(&query, &pq.decode(&codes));
        assert!((d1 - d2).abs() < 1e-4);
    }
}
