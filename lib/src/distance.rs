//! Distance Kernels
//!
//! The three metrics Vivy uses to compare vectors:
//!
//! **L2 Squared** — sum of squared differences. Drops the final sqrt because
//! it's monotonic — preserves nearest-neighbour ordering and saves one sqrt
//! per comparison. Use when absolute distances matter.
//!
//! **Cosine** — `1 − (a·b / (|a|·|b|))`. Standard for text embeddings where
//! direction carries semantics and length is an artifact of padding. Returns
//! `1 − similarity` so smaller = closer, matching the other metrics.
//!
//! **Dot Product** — negated inner product. For unit-normalised vectors this
//! is equivalent to cosine but avoids the norm computation (~30% fewer flops).
//! Negated so that largest positive dot → smallest distance.
//!
//! All three are currently scalar loops. The portable-simd and hand-written
//! AVX2/AVX-512/NEON paths will go behind `#[cfg(target_feature)]` gates
//! with runtime CPUID dispatch. The scalar loop autovectorises adequately
//! (LLVM unrolls and uses SSE/AVX for obvious reductions), so the gap is
//! 2-4x, not 10x. We close it when it shows up as a p99 driver.

#[derive(Clone, Copy)]
pub enum Metric {
    L2,
    Cosine,
    Dot,
}

pub fn compute(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        Metric::L2 => l2_squared(a, b),
        Metric::Cosine => cosine(a, b),
        Metric::Dot => neg_dot(a, b),
    }
}

// Σ (aᵢ − bᵢ)² — squared Euclidean, no sqrt (monotonic, saves one sqrt per cmp).
// Call sqrt() at the application layer if you need true Euclidean.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

// 1 − cos(θ) = 1 − (a·b) / (|a|·|b|).
// Returns 1 − similarity so that 0 = identical direction, 1 = orthogonal, 2 = opposite.
// Denominator clamped at f32::EPSILON for the zero-vector edge case.
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    1.0 - dot / (na.sqrt() * nb.sqrt()).max(f32::EPSILON)
}

// −(a·b). For unit-normalised vectors this equals cosine distance without
// the norm computation. Smaller = closer, so MIP becomes min-negated-dot.
// Caller must match metric to their embedding model.
#[inline]
pub fn neg_dot(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
    }
    -dot
}

#[cfg(test)]
mod tests {
    use super::*;

    /*
     * L2: (4−1)² + (5−2)² + (6−3)² = 9 + 9 + 9 = 27
     */
    #[test]
    fn test_l2() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((l2_squared(&a, &b) - 27.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_zero() {
        let a = vec![1.0, 2.0];
        assert!(l2_squared(&a, &a).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_same() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(cosine(&a, &a).abs() < 1e-5);
    }

    /*
     * Orthogonal vectors [1,0] and [0,1] have dot = 0,
     * so cosine similarity = 0 and distance = 1.
     */
    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-5);
    }
}
