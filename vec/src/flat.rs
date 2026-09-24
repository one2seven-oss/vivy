//! Brute-force flat index: every vector in a flat array, search = full scan.
//!
//! Not for production queries — it's the "correctness oracle" that every
//! optimised index (HNSW, PQ, filtered search) is measured against.
//! `distance_to()` exists so tests can ask "what's the exact distance to
//! vector with this id?" without running a full search.

use crate::distance::{self, Metric};

pub struct FlatIndex {
    vectors: Vec<Vec<f32>>,
    ids: Vec<u64>,
    metric: Metric,
}

impl FlatIndex {
    pub fn new(metric: Metric) -> Self {
        Self {
            vectors: Vec::new(),
            ids: Vec::new(),
            metric,
        }
    }

    // IDs need not be unique, but duplicate IDs make distance_to() undefined.
    pub fn insert(&mut self, id: u64, vector: Vec<f32>) {
        self.ids.push(id);
        self.vectors.push(vector);
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    // Full scan: distance to every stored vector, sort, return top-k.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let mut results: Vec<(u64, f32)> = self
            .vectors
            .iter()
            .zip(self.ids.iter())
            .map(|(v, &id)| (id, distance::compute(self.metric, v, query)))
            .collect();
        results.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }

    // Exact distance to a single stored vector by id. Useful for unit tests.
    pub fn distance_to(&self, id: u64, query: &[f32]) -> Option<f32> {
        self.vectors
            .get(self.ids.iter().position(|&x| x == id)?)
            .map(|v| distance::compute(self.metric, v, query))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Insert two vectors at distances 1 and 10 from origin; closer one (id=0) should come first.
    #[test]
    fn test_flat_insert_search() {
        let mut idx = FlatIndex::new(Metric::L2);
        idx.insert(0, vec![1.0, 0.0]);
        idx.insert(1, vec![10.0, 0.0]);

        let hits = idx.search(&[0.0, 0.0], 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, 0);
    }
}
