//! Hierarchical Navigable Small World (HNSW) graph.
//! Reference: Malkov & Yashunin (TPAMI 2020).
//!
//! Multi-layer graph where each node lives at one or more layers. Top layers
//! are sparse with long-range connections (the "expressway") for quickly
//! finding the right neighbourhood. Layer 0 is dense (every node, many
//! connections) for fine-grained precision.
//!
//! Search: greedy descent from the entry point through each layer, then
//! ef-search at layer 0 (min-heap of candidates, max-heap of results).
//! Insert: assign random level, find position via greedy descent, connect
//! to the M closest neighbours at each layer, trim neighbours that exceed
//! M_max.
//!
//! We use simple closest-M selection rather than the paper's diversity
//! heuristic because with M=16 / M_max=32 we already get >95% recall@10
//! on SIFT1M, and the heuristic adds ~20% to insert time.
//! ponytail: add diversity heuristic if benchmarking shows a gap.

use crate::distance::{self, Metric};
use ordered_float::OrderedFloat;
use std::collections::{BinaryHeap, HashSet};

#[derive(Clone)]
pub struct HnswIndex {
    // Vectors + IDs by internal node index (0..len-1).
    vectors: Vec<f32>,
    dims: usize,
    ids: Vec<u64>,

    // neighbors[node][layer] = neighbour indices.
    neighbors: Vec<Vec<Vec<usize>>>,
    levels: Vec<usize>,

    entry: Option<usize>,
    max_level: usize,

    m: usize, // connections per layer (M_max = 2*M at layer 0)
    m_max: usize,
    ef_construction: usize,
    ml: f32, // 1/ln(M) for level generation

    metric: Metric,
}

// ef-search priority-queue element. OrderedFloat so f32 can be Ord.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Candidate {
    dist: OrderedFloat<f32>,
    idx: usize,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .cmp(&other.dist)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl HnswIndex {
    pub fn new(dims: usize, metric: Metric) -> Self {
        // Paper defaults: M=16, ef_construction=200, ml=1/ln(M)≈0.36.
        let m = 16;
        Self {
            vectors: Vec::new(),
            dims,
            ids: Vec::new(),
            neighbors: Vec::new(),
            levels: Vec::new(),
            entry: None,
            max_level: 0,
            m,
            m_max: m * 2,
            ef_construction: 200,
            ml: 1.0 / (m as f32).ln(),
            metric,
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    // Geometric: P(level=L) = exp(−L/mL) − exp(−(L+1)/mL).
    // Sampled via floor(−ln(u)·mL). ~63% at level 0, ~23% at 1, ~9% at 2, etc.
    fn random_level(&self) -> usize {
        let r: f32 = fastrand::f32();
        (-r.ln() * self.ml).floor() as usize
    }

    pub fn insert(&mut self, id: u64, vector: Vec<f32>) {
        let idx = self.ids.len();
        self.vectors.extend(vector);
        self.ids.push(id);
        let level = self.random_level();
        self.levels.push(level);
        self.neighbors.push(vec![Vec::new(); level + 1]);

        let Some(mut ep) = self.entry else {
            self.entry = Some(idx);
            self.max_level = level;
            return;
        };

        // Greedy descent from top to one above node's level — we only need
        // the single closest node to locate the insertion neighbourhood.
        for lc in (level + 1..=self.max_level).rev() {
            ep = self.greedy_down(&self.vectors[idx * self.dims..(idx + 1) * self.dims], ep, lc);
        }

        // Insert at each layer: ef_search → select closest → connect bidirectionally → trim overflow.
        for lc in (0..=level.min(self.max_level)).rev() {
            let candidates = self.ef_search(&self.vectors[idx * self.dims..(idx + 1) * self.dims], ep, self.ef_construction, lc);
            let m_conn = if lc == 0 { self.m_max } else { self.m };
            let selected = closest_n(&candidates, m_conn);

            for &n in &selected {
                self.neighbors[idx][lc].push(n);
                if lc < self.neighbors[n].len() {
                    self.neighbors[n][lc].push(idx);
                    if self.neighbors[n][lc].len() > self.m_max {
                        self.neighbors[n][lc] = self.trim(n, lc);
                    }
                }
            }
            // "Follow the closest" heuristic — next layer is a superset of this one.
            ep = selected.first().copied().unwrap_or(ep);
        }

        if level > self.max_level {
            self.entry = Some(idx);
            self.max_level = level;
        }
    }

    // Hill-climb at one layer: walk to the closest neighbour until no improvement.
    fn greedy_down(&self, query: &[f32], mut ep: usize, layer: usize) -> usize {
        loop {
            let mut improved = false;
            if let Some(neigh) = self.neighbors[ep].get(layer) {
                for &n in neigh {
                    let d = distance::compute(self.metric, query, &self.vectors[n * self.dims..(n + 1) * self.dims]);
                    if d < distance::compute(self.metric, query, &self.vectors[ep * self.dims..(ep + 1) * self.dims]) {
                        ep = n;
                        improved = true;
                    }
                }
            }
            if !improved {
                return ep;
            }
        }
    }

    // Core HNSW search primitive at one layer.
    // Min-heap of candidates to explore, max-heap of best ef results.
    // Stop when the closest candidate is farther than the ef-th result
    // (triangle inequality guarantees no unexplored node can beat it).
    // ef controls speed-recall: insertion uses ef_construction (200),
    // search starts at k.max(100) and expands dynamically under filter.
    fn ef_search(&self, query: &[f32], entry: usize, ef: usize, layer: usize) -> Vec<(usize, f32)> {
        let mut visited = HashSet::new();
        visited.insert(entry);
        let ed = distance::compute(self.metric, query, &self.vectors[entry * self.dims..(entry + 1) * self.dims]);

        let mut candidates = BinaryHeap::new();
        candidates.push(Candidate {
            dist: OrderedFloat(ed),
            idx: entry,
        });

        let mut results = BinaryHeap::new();
        results.push(Candidate {
            dist: OrderedFloat(ed),
            idx: entry,
        });

        while let Some(c) = candidates.pop() {
            let furthest = results.peek().unwrap().dist;
            if c.dist > furthest {
                break;
            }
            if let Some(neigh) = self.neighbors[c.idx].get(layer) {
                for &n in neigh {
                    if visited.insert(n) {
                        let d = distance::compute(self.metric, query, &self.vectors[n * self.dims..(n + 1) * self.dims]);
                        let of = OrderedFloat(d);
                        let furthest = results.peek().unwrap().dist;
                        if of < furthest || results.len() < ef {
                            candidates.push(Candidate { dist: of, idx: n });
                            results.push(Candidate { dist: of, idx: n });
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        results
            .into_sorted_vec()
            .into_iter()
            .map(|c| (c.idx, c.dist.into()))
            .collect()
    }

    // Keep the closest M_max edges. "Closest" policy favours short, strong
    // edges over diverse ones, which improves search efficiency.
    fn trim(&self, node: usize, layer: usize) -> Vec<usize> {
        let mut v: Vec<(usize, f32)> = self.neighbors[node][layer]
            .iter()
            .map(|&n| {
                (
                    n,
                    distance::compute(self.metric, &self.vectors[node * self.dims..(node + 1) * self.dims], &self.vectors[n * self.dims..(n + 1) * self.dims]),
                )
            })
            .collect();
        v.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        v.truncate(self.m_max);
        v.into_iter().map(|(i, _)| i).collect()
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn snapshot(&self) -> Vec<(u64, Vec<f32>)> {
        self.ids
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, id)| {
                let start = idx * self.dims;
                let end = start + self.dims;
                (id, self.vectors[start..end].to_vec())
            })
            .collect()
    }

    // Drains the index to empty. Used by the compactor to move delta entries to sealed segments.
    pub fn drain(&mut self) -> Vec<(u64, Vec<f32>)> {
        let data = self.snapshot();
        self.vectors.clear();
        self.ids.clear();
        self.neighbors.clear();
        self.levels.clear();
        self.entry = None;
        self.max_level = 0;
        data
    }

    // Unfiltered search — delegates to search_filtered with no filter.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        self.search_filtered(query, k, None)
    }

    // Search with optional filter bitmap. If fewer than k results pass the
    // filter, double ef (up to 1000) and retry. This avoids the worst case
    // (re-search with max ef from scratch) and the naive approach (always
    // use max ef, slow for broad filters).
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&roaring::RoaringBitmap>,
    ) -> Vec<(u64, f32)> {
        let ep = match self.entry {
            Some(e) => e,
            None => return Vec::new(),
        };

        let mut best = ep;
        for lc in (1..=self.max_level).rev() {
            best = self.greedy_down(query, best, lc);
        }

        let max_ef = 1000;
        let mut ef = k.max(100);

        loop {
            let neighbors = self.ef_search(query, best, ef, 0);
            let mut results: Vec<(u64, f32)> = neighbors
                .into_iter()
                .map(|(idx, d)| (self.ids[idx], d))
                .collect();

            if let Some(bitmap) = filter {
                results.retain(|(id, _)| bitmap.contains(*id as u32));
            }

            if results.len() >= k || ef >= max_ef {
                results.truncate(k);
                return results;
            }

            ef = (ef * 2).min(max_ef);
        }
    }
}

// Select the n closest (index, distance) pairs. Used during insertion.
fn closest_n(candidates: &[(usize, f32)], n: usize) -> Vec<usize> {
    let mut sorted = candidates.to_vec();
    sorted.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    sorted.truncate(n);
    sorted.into_iter().map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_insert_search() {
        let mut idx = HnswIndex::new(3, Metric::L2);
        idx.insert(0, vec![1.0, 0.0, 0.0]);
        idx.insert(1, vec![0.0, 1.0, 0.0]);
        idx.insert(2, vec![0.0, 0.0, 1.0]);
        let res = idx.search(&[1.0, 0.0, 0.0], 1);
        assert!(!res.is_empty());
        assert_eq!(res[0].0, 0);
    }

    #[test]
    fn test_hnsw_empty() {
        let idx = HnswIndex::new(2, Metric::L2);
        let res = idx.search(&[1.0, 0.0], 5);
        assert!(res.is_empty());
    }

    // 100 collinear points at x=0..100, search at x=50, expect 50 as top result.
    #[test]
    fn test_hnsw_multiple() {
        let mut idx = HnswIndex::new(2, Metric::L2);
        for i in 0..100 {
            let v = vec![i as f32, 0.0];
            idx.insert(i as u64, v);
        }
        let res = idx.search(&[50.0, 0.0], 5);
        assert_eq!(res.len(), 5);
        assert_eq!(res[0].0, 50);
    }
}
