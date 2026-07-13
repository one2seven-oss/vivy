//! Metadata filter index using Roaring bitmaps.
//!
//! For each metadata field we maintain a `Map<value → RoaringBitmap>` of
//! vector IDs. A filter expression like `{color: "red", size: "large"}` is
//! evaluated by looking up each bitmap and intersecting them. The resulting
//! bitmap is passed to HNSW search — candidates not in the bitmap are skipped.
//!
//! Roaring bitmaps offer fast set operations, good compression for both dense
//! and sparse data (~2-4 KB for 1M sparse IDs), and a well-tested Rust crate.
//!
//! FilterExpr is a simple AST: Equals, In, And, Or, Not. Only Equals and And
//! are wired into the Python bindings; the rest work via the Rust API.
//!
//! Selectivity estimation (<30% → push predicate into traversal, >30% →
//! post-filter) lives in concurrent.rs.

use roaring::RoaringBitmap;
use std::collections::HashMap;

// Filter expression AST. Evaluated bottom-up: children first, then combine bitmaps.
#[derive(Debug, Clone)]
pub enum FilterExpr {
    Equals { field: String, value: String },
    In { field: String, values: Vec<String> },
    And(Vec<FilterExpr>),
    Or(Vec<FilterExpr>),
    Not(Box<FilterExpr>),
}

// index[field][value] → RoaringBitmap. Wrapped in RwLock inside VivyIndex.
#[derive(Clone)]
pub struct FilterIndex {
    index: HashMap<String, HashMap<String, RoaringBitmap>>,
}

impl FilterIndex {
    pub fn new() -> Self {
        Self { index: HashMap::new() }
    }

    // Insert id → bitmap[field][value]. Idempotent if already set.
    pub fn insert(&mut self, id: u64, field: &str, value: &str) {
        let field_map = self.index.entry(field.to_string()).or_default();
        field_map.entry(value.to_string()).or_default().insert(id as u32);
    }

    // Evaluate the expression tree bottom-up.
    pub fn evaluate(&self, expr: &FilterExpr) -> RoaringBitmap {
        match expr {
            FilterExpr::Equals { field, value } => {
                self.index
                    .get(field)
                    .and_then(|m| m.get(value))
                    .cloned()
                    .unwrap_or_default()
            }
            FilterExpr::In { field, values } => {
                let mut result = RoaringBitmap::new();
                if let Some(field_map) = self.index.get(field) {
                    for v in values {
                        if let Some(bitmap) = field_map.get(v) {
                            result |= bitmap;
                        }
                    }
                }
                result
            }
            FilterExpr::And(exprs) => {
                let mut iter = exprs.iter().map(|e| self.evaluate(e));
                let first = match iter.next() {
                    Some(b) => b,
                    None => return RoaringBitmap::new(),
                };
                iter.fold(first, |acc, b| acc & b)
            }
            FilterExpr::Or(exprs) => {
                let mut result = RoaringBitmap::new();
                for e in exprs {
                    result |= self.evaluate(e);
                }
                result
            }
            FilterExpr::Not(expr) => {
                let all = self.all_ids();
                all - self.evaluate(expr)
            }
        }
    }

    // Fraction of all IDs that pass the filter (0.0–1.0).
    // Used by the query planner to choose post-filter vs predicate-push.
    pub fn selectivity(&self, expr: &FilterExpr) -> f64 {
        let total = self.total_ids() as f64;
        if total == 0.0 {
            return 1.0;
        }
        self.evaluate(expr).len() as f64 / total
    }

    // Union of all per-value bitmaps. Used by Not expressions.
    fn all_ids(&self) -> RoaringBitmap {
        let mut all = RoaringBitmap::new();
        for field_map in self.index.values() {
            for bitmap in field_map.values() {
                all |= bitmap;
            }
        }
        all
    }

    fn total_ids(&self) -> u64 {
        let mut all = RoaringBitmap::new();
        for field_map in self.index.values() {
            for bitmap in field_map.values() {
                all |= bitmap;
            }
        }
        all.len()
    }
}

impl Default for FilterIndex {
    fn default() -> Self {
        Self::new()
    }
}

// Check a single ID against a filter. Useful for sealed-segment post-filtering.
pub fn passes_filter(filter_index: &FilterIndex, expr: &FilterExpr, id: u64) -> bool {
    let bitmap = filter_index.evaluate(expr);
    bitmap.contains(id as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    // color="red" should match IDs 1 and 3 but not 2.
    #[test]
    fn test_filter_basic() {
        let mut fi = FilterIndex::new();
        fi.insert(1, "color", "red");
        fi.insert(2, "color", "blue");
        fi.insert(3, "color", "red");

        let expr = FilterExpr::Equals { field: "color".into(), value: "red".into() };
        let result = fi.evaluate(&expr);
        assert!(result.contains(1));
        assert!(result.contains(3));
        assert!(!result.contains(2));
    }

    // AND filter: only ID 1 matches both color="red" and size="large".
    #[test]
    fn test_filter_and() {
        let mut fi = FilterIndex::new();
        fi.insert(1, "color", "red");
        fi.insert(1, "size", "large");
        fi.insert(2, "color", "red");
        fi.insert(2, "size", "small");
        fi.insert(3, "color", "blue");

        let expr = FilterExpr::And(vec![
            FilterExpr::Equals { field: "color".into(), value: "red".into() },
            FilterExpr::Equals { field: "size".into(), value: "large".into() },
        ]);
        let result = fi.evaluate(&expr);
        assert_eq!(result.len(), 1);
        assert!(result.contains(1));
    }

    // 100 vectors split even/odd → selectivity("even") = 0.5.
    #[test]
    fn test_selectivity() {
        let mut fi = FilterIndex::new();
        for i in 1..=100 {
            fi.insert(i, "group", if i % 2 == 0 { "even" } else { "odd" });
        }
        let expr = FilterExpr::Equals { field: "group".into(), value: "even".into() };
        let sel = fi.selectivity(&expr);
        assert!((sel - 0.5).abs() < 0.01);
    }
}
