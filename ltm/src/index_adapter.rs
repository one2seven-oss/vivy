use crate::error::{ErrorCode, MemoryError, Result};
use crate::model::MemoryRecord;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use vivy_core::concurrent::VivyIndex;
use vivy_core::distance::Metric;

/// Trait abstracting vector indexing operations for long-term memory retrieval.
pub trait VectorIndex: Send + Sync {
    /// Upsert a vector associated with a memory record ID.
    fn upsert(&self, memory_id: &str, vector: &[f32]) -> Result<()>;

    /// Mark a memory ID as removed from candidate retrieval.
    fn remove(&self, memory_id: &str) -> Result<()>;

    /// Search for candidate memory IDs nearest to the query vector.
    /// Returns pairs of (memory_id, distance).
    fn search(&self, query: &[f32], limit: usize) -> Result<Vec<(String, f32)>>;

    /// Completely rebuild index from an iterator of canonical memory records.
    fn rebuild<'a>(&self, records: Box<dyn Iterator<Item = &'a MemoryRecord> + 'a>) -> Result<()>;

    /// Return index diagnostic/health status.
    fn is_healthy(&self) -> bool;

    /// Upsert multiple vectors associated with memory record IDs in batch.
    fn upsert_batch(&self, items: &[(&str, &[f32])]) -> Result<()> {
        for (id, vec) in items {
            self.upsert(id, vec)?;
        }
        Ok(())
    }

    /// Mark multiple memory IDs as removed in batch.
    fn remove_batch(&self, memory_ids: &[&str]) -> Result<()> {
        for id in memory_ids {
            self.remove(id)?;
        }
        Ok(())
    }

    /// Flush and copy vector segments and manifest to target directory.
    fn backup_segments(&self, _target_dir: &std::path::Path) -> Result<()> {
        Ok(())
    }
}

/// Vivy-backed implementation of VectorIndex.
pub struct VivyVectorIndex {
    dims: usize,
    metric: Metric,
    index: RwLock<VivyIndex>,
    // Bidirectional collision-safe ID mapping
    id_to_u64: RwLock<HashMap<String, u64>>,
    u64_to_id: RwLock<HashMap<u64, String>>,
    tombstones: RwLock<HashMap<String, bool>>,
    next_u64: AtomicU64,
}

impl VivyVectorIndex {
    pub fn new(dims: usize, metric: Metric) -> Result<Self> {
        Self::new_with_dir(dims, metric, Option::<&std::path::Path>::None)
    }

    pub fn new_with_dir(
        dims: usize,
        metric: Metric,
        data_dir: Option<impl AsRef<std::path::Path>>,
    ) -> Result<Self> {
        let wal_path = data_dir.as_ref().map(|p| p.as_ref().join("index.wal"));
        let vivy_idx = VivyIndex::new(dims, metric, wal_path, data_dir)
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to create VivyIndex: {:?}", e),
            })?;

        Ok(Self {
            dims,
            metric,
            index: RwLock::new(vivy_idx),
            id_to_u64: RwLock::new(HashMap::new()),
            u64_to_id: RwLock::new(HashMap::new()),
            tombstones: RwLock::new(HashMap::new()),
            next_u64: AtomicU64::new(1),
        })
    }
}

impl VectorIndex for VivyVectorIndex {
    fn upsert(&self, memory_id: &str, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dims {
            return Err(MemoryError::DimensionMismatch {
                code: ErrorCode::DimensionMismatch,
                expected: self.dims,
                actual: vector.len(),
            });
        }

        let num_id = {
            let mut id_map = self.id_to_u64.write();
            let mut u64_map = self.u64_to_id.write();
            let id = *id_map
                .entry(memory_id.to_string())
                .or_insert_with(|| self.next_u64.fetch_add(1, Ordering::Relaxed));
            u64_map.insert(id, memory_id.to_string());
            id
        };

        // Clear any prior tombstone
        self.tombstones.write().remove(memory_id);

        let idx = self.index.read();
        idx.insert_with_id(num_id, vector.to_vec())
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("VivyIndex insert failed: {:?}", e),
            })?;

        Ok(())
    }

    fn upsert_batch(&self, items: &[(&str, &[f32])]) -> Result<()> {
        for (_, vector) in items {
            if vector.len() != self.dims {
                return Err(MemoryError::DimensionMismatch {
                    code: ErrorCode::DimensionMismatch,
                    expected: self.dims,
                    actual: vector.len(),
                });
            }
        }

        let num_ids: Vec<u64> = {
            let mut id_map = self.id_to_u64.write();
            let mut u64_map = self.u64_to_id.write();
            items
                .iter()
                .map(|(mem_id, _)| {
                    let id = *id_map
                        .entry(mem_id.to_string())
                        .or_insert_with(|| self.next_u64.fetch_add(1, Ordering::Relaxed));
                    u64_map.insert(id, mem_id.to_string());
                    id
                })
                .collect()
        };

        {
            let mut tombstones = self.tombstones.write();
            for (mem_id, _) in items {
                tombstones.remove(*mem_id);
            }
        }

        let idx = self.index.read();
        for (i, (_, vector)) in items.iter().enumerate() {
            idx.insert_with_id(num_ids[i], vector.to_vec())
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("VivyIndex batch insert failed: {:?}", e),
                })?;
        }

        Ok(())
    }

    fn remove(&self, memory_id: &str) -> Result<()> {
        self.tombstones
            .write()
            .insert(memory_id.to_string(), true);
        Ok(())
    }

    fn remove_batch(&self, memory_ids: &[&str]) -> Result<()> {
        let mut tombstones = self.tombstones.write();
        for id in memory_ids {
            tombstones.insert(id.to_string(), true);
        }
        Ok(())
    }

    fn search(&self, query: &[f32], limit: usize) -> Result<Vec<(String, f32)>> {
        if query.len() != self.dims {
            return Err(MemoryError::DimensionMismatch {
                code: ErrorCode::DimensionMismatch,
                expected: self.dims,
                actual: query.len(),
            });
        }

        let k_fetch = limit * 3 + 10; // Overfetch to account for tombstones/updates
        let raw_results = {
            let idx = self.index.read();
            idx.search(query, k_fetch)
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("VivyIndex search failed: {:?}", e),
                })?
        };

        let u64_map = self.u64_to_id.read();
        let tombstones = self.tombstones.read();

        let mut final_results = Vec::with_capacity(limit);
        let mut seen = HashMap::new();

        for (u64_id, dist) in raw_results {
            if let Some(mem_id) = u64_map.get(&u64_id) {
                if tombstones.contains_key(mem_id) {
                    continue;
                }
                if seen.contains_key(mem_id) {
                    continue;
                }
                seen.insert(mem_id.clone(), true);
                final_results.push((mem_id.clone(), dist));
                if final_results.len() == limit {
                    break;
                }
            }
        }

        Ok(final_results)
    }

    fn rebuild<'a>(&self, records: Box<dyn Iterator<Item = &'a MemoryRecord> + 'a>) -> Result<()> {
        let new_idx = VivyIndex::new(self.dims, self.metric, None::<&str>, None::<&str>)
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to create new VivyIndex during rebuild: {:?}", e),
            })?;

        let mut new_id_map = HashMap::new();
        let mut new_u64_map = HashMap::new();
        let mut counter = 1u64;

        for record in records {
            if record.embedding.len() != self.dims {
                continue;
            }
            let num_id = counter;
            counter += 1;

            new_idx
                .insert_with_id(num_id, record.embedding.clone())
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to insert during rebuild: {:?}", e),
                })?;

            new_id_map.insert(record.id.clone(), num_id);
            new_u64_map.insert(num_id, record.id.clone());
        }

        *self.id_to_u64.write() = new_id_map;
        *self.u64_to_id.write() = new_u64_map;
        self.tombstones.write().clear();
        self.next_u64.store(counter, Ordering::Relaxed);
        *self.index.write() = new_idx;

        Ok(())
    }

    fn is_healthy(&self) -> bool {
        true
    }

    fn backup_segments(&self, target_dir: &std::path::Path) -> Result<()> {
        let idx = self.index.read();
        idx.flush_and_copy_segments(target_dir)
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Vector segment backup failed: {:?}", e),
            })?;
        Ok(())
    }
}

/// Deterministic in-memory test double for unit testing.
pub struct InMemoryTestIndex {
    dims: usize,
    vectors: RwLock<HashMap<String, Vec<f32>>>,
}

impl InMemoryTestIndex {
    pub fn new(dims: usize) -> Self {
        Self {
            dims,
            vectors: RwLock::new(HashMap::new()),
        }
    }
}

impl VectorIndex for InMemoryTestIndex {
    fn upsert(&self, memory_id: &str, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dims {
            return Err(MemoryError::DimensionMismatch {
                code: ErrorCode::DimensionMismatch,
                expected: self.dims,
                actual: vector.len(),
            });
        }
        self.vectors
            .write()
            .insert(memory_id.to_string(), vector.to_vec());
        Ok(())
    }

    fn remove(&self, memory_id: &str) -> Result<()> {
        self.vectors.write().remove(memory_id);
        Ok(())
    }

    fn search(&self, query: &[f32], limit: usize) -> Result<Vec<(String, f32)>> {
        if query.len() != self.dims {
            return Err(MemoryError::DimensionMismatch {
                code: ErrorCode::DimensionMismatch,
                expected: self.dims,
                actual: query.len(),
            });
        }

        let map = self.vectors.read();
        let mut scored: Vec<(String, f32)> = map
            .iter()
            .map(|(id, vec)| {
                let dist = vivy_core::distance::cosine(query, vec);
                (id.clone(), dist)
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.truncate(limit);
        Ok(scored)
    }

    fn rebuild<'a>(&self, records: Box<dyn Iterator<Item = &'a MemoryRecord> + 'a>) -> Result<()> {
        let mut map = HashMap::new();
        for record in records {
            if record.embedding.len() == self.dims {
                map.insert(record.id.clone(), record.embedding.clone());
            }
        }
        *self.vectors.write() = map;
        Ok(())
    }

    fn is_healthy(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MemoryKind, MemoryStatus};
    use crate::namespace::MemoryScope;
    use std::sync::Arc;

    fn test_index_contract(index: Arc<dyn VectorIndex>) {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];
        let q = vec![1.0, 0.1, 0.0];

        index.upsert("doc-1", &v1).unwrap();
        index.upsert("doc-2", &v2).unwrap();

        let results = index.search(&q, 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "doc-1");

        // Test remove
        index.remove("doc-1").unwrap();
        let results_after = index.search(&q, 2).unwrap();
        assert_eq!(results_after.len(), 1);
        assert_eq!(results_after[0].0, "doc-2");

        // Test rebuild
        let rec = MemoryRecord {
            id: "doc-rebuilt".into(),
            scope: MemoryScope::new("t", "ns").unwrap(),
            kind: MemoryKind::Fact,
            content: "rebuilt".into(),
            content_hash: vec![],
            embedding: vec![0.9, 0.1, 0.0],
            embedding_model: "test".into(),
            embedding_dims: 3,
            importance: 1.0,
            created_at_ms: 0,
            updated_at_ms: 0,
            last_accessed_at_ms: None,
            access_count: 0,
            expires_at_ms: None,
            status: MemoryStatus::Active,
            revision: 1,
            metadata: HashMap::new(),
            source: HashMap::new(),
        };

        let records = [rec];
        index.rebuild(Box::new(records.iter())).unwrap();
        let results_rebuilt = index.search(&q, 2).unwrap();
        assert_eq!(results_rebuilt.len(), 1);
        assert_eq!(results_rebuilt[0].0, "doc-rebuilt");
    }

    #[test]
    fn test_in_memory_index() {
        let index = Arc::new(InMemoryTestIndex::new(3));
        test_index_contract(index);
    }

    #[test]
    fn test_vivy_vector_index() {
        let index = Arc::new(VivyVectorIndex::new(3, Metric::Cosine).unwrap());
        test_index_contract(index);
    }
}
