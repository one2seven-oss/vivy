use crate::distance::{self, Metric};
use crate::filter::{FilterExpr, FilterIndex};
use crate::hnsw::HnswIndex;
use crate::storage::manifest::Manifest;
use crate::storage::segments::{SealedSegment, SegmentWriter};
use crate::storage::wal::{WalEntry, WalError, WalWriter};
use arc_swap::ArcSwap;
use log::{info, warn};
use parking_lot::{Mutex, RwLock};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::error::VivyError;

pub const NUM_SHARDS: usize = 8;

#[inline]
fn shard_idx(id: u64, num_shards: usize) -> usize {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    id.hash(&mut hasher);
    (hasher.finish() as usize) % num_shards
}

/// Thread-safe vector index (mutable delta segment + atomically-swappable sealed list)
pub struct VivyIndex {
    dims: usize,
    pub(crate) shards: Arc<Vec<RwLock<HnswIndex>>>,
    sealed: Arc<ArcSwap<Vec<Arc<SealedSegment>>>>,
    pub(crate) filter_index: RwLock<FilterIndex>,
    wal: Option<Arc<Mutex<WalWriter>>>,
    running: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
    pub(crate) next_id: AtomicU64,
    data_dir: Option<PathBuf>,
}

impl VivyIndex {
    pub fn new(
        dims: usize,
        metric: Metric,
        wal_path: Option<impl AsRef<Path>>,
        data_dir: Option<impl AsRef<Path>>,
    ) -> Result<Self, WalError> {
        let shards = (0..NUM_SHARDS)
            .map(|_| RwLock::new(HnswIndex::new(dims, metric)))
            .collect::<Vec<_>>();
        let shards = Arc::new(shards);

        let mut next_id = 1u64;

        // 1. Replay uncompacted WAL entries if existing WAL is present
        if let Some(ref path) = wal_path {
            WalWriter::replay(path.as_ref(), |entry| match entry {
                WalEntry::Insert { id, vector } => {
                    let shard = shard_idx(id, NUM_SHARDS);
                    shards[shard].write().insert(id, vector);
                    if id >= next_id {
                        next_id = id.saturating_add(1);
                    }
                }
            })?;
        }

        let wal = wal_path
            .as_ref()
            .map(WalWriter::open)
            .transpose()?
            .map(|w| Arc::new(Mutex::new(w)));
        let data_dir = data_dir.map(|p| p.as_ref().to_path_buf());

        // 2. Discover existing sealed segments in data directory using manifest
        let mut initial_sealed = Vec::new();
        if let Some(ref dir) = data_dir {
            if dir.exists() {
                let manifest = Manifest::load(dir).unwrap_or(None);
                let seg_files: Vec<String> = match manifest {
                    Some(m) => m.segments,
                    None => {
                        let mut found = Vec::new();
                        if let Ok(entries) = std::fs::read_dir(dir) {
                            for entry in entries.flatten() {
                                let path = entry.path();
                                if path.extension().and_then(|s| s.to_str()) == Some("vivy") {
                                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                                        found.push(name.to_string());
                                    }
                                }
                            }
                        }
                        found.sort();
                        if !found.is_empty() {
                            let m = Manifest::new(found.clone());
                            let _ = m.save(dir);
                        }
                        found
                    }
                };

                for fname in seg_files {
                    let path = dir.join(&fname);
                    match SealedSegment::open(&path) {
                        Ok(seg) => {
                            for idx in 0..seg.num_nodes() {
                                if let Ok(id) = seg.id_at(idx) {
                                    if id >= next_id {
                                        next_id = id.saturating_add(1);
                                    }
                                }
                            }
                            initial_sealed.push(Arc::new(seg));
                        }
                        Err(e) => {
                            warn!("Failed to open existing sealed segment {:?}: {:?}", path, e);
                        }
                    }
                }
            }
        }

        let sealed: Arc<ArcSwap<Vec<Arc<SealedSegment>>>> =
            Arc::new(ArcSwap::new(Arc::new(initial_sealed)));

        let mut idx = Self {
            dims,
            shards: shards.clone(),
            sealed: sealed.clone(),
            filter_index: RwLock::new(FilterIndex::new()),
            wal: wal.clone(),
            running: Arc::new(AtomicBool::new(true)),
            handle: None,
            next_id: AtomicU64::new(next_id),
            data_dir,
        };

        let running = idx.running.clone();
        let sealed = idx.sealed.clone();
        let dir = idx.data_dir.clone();
        let m = metric;

        let h = thread::Builder::new()
            .name("vivy-compactor".into())
            .spawn(move || compactor_loop(running, dims, shards, sealed, dir, wal, m))
            .expect("compactor thread");
        idx.handle = Some(h);

        Ok(idx)
    }

    /// Insert with auto-generated ID
    pub fn insert(&self, vector: Vec<f32>) -> Result<u64, VivyError> {
        if vector.len() != self.dims {
            return Err(VivyError::DimensionMismatch);
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        if let Some(ref wal_mutex) = self.wal {
            let mut wal = wal_mutex.lock();
            wal.append(&WalEntry::Insert {
                id,
                vector: vector.clone(),
            })?;
            wal.commit()?;
        }

        let shard = shard_idx(id, NUM_SHARDS);
        self.shards[shard].write().insert(id, vector);
        Ok(id)
    }

    /// Insert with explicit ID, advancing the next_id high-water mark to prevent collisions.
    pub fn insert_with_id(&self, id: u64, vector: Vec<f32>) -> Result<(), VivyError> {
        if vector.len() != self.dims {
            return Err(VivyError::DimensionMismatch);
        }
        self.next_id.fetch_max(id.saturating_add(1), Ordering::SeqCst);

        if let Some(ref wal_mutex) = self.wal {
            let mut wal = wal_mutex.lock();
            wal.append(&WalEntry::Insert {
                id,
                vector: vector.clone(),
            })?;
            wal.commit()?;
        }
        let shard = shard_idx(id, NUM_SHARDS);
        self.shards[shard].write().insert(id, vector);
        Ok(())
    }

    /// Insert with metadata fields for filtering
    pub fn insert_with_metadata(
        &self,
        vector: Vec<f32>,
        metadata: Vec<(String, String)>,
    ) -> Result<u64, VivyError> {
        if vector.len() != self.dims {
            return Err(VivyError::DimensionMismatch);
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);

        if let Some(ref wal_mutex) = self.wal {
            let mut wal = wal_mutex.lock();
            wal.append(&WalEntry::Insert {
                id,
                vector: vector.clone(),
            })?;
            wal.commit()?;
        }

        {
            let mut fi = self.filter_index.write();
            for (field, value) in &metadata {
                fi.insert(id, field, value);
            }
        }

        let shard = shard_idx(id, NUM_SHARDS);
        self.shards[shard].write().insert(id, vector);
        Ok(id)
    }

    /// Batch insert with auto-generated IDs
    pub fn insert_batch(&self, vectors: Vec<Vec<f32>>) -> Result<Vec<u64>, VivyError> {
        self.insert_batch_with_metadata(vectors, None)
    }

    /// Batch insert with metadata fields
    pub fn insert_batch_with_metadata(
        &self,
        vectors: Vec<Vec<f32>>,
        metadata: Option<Vec<Vec<(String, String)>>>,
    ) -> Result<Vec<u64>, VivyError> {
        let mut ids = Vec::with_capacity(vectors.len());
        for _ in 0..vectors.len() {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            ids.push(id);
        }

        for v in &vectors {
            if v.len() != self.dims {
                return Err(VivyError::DimensionMismatch);
            }
        }

        if let Some(ref wal_mutex) = self.wal {
            let mut wal = wal_mutex.lock();
            for (id, vector) in ids.iter().zip(vectors.iter()) {
                wal.append(&WalEntry::Insert {
                    id: *id,
                    vector: vector.clone(),
                })?;
            }
            wal.commit()?;
        }

        if let Some(ref metas) = metadata {
            let mut fi = self.filter_index.write();
            for (id, meta) in ids.iter().zip(metas.iter()) {
                for (field, value) in meta {
                    fi.insert(*id, field, value);
                }
            }
        }

        for (id, vector) in ids.iter().zip(vectors) {
            let shard = shard_idx(*id, NUM_SHARDS);
            self.shards[shard].write().insert(*id, vector);
        }

        Ok(ids)
    }

    /// Search with an optional filter expression
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&FilterExpr>,
    ) -> Result<Vec<(u64, f32)>, VivyError> {
        if query.len() != self.dims {
            return Err(VivyError::DimensionMismatch);
        }
        let (filter_bitmap, has_filter) = match filter {
            Some(expr) => {
                let fi = self.filter_index.read();
                let bitmap = fi.evaluate(expr);
                let empty = bitmap.is_empty();
                (if empty { None } else { Some(bitmap) }, true)
            }
            None => (None, false),
        };

        // If a filter was provided but no items match, return empty
        if has_filter && filter_bitmap.is_none() {
            return Ok(Vec::new());
        }

        let (mut results, metric) = {
            let mut results = Vec::new();
            let metric = self.shards[0].read().metric();
            for shard in self.shards.iter() {
                let shard_results = shard.read().search_filtered(query, k, filter_bitmap.as_ref());
                results.extend(shard_results);
            }
            (results, metric)
        };

        // Search sealed segments
        let sealed_list = self.sealed.load();
        for seg in sealed_list.iter() {
            for idx in 0..seg.num_nodes() {
                if let Ok(id) = seg.id_at(idx) {
                    let passes_filter = filter_bitmap
                        .as_ref()
                        .is_none_or(|bm| bm.contains(id));
                    if !passes_filter {
                        continue;
                    }
                    if let Ok(v) = seg.vector_at(idx) {
                        let d = distance::compute(metric, query, v);
                        results.push((id, d));
                    }
                }
            }
        }

        results.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        Ok(results)
    }

    /// Search across delta + all sealed segments.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>, VivyError> {
        if query.len() != self.dims {
            return Err(VivyError::DimensionMismatch);
        }
        let (mut results, metric) = {
            let mut results = Vec::new();
            let metric = self.shards[0].read().metric();
            for shard in self.shards.iter() {
                let shard_results = shard.read().search(query, k);
                results.extend(shard_results);
            }
            (results, metric)
        };

        let sealed_list = self.sealed.load();
        for seg in sealed_list.iter() {
            for idx in 0..seg.num_nodes() {
                if let Ok(id) = seg.id_at(idx) {
                    if let Ok(v) = seg.vector_at(idx) {
                        let d = distance::compute(metric, query, v);
                        results.push((id, d));
                    }
                }
            }
        }

        results.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        Ok(results)
    }

    pub fn delta_len(&self) -> usize {
        self.shards.iter().map(|s| s.read().len()).sum()
    }

    pub fn num_sealed(&self) -> usize {
        self.sealed.load().len()
    }

    /// Force immediate compaction.
    pub fn compact_now(&self) {
        if let Some(ref dir) = self.data_dir {
            let m = self.shards[0].read().metric();
            let wal_ref = self.wal.as_deref();
            if let Err(e) = run_compaction(self.dims, &self.shards, &self.sealed, dir, wal_ref, m) {
                warn!("compaction failed: {e}");
            }
        }
    }
}

impl Drop for VivyIndex {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Release);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn compactor_loop(
    running: Arc<AtomicBool>,
    dims: usize,
    shards: Arc<Vec<RwLock<HnswIndex>>>,
    sealed: Arc<ArcSwap<Vec<Arc<SealedSegment>>>>,
    data_dir: Option<PathBuf>,
    wal: Option<Arc<Mutex<WalWriter>>>,
    metric: Metric,
) {
    let threshold = 10_000usize;
    while running.load(Ordering::Acquire) {
        thread::sleep(Duration::from_secs(5));
        let total_len: usize = shards.iter().map(|s| s.read().len()).sum();
        if total_len >= threshold {
            if let Some(ref dir) = data_dir {
                let wal_ref = wal.as_deref();
                if let Err(e) = run_compaction(dims, &shards, &sealed, dir, wal_ref, metric) {
                    warn!("compaction failed: {e}");
                }
            }
        }
    }
}

fn run_compaction(
    dims: usize,
    shards: &[RwLock<HnswIndex>],
    sealed: &ArcSwap<Vec<Arc<SealedSegment>>>,
    data_dir: &Path,
    wal: Option<&Mutex<WalWriter>>,
    metric: Metric,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut entries = Vec::new();
    for shard in shards {
        let mut old = {
            let mut guard = shard.write();
            let fresh = HnswIndex::new(dims, metric);
            std::mem::replace(&mut *guard, fresh)
        };
        entries.extend(old.drain());
    }
    if entries.is_empty() {
        return Ok(());
    }
    info!("compacting {} vectors into sealed segment", entries.len());

    let seg_filename = format!("seg-{}.vivy", timestamp_ns());
    let seg_path = data_dir.join(&seg_filename);
    let file = std::fs::File::create(&seg_path)?;
    let writer = BufWriter::new(file.try_clone()?);

    let mut seg_writer = SegmentWriter::new(writer, dims as u32, 16, 32);
    for (id, vector) in &entries {
        seg_writer.push(*id, 0, vec![Vec::new()], vector.clone())?;
    }
    seg_writer.write()?;
    drop(file);

    // Atomically record newly sealed segment in manifest
    let mut manifest = Manifest::load(data_dir)?.unwrap_or_else(|| Manifest::new(Vec::new()));
    manifest.segments.push(seg_filename);
    manifest.save(data_dir)?;

    // Reset WAL to clear compacted delta entries
    if let Some(wal_mutex) = wal {
        let mut wal_guard = wal_mutex.lock();
        wal_guard.reset()?;
    }

    let new_seg = Arc::new(SealedSegment::open(&seg_path)?);

    sealed.rcu(|list| {
        let mut new_list = (**list).clone();
        new_list.push(new_seg.clone());
        new_list
    });

    info!(
        "sealed segment written: {} nodes at {:?}",
        entries.len(),
        seg_path
    );
    Ok(())
}

fn timestamp_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_basic_insert_search() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("test.wal");
        let data = dir.path().join("segments");
        std::fs::create_dir_all(&data).unwrap();

        let idx = VivyIndex::new(3, Metric::L2, Some(&wal), Some(&data)).unwrap();
        idx.insert(vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert(vec![0.0, 1.0, 0.0]).unwrap();

        let res = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert!(!res.is_empty());
        drop(idx);
    }

    #[test]
    fn test_compaction() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("test.wal");
        let data = dir.path().join("segments");
        std::fs::create_dir_all(&data).unwrap();

        let idx = VivyIndex::new(3, Metric::L2, Some(&wal), Some(&data)).unwrap();
        for i in 0..5 {
            idx.insert(vec![i as f32, 0.0, 0.0]).unwrap();
        }
        assert_eq!(idx.delta_len(), 5);
        idx.compact_now();
        assert_eq!(idx.delta_len(), 0);
        assert_eq!(idx.num_sealed(), 1);

        let res = idx.search(&[3.0, 0.0, 0.0], 1).unwrap();
        assert!(!res.is_empty());
        drop(idx);
    }

    #[test]
    fn test_512_dim() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("test.wal");
        let data = dir.path().join("segments");
        std::fs::create_dir_all(&data).unwrap();

        let idx = VivyIndex::new(512, Metric::L2, Some(&wal), Some(&data)).unwrap();
        let v1 = vec![1.0; 512];
        let v2 = vec![0.0; 512];

        idx.insert(v1.clone()).unwrap();
        idx.insert(v2.clone()).unwrap();

        let res = idx.search(&v1, 1).unwrap();
        assert_eq!(res[0].0, 1);
    }

    #[test]
    fn test_64bit_id_preservation_and_filtering() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("test.wal");
        let data = dir.path().join("segments");
        std::fs::create_dir_all(&data).unwrap();

        let idx = VivyIndex::new(3, Metric::L2, Some(&wal), Some(&data)).unwrap();
        let large_id_1 = (1u64 << 40) + 123;
        let large_id_2 = u64::MAX - 99;

        idx.insert_with_id(large_id_1, vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert_with_id(large_id_2, vec![0.0, 1.0, 0.0]).unwrap();

        // Check search returns exact 64-bit IDs without truncation
        let res = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(res[0].0, large_id_1);

        let res2 = idx.search(&[0.0, 1.0, 0.0], 1).unwrap();
        assert_eq!(res2[0].0, large_id_2);

        // Compact to sealed segment and verify 64-bit IDs in sealed segment
        idx.compact_now();
        assert_eq!(idx.num_sealed(), 1);

        let res_sealed = idx.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(res_sealed[0].0, large_id_1);
    }

    #[test]
    fn test_explicit_id_high_water_mark() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("test.wal");
        let idx = VivyIndex::new(3, Metric::L2, Some(&wal), Option::<&Path>::None).unwrap();

        idx.insert_with_id(500, vec![1.0, 0.0, 0.0]).unwrap();
        let auto_id = idx.insert(vec![0.0, 1.0, 0.0]).unwrap();
        assert!(auto_id > 500, "auto-generated ID must be strictly greater than explicit high-water mark, got {}", auto_id);
    }

    #[test]
    fn test_reopen_wal_and_sealed_recovery() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("test.wal");
        let data = dir.path().join("segments");
        std::fs::create_dir_all(&data).unwrap();

        {
            let idx = VivyIndex::new(3, Metric::L2, Some(&wal), Some(&data)).unwrap();
            idx.insert_with_id(10, vec![1.0, 0.0, 0.0]).unwrap();
            idx.insert_with_id(20, vec![0.0, 1.0, 0.0]).unwrap();
            // Compact 10 & 20 into a sealed segment
            idx.compact_now();
            assert_eq!(idx.num_sealed(), 1);

            // Insert 30 into WAL (uncompacted delta)
            idx.insert_with_id(30, vec![0.0, 0.0, 1.0]).unwrap();
            assert_eq!(idx.delta_len(), 1);
        }

        // Reopen index from the same directory & wal
        let reopened = VivyIndex::new(3, Metric::L2, Some(&wal), Some(&data)).unwrap();
        assert_eq!(reopened.num_sealed(), 1);
        assert_eq!(reopened.delta_len(), 1);

        // Search for all 3 vectors
        let res1 = reopened.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(res1[0].0, 10);

        let res2 = reopened.search(&[0.0, 1.0, 0.0], 1).unwrap();
        assert_eq!(res2[0].0, 20);

        let res3 = reopened.search(&[0.0, 0.0, 1.0], 1).unwrap();
        assert_eq!(res3[0].0, 30);

        // Verify high-water mark after reopen
        let next_auto = reopened.insert(vec![0.5, 0.5, 0.0]).unwrap();
        assert!(next_auto > 30, "next auto ID must be > 30, got {}", next_auto);
    }
}
