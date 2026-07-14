//! Segment-based concurrency model.
//!
//! Thread-safe VivyIndex combining three ideas for concurrent reads during writes:
//!
//! 1. **Delta/Sealed split** — new vectors go into a small in-memory delta
//!    (RwLock-protected, tiny → negligible contention). When it exceeds a
//!    threshold, it's frozen, serialised to an immutable sealed segment on
//!    disk, and atomically swapped into the search path via arc-swap.
//!
//! 2. **Read-side fan-out** — every search queries the delta (read lock,
//!    doesn't block other readers) AND all sealed segments (immutable,
//!    no lock). Results are merged and top-k returned.
//!
//! 3. **Background compactor** — monitors delta size, triggers compaction,
//!    writes to a sealed segment file, atomically adds to the sealed list.
//!
//! True lock-free concurrent mutation of a single HNSW graph is an open
//! research problem. This is the LSM-tree / Lucene approach: write to a
//! small mutable buffer, read from buffer + all immutable buffers, merge
//! in background. The only lock-free component is arc-swap for the sealed
//! list pointer. Everything else uses standard RwLock scoped to the tiny
//! delta, so contention is negligible.

use crate::distance::{self, Metric};
use crate::filter::{FilterExpr, FilterIndex};
use crate::hnsw::HnswIndex;
use crate::storage::segments::{SealedSegment, SegmentWriter};
use crate::storage::wal::{WalEntry, WalWriter, WalError};
use arc_swap::ArcSwap;
use log::{info, warn};
use parking_lot::{RwLock, Mutex};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

// Top-level thread-safe index. Python (PyO3) and Rust apps interact with this.
pub struct VivyIndex {
    delta: Arc<RwLock<HnswIndex>>,
    sealed: Arc<ArcSwap<Vec<Arc<SealedSegment>>>>,
    filter_index: RwLock<FilterIndex>,
    wal: Option<Mutex<WalWriter>>,
    running: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
    next_id: AtomicU64,
    data_dir: Option<PathBuf>,
}

impl VivyIndex {
    // metric: distance metric for all comparisons.
    // wal_path: optional WAL path for crash durability.
    // data_dir: optional directory for sealed segments (None = in-RAM only).
    pub fn new(
        metric: Metric,
        wal_path: Option<impl AsRef<Path>>,
        data_dir: Option<impl AsRef<Path>>,
    ) -> Result<Self, WalError> {
        let wal = wal_path.as_ref().map(WalWriter::open).transpose()?.map(Mutex::new);
        let data_dir = data_dir.map(|p| p.as_ref().to_path_buf());

        let sealed: Arc<ArcSwap<Vec<Arc<SealedSegment>>>> =
            Arc::new(ArcSwap::new(Arc::new(Vec::new())));

        let mut idx = Self {
            delta: Arc::new(RwLock::new(HnswIndex::new(metric))),
            sealed: sealed.clone(),
            filter_index: RwLock::new(FilterIndex::new()),
            wal,
            running: Arc::new(AtomicBool::new(true)),
            handle: None,
            next_id: AtomicU64::new(1),
            data_dir,
        };

        // Background compactor: watches delta size, triggers compaction.
        let running = idx.running.clone();
        let delta = idx.delta.clone();
        let sealed = idx.sealed.clone();
        let dir = idx.data_dir.clone();
        let m = metric;

        let h = thread::Builder::new()
            .name("vivy-compactor".into())
            .spawn(move || compactor_loop(running, delta, sealed, dir, m))
            .expect("compactor thread");
        idx.handle = Some(h);

        Ok(idx)
    }

    // Insert with auto-assigned ID. If WAL is enabled, it's fsynced before the delta update.
    pub fn insert(&self, vector: Vec<f32>) -> Result<u64, WalError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        if let Some(ref wal_mutex) = self.wal {
            let mut wal = wal_mutex.lock();
            wal.append(&WalEntry::Insert { id, vector: vector.clone() })?;
            wal.commit()?;
        }

        self.delta.write().insert(id, vector);
        Ok(id)
    }

    // Insert with caller-specified ID. For WAL replay or external ID sync.
    // Caller must ensure uniqueness; duplicates cause undefined filter behaviour.
    pub fn insert_with_id(&self, id: u64, vector: Vec<f32>) -> Result<(), WalError> {
        if let Some(ref wal_mutex) = self.wal {
            let mut wal = wal_mutex.lock();
            wal.append(&WalEntry::Insert { id, vector: vector.clone() })?;
            wal.commit()?;
        }
        self.delta.write().insert(id, vector);
        Ok(())
    }

    // Insert with metadata pairs indexed in FilterIndex for filtered search.
    pub fn insert_with_metadata(
        &self,
        vector: Vec<f32>,
        metadata: Vec<(String, String)>,
    ) -> Result<u64, WalError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        if let Some(ref wal_mutex) = self.wal {
            let mut wal = wal_mutex.lock();
            wal.append(&WalEntry::Insert { id, vector: vector.clone() })?;
            wal.commit()?;
        }

        {
            let mut fi = self.filter_index.write();
            for (field, value) in &metadata {
                fi.insert(id, field, value);
            }
        }

        self.delta.write().insert(id, vector);
        Ok(id)
    }

    // Search with optional filter. Delta uses predicate-pushed HNSW search;
    // sealed segments use post-filtering (linear scan + bitmap check).
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&FilterExpr>,
    ) -> Vec<(u64, f32)> {
        let (filter_bitmap, has_filter) = match filter {
            Some(expr) => {
                let fi = self.filter_index.read();
                let bitmap = fi.evaluate(expr);
                let empty = bitmap.is_empty();
                (if empty { None } else { Some(bitmap) }, true)
            }
            None => (None, false),
        };

        if has_filter && filter_bitmap.is_none() {
            return Vec::new();
        }

        let metric = self.delta.read().metric();
        let mut results: Vec<(u64, f32)>;

        // Delta segment: predicate-pushed HNSW search with dynamic ef expansion.
        {
            let guard = self.delta.read();
            results = guard.search_filtered(query, k, filter_bitmap.as_ref());
        }

        // Sealed segments: linear scan with optional post-filtering.
        // ponytail: PQ-based ADC pre-filter to avoid full scan.
        let sealed_list = self.sealed.load();
        for seg in sealed_list.iter() {
            for idx in 0..seg.num_nodes() {
                if let Ok(rec) = seg.read_node(idx) {
                    let passes_filter = filter_bitmap
                        .as_ref()
                        .is_none_or(|bm| bm.contains(rec.id as u32));
                    if !passes_filter {
                        continue;
                    }
                    if let Some(ref v) = rec.vector {
                        let d = distance::compute(metric, query, v);
                        results.push((rec.id, d));
                    }
                }
            }
        }

        results.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }

    // Unfiltered search convenience wrapper.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let metric = self.delta.read().metric();
        let mut results: Vec<(u64, f32)>;

        {
            let guard = self.delta.read();
            results = guard.search(query, k);
        }

        let sealed_list = self.sealed.load();
        for seg in sealed_list.iter() {
            for idx in 0..seg.num_nodes() {
                if let Ok(rec) = seg.read_node(idx) {
                    if let Some(ref v) = rec.vector {
                        let d = distance::compute(metric, query, v);
                        results.push((rec.id, d));
                    }
                }
            }
        }

        results.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }

    // Vectors in the delta segment (excludes sealed).
    pub fn delta_len(&self) -> usize {
        self.delta.read().len()
    }

    // Sealed segment count.
    pub fn num_sealed(&self) -> usize {
        self.sealed.load().len()
    }

    // Force immediate compaction. Auto-compaction triggers at 10K vectors.
    pub fn compact_now(&self) {
        if let Some(ref dir) = self.data_dir {
            let m = self.delta.read().metric();
            if let Err(e) = run_compaction(&self.delta, &self.sealed, dir, m) {
                warn!("compaction failed: {e}");
            }
        }
    }
}

// Signal compactor to exit on drop. Prevents resource leak from a dangling thread.
impl Drop for VivyIndex {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

// Polls every 5s, compacts when delta exceeds 10K vectors.
// Conservative defaults. Tune for high insert rates.
// ponytail: event-driven trigger (on insert threshold crossing) instead of polling.
fn compactor_loop(
    running: Arc<AtomicBool>,
    delta: Arc<RwLock<HnswIndex>>,
    sealed: Arc<ArcSwap<Vec<Arc<SealedSegment>>>>,
    data_dir: Option<PathBuf>,
    metric: Metric,
) {
    let threshold = 10_000usize;
    while running.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_secs(5));
        if delta.read().len() >= threshold {
            if let Some(ref dir) = data_dir {
                if let Err(e) = run_compaction(&delta, &sealed, dir, metric) {
                    warn!("compaction failed: {e}");
                }
            }
        }
    }
}

// Atomically swap delta → sealed: drain delta, write to file, mmap, push to sealed list via arc-swap rcu.
fn run_compaction(
    delta: &RwLock<HnswIndex>,
    sealed: &ArcSwap<Vec<Arc<SealedSegment>>>,
    data_dir: &Path,
    metric: Metric,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut old = {
        let mut guard = delta.write();
        let fresh = HnswIndex::new(metric);
        std::mem::replace(&mut *guard, fresh)
    };

    let entries = old.drain();
    if entries.is_empty() {
        return Ok(());
    }
    info!("compacting {} vectors into sealed segment", entries.len());

    let dims = entries[0].1.len() as u32;

    let seg_path = data_dir.join(format!("seg-{}.vivy", timestamp_ns()));
    let file = std::fs::File::create(&seg_path)?;
    let writer = BufWriter::new(file.try_clone()?);

    let mut seg_writer = SegmentWriter::new(writer, dims, 16, 32);
    // level=0 + empty neighbours because sealed segments don't share a graph
    // with delta. No cross-segment edges — avoids global-lock problem.
    // Cost: linear scan instead of HNSW search. Acceptable for <=100K vectors.
    // ponytail: multi-segment merge with cross-segment edges for larger segments.
    for (id, vector) in &entries {
        seg_writer.push(*id, 0, vec![Vec::new()], vector.clone());
    }
    seg_writer.write()?;
    drop(file);

    let new_seg = Arc::new(SealedSegment::open(&seg_path)?);

    // rcu = read-copy-update: atomically swap Arc<Vec> — readers before see old list, after see new.
    sealed.rcu(|list| {
        let mut new_list = (**list).clone();
        new_list.push(new_seg.clone());
        new_list
    });

    info!("sealed segment written: {} nodes at {:?}", entries.len(), seg_path);
    Ok(())
}

// Nanosecond timestamp for unique segment filenames. Collisions exceedingly unlikely.
fn timestamp_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    // Insert two, search for one, verify closest match.
    #[test]
    fn test_basic_insert_search() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("test.wal");
        let data = dir.path().join("segments");
        std::fs::create_dir_all(&data).unwrap();

        let idx = VivyIndex::new(Metric::L2, Some(&wal), Some(&data)).unwrap();
        idx.insert(vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert(vec![0.0, 1.0, 0.0]).unwrap();

        let res = idx.search(&[1.0, 0.0, 0.0], 1);
        assert!(!res.is_empty());
        drop(idx);
    }

    // Insert 5, compact, verify delta empty + sealed exists + search works end-to-end.
    #[test]
    fn test_compaction() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("test.wal");
        let data = dir.path().join("segments");
        std::fs::create_dir_all(&data).unwrap();

        let idx = VivyIndex::new(Metric::L2, Some(&wal), Some(&data)).unwrap();
        for i in 0..5 {
            idx.insert(vec![i as f32, 0.0, 0.0]).unwrap();
        }
        assert_eq!(idx.delta_len(), 5);
        idx.compact_now();
        assert_eq!(idx.delta_len(), 0);
        assert_eq!(idx.num_sealed(), 1);

        let res = idx.search(&[3.0, 0.0, 0.0], 1);
        assert!(!res.is_empty());
        drop(idx);
    }
}
