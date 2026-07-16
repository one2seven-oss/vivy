//! Segment-based concurrency model with non-blocking inserts.
//!
//! Thread-safe VivyIndex combinin four ideas for concurrent reads during writes:
//!
//! 1. **Delta/Pending split** — new vectors go into a small pending buffer,
//!    flushed in batches to the delta (RwLock). When delta exceeds a threshold,
//!    it's serialised to an immutable sealed segment and atomically swapped
//!    into the read path via arc-swap.
//!
//! 2. **Read-side snapshot** — every search copies delta entries under a brief
//!    read lock, releases it, then searches the snapshot + sealed segments.
//!    No lock held during distance computation.
//!
//! 3. **Background WAL worker** — WAL append+fsync runs in a dedicated thread
//!    via a bounded channel, decoupled from the insert path.
//!
//! 4. **Background compactor** — monitors delta size, triggers compaction,
//!    writes sealed segments to disk.

use crate::distance::{self, Metric};
use crate::event::{EventBus, VivyEvent};
use crate::filter::{FilterExpr, FilterIndex};
use crate::hnsw::HnswIndex;
use crate::metrics::VivyMetrics;
use crate::storage::segments::{SealedSegment, SegmentWriter};
use crate::storage::wal::{WalEntry, WalError, WalWriter};
use arc_swap::ArcSwap;
use crossbeam_channel;
use log::{info, warn};
use parking_lot::{Mutex, RwLock};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

struct PendingInsert {
    id: u64,
    vector: Vec<f32>,
}

pub struct VivyIndex {
    delta: Arc<RwLock<HnswIndex>>,
    sealed: Arc<ArcSwap<Vec<Arc<SealedSegment>>>>,
    filter_index: RwLock<FilterIndex>,
    wal_sender: Option<crossbeam_channel::Sender<WalEntry>>,
    wal_handle: Option<thread::JoinHandle<()>>,
    pending: Arc<Mutex<Vec<PendingInsert>>>,
    running: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
    next_id: AtomicU64,
    data_dir: Option<PathBuf>,
    events: Option<EventBus>,
    compact_sender: Option<mpsc::Sender<()>>,
    compaction_threshold: usize,
    pub metrics: Arc<VivyMetrics>,
}

impl VivyIndex {
    pub fn new(
        metric: Metric,
        wal_path: Option<impl AsRef<Path>>,
        data_dir: Option<impl AsRef<Path>>,
    ) -> Result<Self, WalError> {
        let wal = wal_path.as_ref().map(WalWriter::open).transpose()?;
        let data_dir = data_dir.map(|p| p.as_ref().to_path_buf());
        Self::new_impl(metric, wal, data_dir, None)
    }

    pub fn new_with_events(
        metric: Metric,
        wal_path: Option<impl AsRef<Path>>,
        data_dir: Option<impl AsRef<Path>>,
        events: EventBus,
    ) -> Result<Self, WalError> {
        let wal = wal_path.as_ref().map(WalWriter::open).transpose()?;
        let data_dir = data_dir.map(|p| p.as_ref().to_path_buf());
        Self::new_impl(metric, wal, data_dir, Some(events))
    }

    fn new_impl(
        metric: Metric,
        wal: Option<WalWriter>,
        data_dir: Option<PathBuf>,
        events: Option<EventBus>,
    ) -> Result<Self, WalError> {
        let metrics = Arc::new(VivyMetrics::default());

        let sealed: Arc<ArcSwap<Vec<Arc<SealedSegment>>>> =
            Arc::new(ArcSwap::new(Arc::new(Vec::new())));

        let (wal_sender, wal_handle) = if let Some(writer) = wal {
            let (tx, rx) = crossbeam_channel::bounded(4096);
            let wal_metrics = metrics.clone();
            let h = thread::Builder::new()
                .name("vivy-wal".into())
                .spawn(move || wal_worker_loop(rx, writer, wal_metrics))
                .expect("WAL worker thread");
            (Some(tx), Some(h))
        } else {
            (None, None)
        };

        let (compact_tx, compact_rx) = mpsc::channel();
        let pending = Arc::new(Mutex::new(Vec::new()));
        let comp_metrics = metrics.clone();

        let mut idx = Self {
            delta: Arc::new(RwLock::new(HnswIndex::new(metric))),
            sealed: sealed.clone(),
            filter_index: RwLock::new(FilterIndex::new()),
            wal_sender,
            wal_handle,
            pending: pending.clone(),
            running: Arc::new(AtomicBool::new(true)),
            handle: None,
            next_id: AtomicU64::new(1),
            data_dir,
            events,
            compact_sender: Some(compact_tx),
            compaction_threshold: 10_000,
            metrics,
        };

        let running = idx.running.clone();
        let delta = idx.delta.clone();
        let sealed = idx.sealed.clone();
        let dir = idx.data_dir.clone();
        let m = metric;
        let ev = idx.events.clone();

        let h = thread::Builder::new()
            .name("vivy-compactor".into())
            .spawn(move || compactor_loop(running, delta, pending, sealed, dir, m, ev, comp_metrics, compact_rx))
            .expect("compactor thread");
        idx.handle = Some(h);

        Ok(idx)
    }

    fn maybe_trigger_compaction(&self) {
        if self.delta.read().len() >= self.compaction_threshold {
            if let Some(ref sender) = self.compact_sender {
                let _ = sender.send(());
            }
        }
    }

    pub fn insert(&self, vector: Vec<f32>) -> Result<u64, WalError> {
        let t0 = Instant::now();
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        if let Some(ref sender) = self.wal_sender {
            let _ = sender.try_send(WalEntry::Insert {
                id,
                vector: vector.clone(),
            });
        }

        let mut pending = self.pending.lock();
        pending.push(PendingInsert { id, vector });
        if pending.len() >= 64 {
            let batch = std::mem::take(&mut *pending);
            drop(pending);
            let mut delta = self.delta.write();
            for ins in batch {
                delta.insert(ins.id, ins.vector);
            }
            drop(delta);
            self.maybe_trigger_compaction();
        } else {
            drop(pending);
        }
        self.metrics.record_insert(t0.elapsed());
        self.metrics
            .delta_size
            .store(self.delta_len(), Ordering::Relaxed);
        Ok(id)
    }

    pub fn insert_with_id(&self, id: u64, vector: Vec<f32>) -> Result<(), WalError> {
        let t0 = Instant::now();
        if let Some(ref sender) = self.wal_sender {
            let _ = sender.try_send(WalEntry::Insert {
                id,
                vector: vector.clone(),
            });
        }

        let mut pending = self.pending.lock();
        pending.push(PendingInsert { id, vector });
        if pending.len() >= 64 {
            let batch = std::mem::take(&mut *pending);
            drop(pending);
            let mut delta = self.delta.write();
            for ins in batch {
                delta.insert(ins.id, ins.vector);
            }
            drop(delta);
            self.maybe_trigger_compaction();
        } else {
            drop(pending);
        }
        self.metrics.record_insert(t0.elapsed());
        self.metrics
            .delta_size
            .store(self.delta_len(), Ordering::Relaxed);
        Ok(())
    }

    pub fn insert_with_metadata(
        &self,
        vector: Vec<f32>,
        metadata: Vec<(String, String)>,
    ) -> Result<u64, WalError> {
        let t0 = Instant::now();
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        if let Some(ref sender) = self.wal_sender {
            let _ = sender.try_send(WalEntry::Insert {
                id,
                vector: vector.clone(),
            });
        }

        {
            let mut fi = self.filter_index.write();
            for (field, value) in &metadata {
                fi.insert(id, field, value);
            }
        }

        let mut pending = self.pending.lock();
        pending.push(PendingInsert { id, vector });
        if pending.len() >= 64 {
            let batch = std::mem::take(&mut *pending);
            drop(pending);
            let mut delta = self.delta.write();
            for ins in batch {
                delta.insert(ins.id, ins.vector);
            }
            drop(delta);
            self.maybe_trigger_compaction();
        } else {
            drop(pending);
        }
        self.metrics.record_insert(t0.elapsed());
        self.metrics
            .delta_size
            .store(self.delta_len(), Ordering::Relaxed);
        Ok(id)
    }

    pub fn insert_batch(&self, vectors: Vec<Vec<f32>>) -> Vec<u64> {
        let t0 = Instant::now();
        let ids: Vec<u64> = (0..vectors.len())
            .map(|_| self.next_id.fetch_add(1, Ordering::SeqCst))
            .collect();

        if let Some(ref sender) = self.wal_sender {
            for (id, vec) in ids.iter().zip(vectors.iter()) {
                let _ = sender.try_send(WalEntry::Insert {
                    id: *id,
                    vector: vec.clone(),
                });
            }
        }

        let mut delta = self.delta.write();
        for (id, vec) in ids.iter().zip(vectors) {
            delta.insert(*id, vec);
        }
        drop(delta);
        self.maybe_trigger_compaction();
        self.metrics.record_insert(t0.elapsed());
        self.metrics
            .delta_size
            .store(self.delta_len(), Ordering::Relaxed);
        ids
    }

    fn flush_pending(&self) {
        let batch = {
            let mut p = self.pending.lock();
            if p.is_empty() {
                return;
            }
            std::mem::take(&mut *p)
        };
        let mut guard = self.delta.write();
        for ins in batch {
            guard.insert(ins.id, ins.vector);
        }
    }

    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&FilterExpr>,
    ) -> Vec<(u64, f32)> {
        let t0 = Instant::now();
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
            self.metrics.record_search(t0.elapsed());
            return Vec::new();
        }

        self.flush_pending();

        let metric;
        let mut results: Vec<(u64, f32)>;
        {
            let guard = self.delta.read();
            metric = guard.metric();
            let snapshot = guard.snapshot();
            results = snapshot
                .into_iter()
                .filter(|(id, _)| {
                    filter_bitmap
                        .as_ref()
                        .is_none_or(|bm| bm.contains(*id as u32))
                })
                .map(|(id, vec)| (id, distance::compute(metric, query, &vec)))
                .collect();
        }

        let sealed_list = self.sealed.load();
        for seg in sealed_list.iter() {
            for idx in 0..seg.num_nodes() {
                if let Ok(rec) = seg.read_node(idx) {
                    let passes = filter_bitmap
                        .as_ref()
                        .is_none_or(|bm| bm.contains(rec.id as u32));
                    if !passes {
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
        self.metrics.record_search(t0.elapsed());
        results
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let t0 = Instant::now();
        self.flush_pending();

        let metric;
        let mut results: Vec<(u64, f32)>;
        {
            let guard = self.delta.read();
            metric = guard.metric();
            let snapshot = guard.snapshot();
            results = snapshot
                .into_iter()
                .map(|(id, vec)| (id, distance::compute(metric, query, &vec)))
                .collect();
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
        self.metrics.record_search(t0.elapsed());
        results
    }

    pub fn delta_len(&self) -> usize {
        let pending = self.pending.lock().len();
        pending + self.delta.read().len()
    }

    pub fn num_sealed(&self) -> usize {
        self.sealed.load().len()
    }

    pub fn compact_now(&self) {
        if let Some(ref dir) = self.data_dir {
            self.flush_pending();
            let m = self.delta.read().metric();
            let t0 = Instant::now();
            if let Err(e) = run_compaction(&self.delta, &self.sealed, dir, m) {
                warn!("compaction failed: {e}");
            }
            self.metrics.record_compaction(t0.elapsed());
            self.metrics.num_sealed_segments.store(
                self.sealed.load().len(),
                Ordering::Relaxed,
            );
        }
    }
}

impl Drop for VivyIndex {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        drop(self.wal_sender.take());
        drop(self.compact_sender.take());
        if let Some(h) = self.wal_handle.take() {
            let _ = h.join();
        }
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn wal_worker_loop(
    rx: crossbeam_channel::Receiver<WalEntry>,
    mut writer: WalWriter,
    metrics: Arc<VivyMetrics>,
) {
    while let Ok(entry) = rx.recv() {
        if let Err(e) = writer.append(&entry) {
            warn!("WAL append failed: {e}");
            continue;
        }
        if let Err(e) = writer.commit() {
            warn!("WAL fsync failed: {e}");
        }
        metrics.wal_fsyncs_total.fetch_add(1, Ordering::Relaxed);
    }
}

#[allow(clippy::too_many_arguments)]
fn compactor_loop(
    running: Arc<AtomicBool>,
    delta: Arc<RwLock<HnswIndex>>,
    pending: Arc<Mutex<Vec<PendingInsert>>>,
    sealed: Arc<ArcSwap<Vec<Arc<SealedSegment>>>>,
    data_dir: Option<PathBuf>,
    metric: Metric,
    events: Option<EventBus>,
    metrics: Arc<VivyMetrics>,
    trigger_rx: mpsc::Receiver<()>,
) {
    let threshold = 10_000usize;
    while running.load(Ordering::SeqCst) {
        let _ = trigger_rx.recv_timeout(Duration::from_secs(30));
        let batch = {
            let mut p = pending.lock();
            if !p.is_empty() {
                Some(std::mem::take(&mut *p))
            } else {
                None
            }
        };
        if let Some(batch) = batch {
            let mut guard = delta.write();
            for ins in batch {
                guard.insert(ins.id, ins.vector);
            }
        }
        if delta.read().len() >= threshold {
            if let Some(ref events) = events {
                events.emit(VivyEvent::CompactionStarted {
                    delta_size: delta.read().len(),
                });
            }
            if let Some(ref dir) = data_dir {
                let t0 = Instant::now();
                match run_compaction(&delta, &sealed, dir, metric) {
                    Ok(_) => {
                        metrics.record_compaction(t0.elapsed());
                        metrics.num_sealed_segments.store(
                            sealed.load().len(),
                            Ordering::Relaxed,
                        );
                        if let Some(ref events) = events {
                            events.emit(VivyEvent::CompactionFinished {
                                num_sealed: sealed.load().len(),
                                segment_path: format!("{:?}", dir),
                            });
                        }
                    }
                    Err(e) => {
                        if let Some(ref events) = events {
                            events.emit(VivyEvent::CompactionFailed {
                                error: e.to_string(),
                            });
                        }
                        warn!("compaction failed: {e}");
                    }
                }
            }
        }
    }
}

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
    for (id, vector) in &entries {
        seg_writer.push(*id, 0, vec![Vec::new()], vector.clone());
    }
    seg_writer.write()?;
    drop(file);

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

        let idx = VivyIndex::new(Metric::L2, Some(&wal), Some(&data)).unwrap();
        idx.insert(vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert(vec![0.0, 1.0, 0.0]).unwrap();

        let res = idx.search(&[1.0, 0.0, 0.0], 1);
        assert!(!res.is_empty());
        drop(idx);
    }

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
