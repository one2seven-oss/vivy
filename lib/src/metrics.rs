use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

#[derive(Default)]
pub struct VivyMetrics {
    pub inserts_total: AtomicU64,
    pub searches_total: AtomicU64,
    pub compactions_total: AtomicU64,
    pub wal_fsyncs_total: AtomicU64,
    pub delta_size: AtomicUsize,
    pub num_sealed_segments: AtomicUsize,
    pub search_time_ns_sum: AtomicU64,
    pub search_time_ns_count: AtomicU64,
    pub insert_time_ns_sum: AtomicU64,
    pub insert_time_ns_count: AtomicU64,
    pub compaction_time_ns_sum: AtomicU64,
    pub compaction_time_ns_count: AtomicU64,
}

impl VivyMetrics {
    pub fn record_insert(&self, dur: Duration) {
        self.inserts_total.fetch_add(1, Ordering::Relaxed);
        self.insert_time_ns_sum
            .fetch_add(dur.as_nanos() as u64, Ordering::Relaxed);
        self.insert_time_ns_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_search(&self, dur: Duration) {
        self.searches_total.fetch_add(1, Ordering::Relaxed);
        self.search_time_ns_sum
            .fetch_add(dur.as_nanos() as u64, Ordering::Relaxed);
        self.search_time_ns_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_compaction(&self, dur: Duration) {
        self.compactions_total.fetch_add(1, Ordering::Relaxed);
        self.compaction_time_ns_sum
            .fetch_add(dur.as_nanos() as u64, Ordering::Relaxed);
        self.compaction_time_ns_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            inserts_total: self.inserts_total.load(Ordering::Relaxed),
            searches_total: self.searches_total.load(Ordering::Relaxed),
            compactions_total: self.compactions_total.load(Ordering::Relaxed),
            avg_insert_time_ns: self.avg_insert_time_ns(),
            avg_search_time_ns: self.avg_search_time_ns(),
            delta_size: self.delta_size.load(Ordering::Relaxed),
            num_sealed: self.num_sealed_segments.load(Ordering::Relaxed),
        }
    }

    fn avg_insert_time_ns(&self) -> f64 {
        let count = self.insert_time_ns_count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        self.insert_time_ns_sum.load(Ordering::Relaxed) as f64 / count as f64
    }

    fn avg_search_time_ns(&self) -> f64 {
        let count = self.search_time_ns_count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        self.search_time_ns_sum.load(Ordering::Relaxed) as f64 / count as f64
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct MetricsSnapshot {
    pub inserts_total: u64,
    pub searches_total: u64,
    pub compactions_total: u64,
    pub avg_insert_time_ns: f64,
    pub avg_search_time_ns: f64,
    pub delta_size: usize,
    pub num_sealed: usize,
}

#[cfg(feature = "prometheus")]
impl VivyMetrics {
    pub fn register_prometheus(&self, _registry: &prometheus::Registry) {
        // ponytail: stub, wire counters/gauges when prometheus integration is needed
    }
}
