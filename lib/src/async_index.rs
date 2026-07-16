use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Notify;

use crate::concurrent::VivyIndex;
use crate::distance::Metric;
use crate::event::{EventBus, VivyEvent};
use crate::storage::wal::WalError;

pub struct AsyncVivyIndex {
    inner: Arc<VivyIndex>,
    insert_notify: Arc<Notify>,
    compact_notify: Arc<Notify>,
}

impl AsyncVivyIndex {
    pub fn new(
        metric: Metric,
        wal: Option<PathBuf>,
        data: Option<PathBuf>,
    ) -> Result<Self, WalError> {
        let (bus, sub) = EventBus::new();
        let inner = Arc::new(VivyIndex::new_with_events(metric, wal, data, bus)?);

        let insert_notify = Arc::new(Notify::new());
        let compact_notify = Arc::new(Notify::new());

        let n_insert = insert_notify.clone();
        let n_compact = compact_notify.clone();
        std::thread::Builder::new()
            .name("vivy-async-events".into())
            .spawn(move || {
                for event in sub.iter() {
                    match event {
                        VivyEvent::Inserted { .. } => n_insert.notify_waiters(),
                        VivyEvent::CompactionStarted { .. }
                        | VivyEvent::CompactionFinished { .. } => n_compact.notify_waiters(),
                        _ => {}
                    }
                }
            })
            .expect("event listener thread");

        Ok(Self { inner, insert_notify, compact_notify })
    }

    pub async fn insert(&self, vector: Vec<f32>) -> Result<u64, WalError> {
        let inner = self.inner.clone();
        let id = tokio::task::spawn_blocking(move || inner.insert(vector))
            .await
            .unwrap()?;
        self.insert_notify.notify_waiters();
        Ok(id)
    }

    pub async fn search(&self, query: Vec<f32>, k: usize) -> Vec<(u64, f32)> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || inner.search(&query, k))
            .await
            .unwrap()
    }

    pub async fn search_timeout(
        &self,
        query: Vec<f32>,
        k: usize,
        timeout: Duration,
    ) -> Result<Vec<(u64, f32)>, tokio::time::error::Elapsed> {
        let inner = self.inner.clone();
        tokio::time::timeout(timeout, tokio::task::spawn_blocking(move || {
            inner.search(&query, k)
        }))
        .await?
        .map_err(|_| panic!("search panicked"))
    }

    // ponytail: eager fetch, stream all results. Streaming HNSW would need incremental search.
    pub async fn search_stream(
        &self,
        query: Vec<f32>,
        k: usize,
    ) -> impl futures::Stream<Item = (u64, f32)> {
        let results = self.search(query, k).await;
        futures::stream::iter(results)
    }

    pub async fn wait_for_insert(&self) {
        self.insert_notify.notified().await;
    }

    pub async fn wait_for_compaction(&self) {
        self.compact_notify.notified().await;
    }
}
