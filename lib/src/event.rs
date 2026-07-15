use std::sync::mpsc::{self, Receiver, Sender};

#[derive(Debug, Clone)]
pub enum VivyEvent {
    Inserted { id: u64, num_vectors: usize },
    MetadataInserted { id: u64, fields: Vec<(String, String)> },
    CompactionStarted { delta_size: usize },
    CompactionFinished { num_sealed: usize, segment_path: String },
    CompactionFailed { error: String },
    SearchThreshold { latency_ms: f64 },
    IndexFull { threshold_hit: usize },
    TenantCreated { tenant_id: String },
    TenantDropped { tenant_id: String },
}

#[derive(Clone)]
pub struct EventBus {
    tx: Sender<VivyEvent>,
}

impl EventBus {
    pub fn new() -> (Self, EventSubscriber) {
        let (tx, rx) = mpsc::channel();
        (EventBus { tx }, EventSubscriber { rx })
    }

    pub fn emit(&self, event: VivyEvent) {
        let _ = self.tx.send(event);
    }
}

pub struct EventSubscriber {
    rx: Receiver<VivyEvent>,
}

impl EventSubscriber {
    pub fn try_recv(&self) -> Option<VivyEvent> {
        self.rx.try_recv().ok()
    }

    pub fn iter(&self) -> impl Iterator<Item = VivyEvent> + '_ {
        std::iter::from_fn(|| self.try_recv())
    }
}
