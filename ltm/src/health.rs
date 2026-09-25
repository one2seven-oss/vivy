use serde::{Deserialize, Serialize};

/// Non-blocking operational health snapshot for a `MemoryStore`.
///
/// **Privacy Invariant**: Health reporting includes counts, status flags, and disk
/// sizes, but NEVER contains raw memory text, embeddings, or keys.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreHealth {
    pub is_healthy: bool,
    pub total_active_records: usize,
    pub total_tombstoned_records: usize,
    pub pending_operations_count: usize,
    pub index_rebuild_required: bool,
    pub db_size_bytes: u64,
    pub wal_size_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_health_serialization() {
        let health = StoreHealth {
            is_healthy: true,
            total_active_records: 150,
            total_tombstoned_records: 12,
            pending_operations_count: 0,
            index_rebuild_required: false,
            db_size_bytes: 40960,
            wal_size_bytes: 8192,
        };

        let json = serde_json::to_string(&health).unwrap();
        assert!(json.contains("is_healthy"));
        assert!(json.contains("total_active_records"));

        // Assert zero data content leakage
        assert!(!json.contains("content"));
        assert!(!json.contains("embedding"));
    }
}
