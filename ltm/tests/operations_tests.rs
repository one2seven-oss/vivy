use std::collections::HashMap;
use tempfile::tempdir;
use vivy_memory::*;

#[test]
fn test_health_snapshot_reports_accurate_counts_and_sizes() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("tenant-ops", "ns-1").unwrap();

    // Check initial health report
    let health_initial = store.health().unwrap();
    assert!(health_initial.is_healthy);
    assert_eq!(health_initial.total_active_records, 0);
    assert_eq!(health_initial.total_tombstoned_records, 0);
    assert!(!health_initial.index_rebuild_required);
    assert!(health_initial.db_size_bytes > 0);

    // Insert 2 records
    let id1 = store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Fact 1".into(),
            embedding: vec![1.0, 0.0, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.5,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    let _id2 = store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Fact 2".into(),
            embedding: vec![0.0, 1.0, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.5,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    let health_after_insert = store.health().unwrap();
    assert_eq!(health_after_insert.total_active_records, 2);
    assert_eq!(health_after_insert.total_tombstoned_records, 0);

    // Tombstone 1 record
    store
        .forget(ForgetRequest {
            operation_id: None,
            scope: scope.clone(),
            id: id1,
        })
        .unwrap();

    let health_after_forget = store.health().unwrap();
    assert_eq!(health_after_forget.total_active_records, 1);
    assert_eq!(health_after_forget.total_tombstoned_records, 1);
}

#[test]
fn test_resumable_vacuum_tombstones_scrubs_deleted_records() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("tenant-ops", "ns-vacuum").unwrap();

    // Insert 5 records
    let mut ids = Vec::new();
    for i in 0..5 {
        let id = store
            .remember(RememberRequest {
                operation_id: None,
                scope: scope.clone(),
                content: format!("Record {i}"),
                embedding: vec![i as f32, 0.0, 0.0],
                kind: MemoryKind::Fact,
                importance: 0.5,
                expires_at_ms: None,
                metadata: HashMap::new(),
                source: HashMap::new(),
            })
            .unwrap();
        ids.push(id);
    }

    // Tombstone first 3 records
    for id in ids.iter().take(3) {
        store
            .forget(ForgetRequest {
                operation_id: None,
                scope: scope.clone(),
                id: id.clone(),
            })
            .unwrap();
    }

    let health_before = store.health().unwrap();
    assert_eq!(health_before.total_tombstoned_records, 3);

    // Vacuum batch of 2 tombstoned records
    let purged_batch_1 = store.vacuum_tombstones(2).unwrap();
    assert_eq!(purged_batch_1, 2);

    let health_mid = store.health().unwrap();
    assert_eq!(health_mid.total_tombstoned_records, 1);

    // Vacuum remaining tombstoned record
    let purged_batch_2 = store.vacuum_tombstones(10).unwrap();
    assert_eq!(purged_batch_2, 1);

    let health_final = store.health().unwrap();
    assert_eq!(health_final.total_tombstoned_records, 0);
    assert_eq!(health_final.total_active_records, 2);

    // Verify recall returns active records properly
    let recall = store
        .recall(RecallRequest {
            scope: scope.clone(),
            query_embedding: vec![3.0, 0.0, 0.0],
            query_text: None,
            limit: 5,
            filters: MemoryFilter::default(),
            include_explanations: false,
            mmr_lambda: None,
        })
        .unwrap();

    assert_eq!(recall.items.len(), 2);
}

#[test]
fn test_rebuild_index_restores_active_records() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("tenant-ops", "ns-rebuild").unwrap();

    store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Rebuild target".into(),
            embedding: vec![1.0, 1.0, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.8,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    // Manual rebuild index
    store.rebuild_index().unwrap();

    let recall = store
        .recall(RecallRequest {
            scope,
            query_embedding: vec![1.0, 1.0, 0.0],
            query_text: None,
            limit: 5,
            filters: MemoryFilter::default(),
            include_explanations: false,
            mmr_lambda: None,
        })
        .unwrap();

    assert_eq!(recall.items.len(), 1);
    assert_eq!(recall.items[0].memory.content, "Rebuild target");
}
