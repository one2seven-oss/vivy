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

#[test]
fn test_atomic_online_backup_and_restore() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;

    let primary_dir = tempdir().unwrap();
    let backup_dir = tempdir().unwrap();

    let config = MemoryConfig::builder(primary_dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = Arc::new(MemoryStore::open(config).unwrap());
    let scope = MemoryScope::new("tenant-backup", "ns-live").unwrap();

    // 1. Seed initial memory
    let initial_id = store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Primary seed observation".into(),
            embedding: vec![1.0, 0.0, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.9,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    // 2. Spawn concurrent background writer thread
    let running = Arc::new(AtomicBool::new(true));
    let writer_store = store.clone();
    let writer_scope = scope.clone();
    let writer_running = running.clone();

    let writer_handle = thread::spawn(move || {
        let mut count = 0;
        while writer_running.load(Ordering::Acquire) {
            let req = RememberRequest {
                operation_id: None,
                scope: writer_scope.clone(),
                content: format!("Concurrent memory write {count}"),
                embedding: vec![0.0, 1.0, 0.0],
                kind: MemoryKind::Fact,
                importance: 0.5,
                expires_at_ms: None,
                metadata: HashMap::new(),
                source: HashMap::new(),
            };
            let _ = writer_store.remember(req);
            count += 1;
            thread::sleep(std::time::Duration::from_millis(1));
        }
    });

    // Let concurrent writes run briefly
    thread::sleep(std::time::Duration::from_millis(20));

    // 3. Perform live zero-downtime atomic backup
    store.backup(backup_dir.path()).unwrap();

    // Stop writer thread
    running.store(false, Ordering::Release);
    writer_handle.join().unwrap();

    // 4. Open the backed-up directory as a new independent MemoryStore
    let backup_config = MemoryConfig::builder(backup_dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let restored_store = MemoryStore::open(backup_config).unwrap();

    // 5. Verify integrity and data of restored store
    let health = restored_store.health().unwrap();
    assert!(health.is_healthy);
    assert!(health.total_active_records >= 1);

    // Verify initial seed record exists in restored store
    let seed_record = restored_store.get(&scope, &initial_id).unwrap();
    assert!(seed_record.is_some());
    assert_eq!(
        seed_record.unwrap().content,
        "Primary seed observation"
    );

    // Verify recall succeeds on restored store
    let recall_res = restored_store
        .recall(RecallRequest {
            scope,
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: None,
            limit: 5,
            filters: MemoryFilter::default(),
            include_explanations: false,
            mmr_lambda: None,
        })
        .unwrap();

    assert!(!recall_res.items.is_empty());
}
