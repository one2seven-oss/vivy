use std::collections::HashMap;
use tempfile::tempdir;
use vivy_memory::*;

#[test]
fn test_idempotent_remember() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("acme", "chat-support").unwrap();

    let req = RememberRequest {
        operation_id: Some("op-fixed-123".into()),
        scope: scope.clone(),
        content: "User loves dark mode.".into(),
        embedding: vec![1.0, 0.0, 0.0],
        kind: MemoryKind::Preference,
        importance: 0.9,
        expires_at_ms: None,
        metadata: HashMap::new(),
        source: HashMap::new(),
    };

    // First remember call
    let id1 = store.remember(req.clone()).unwrap();

    // Repeated remember call with same operation_id returns exact same ID
    let id2 = store.remember(req.clone()).unwrap();
    assert_eq!(id1, id2);

    // Verify record exists and is active
    let record = store.get(&scope, &id1).unwrap().expect("record found");
    assert_eq!(record.content, "User loves dark mode.");
    assert_eq!(record.status, MemoryStatus::Active);
}

#[test]
fn test_recall_and_explainability() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config.clone()).unwrap();
    let scope_a = MemoryScope::new("acme", "support").unwrap();
    let scope_b = MemoryScope::new("globex", "support").unwrap();

    // Insert record for scope_a
    store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope_a.clone(),
            content: "Acme record".into(),
            embedding: vec![1.0, 0.0, 0.0],
            kind: MemoryKind::Preference,
            importance: 0.8,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    // Recall from scope_a finds the record
    let recall_a = store
        .recall(RecallRequest {
            scope: scope_a.clone(),
            query_embedding: vec![1.0, 0.0, 0.0],
            limit: 5,
            filters: MemoryFilter::default(),
            include_explanations: true,
            mmr_lambda: None,
        })
        .unwrap();

    assert_eq!(recall_a.items.len(), 1);
    assert_eq!(recall_a.items[0].memory.content, "Acme record");
    assert!(recall_a.items[0].explanation.is_some());
    let exp = recall_a.items[0].explanation.as_ref().unwrap();
    assert!(exp.total_score > 0.8);
    assert_eq!(exp.similarity_score, 1.0);

    // Recall from scope_b returns ZERO records (strict multi-tenant isolation)
    let recall_b = store
        .recall(RecallRequest {
            scope: scope_b,
            query_embedding: vec![1.0, 0.0, 0.0],
            limit: 5,
            filters: MemoryFilter::default(),
            include_explanations: false,
            mmr_lambda: None,
        })
        .unwrap();

    assert_eq!(recall_b.items.len(), 0);
}

#[test]
fn test_crash_recovery_rebuild() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let scope = MemoryScope::new("tenant-x", "ns-1").unwrap();

    {
        let store = MemoryStore::open(config.clone()).unwrap();
        store
            .remember(RememberRequest {
                operation_id: Some("op-rec-1".into()),
                scope: scope.clone(),
                content: "Persistent fact".into(),
                embedding: vec![0.0, 1.0, 0.0],
                kind: MemoryKind::Fact,
                importance: 0.7,
                expires_at_ms: None,
                metadata: HashMap::new(),
                source: HashMap::new(),
            })
            .unwrap();
    }

    // Reopen store from disk
    {
        let store = MemoryStore::open(config).unwrap();
        let recall = store
            .recall(RecallRequest {
                scope: scope.clone(),
                query_embedding: vec![0.0, 1.0, 0.0],
                limit: 5,
                filters: MemoryFilter::default(),
                include_explanations: true,
                mmr_lambda: None,
            })
            .unwrap();

        assert_eq!(recall.items.len(), 1);
        assert_eq!(recall.items[0].memory.content, "Persistent fact");
    }
}

#[test]
fn test_update_revision_concurrency_and_forget() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config.clone()).unwrap();
    let scope = MemoryScope::new("acme", "chat").unwrap();

    let id = store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Initial observation".into(),
            embedding: vec![1.0, 0.0, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.5,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    // Stale update with wrong revision fails with RevisionConflict
    let stale_res = store.update(UpdateRequest {
        operation_id: None,
        scope: scope.clone(),
        id: id.clone(),
        expected_revision: 99,
        content: Some("Wrong revision update".into()),
        embedding: None,
        kind: None,
        importance: None,
        expires_at_ms: None,
        metadata_patch: None,
    });
    assert!(matches!(
        stale_res.unwrap_err(),
        MemoryError::RevisionConflict { .. }
    ));

    // Valid update with expected_revision 1 succeeds and bumps to revision 2
    store
        .update(UpdateRequest {
            operation_id: None,
            scope: scope.clone(),
            id: id.clone(),
            expected_revision: 1,
            content: Some("Updated observation".into()),
            embedding: Some(vec![0.0, 1.0, 0.0]),
            kind: None,
            importance: Some(0.9),
            expires_at_ms: None,
            metadata_patch: None,
        })
        .unwrap();

    let updated = store.get(&scope, &id).unwrap().unwrap();
    assert_eq!(updated.content, "Updated observation");
    assert_eq!(updated.revision, 2);
    assert_eq!(updated.importance, 0.9);

    // Forget memory immediately hides it
    store
        .forget(ForgetRequest {
            operation_id: None,
            scope: scope.clone(),
            id: id.clone(),
        })
        .unwrap();

    let after_forget = store.get(&scope, &id).unwrap();
    assert_eq!(after_forget.unwrap().status, MemoryStatus::Deleted);

    // Recall returns 0 items
    let recall = store
        .recall(RecallRequest {
            scope: scope.clone(),
            query_embedding: vec![0.0, 1.0, 0.0],
            limit: 5,
            filters: MemoryFilter::default(),
            include_explanations: false,
            mmr_lambda: None,
        })
        .unwrap();
    assert_eq!(recall.items.len(), 0);
}

#[test]
fn test_mmr_diversity_ranking() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("acme", "docs").unwrap();

    // Add 2 very similar records and 1 distinct record
    store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Rust concurrency 1".into(),
            embedding: vec![1.0, 0.0, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.9,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Rust concurrency 2 (near duplicate)".into(),
            embedding: vec![0.99, 0.01, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.89,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Python async IO".into(),
            embedding: vec![0.7, 0.7, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.85,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    // Recall with MMR diversity (lambda = 0.5)
    let recall = store
        .recall(RecallRequest {
            scope,
            query_embedding: vec![1.0, 0.0, 0.0],
            limit: 2,
            filters: MemoryFilter::default(),
            include_explanations: false,
            mmr_lambda: Some(0.5),
        })
        .unwrap();

    assert_eq!(recall.items.len(), 2);
    assert_eq!(recall.items[0].memory.content, "Rust concurrency 1");
    // With MMR, the second selected item should be Python async IO rather than the duplicate
    assert_eq!(recall.items[1].memory.content, "Python async IO");
}
