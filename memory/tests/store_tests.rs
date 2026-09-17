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
            })
            .unwrap();

        assert_eq!(recall.items.len(), 1);
        assert_eq!(recall.items[0].memory.content, "Persistent fact");
    }
}
