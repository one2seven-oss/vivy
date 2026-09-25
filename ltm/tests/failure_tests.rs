use std::collections::HashMap;
use tempfile::tempdir;
use vivy_memory::*;

#[test]
fn test_fault_injection_remember_unapplied_op() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("memory.db");
    let dimensions = 4;

    // 1. Open repository directly and inject a pending remember operation
    let repo = Repository::open(&db_path).unwrap();
    let scope = MemoryScope::new("tenant-crash", "ns-crash").unwrap();
    let record_id = "mem-crash-001".to_string();

    let record = MemoryRecord {
        id: record_id.clone(),
        scope: scope.clone(),
        kind: MemoryKind::Fact,
        content: "Critical crash recovery memory".into(),
        content_hash: vec![1, 2, 3, 4],
        embedding: vec![1.0, 0.0, 0.0, 0.0],
        embedding_model: "test-model".into(),
        embedding_dims: dimensions,
        importance: 0.9,
        created_at_ms: 1000,
        updated_at_ms: 1000,
        last_accessed_at_ms: None,
        access_count: 0,
        expires_at_ms: None,
        status: MemoryStatus::Pending,
        revision: 1,
        metadata: HashMap::new(),
        source: HashMap::new(),
    };

    // Simulate crash after SQLite transaction commit but before vector index upsert or status update
    repo.insert_pending_memory(&record, Some("op-pending-rem-1"), 1000)
        .unwrap();

    // Verify raw status in DB is still Pending before recovery
    let raw_record = repo.get_by_scope_and_id(&scope, &record_id).unwrap().unwrap();
    assert_eq!(raw_record.status, MemoryStatus::Pending);

    drop(repo);

    // 2. Open MemoryStore (triggers JournalCoordinator::recover_startup)
    let config = MemoryConfig::builder(dir.path())
        .dimensions(dimensions)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();

    // 3. Verify record was recovered, marked Active, and populated in Vector Index
    let recovered_record = store.get(&scope, &record_id).unwrap().expect("record recovered");
    assert_eq!(recovered_record.status, MemoryStatus::Active);

    // 4. Verify recall can locate it via vector search
    let recall_res = store
        .recall(RecallRequest {
            scope: scope.clone(),
            query_embedding: vec![1.0, 0.0, 0.0, 0.0],
            query_text: None,
            limit: 5,
            filters: MemoryFilter::default(),
            include_explanations: false,
            mmr_lambda: None,
        })
        .unwrap();

    assert_eq!(recall_res.items.len(), 1);
    assert_eq!(recall_res.items[0].memory.id, record_id);
}

#[test]
fn test_fault_injection_update_unapplied_op() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(4)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let scope = MemoryScope::new("tenant-update-crash", "ns-main").unwrap();

    // 1. Insert initial record normally
    let mem_id = {
        let store = MemoryStore::open(config.clone()).unwrap();
        store
            .remember(RememberRequest {
                operation_id: Some("op-init-1".into()),
                scope: scope.clone(),
                content: "Original text before crash".into(),
                embedding: vec![1.0, 0.0, 0.0, 0.0],
                kind: MemoryKind::Fact,
                importance: 0.5,
                expires_at_ms: None,
                metadata: HashMap::new(),
                source: HashMap::new(),
            })
            .unwrap()
    };

    // 2. Open repository directly and insert an unapplied update operation
    let db_path = dir.path().join("memory.db");
    let repo = Repository::open(&db_path).unwrap();

    let mut updated_record = repo.get_by_scope_and_id(&scope, &mem_id).unwrap().unwrap();
    updated_record.content = "Updated text after crash recovery".into();
    updated_record.embedding = vec![0.0, 1.0, 0.0, 0.0];
    updated_record.importance = 0.95;

    // Execute update in SQLite with operation_id but DO NOT update vector index
    repo.update_memory(&updated_record, 1, Some("op-update-crash-1"), 2000)
        .unwrap();
    drop(repo);

    // 3. Reopen MemoryStore (triggers startup recovery)
    let store = MemoryStore::open(config).unwrap();

    // 4. Verify vector index has new embedding via recall
    let recall_res = store
        .recall(RecallRequest {
            scope: scope.clone(),
            query_embedding: vec![0.0, 1.0, 0.0, 0.0],
            query_text: None,
            limit: 5,
            filters: MemoryFilter::default(),
            include_explanations: false,
            mmr_lambda: None,
        })
        .unwrap();

    assert_eq!(recall_res.items.len(), 1);
    assert_eq!(recall_res.items[0].memory.content, "Updated text after crash recovery");
}

#[test]
fn test_fault_injection_delete_unapplied_op() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(4)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let scope = MemoryScope::new("tenant-del-crash", "ns-main").unwrap();

    // 1. Create record
    let mem_id = {
        let store = MemoryStore::open(config.clone()).unwrap();
        store
            .remember(RememberRequest {
                operation_id: None,
                scope: scope.clone(),
                content: "Record to be deleted in crash test".into(),
                embedding: vec![1.0, 0.0, 0.0, 0.0],
                kind: MemoryKind::Context,
                importance: 0.5,
                expires_at_ms: None,
                metadata: HashMap::new(),
                source: HashMap::new(),
            })
            .unwrap()
    };

    // 2. Open repository directly and execute delete in SQLite without updating index
    let db_path = dir.path().join("memory.db");
    let repo = Repository::open(&db_path).unwrap();
    repo.delete_memory(&scope, &mem_id, Some("op-del-crash-1"), 2000)
        .unwrap();
    drop(repo);

    // 3. Reopen MemoryStore
    let store = MemoryStore::open(config).unwrap();

    // 4. Verify record is gone from recall
    let recall_res = store
        .recall(RecallRequest {
            scope: scope.clone(),
            query_embedding: vec![1.0, 0.0, 0.0, 0.0],
            query_text: None,
            limit: 5,
            filters: MemoryFilter::default(),
            include_explanations: false,
            mmr_lambda: None,
        })
        .unwrap();

    assert_eq!(recall_res.items.len(), 0);
}

#[test]
fn test_property_multi_tenant_isolation_fuzz() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(4)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();

    let tenants = ["tenant-alpha", "tenant-beta", "tenant-gamma", "tenant-delta", "tenant-epsilon"];
    let namespaces = ["support", "billing", "analytics"];
    let agents = ["agent-1", "agent-2"];
    let users = ["user-x", "user-y"];

    let keywords = ["confidential", "api_key", "password", "secret", "token"];

    let mut seed: u64 = 0xDEADBEEF;
    let mut rng = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        seed
    };

    let mut inserted_records = Vec::new();

    // Generate 100 randomized memory records across multi-tenant partitions
    for i in 0..100 {
        let t_idx = (rng() as usize) % tenants.len();
        let ns_idx = (rng() as usize) % namespaces.len();
        let agent_idx = (rng() as usize) % agents.len();
        let user_idx = (rng() as usize) % users.len();
        let kw_idx = (rng() as usize) % keywords.len();

        let mut scope = MemoryScope::new(tenants[t_idx], namespaces[ns_idx]).unwrap();
        if i % 2 == 0 {
            scope = scope.with_agent(agents[agent_idx]).unwrap();
        }
        if i % 3 == 0 {
            scope = scope.with_user(users[user_idx]).unwrap();
        }

        let content = format!(
            "Memory {} for {} scope with keyword {} and secret payload {}",
            i,
            scope.tenant_id(),
            keywords[kw_idx],
            rng()
        );

        let v1 = ((rng() % 1000) as f32) / 1000.0;
        let v2 = ((rng() % 1000) as f32) / 1000.0;
        let v3 = ((rng() % 1000) as f32) / 1000.0;
        let v4 = ((rng() % 1000) as f32) / 1000.0;

        let id = store
            .remember(RememberRequest {
                operation_id: None,
                scope: scope.clone(),
                content: content.clone(),
                embedding: vec![v1, v2, v3, v4],
                kind: MemoryKind::Fact,
                importance: 0.5 + (((rng() % 50) as f32) / 100.0),
                expires_at_ms: None,
                metadata: HashMap::new(),
                source: HashMap::new(),
            })
            .unwrap();

        inserted_records.push((id, scope, keywords[kw_idx].to_string()));
    }

    // Property Invariant Checks
    for tenant in &tenants {
        for ns in &namespaces {
            let query_scope = MemoryScope::new(*tenant, *ns).unwrap();

            // Check FTS query for all keywords
            for kw in &keywords {
                let recall_res = store
                    .recall(RecallRequest {
                        scope: query_scope.clone(),
                        query_embedding: vec![0.5, 0.5, 0.5, 0.5],
                        query_text: Some(kw.to_string()),
                        limit: 50,
                        filters: MemoryFilter::default(),
                        include_explanations: false,
                        mmr_lambda: None,
                    })
                    .unwrap();

                // STABILITY & ISOLATION INVARIANT:
                // Every single returned record MUST strictly match query_scope.tenant_id() and namespace()
                for item in &recall_res.items {
                    assert_eq!(
                        item.memory.scope.tenant_id(),
                        *tenant,
                        "CRITICAL INVARIANT VIOLATION: Cross-tenant data leak detected!"
                    );
                    assert_eq!(
                        item.memory.scope.namespace(),
                        *ns,
                        "CRITICAL INVARIANT VIOLATION: Cross-namespace data leak detected!"
                    );
                }
            }
        }
    }
}

#[test]
fn test_hostile_fts_inputs() {
    let dir = tempdir().unwrap();
    let config = MemoryConfig::builder(dir.path())
        .dimensions(3)
        .embedding_model("test-model")
        .build()
        .unwrap();

    let store = MemoryStore::open(config).unwrap();
    let scope = MemoryScope::new("tenant-hostile", "ns-security").unwrap();

    store
        .remember(RememberRequest {
            operation_id: None,
            scope: scope.clone(),
            content: "Normal system memory record with confidential data".into(),
            embedding: vec![1.0, 0.0, 0.0],
            kind: MemoryKind::Fact,
            importance: 0.8,
            expires_at_ms: None,
            metadata: HashMap::new(),
            source: HashMap::new(),
        })
        .unwrap();

    // List of hostile, malformed, or injection queries
    let hostile_queries = vec![
        "' OR '1'='1".to_string(),
        "\" OR 1=1 --".to_string(),
        "NEAR(a, b, 10)".to_string(),
        "MATCH 'test'".to_string(),
        "AND OR NOT".to_string(),
        "(((((((".to_string(),
        "***???---+++".to_string(),
        "unicode_test_😀_🤖_\u{0000}_null".to_string(),
        "a".repeat(5000),
    ];

    for q in hostile_queries {
        let recall_result = store.recall(RecallRequest {
            scope: scope.clone(),
            query_embedding: vec![1.0, 0.0, 0.0],
            query_text: Some(q),
            limit: 5,
            filters: MemoryFilter::default(),
            include_explanations: false,
            mmr_lambda: None,
        });

        // Query execution MUST NOT panic or crash SQLite
        assert!(recall_result.is_ok(), "Hostile query caused failure");
    }
}
