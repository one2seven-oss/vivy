import tempfile
import pytest
import vivy


def test_memory_store_full_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = vivy.MemoryStore.open(tmpdir, 3, "test-model")

        # 1. Remember
        mem_id = store.remember(
            tenant_id="tenant-a",
            namespace="support",
            content="Original content text",
            embedding=[1.0, 0.0, 0.0],
            kind="fact",
            importance=0.8
        )
        assert mem_id is not None

        # 2. Get
        record = store.get("tenant-a", "support", mem_id)
        assert record is not None
        assert record["id"] == mem_id
        assert record["content"] == "Original content text"
        assert record["revision"] == 1

        # 3. Update
        store.update(
            tenant_id="tenant-a",
            namespace="support",
            id=mem_id,
            expected_revision=1,
            content="Updated content text",
            embedding=[0.9, 0.1, 0.0],
            importance=0.95
        )

        # Verify update via Get
        updated_record = store.get("tenant-a", "support", mem_id)
        assert updated_record is not None
        assert updated_record["content"] == "Updated content text"
        assert updated_record["revision"] == 2

        # 4. Recall
        results = store.recall(
            tenant_id="tenant-a",
            namespace="support",
            query_embedding=[1.0, 0.0, 0.0],
            query_text="Updated content",
            limit=5
        )
        assert len(results) == 1
        assert results[0][0] == mem_id
        assert results[0][1] == "Updated content text"

        # 5. Forget
        store.forget("tenant-a", "support", mem_id)
        assert store.get("tenant-a", "support", mem_id) is None

        # 6. Health & Vacuum
        health = store.health()
        assert health["is_healthy"] is True
        assert health["total_tombstoned_records"] == 1

        purged = store.vacuum_tombstones(100)
        assert purged == 1

        health_after = store.health()
        assert health_after["total_tombstoned_records"] == 0


def test_remember_reopen_recall():
    """Verify remember -> close (drop) -> reopen -> recall workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Phase 1: Open store, remember item, drop reference
        store1 = vivy.MemoryStore.open(tmpdir, 4, "embedding-v1")
        mem_id = store1.remember(
            tenant_id="tenant-acme",
            namespace="docs",
            content="Persistent memory across store reopens",
            embedding=[0.1, 0.9, 0.0, 0.0],
            kind="instruction",
            importance=0.9
        )
        del store1

        # Phase 2: Reopen store from same directory, query recall
        store2 = vivy.MemoryStore.open(tmpdir, 4, "embedding-v1")
        results = store2.recall(
            tenant_id="tenant-acme",
            namespace="docs",
            query_embedding=[0.1, 0.85, 0.0, 0.0],
            query_text="Persistent memory",
            limit=5
        )
        assert len(results) == 1
        assert results[0][0] == mem_id
        assert "Persistent memory" in results[0][1]


def test_dimension_mismatch_error():
    """Verify passing embedding with invalid dimension raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = vivy.MemoryStore.open(tmpdir, 4, "embedding-v1")

        with pytest.raises(ValueError, match="Dimension mismatch"):
            store.remember(
                tenant_id="tenant-a",
                namespace="ns-1",
                content="Invalid dimension test",
                embedding=[1.0, 0.0]  # Expected 4, got 2
            )

        with pytest.raises(ValueError, match="Dimension mismatch"):
            store.recall(
                tenant_id="tenant-a",
                namespace="ns-1",
                query_embedding=[1.0, 0.0, 0.0]  # Expected 4, got 3
            )


def test_invalid_scope_error():
    """Verify empty tenant_id or namespace raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = vivy.MemoryStore.open(tmpdir, 3, "embedding-v1")

        with pytest.raises(ValueError, match="tenant_id must not be empty"):
            store.remember(
                tenant_id="",
                namespace="support",
                content="Empty tenant test",
                embedding=[1.0, 0.0, 0.0]
            )

        with pytest.raises(ValueError, match="namespace must not be empty"):
            store.remember(
                tenant_id="tenant-a",
                namespace="   ",
                content="Empty namespace test",
                embedding=[1.0, 0.0, 0.0]
            )


def test_revision_conflict_error():
    """Verify update with mismatched expected_revision raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = vivy.MemoryStore.open(tmpdir, 3, "embedding-v1")
        mem_id = store.remember(
            tenant_id="tenant-a",
            namespace="ns-1",
            content="Initial content",
            embedding=[1.0, 0.0, 0.0]
        )

        with pytest.raises(ValueError, match="Revision conflict"):
            store.update(
                tenant_id="tenant-a",
                namespace="ns-1",
                id=mem_id,
                expected_revision=999,  # Mismatch (current is 1)
                content="Conflicting update"
            )


def test_cross_tenant_isolation():
    """Verify zero cross-tenant retrieval leakage in recall and get."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = vivy.MemoryStore.open(tmpdir, 3, "test-model")

        mem_id = store.remember(
            tenant_id="tenant-alpha",
            namespace="support",
            content="Secret alpha data",
            embedding=[1.0, 0.0, 0.0]
        )

        # Querying tenant-beta returns zero results
        beta_results = store.recall(
            tenant_id="tenant-beta",
            namespace="support",
            query_embedding=[1.0, 0.0, 0.0]
        )
        assert len(beta_results) == 0

        # Get from tenant-beta returns None
        assert store.get("tenant-beta", "support", mem_id) is None


def test_memory_store_batch_operations():
    """Verify batch memory insertion (remember_batch) and deletion (forget_batch)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = vivy.MemoryStore.open(tmpdir, 3, "test-model")

        records = [
            {
                "tenant_id": "tenant-batch",
                "namespace": "support",
                "content": f"Batch item {i}",
                "embedding": [1.0, 0.0, 0.0],
                "kind": "fact",
                "importance": 0.8
            }
            for i in range(10)
        ]

        # Batch remember
        ids = store.remember_batch(records)
        assert len(ids) == 10

        # Recall all 10
        results = store.recall(
            tenant_id="tenant-batch",
            namespace="support",
            query_embedding=[1.0, 0.0, 0.0],
            limit=15
        )
        assert len(results) == 10

        # Batch forget first 4 items
        store.forget_batch("tenant-batch", "support", ids[:4])

        # Recall remaining 6
        results_after = store.recall(
            tenant_id="tenant-batch",
            namespace="support",
            query_embedding=[1.0, 0.0, 0.0],
            limit=15
        )
        assert len(results_after) == 6


def test_memory_store_filter_metadata():
    """Verify metadata dictionary filtering in MemoryStore.recall()."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = vivy.MemoryStore.open(tmpdir, 3, "test-model")

        store.remember(
            tenant_id="acme",
            namespace="support",
            content="Alpha project memory",
            embedding=[1.0, 0.0, 0.0],
            kind="fact",
            metadata={"project": "alpha", "session_id": 42, "confidential": True}
        )

        store.remember(
            tenant_id="acme",
            namespace="support",
            content="Beta project memory",
            embedding=[1.0, 0.0, 0.0],
            kind="fact",
            metadata={"project": "beta", "session_id": 99, "confidential": False}
        )

        # Filter by project == "alpha"
        alpha_res = store.recall(
            tenant_id="acme",
            namespace="support",
            query_embedding=[1.0, 0.0, 0.0],
            filter_metadata={"project": "alpha"}
        )
        assert len(alpha_res) == 1
        assert alpha_res[0][1] == "Alpha project memory"

        # Filter by int metadata session_id == 99
        beta_res = store.recall(
            tenant_id="acme",
            namespace="support",
            query_embedding=[1.0, 0.0, 0.0],
            filter_metadata={"session_id": 99}
        )
        assert len(beta_res) == 1
        assert beta_res[0][1] == "Beta project memory"

        # Filter by boolean metadata confidential == True
        conf_res = store.recall(
            tenant_id="acme",
            namespace="support",
            query_embedding=[1.0, 0.0, 0.0],
            filter_metadata={"confidential": True}
        )
        assert len(conf_res) == 1
        assert conf_res[0][1] == "Alpha project memory"

        # Filter with non-matching metadata
        none_res = store.recall(
            tenant_id="acme",
            namespace="support",
            query_embedding=[1.0, 0.0, 0.0],
            filter_metadata={"project": "gamma"}
        )
        assert len(none_res) == 0


def test_memory_store_format_context():
    """Verify PyMemoryStore.format_context() string formatting and filtering options."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = vivy.MemoryStore.open(tmpdir, 3, "test-model")

        store.remember(
            tenant_id="tenant-ctx",
            namespace="support",
            content="Alpha context memory block",
            embedding=[1.0, 0.0, 0.0],
            kind="fact",
            metadata={"source": "user"}
        )

        store.remember(
            tenant_id="tenant-ctx",
            namespace="support",
            content="Beta context memory block",
            embedding=[0.9, 0.1, 0.0],
            kind="rule",
            metadata={"source": "system"}
        )

        # Basic context formatting with default settings
        formatted = store.format_context(
            tenant_id="tenant-ctx",
            namespace="support",
            query_embedding=[1.0, 0.0, 0.0]
        )
        assert "Alpha context memory block" in formatted
        assert "Beta context memory block" in formatted

        # Custom header, footer, template, and metadata filtering
        custom_fmt = store.format_context(
            tenant_id="tenant-ctx",
            namespace="support",
            query_embedding=[1.0, 0.0, 0.0],
            header="=== CONTEXT START ===",
            footer="=== CONTEXT END ===",
            template="[{kind}] {content}",
            filter_metadata={"source": "user"}
        )
        assert custom_fmt.startswith("=== CONTEXT START ===")
        assert custom_fmt.endswith("=== CONTEXT END ===")
        assert "[fact] Alpha context memory block" in custom_fmt
        assert "Beta context memory block" not in custom_fmt

        # Token budget constraint
        short_fmt = store.format_context(
            tenant_id="tenant-ctx",
            namespace="support",
            query_embedding=[1.0, 0.0, 0.0],
            max_tokens=10
        )
        # Should truncate to at most 1 item due to low max_tokens
        assert short_fmt.count("context memory block") <= 1



