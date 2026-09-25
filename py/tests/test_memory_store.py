import tempfile
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
