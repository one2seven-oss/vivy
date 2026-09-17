use crate::error::Result;
use crate::index_adapter::VectorIndex;
use crate::model::{MemoryRecord, MemoryStatus};
use crate::repository::Repository;
use std::sync::Arc;

pub struct JournalCoordinator {
    repo: Arc<Repository>,
    index: Arc<dyn VectorIndex>,
}

impl JournalCoordinator {
    pub fn new(repo: Arc<Repository>, index: Arc<dyn VectorIndex>) -> Self {
        Self { repo, index }
    }

    /// Process a remember operation through the 2-phase state machine:
    /// 1. Commit pending record + operation in SQLite
    /// 2. Upsert vector into index
    /// 3. Mark memory active + operation applied
    pub fn coordinate_remember(
        &self,
        record: MemoryRecord,
        operation_id: Option<String>,
    ) -> Result<String> {
        let memory_id = record.id.clone();
        let embedding = record.embedding.clone();

        // 1. If operation_id provided, check if already applied or pending
        if let Some(ref op_id) = operation_id {
            if let Some(existing_mem_id) = self.repo.get_memory_id_by_operation_id(op_id)? {
                return Ok(existing_mem_id);
            }
        }

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        // 2. Commit SQLite pending record & operation
        self.repo
            .insert_pending_memory(&record, operation_id.as_deref(), now_ms)?;

        // 3. Upsert into Vector Index
        self.index.upsert(&memory_id, &embedding)?;

        // 4. Mark active and operation applied
        self.repo.set_status(&memory_id, MemoryStatus::Active)?;
        if let Some(ref op_id) = operation_id {
            self.repo.mark_operation_applied(op_id, now_ms)?;
        }

        Ok(memory_id)
    }

    /// Process an update operation through the 2-phase state machine
    pub fn coordinate_update(
        &self,
        record: MemoryRecord,
        expected_revision: u64,
        operation_id: Option<String>,
    ) -> Result<()> {
        let memory_id = record.id.clone();
        let embedding = record.embedding.clone();

        if let Some(ref op_id) = operation_id {
            if let Some(existing_mem_id) = self.repo.get_memory_id_by_operation_id(op_id)? {
                if existing_mem_id == memory_id {
                    return Ok(());
                }
            }
        }

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        self.repo
            .update_memory(&record, expected_revision, operation_id.as_deref(), now_ms)?;

        self.index.upsert(&memory_id, &embedding)?;

        if let Some(ref op_id) = operation_id {
            self.repo.mark_operation_applied(op_id, now_ms)?;
        }

        Ok(())
    }

    /// Process a forget/delete operation
    pub fn coordinate_forget(
        &self,
        scope: &crate::namespace::MemoryScope,
        id: &str,
        operation_id: Option<String>,
    ) -> Result<()> {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        self.repo
            .delete_memory(scope, id, operation_id.as_deref(), now_ms)?;
        self.index.remove(id)?;
        Ok(())
    }

    /// Reconcile unapplied journal operations and restore index consistency on startup.
    pub fn recover_startup(&self) -> Result<usize> {
        let pending_ops = self.repo.get_pending_operations()?;
        let mut recovered_count = 0;

        for (op_id, mem_id, kind) in pending_ops {
            if kind == "remember" || kind == "update" {
                if let Some(record) = self.repo.get_record_by_id_internal(&mem_id)? {
                    if record.status == MemoryStatus::Pending || record.status == MemoryStatus::Active {
                        self.index.upsert(&record.id, &record.embedding)?;
                        if record.status == MemoryStatus::Pending {
                            self.repo.set_status(&record.id, MemoryStatus::Active)?;
                        }
                    }
                    let now_ms = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as i64;
                    self.repo.mark_operation_applied(&op_id, now_ms)?;
                    recovered_count += 1;
                }
            } else if kind == "delete" {
                self.index.remove(&mem_id)?;
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as i64;
                self.repo.mark_operation_applied(&op_id, now_ms)?;
                recovered_count += 1;
            }
        }

        // Rebuild active records in index
        let active_records = self.repo.get_all_active_records()?;
        self.index.rebuild(Box::new(active_records.iter()))?;

        Ok(recovered_count)
    }
}
