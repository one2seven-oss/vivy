use crate::error::{ErrorCode, MemoryError, Result};
use crate::model::{MemoryKind, MemoryRecord, MemoryStatus};
use crate::namespace::MemoryScope;
use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::HashMap;
use std::path::Path;

fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks.iter().map(|&chunk| f32::from_le_bytes(chunk)).collect()
}

const SCHEMA_V1: &str = r#"
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    agent_id TEXT,
    user_id TEXT,
    kind TEXT NOT NULL,
    content BLOB NOT NULL,
    content_hash BLOB NOT NULL,
    embedding BLOB NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dims INTEGER NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    last_accessed_at_ms INTEGER,
    access_count INTEGER NOT NULL DEFAULT 0,
    expires_at_ms INTEGER,
    status TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    source_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS memories_scope_active
    ON memories(tenant_id, namespace, status, expires_at_ms);

CREATE TABLE IF NOT EXISTS operations (
    operation_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    applied_at_ms INTEGER
);
"#;

const SCHEMA_V2: &str = r#"
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    id UNINDEXED,
    tenant_id UNINDEXED,
    namespace UNINDEXED,
    content,
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(id, tenant_id, namespace, content)
    VALUES (new.id, new.tenant_id, new.namespace, CAST(new.content AS TEXT));
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    DELETE FROM memories_fts WHERE id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    DELETE FROM memories_fts WHERE id = old.id;
    INSERT INTO memories_fts(id, tenant_id, namespace, content)
    VALUES (new.id, new.tenant_id, new.namespace, CAST(new.content AS TEXT));
END;
"#;

fn sanitize_fts_query(input: &str) -> String {
    let clean: String = input
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { ' ' })
        .collect();
    let tokens: Vec<&str> = clean.split_whitespace().collect();
    if tokens.is_empty() {
        return String::new();
    }
    let phrase = format!("\"{}\"", tokens.join(" "));
    let term_ors = tokens
        .iter()
        .map(|t| format!("\"{}\"", t))
        .collect::<Vec<_>>()
        .join(" OR ");
    format!("{} OR {}", phrase, term_ors)
}

pub struct Repository {
    conn: Mutex<Connection>,
}

impl Repository {
    pub fn open(db_path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(db_path.as_ref()).map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to open SQLite database: {}", e),
        })?;

        // Configure WAL mode, synchronous=NORMAL, and foreign keys
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to set WAL mode: {}", e),
            })?;
        conn.pragma_update(None, "synchronous", "NORMAL")
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to set synchronous mode: {}", e),
            })?;
        conn.pragma_update(None, "foreign_keys", "ON")
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to enable foreign keys: {}", e),
            })?;

        let repo = Self {
            conn: Mutex::new(conn),
        };
        repo.migrate()?;
        repo.integrity_check()?;
        Ok(repo)
    }

    pub fn migrate(&self) -> Result<()> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to start migration transaction: {}", e),
        })?;

        let table_exists: bool = tx
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'",
                [],
                |_| Ok(true),
            )
            .optional()
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to check table existence: {}", e),
            })?
            .unwrap_or(false);

        let current_ver: i64 = if table_exists {
            tx.query_row(
                "SELECT MAX(version) FROM schema_migrations",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to check migration version: {}", e),
            })?
            .flatten()
            .unwrap_or(0)
        } else {
            0
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        if current_ver < 1 {
            tx.execute_batch(SCHEMA_V1)
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to apply schema v1: {}", e),
                })?;
            tx.execute(
                "INSERT INTO schema_migrations (version, applied_at_ms) VALUES (1, ?)",
                params![now],
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to record schema v1 migration: {}", e),
            })?;
        }

        if current_ver < 2 {
            tx.execute_batch(SCHEMA_V2)
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to apply schema v2: {}", e),
                })?;
            tx.execute(
                r#"
                INSERT OR IGNORE INTO memories_fts(id, tenant_id, namespace, content)
                SELECT id, tenant_id, namespace, CAST(content AS TEXT)
                FROM memories
                WHERE status != 'deleted'
                "#,
                [],
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to backfill memories_fts: {}", e),
            })?;
            tx.execute(
                "INSERT INTO schema_migrations (version, applied_at_ms) VALUES (2, ?)",
                params![now],
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to record schema v2 migration: {}", e),
            })?;
        }

        tx.commit().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to commit migration transaction: {}", e),
        })?;

        Ok(())
    }

    pub fn integrity_check(&self) -> Result<()> {
        let conn = self.conn.lock();
        let check_status: String = conn
            .query_row("PRAGMA integrity_check;", [], |row| row.get(0))
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to run PRAGMA integrity_check: {}", e),
            })?;

        if check_status.to_lowercase() != "ok" {
            return Err(MemoryError::CorruptStore {
                code: ErrorCode::CorruptStore,
                message: format!("SQLite integrity check failed: {}", check_status),
            });
        }
        Ok(())
    }

    pub fn insert_pending_memory(
        &self,
        record: &MemoryRecord,
        operation_id: Option<&str>,
        now_ms: i64,
    ) -> Result<()> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to begin transaction: {}", e),
        })?;

        let embedding_bytes: Vec<u8> = record
            .embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let metadata_json = serde_json::to_string(&record.metadata).unwrap_or_else(|_| "{}".into());
        let source_json = serde_json::to_string(&record.source).unwrap_or_else(|_| "{}".into());

        tx.execute(
            r#"
            INSERT INTO memories (
                id, tenant_id, namespace, agent_id, user_id, kind,
                content, content_hash, embedding, embedding_model, embedding_dims,
                importance, created_at_ms, updated_at_ms, last_accessed_at_ms,
                access_count, expires_at_ms, status, revision, metadata_json, source_json
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?
            )
            "#,
            params![
                record.id,
                record.scope.tenant_id(),
                record.scope.namespace(),
                record.scope.agent_id(),
                record.scope.user_id(),
                format!("{:?}", record.kind).to_lowercase(),
                record.content.as_bytes(),
                record.content_hash,
                embedding_bytes,
                record.embedding_model,
                record.embedding_dims as i64,
                record.importance,
                record.created_at_ms,
                record.updated_at_ms,
                record.last_accessed_at_ms,
                record.access_count as i64,
                record.expires_at_ms,
                record.status.as_str(),
                record.revision as i64,
                metadata_json,
                source_json,
            ],
        )
        .map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to insert memory: {}", e),
        })?;

        if let Some(op_id) = operation_id {
            tx.execute(
                r#"
                INSERT INTO operations (
                    operation_id, memory_id, kind, payload_json, state, created_at_ms, applied_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, NULL)
                "#,
                params![
                    op_id,
                    record.id,
                    "remember",
                    serde_json::to_string(record).unwrap_or_else(|_| "{}".into()),
                    "pending",
                    now_ms,
                ],
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to insert operation journal: {}", e),
            })?;
        }

        tx.commit().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to commit memory transaction: {}", e),
        })?;

        Ok(())
    }

    pub fn insert_pending_memory_batch(
        &self,
        records: &[MemoryRecord],
        operation_ids: &[Option<String>],
        now_ms: i64,
    ) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to begin transaction: {}", e),
        })?;

        {
            let mut mem_stmt = tx.prepare(
                r#"
                INSERT INTO memories (
                    id, tenant_id, namespace, agent_id, user_id, kind,
                    content, content_hash, embedding, embedding_model, embedding_dims,
                    importance, created_at_ms, updated_at_ms, last_accessed_at_ms,
                    access_count, expires_at_ms, status, revision, metadata_json, source_json
                ) VALUES (
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
                )
                "#,
            ).map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to prepare batch memory insert statement: {}", e),
            })?;

            let mut op_stmt = tx.prepare(
                r#"
                INSERT INTO operations (
                    operation_id, memory_id, kind, payload_json, state, created_at_ms, applied_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, NULL)
                "#,
            ).map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to prepare batch operation insert statement: {}", e),
            })?;

            for (i, record) in records.iter().enumerate() {
                let embedding_bytes: Vec<u8> = record
                    .embedding
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect();
                let metadata_json = serde_json::to_string(&record.metadata).unwrap_or_else(|_| "{}".into());
                let source_json = serde_json::to_string(&record.source).unwrap_or_else(|_| "{}".into());

                mem_stmt.execute(params![
                    record.id,
                    record.scope.tenant_id(),
                    record.scope.namespace(),
                    record.scope.agent_id(),
                    record.scope.user_id(),
                    format!("{:?}", record.kind).to_lowercase(),
                    record.content.as_bytes(),
                    record.content_hash,
                    embedding_bytes,
                    record.embedding_model,
                    record.embedding_dims as i64,
                    record.importance,
                    record.created_at_ms,
                    record.updated_at_ms,
                    record.last_accessed_at_ms,
                    record.access_count as i64,
                    record.expires_at_ms,
                    record.status.as_str(),
                    record.revision as i64,
                    metadata_json,
                    source_json,
                ]).map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to insert memory in batch: {}", e),
                })?;

                if let Some(op_id) = operation_ids.get(i).and_then(|opt| opt.as_deref()) {
                    op_stmt.execute(params![
                        op_id,
                        record.id,
                        "remember",
                        serde_json::to_string(record).unwrap_or_else(|_| "{}".into()),
                        "pending",
                        now_ms,
                    ]).map_err(|e| MemoryError::DatabaseError {
                        code: ErrorCode::DatabaseError,
                        message: format!("Failed to insert operation journal in batch: {}", e),
                    })?;
                }
            }
        }

        tx.commit().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to commit batch memory transaction: {}", e),
        })?;

        Ok(())
    }

    pub fn get_by_scope_and_id(
        &self,
        scope: &MemoryScope,
        id: &str,
    ) -> Result<Option<MemoryRecord>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
            SELECT id, tenant_id, namespace, agent_id, user_id, kind,
                   content, content_hash, embedding, embedding_model, embedding_dims,
                   importance, created_at_ms, updated_at_ms, last_accessed_at_ms,
                   access_count, expires_at_ms, status, revision, metadata_json, source_json
            FROM memories
            WHERE id = ? AND tenant_id = ? AND namespace = ?
            "#,
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to prepare query: {}", e),
            })?;

        let record = stmt
            .query_row(params![id, scope.tenant_id(), scope.namespace()], |row| {
                let id: String = row.get(0)?;
                let tenant_id: String = row.get(1)?;
                let namespace: String = row.get(2)?;
                let agent_id: Option<String> = row.get(3)?;
                let user_id: Option<String> = row.get(4)?;
                let kind_str: String = row.get(5)?;
                let content_bytes: Vec<u8> = row.get(6)?;
                let content_hash: Vec<u8> = row.get(7)?;
                let embedding_bytes: Vec<u8> = row.get(8)?;
                let embedding_model: String = row.get(9)?;
                let embedding_dims: i64 = row.get(10)?;
                let importance: f64 = row.get(11)?;
                let created_at_ms: i64 = row.get(12)?;
                let updated_at_ms: i64 = row.get(13)?;
                let last_accessed_at_ms: Option<i64> = row.get(14)?;
                let access_count: i64 = row.get(15)?;
                let expires_at_ms: Option<i64> = row.get(16)?;
                let status_str: String = row.get(17)?;
                let revision: i64 = row.get(18)?;
                let metadata_json: String = row.get(19)?;
                let source_json: String = row.get(20)?;

                let mut scope = MemoryScope::new(tenant_id, namespace).map_err(|_| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "Invalid scope",
                        )),
                    )
                })?;
                if let Some(agent) = agent_id {
                    scope = scope.with_agent(agent).unwrap();
                }
                if let Some(user) = user_id {
                    scope = scope.with_user(user).unwrap();
                }

                let kind = match kind_str.as_str() {
                    "preference" => MemoryKind::Preference,
                    "fact" => MemoryKind::Fact,
                    "instruction" => MemoryKind::Instruction,
                    "context" => MemoryKind::Context,
                    _ => MemoryKind::Episodic,
                };

                let embedding = bytes_to_embedding(&embedding_bytes);

                let metadata: HashMap<String, serde_json::Value> =
                    serde_json::from_str(&metadata_json).unwrap_or_default();
                let source: HashMap<String, serde_json::Value> =
                    serde_json::from_str(&source_json).unwrap_or_default();

                Ok(MemoryRecord {
                    id,
                    scope,
                    kind,
                    content: String::from_utf8_lossy(&content_bytes).to_string(),
                    content_hash,
                    embedding,
                    embedding_model,
                    embedding_dims: embedding_dims as usize,
                    importance: importance as f32,
                    created_at_ms,
                    updated_at_ms,
                    last_accessed_at_ms,
                    access_count: access_count as u64,
                    expires_at_ms,
                    status: MemoryStatus::parse(&status_str).unwrap_or(MemoryStatus::Pending),
                    revision: revision as u64,
                    metadata,
                    source,
                })
            })
            .optional()
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Query failed: {}", e),
            })?;

        Ok(record)
    }

    pub fn set_status(&self, id: &str, status: MemoryStatus) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE memories SET status = ? WHERE id = ?",
            params![status.as_str(), id],
        )
        .map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to update status: {}", e),
        })?;
        Ok(())
    }

    pub fn set_status_batch(&self, ids: &[&str], status: MemoryStatus) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to begin transaction for set_status_batch: {}", e),
        })?;

        {
            let mut stmt = tx.prepare("UPDATE memories SET status = ? WHERE id = ?")
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to prepare set_status_batch statement: {}", e),
                })?;

            for id in ids {
                stmt.execute(params![status.as_str(), id])
                    .map_err(|e| MemoryError::DatabaseError {
                        code: ErrorCode::DatabaseError,
                        message: format!("Failed to update status in batch: {}", e),
                    })?;
            }
        }

        tx.commit().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to commit set_status_batch transaction: {}", e),
        })?;
        Ok(())
    }

    pub fn get_memory_id_by_operation_id(&self, op_id: &str) -> Result<Option<String>> {
        let conn = self.conn.lock();
        let mem_id = conn
            .query_row(
                "SELECT memory_id FROM operations WHERE operation_id = ?",
                params![op_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to query operation: {}", e),
            })?;
        Ok(mem_id)
    }

    pub fn mark_operation_applied(&self, op_id: &str, applied_at_ms: i64) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE operations SET state = 'applied', applied_at_ms = ? WHERE operation_id = ?",
            params![applied_at_ms, op_id],
        )
        .map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to mark operation applied: {}", e),
        })?;
        Ok(())
    }

    pub fn mark_operation_applied_batch(&self, op_ids: &[&str], applied_at_ms: i64) -> Result<()> {
        if op_ids.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to begin transaction for mark_operation_applied_batch: {}", e),
        })?;

        {
            let mut stmt = tx.prepare("UPDATE operations SET state = 'applied', applied_at_ms = ? WHERE operation_id = ?")
                .map_err(|e| MemoryError::DatabaseError {
                    code: ErrorCode::DatabaseError,
                    message: format!("Failed to prepare mark_operation_applied_batch statement: {}", e),
                })?;

            for op_id in op_ids {
                stmt.execute(params![applied_at_ms, op_id])
                    .map_err(|e| MemoryError::DatabaseError {
                        code: ErrorCode::DatabaseError,
                        message: format!("Failed to mark operation applied in batch: {}", e),
                    })?;
            }
        }

        tx.commit().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to commit mark_operation_applied_batch transaction: {}", e),
        })?;
        Ok(())
    }

    pub fn get_pending_operations(&self) -> Result<Vec<(String, String, String)>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare("SELECT operation_id, memory_id, kind FROM operations WHERE state = 'pending' ORDER BY created_at_ms ASC")
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to prepare pending operations query: {}", e),
            })?;

        let rows = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Query pending operations failed: {}", e),
            })?;

        let mut ops = Vec::new();
        for r in rows {
            ops.push(r.map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Row error: {}", e),
            })?);
        }
        Ok(ops)
    }

    pub fn get_record_by_id_internal(&self, id: &str) -> Result<Option<MemoryRecord>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
            SELECT id, tenant_id, namespace, agent_id, user_id, kind,
                   content, content_hash, embedding, embedding_model, embedding_dims,
                   importance, created_at_ms, updated_at_ms, last_accessed_at_ms,
                   access_count, expires_at_ms, status, revision, metadata_json, source_json
            FROM memories
            WHERE id = ?
            "#,
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to prepare internal record query: {}", e),
            })?;

        let record = stmt
            .query_row(params![id], |row| {
                let id: String = row.get(0)?;
                let tenant_id: String = row.get(1)?;
                let namespace: String = row.get(2)?;
                let agent_id: Option<String> = row.get(3)?;
                let user_id: Option<String> = row.get(4)?;
                let kind_str: String = row.get(5)?;
                let content_bytes: Vec<u8> = row.get(6)?;
                let content_hash: Vec<u8> = row.get(7)?;
                let embedding_bytes: Vec<u8> = row.get(8)?;
                let embedding_model: String = row.get(9)?;
                let embedding_dims: i64 = row.get(10)?;
                let importance: f64 = row.get(11)?;
                let created_at_ms: i64 = row.get(12)?;
                let updated_at_ms: i64 = row.get(13)?;
                let last_accessed_at_ms: Option<i64> = row.get(14)?;
                let access_count: i64 = row.get(15)?;
                let expires_at_ms: Option<i64> = row.get(16)?;
                let status_str: String = row.get(17)?;
                let revision: i64 = row.get(18)?;
                let metadata_json: String = row.get(19)?;
                let source_json: String = row.get(20)?;

                let mut scope = MemoryScope::new(tenant_id, namespace).map_err(|_| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "Invalid scope",
                        )),
                    )
                })?;
                if let Some(agent) = agent_id {
                    scope = scope.with_agent(agent).unwrap();
                }
                if let Some(user) = user_id {
                    scope = scope.with_user(user).unwrap();
                }

                let kind = match kind_str.as_str() {
                    "preference" => MemoryKind::Preference,
                    "fact" => MemoryKind::Fact,
                    "instruction" => MemoryKind::Instruction,
                    "context" => MemoryKind::Context,
                    _ => MemoryKind::Episodic,
                };

                let embedding = bytes_to_embedding(&embedding_bytes);

                let metadata: HashMap<String, serde_json::Value> =
                    serde_json::from_str(&metadata_json).unwrap_or_default();
                let source: HashMap<String, serde_json::Value> =
                    serde_json::from_str(&source_json).unwrap_or_default();

                Ok(MemoryRecord {
                    id,
                    scope,
                    kind,
                    content: String::from_utf8_lossy(&content_bytes).to_string(),
                    content_hash,
                    embedding,
                    embedding_model,
                    embedding_dims: embedding_dims as usize,
                    importance: importance as f32,
                    created_at_ms,
                    updated_at_ms,
                    last_accessed_at_ms,
                    access_count: access_count as u64,
                    expires_at_ms,
                    status: MemoryStatus::parse(&status_str).unwrap_or(MemoryStatus::Pending),
                    revision: revision as u64,
                    metadata,
                    source,
                })
            })
            .optional()
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Internal query failed: {}", e),
            })?;

        Ok(record)
    }

    pub fn get_all_active_records(&self) -> Result<Vec<MemoryRecord>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
            SELECT id, tenant_id, namespace, agent_id, user_id, kind,
                   content, content_hash, embedding, embedding_model, embedding_dims,
                   importance, created_at_ms, updated_at_ms, last_accessed_at_ms,
                   access_count, expires_at_ms, status, revision, metadata_json, source_json
            FROM memories
            WHERE status = 'active'
            "#,
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to prepare active records query: {}", e),
            })?;

        let rows = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let tenant_id: String = row.get(1)?;
                let namespace: String = row.get(2)?;
                let agent_id: Option<String> = row.get(3)?;
                let user_id: Option<String> = row.get(4)?;
                let kind_str: String = row.get(5)?;
                let content_bytes: Vec<u8> = row.get(6)?;
                let content_hash: Vec<u8> = row.get(7)?;
                let embedding_bytes: Vec<u8> = row.get(8)?;
                let embedding_model: String = row.get(9)?;
                let embedding_dims: i64 = row.get(10)?;
                let importance: f64 = row.get(11)?;
                let created_at_ms: i64 = row.get(12)?;
                let updated_at_ms: i64 = row.get(13)?;
                let last_accessed_at_ms: Option<i64> = row.get(14)?;
                let access_count: i64 = row.get(15)?;
                let expires_at_ms: Option<i64> = row.get(16)?;
                let status_str: String = row.get(17)?;
                let revision: i64 = row.get(18)?;
                let metadata_json: String = row.get(19)?;
                let source_json: String = row.get(20)?;

                let mut scope = MemoryScope::new(tenant_id, namespace).map_err(|_| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "Invalid scope",
                        )),
                    )
                })?;
                if let Some(agent) = agent_id {
                    scope = scope.with_agent(agent).unwrap();
                }
                if let Some(user) = user_id {
                    scope = scope.with_user(user).unwrap();
                }

                let kind = match kind_str.as_str() {
                    "preference" => MemoryKind::Preference,
                    "fact" => MemoryKind::Fact,
                    "instruction" => MemoryKind::Instruction,
                    "context" => MemoryKind::Context,
                    _ => MemoryKind::Episodic,
                };

                let embedding = bytes_to_embedding(&embedding_bytes);

                let metadata: HashMap<String, serde_json::Value> =
                    serde_json::from_str(&metadata_json).unwrap_or_default();
                let source: HashMap<String, serde_json::Value> =
                    serde_json::from_str(&source_json).unwrap_or_default();

                Ok(MemoryRecord {
                    id,
                    scope,
                    kind,
                    content: String::from_utf8_lossy(&content_bytes).to_string(),
                    content_hash,
                    embedding,
                    embedding_model,
                    embedding_dims: embedding_dims as usize,
                    importance: importance as f32,
                    created_at_ms,
                    updated_at_ms,
                    last_accessed_at_ms,
                    access_count: access_count as u64,
                    expires_at_ms,
                    status: MemoryStatus::parse(&status_str).unwrap_or(MemoryStatus::Active),
                    revision: revision as u64,
                    metadata,
                    source,
                })
            })
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Query active records failed: {}", e),
            })?;

        let mut records = Vec::new();
        for r in rows {
            records.push(r.map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Row error: {}", e),
            })?);
        }
        Ok(records)
    }

    pub fn update_memory(
        &self,
        record: &MemoryRecord,
        expected_revision: u64,
        operation_id: Option<&str>,
        now_ms: i64,
    ) -> Result<()> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to begin update transaction: {}", e),
        })?;

        // Verify current revision and scope
        let (current_revision, current_status): (i64, String) = tx
            .query_row(
                "SELECT revision, status FROM memories WHERE id = ? AND tenant_id = ? AND namespace = ?",
                params![record.id, record.scope.tenant_id(), record.scope.namespace()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to fetch current revision: {}", e),
            })?
            .ok_or_else(|| MemoryError::not_found(&record.id))?;

        if current_status == "deleted" {
            return Err(MemoryError::not_found(&record.id));
        }

        if current_revision as u64 != expected_revision {
            return Err(MemoryError::RevisionConflict {
                code: ErrorCode::RevisionConflict,
                id: record.id.clone(),
                expected: expected_revision,
                current: current_revision as u64,
            });
        }

        let embedding_bytes: Vec<u8> = record
            .embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let metadata_json = serde_json::to_string(&record.metadata).unwrap_or_else(|_| "{}".into());

        tx.execute(
            r#"
            UPDATE memories SET
                content = ?,
                content_hash = ?,
                embedding = ?,
                importance = ?,
                updated_at_ms = ?,
                expires_at_ms = ?,
                status = ?,
                revision = ?,
                metadata_json = ?
            WHERE id = ? AND revision = ?
            "#,
            params![
                record.content.as_bytes(),
                record.content_hash,
                embedding_bytes,
                record.importance,
                record.updated_at_ms,
                record.expires_at_ms,
                record.status.as_str(),
                (expected_revision + 1) as i64,
                metadata_json,
                record.id,
                expected_revision as i64,
            ],
        )
        .map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to update memory record: {}", e),
        })?;

        if let Some(op_id) = operation_id {
            tx.execute(
                r#"
                INSERT INTO operations (
                    operation_id, memory_id, kind, payload_json, state, created_at_ms, applied_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, NULL)
                "#,
                params![
                    op_id,
                    record.id,
                    "update",
                    serde_json::to_string(record).unwrap_or_else(|_| "{}".into()),
                    "pending",
                    now_ms,
                ],
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to record update operation: {}", e),
            })?;
        }

        tx.commit().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to commit update transaction: {}", e),
        })?;

        Ok(())
    }

    pub fn delete_memory(
        &self,
        scope: &MemoryScope,
        id: &str,
        operation_id: Option<&str>,
        now_ms: i64,
    ) -> Result<()> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to begin delete transaction: {}", e),
        })?;

        tx.execute(
            r#"
            UPDATE memories SET
                status = 'deleted',
                updated_at_ms = ?
            WHERE id = ? AND tenant_id = ? AND namespace = ?
            "#,
            params![now_ms, id, scope.tenant_id(), scope.namespace()],
        )
        .map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to tombstone memory: {}", e),
        })?;

        if let Some(op_id) = operation_id {
            tx.execute(
                r#"
                INSERT INTO operations (
                    operation_id, memory_id, kind, payload_json, state, created_at_ms, applied_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                "#,
                params![op_id, id, "delete", "{}", "applied", now_ms, now_ms],
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to insert delete operation: {}", e),
            })?;
        }

        tx.commit().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to commit delete transaction: {}", e),
        })?;

        Ok(())
    }

    pub fn delete_memory_batch(
        &self,
        scope: &MemoryScope,
        ids: &[&str],
        operation_ids: &[Option<String>],
        now_ms: i64,
    ) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }

        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to begin delete batch transaction: {}", e),
        })?;

        {
            let mut del_stmt = tx.prepare(
                r#"
                UPDATE memories SET
                    status = 'deleted',
                    updated_at_ms = ?
                WHERE id = ? AND tenant_id = ? AND namespace = ?
                "#,
            ).map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to prepare delete batch statement: {}", e),
            })?;

            let mut op_stmt = tx.prepare(
                r#"
                INSERT INTO operations (
                    operation_id, memory_id, kind, payload_json, state, created_at_ms, applied_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                "#,
            ).map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to prepare delete operation batch statement: {}", e),
            })?;

            for (i, id) in ids.iter().enumerate() {
                del_stmt.execute(params![now_ms, id, scope.tenant_id(), scope.namespace()])
                    .map_err(|e| MemoryError::DatabaseError {
                        code: ErrorCode::DatabaseError,
                        message: format!("Failed to tombstone memory in batch: {}", e),
                    })?;

                if let Some(op_id) = operation_ids.get(i).and_then(|opt| opt.as_deref()) {
                    op_stmt.execute(params![op_id, id, "delete", "{}", "applied", now_ms, now_ms])
                        .map_err(|e| MemoryError::DatabaseError {
                            code: ErrorCode::DatabaseError,
                            message: format!("Failed to insert delete operation in batch: {}", e),
                        })?;
                }
            }
        }

        tx.commit().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to commit delete batch transaction: {}", e),
        })?;

        Ok(())
    }

    pub fn search_fts(
        &self,
        scope: &MemoryScope,
        query_text: &str,
        limit: usize,
    ) -> Result<Vec<String>> {
        let sanitized = sanitize_fts_query(query_text);
        if sanitized.is_empty() {
            return Ok(Vec::new());
        }

        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
            SELECT f.id
            FROM memories_fts f
            JOIN memories m ON f.id = m.id
            WHERE f.tenant_id = ?
              AND f.namespace = ?
              AND f.memories_fts MATCH ?
              AND m.status = 'active'
            ORDER BY rank
            LIMIT ?
            "#,
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to prepare FTS query: {}", e),
            })?;

        let rows = stmt
            .query_map(
                params![
                    scope.tenant_id(),
                    scope.namespace(),
                    sanitized,
                    limit as i64,
                ],
                |row| row.get(0),
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("FTS query failed: {}", e),
            })?;

        let ids = rows.flatten().collect();
        Ok(ids)
    }

    pub fn get_health_counts(&self) -> Result<(usize, usize, usize)> {
        let conn = self.conn.lock();
        let active_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE status = 'active'",
                [],
                |row| row.get(0),
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to count active records: {}", e),
            })?;

        let tombstoned_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE status = 'deleted'",
                [],
                |row| row.get(0),
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to count tombstoned records: {}", e),
            })?;

        let pending_ops_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM operations WHERE state = 'pending'",
                [],
                |row| row.get(0),
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to count pending operations: {}", e),
            })?;

        Ok((
            active_count as usize,
            tombstoned_count as usize,
            pending_ops_count as usize,
        ))
    }

    pub fn vacuum_tombstoned_records(&self, batch_size: usize, now_ms: i64) -> Result<usize> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to start vacuum transaction: {}", e),
        })?;

        let purged_count = tx
            .execute(
                r#"
            DELETE FROM memories
            WHERE id IN (
                SELECT id FROM memories
                WHERE status = 'deleted'
                   OR (expires_at_ms IS NOT NULL AND expires_at_ms <= ?)
                LIMIT ?
            )
            "#,
                params![now_ms, batch_size as i64],
            )
            .map_err(|e| MemoryError::DatabaseError {
                code: ErrorCode::DatabaseError,
                message: format!("Failed to vacuum tombstoned records: {}", e),
            })?;

        tx.commit().map_err(|e| MemoryError::DatabaseError {
            code: ErrorCode::DatabaseError,
            message: format!("Failed to commit vacuum transaction: {}", e),
        })?;

        Ok(purged_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_repository_lifecycle_and_reopen() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memory.db");

        let record = MemoryRecord {
            id: "mem-123".into(),
            scope: MemoryScope::new("tenant-a", "ns-1").unwrap(),
            kind: MemoryKind::Fact,
            content: "The sky is blue".into(),
            content_hash: vec![1, 2, 3, 4],
            embedding: vec![0.1, 0.2, 0.3],
            embedding_model: "test-model".into(),
            embedding_dims: 3,
            importance: 0.8,
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

        {
            let repo = Repository::open(&db_path).unwrap();
            repo.insert_pending_memory(&record, Some("op-1"), 1000)
                .unwrap();
            let fetched = repo
                .get_by_scope_and_id(&record.scope, &record.id)
                .unwrap()
                .expect("record should exist");
            assert_eq!(fetched.id, "mem-123");
            assert_eq!(fetched.content, "The sky is blue");
            assert_eq!(fetched.embedding, vec![0.1, 0.2, 0.3]);
            assert_eq!(fetched.status, MemoryStatus::Pending);
        }

        // Reopen database
        {
            let repo = Repository::open(&db_path).unwrap();
            let fetched = repo
                .get_by_scope_and_id(&record.scope, &record.id)
                .unwrap()
                .expect("record should persist across reopen");
            assert_eq!(fetched.id, "mem-123");
            assert_eq!(fetched.content, "The sky is blue");
            assert_eq!(fetched.embedding, vec![0.1, 0.2, 0.3]);
        }
    }

    #[test]
    fn test_cross_tenant_isolation_in_sql() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memory.db");
        let repo = Repository::open(&db_path).unwrap();

        let record = MemoryRecord {
            id: "mem-secret".into(),
            scope: MemoryScope::new("tenant-a", "ns-1").unwrap(),
            kind: MemoryKind::Fact,
            content: "Secret data".into(),
            content_hash: vec![0],
            embedding: vec![1.0, 0.0],
            embedding_model: "test".into(),
            embedding_dims: 2,
            importance: 0.5,
            created_at_ms: 100,
            updated_at_ms: 100,
            last_accessed_at_ms: None,
            access_count: 0,
            expires_at_ms: None,
            status: MemoryStatus::Pending,
            revision: 1,
            metadata: HashMap::new(),
            source: HashMap::new(),
        };

        repo.insert_pending_memory(&record, None, 100).unwrap();

        // Querying from tenant-b returns None
        let other_scope = MemoryScope::new("tenant-b", "ns-1").unwrap();
        assert!(repo
            .get_by_scope_and_id(&other_scope, "mem-secret")
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_fts_retrieval_and_scope_isolation() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("memory.db");
        let repo = Repository::open(&db_path).unwrap();

        let scope_a = MemoryScope::new("tenant-a", "ns-1").unwrap();
        let scope_b = MemoryScope::new("tenant-b", "ns-1").unwrap();

        let record_a = MemoryRecord {
            id: "mem-fts-1".into(),
            scope: scope_a.clone(),
            kind: MemoryKind::Fact,
            content: "Quantum computing breakthrough in silicon quantum dots".into(),
            content_hash: vec![1],
            embedding: vec![0.1, 0.9],
            embedding_model: "test".into(),
            embedding_dims: 2,
            importance: 0.9,
            created_at_ms: 100,
            updated_at_ms: 100,
            last_accessed_at_ms: None,
            access_count: 0,
            expires_at_ms: None,
            status: MemoryStatus::Active,
            revision: 1,
            metadata: HashMap::new(),
            source: HashMap::new(),
        };

        repo.insert_pending_memory(&record_a, None, 100).unwrap();
        repo.set_status("mem-fts-1", MemoryStatus::Active).unwrap();

        let record_b = MemoryRecord {
            id: "mem-fts-2".into(),
            scope: scope_b.clone(),
            kind: MemoryKind::Fact,
            content: "Quantum computing in silicon".into(),
            content_hash: vec![2],
            embedding: vec![0.1, 0.9],
            embedding_model: "test".into(),
            embedding_dims: 2,
            importance: 0.9,
            created_at_ms: 100,
            updated_at_ms: 100,
            last_accessed_at_ms: None,
            access_count: 0,
            expires_at_ms: None,
            status: MemoryStatus::Active,
            revision: 1,
            metadata: HashMap::new(),
            source: HashMap::new(),
        };

        repo.insert_pending_memory(&record_b, None, 100).unwrap();
        repo.set_status("mem-fts-2", MemoryStatus::Active).unwrap();

        // Search scope_a for "Quantum silicon"
        let matches_a = repo.search_fts(&scope_a, "Quantum silicon", 10).unwrap();
        assert_eq!(matches_a, vec!["mem-fts-1"]);

        // Search scope_b for "Quantum silicon"
        let matches_b = repo.search_fts(&scope_b, "Quantum silicon", 10).unwrap();
        assert_eq!(matches_b, vec!["mem-fts-2"]);

        // Tombstone mem-fts-1 and verify FTS search omits deleted
        repo.delete_memory(&scope_a, "mem-fts-1", None, 200).unwrap();
        let matches_after_del = repo.search_fts(&scope_a, "Quantum silicon", 10).unwrap();
        assert!(matches_after_del.is_empty());
    }
}
