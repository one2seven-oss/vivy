pub mod config;
pub mod error;
pub mod index_adapter;
pub mod journal;
pub mod model;
pub mod namespace;
pub mod repository;

pub use config::MemoryConfig;
pub use error::{ErrorCode, MemoryError, Result};
pub use index_adapter::{InMemoryTestIndex, VectorIndex, VivyVectorIndex};
pub use model::{
    MemoryFilter, MemoryKind, MemoryRecord, MemoryStatus, RecallExplanation, RecallItem,
    RecallRequest, RecallResponse, RememberRequest,
};
pub use namespace::MemoryScope;
pub use repository::Repository;

use crate::journal::JournalCoordinator;
use std::sync::Arc;
use uuid::Uuid;
use vivy_core::distance::Metric;

/// A local, durable, namespace-isolated memory store for AI agents.
pub struct MemoryStore {
    config: Arc<MemoryConfig>,
    repo: Arc<Repository>,
    index: Arc<dyn VectorIndex>,
    coordinator: JournalCoordinator,
}

impl MemoryStore {
    /// Open or initialize a memory store at the configured directory.
    pub fn open(config: MemoryConfig) -> Result<Self> {
        let path = config.path();
        if !path.exists() {
            std::fs::create_dir_all(path).map_err(|e| MemoryError::IoError {
                code: ErrorCode::IoError,
                message: format!("Failed to create directory {:?}: {}", path, e),
            })?;
        }

        let db_path = path.join("memory.db");
        let repo = Arc::new(Repository::open(db_path)?);

        let index = Arc::new(VivyVectorIndex::new(config.dimensions(), Metric::Cosine)?);
        let coordinator = JournalCoordinator::new(repo.clone(), index.clone());

        // Run crash recovery / journal replay
        coordinator.recover_startup()?;

        Ok(Self {
            config: Arc::new(config),
            repo,
            index,
            coordinator,
        })
    }

    /// Access the store's immutable configuration.
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }

    /// Store an observation/fact into durable memory.
    pub fn remember(&self, req: RememberRequest) -> Result<String> {
        req.validate(self.config.dimensions())?;

        let id = Uuid::new_v4().to_string();
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&req.content, &mut hasher);
        let content_hash = std::hash::Hasher::finish(&hasher).to_le_bytes().to_vec();

        let record = MemoryRecord {
            id,
            scope: req.scope,
            kind: req.kind,
            content: req.content,
            content_hash,
            embedding: req.embedding,
            embedding_model: self.config.embedding_model().to_string(),
            embedding_dims: self.config.dimensions(),
            importance: req.importance,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
            last_accessed_at_ms: None,
            access_count: 0,
            expires_at_ms: req.expires_at_ms,
            status: MemoryStatus::Pending,
            revision: 1,
            metadata: req.metadata,
            source: req.source,
        };

        self.coordinator
            .coordinate_remember(record, req.operation_id)
    }

    /// Fetch a memory by ID within scope.
    pub fn get(&self, scope: &MemoryScope, id: &str) -> Result<Option<MemoryRecord>> {
        self.repo.get_by_scope_and_id(scope, id)
    }

    /// Retrieve matching memories for an agent context with explainability.
    pub fn recall(&self, req: RecallRequest) -> Result<RecallResponse> {
        req.validate(self.config.dimensions())?;

        let candidates = self
            .index
            .search(&req.query_embedding, req.limit.min(self.config.max_recall_limit()) * 3)?;

        let mut items = Vec::new();
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        for (mem_id, distance) in candidates {
            if let Some(record) = self.repo.get_by_scope_and_id(&req.scope, &mem_id)? {
                // Must be active
                if record.status != MemoryStatus::Active {
                    continue;
                }

                // Check expiry
                if let Some(exp) = record.expires_at_ms {
                    if now_ms >= exp {
                        continue;
                    }
                }

                // Filter by kind
                if let Some(ref kinds) = req.filters.kinds {
                    if !kinds.contains(&record.kind) {
                        continue;
                    }
                }

                // Filter by min importance
                if let Some(min_imp) = req.filters.min_importance {
                    if record.importance < min_imp {
                        continue;
                    }
                }

                // Cosine similarity in [0, 1]
                let similarity = (1.0 - (distance / 2.0)).clamp(0.0, 1.0);
                let importance = record.importance.clamp(0.0, 1.0);

                // Recency decay: exponential decay with half-life of 7 days (604,800,000 ms)
                let age_ms = (now_ms - record.created_at_ms).max(0) as f32;
                let recency_decay = (-age_ms / (7.0 * 86_400_000.0)).exp();

                // Reinforcement score based on access count
                let reinforcement = (record.access_count as f32 / 10.0).clamp(0.0, 1.0);

                // Default recall scoring formula from 04-api-contract.md:
                // score = 0.70 * similarity + 0.15 * importance + 0.10 * recency_decay + 0.05 * reinforcement
                let total_score = 0.70 * similarity
                    + 0.15 * importance
                    + 0.10 * recency_decay
                    + 0.05 * reinforcement;

                let explanation = if req.include_explanations {
                    Some(RecallExplanation {
                        total_score,
                        similarity_score: similarity,
                        importance_score: importance,
                        recency_score: recency_decay,
                        reinforcement_score: reinforcement,
                        policy_notes: vec!["default_v1_scoring".to_string()],
                    })
                } else {
                    None
                };

                items.push(RecallItem {
                    memory: record,
                    score: total_score,
                    explanation,
                });

                if items.len() == req.limit {
                    break;
                }
            }
        }

        // Sort items by total score descending
        items.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        Ok(RecallResponse {
            total_candidates: items.len(),
            items,
        })
    }
}
