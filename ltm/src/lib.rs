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
    ForgetRequest, MemoryFilter, MemoryKind, MemoryRecord, MemoryStatus, RecallExplanation,
    RecallItem, RecallRequest, RecallResponse, RememberRequest, UpdateRequest,
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
                if record.status != MemoryStatus::Active {
                    continue;
                }

                if let Some(exp) = record.expires_at_ms {
                    if now_ms >= exp {
                        continue;
                    }
                }

                if let Some(ref kinds) = req.filters.kinds {
                    if !kinds.contains(&record.kind) {
                        continue;
                    }
                }

                if let Some(min_imp) = req.filters.min_importance {
                    if record.importance < min_imp {
                        continue;
                    }
                }

                let similarity = (1.0 - (distance / 2.0)).clamp(0.0, 1.0);
                let importance = record.importance.clamp(0.0, 1.0);

                // 7-day half-life decay (604_800_000 ms)
                let age_ms = (now_ms - record.created_at_ms).max(0) as f32;
                let recency_decay = (-age_ms / (7.0 * 86_400_000.0)).exp();

                let reinforcement = (record.access_count as f32 / 10.0).clamp(0.0, 1.0);

                // 04-api-contract: 0.70*sim + 0.15*imp + 0.10*recency + 0.05*reinforcement
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
            }
        }

        // Apply MMR diversity if mmr_lambda is specified
        if let Some(lambda) = req.mmr_lambda {
            let mut selected: Vec<RecallItem> = Vec::new();
            let mut remaining = items;

            while !remaining.is_empty() && selected.len() < req.limit {
                let mut best_idx = 0;
                let mut best_mmr_score = f32::NEG_INFINITY;

                for (idx, candidate) in remaining.iter().enumerate() {
                    // Relevance term
                    let rel = candidate.score;

                    // Redundancy term: max similarity to already selected items
                    let mut max_sim = 0.0f32;
                    for sel in &selected {
                        let dist = vivy_core::distance::cosine(
                            &candidate.memory.embedding,
                            &sel.memory.embedding,
                        );
                        let sim = (1.0 - dist).clamp(0.0, 1.0);
                        if sim > max_sim {
                            max_sim = sim;
                        }
                    }

                    // MMR formula: lambda * rel - (1 - lambda) * max_sim
                    let mmr_score = lambda * rel - (1.0 - lambda) * max_sim;
                    if mmr_score > best_mmr_score {
                        best_mmr_score = mmr_score;
                        best_idx = idx;
                    }
                }

                let chosen = remaining.remove(best_idx);
                selected.push(chosen);
            }

            Ok(RecallResponse {
                total_candidates: selected.len(),
                items: selected,
            })
        } else {
            // Sort items by total score descending
            items.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            items.truncate(req.limit);

            Ok(RecallResponse {
                total_candidates: items.len(),
                items,
            })
        }
    }

    /// Update an existing memory record with optimistic concurrency.
    pub fn update(&self, req: UpdateRequest) -> Result<()> {
        req.validate(self.config.dimensions())?;

        let existing = self
            .repo
            .get_by_scope_and_id(&req.scope, &req.id)?
            .ok_or_else(|| MemoryError::not_found(&req.id))?;

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let new_content = req.content.unwrap_or(existing.content);
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&new_content, &mut hasher);
        let new_content_hash = std::hash::Hasher::finish(&hasher).to_le_bytes().to_vec();

        let new_embedding = req.embedding.unwrap_or(existing.embedding);
        let new_kind = req.kind.unwrap_or(existing.kind);
        let new_importance = req.importance.unwrap_or(existing.importance);
        let new_expires_at = req.expires_at_ms.unwrap_or(existing.expires_at_ms);

        let mut new_metadata = existing.metadata;
        if let Some(patch) = req.metadata_patch {
            for (k, v) in patch {
                new_metadata.insert(k, v);
            }
        }

        let updated_record = MemoryRecord {
            id: req.id,
            scope: req.scope,
            kind: new_kind,
            content: new_content,
            content_hash: new_content_hash,
            embedding: new_embedding,
            embedding_model: self.config.embedding_model().to_string(),
            embedding_dims: self.config.dimensions(),
            importance: new_importance,
            created_at_ms: existing.created_at_ms,
            updated_at_ms: now_ms,
            last_accessed_at_ms: existing.last_accessed_at_ms,
            access_count: existing.access_count,
            expires_at_ms: new_expires_at,
            status: MemoryStatus::Active,
            revision: existing.revision + 1,
            metadata: new_metadata,
            source: existing.source,
        };

        self.coordinator
            .coordinate_update(updated_record, req.expected_revision, req.operation_id)
    }

    /// Forget/delete a memory record immediately hiding it from reads.
    pub fn forget(&self, req: ForgetRequest) -> Result<()> {
        self.coordinator
            .coordinate_forget(&req.scope, &req.id, req.operation_id)
    }
}
