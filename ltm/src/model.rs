use crate::error::{MemoryError, Result};
use crate::namespace::MemoryScope;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Identifies the semantic nature of a stored memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    Preference,
    Fact,
    Instruction,
    Context,
    Episodic,
}

/// Lifecycle status of a memory in the store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStatus {
    Pending,
    Active,
    Deleted,
    Expired,
}

impl MemoryStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Active => "active",
            Self::Deleted => "deleted",
            Self::Expired => "expired",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "pending" => Some(Self::Pending),
            "active" => Some(Self::Active),
            "deleted" => Some(Self::Deleted),
            "expired" => Some(Self::Expired),
            _ => None,
        }
    }
}

/// A stored memory record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: String,
    pub scope: MemoryScope,
    pub kind: MemoryKind,
    pub content: String,
    pub content_hash: Vec<u8>,
    pub embedding: Vec<f32>,
    pub embedding_model: String,
    pub embedding_dims: usize,
    pub importance: f32,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    pub last_accessed_at_ms: Option<i64>,
    pub access_count: u64,
    pub expires_at_ms: Option<i64>,
    pub status: MemoryStatus,
    pub revision: u64,
    pub metadata: HashMap<String, serde_json::Value>,
    pub source: HashMap<String, serde_json::Value>,
}

/// Request to store an observation or fact into memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RememberRequest {
    pub operation_id: Option<String>,
    pub scope: MemoryScope,
    pub content: String,
    pub embedding: Vec<f32>,
    pub kind: MemoryKind,
    pub importance: f32,
    pub expires_at_ms: Option<i64>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub source: HashMap<String, serde_json::Value>,
}

impl RememberRequest {
    pub fn validate(&self, expected_dims: usize) -> Result<()> {
        if self.content.trim().is_empty() {
            return Err(MemoryError::invalid_input("content cannot be empty"));
        }
        if self.embedding.len() != expected_dims {
            return Err(MemoryError::DimensionMismatch {
                code: crate::error::ErrorCode::DimensionMismatch,
                expected: expected_dims,
                actual: self.embedding.len(),
            });
        }
        if !(0.0..=1.0).contains(&self.importance) {
            return Err(MemoryError::invalid_input(
                "importance must be between 0.0 and 1.0",
            ));
        }
        Ok(())
    }
}

/// Filtering criteria for recall operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryFilter {
    pub kinds: Option<Vec<MemoryKind>>,
    pub min_importance: Option<f32>,
    pub metadata_eq: Option<HashMap<String, serde_json::Value>>,
}

impl MemoryFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kind_in(kinds: impl IntoIterator<Item = MemoryKind>) -> Self {
        Self {
            kinds: Some(kinds.into_iter().collect()),
            min_importance: None,
            metadata_eq: None,
        }
    }

    pub fn with_min_importance(mut self, min_importance: f32) -> Self {
        self.min_importance = Some(min_importance);
        self
    }
}

/// Request to update an existing memory record with optimistic revision concurrency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateRequest {
    pub operation_id: Option<String>,
    pub scope: MemoryScope,
    pub id: String,
    pub expected_revision: u64,
    pub content: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub kind: Option<MemoryKind>,
    pub importance: Option<f32>,
    pub expires_at_ms: Option<Option<i64>>,
    pub metadata_patch: Option<HashMap<String, serde_json::Value>>,
}

impl UpdateRequest {
    pub fn validate(&self, expected_dims: usize) -> Result<()> {
        if self.id.trim().is_empty() {
            return Err(MemoryError::invalid_input("id cannot be empty"));
        }
        if let Some(ref c) = self.content {
            if c.trim().is_empty() {
                return Err(MemoryError::invalid_input("content cannot be empty"));
            }
        }
        if let Some(ref emb) = self.embedding {
            if emb.len() != expected_dims {
                return Err(MemoryError::DimensionMismatch {
                    code: crate::error::ErrorCode::DimensionMismatch,
                    expected: expected_dims,
                    actual: emb.len(),
                });
            }
        }
        if let Some(imp) = self.importance {
            if !(0.0..=1.0).contains(&imp) {
                return Err(MemoryError::invalid_input(
                    "importance must be between 0.0 and 1.0",
                ));
            }
        }
        Ok(())
    }
}

/// Request to delete/forget a memory record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgetRequest {
    pub operation_id: Option<String>,
    pub scope: MemoryScope,
    pub id: String,
}

impl ForgetRequest {
    pub fn new(scope: MemoryScope, id: impl Into<String>) -> Result<Self> {
        let id = id.into().trim().to_string();
        if id.is_empty() {
            return Err(MemoryError::invalid_input("id cannot be empty"));
        }
        Ok(Self {
            operation_id: None,
            scope,
            id,
        })
    }
}

/// Request to recall memories for an agent context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallRequest {
    pub scope: MemoryScope,
    pub query_embedding: Vec<f32>,
    pub limit: usize,
    pub filters: MemoryFilter,
    pub include_explanations: bool,
    pub mmr_lambda: Option<f32>,
}

impl RecallRequest {
    pub fn validate(&self, expected_dims: usize) -> Result<()> {
        if self.limit == 0 {
            return Err(MemoryError::invalid_input("limit must be greater than 0"));
        }
        if self.query_embedding.len() != expected_dims {
            return Err(MemoryError::DimensionMismatch {
                code: crate::error::ErrorCode::DimensionMismatch,
                expected: expected_dims,
                actual: self.query_embedding.len(),
            });
        }
        if let Some(lambda) = self.mmr_lambda {
            if !(0.0..=1.0).contains(&lambda) {
                return Err(MemoryError::invalid_input(
                    "mmr_lambda must be between 0.0 and 1.0",
                ));
            }
        }
        Ok(())
    }
}

/// Explanation detailing why a memory was selected during recall.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallExplanation {
    pub total_score: f32,
    pub similarity_score: f32,
    pub importance_score: f32,
    pub recency_score: f32,
    pub reinforcement_score: f32,
    pub policy_notes: Vec<String>,
}

/// Individual recalled item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallItem {
    pub memory: MemoryRecord,
    pub score: f32,
    pub explanation: Option<RecallExplanation>,
}

/// Complete response from a recall query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallResponse {
    pub items: Vec<RecallItem>,
    pub total_candidates: usize,
}
