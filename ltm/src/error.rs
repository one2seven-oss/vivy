use std::fmt;
use thiserror::Error;

/// Stable error codes for programmatic client handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    InvalidScope,
    DimensionMismatch,
    EmbeddingModelMismatch,
    NotFound,
    RevisionConflict,
    StoreBusy,
    RecoveryRequired,
    CorruptStore,
    EncryptionKeyUnavailable,
    PolicyDenied,
    InvalidFilter,
    InvalidInput,
    IoError,
    DatabaseError,
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let code = match self {
            Self::InvalidScope => "INVALID_SCOPE",
            Self::DimensionMismatch => "DIMENSION_MISMATCH",
            Self::EmbeddingModelMismatch => "EMBEDDING_MODEL_MISMATCH",
            Self::NotFound => "NOT_FOUND",
            Self::RevisionConflict => "REVISION_CONFLICT",
            Self::StoreBusy => "STORE_BUSY",
            Self::RecoveryRequired => "RECOVERY_REQUIRED",
            Self::CorruptStore => "CORRUPT_STORE",
            Self::EncryptionKeyUnavailable => "ENCRYPTION_KEY_UNAVAILABLE",
            Self::PolicyDenied => "POLICY_DENIED",
            Self::InvalidFilter => "INVALID_FILTER",
            Self::InvalidInput => "INVALID_INPUT",
            Self::IoError => "IO_ERROR",
            Self::DatabaseError => "DATABASE_ERROR",
        };
        write!(f, "{}", code)
    }
}

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("[{code}] {message}")]
    InvalidScope {
        code: ErrorCode,
        message: String,
    },

    #[error("[{code}] Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        code: ErrorCode,
        expected: usize,
        actual: usize,
    },

    #[error("[{code}] Embedding model mismatch: expected {expected}, got {actual}")]
    EmbeddingModelMismatch {
        code: ErrorCode,
        expected: String,
        actual: String,
    },

    #[error("[{code}] Memory record not found: {id}")]
    NotFound {
        code: ErrorCode,
        id: String,
    },

    #[error("[{code}] Revision conflict for record {id}: expected revision {expected}, current revision {current}")]
    RevisionConflict {
        code: ErrorCode,
        id: String,
        expected: u64,
        current: u64,
    },

    #[error("[{code}] Store is busy or locked: {message}")]
    StoreBusy {
        code: ErrorCode,
        message: String,
    },

    #[error("[{code}] Recovery required before store can be accessed: {message}")]
    RecoveryRequired {
        code: ErrorCode,
        message: String,
    },

    #[error("[{code}] Corrupt store state: {message}")]
    CorruptStore {
        code: ErrorCode,
        message: String,
    },

    #[error("[{code}] Policy denied memory operation: {message}")]
    PolicyDenied {
        code: ErrorCode,
        message: String,
    },

    #[error("[{code}] Invalid filter: {message}")]
    InvalidFilter {
        code: ErrorCode,
        message: String,
    },

    #[error("[{code}] Invalid input parameter: {message}")]
    InvalidInput {
        code: ErrorCode,
        message: String,
    },

    #[error("[{code}] Database error: {message}")]
    DatabaseError {
        code: ErrorCode,
        message: String,
    },

    #[error("[{code}] I/O error: {message}")]
    IoError {
        code: ErrorCode,
        message: String,
    },
}

impl MemoryError {
    pub fn code(&self) -> ErrorCode {
        match self {
            Self::InvalidScope { code, .. } => *code,
            Self::DimensionMismatch { code, .. } => *code,
            Self::EmbeddingModelMismatch { code, .. } => *code,
            Self::NotFound { code, .. } => *code,
            Self::RevisionConflict { code, .. } => *code,
            Self::StoreBusy { code, .. } => *code,
            Self::RecoveryRequired { code, .. } => *code,
            Self::CorruptStore { code, .. } => *code,
            Self::PolicyDenied { code, .. } => *code,
            Self::InvalidFilter { code, .. } => *code,
            Self::InvalidInput { code, .. } => *code,
            Self::DatabaseError { code, .. } => *code,
            Self::IoError { code, .. } => *code,
        }
    }

    pub fn invalid_scope(msg: impl Into<String>) -> Self {
        Self::InvalidScope {
            code: ErrorCode::InvalidScope,
            message: msg.into(),
        }
    }

    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput {
            code: ErrorCode::InvalidInput,
            message: msg.into(),
        }
    }

    pub fn not_found(id: impl Into<String>) -> Self {
        Self::NotFound {
            code: ErrorCode::NotFound,
            id: id.into(),
        }
    }
}

pub type Result<T> = std::result::Result<T, MemoryError>;
