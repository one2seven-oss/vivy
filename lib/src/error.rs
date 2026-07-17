use thiserror::Error;

#[derive(Error, Debug)]
pub enum VivyError {
    #[error("WAL error: {0}")]
    Wal(#[from] crate::storage::wal::WalError),

    #[error("Segment error: {0}")]
    Segment(#[from] crate::storage::segments::SegmentError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Inconsistent state: {0}")]
    Inconsistent(String),

    #[error("Index full: {capacity} vectors, cannot insert more")]
    IndexFull { capacity: usize },

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Compaction failed: {0}")]
    Compaction(String),
}

pub type VivyResult<T> = Result<T, VivyError>;
