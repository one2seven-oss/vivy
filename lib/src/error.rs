use thiserror::Error;

#[derive(Error, Debug)]
pub enum VivyError {
    #[error("WAL error: {0}")]
    Wal(#[from] crate::storage::wal::WalError),

    #[error("Segment error: {0}")]
    Segment(#[from] crate::storage::segments::SegmentError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type VivyResult<T> = Result<T, VivyError>;
