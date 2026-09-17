use crate::error::{MemoryError, Result};
use std::path::{Path, PathBuf};

/// Configuration for opening or creating a `MemoryStore`.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub(crate) path: PathBuf,
    pub(crate) dimensions: usize,
    pub(crate) embedding_model: String,
    pub(crate) max_recall_limit: usize,
}

impl MemoryConfig {
    pub fn builder(path: impl AsRef<Path>) -> MemoryConfigBuilder {
        MemoryConfigBuilder::new(path)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn embedding_model(&self) -> &str {
        &self.embedding_model
    }

    pub fn max_recall_limit(&self) -> usize {
        self.max_recall_limit
    }
}

pub struct MemoryConfigBuilder {
    path: PathBuf,
    dimensions: Option<usize>,
    embedding_model: Option<String>,
    max_recall_limit: usize,
}

impl MemoryConfigBuilder {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            dimensions: None,
            embedding_model: None,
            max_recall_limit: 100,
        }
    }

    pub fn dimensions(mut self, dims: usize) -> Self {
        self.dimensions = Some(dims);
        self
    }

    pub fn embedding_model(mut self, model: impl Into<String>) -> Self {
        self.embedding_model = Some(model.into());
        self
    }

    pub fn max_recall_limit(mut self, limit: usize) -> Self {
        self.max_recall_limit = limit;
        self
    }

    pub fn build(self) -> Result<MemoryConfig> {
        let dimensions = self.dimensions.ok_or_else(|| {
            MemoryError::invalid_input("MemoryConfig: embedding dimensions must be specified")
        })?;

        if dimensions == 0 {
            return Err(MemoryError::invalid_input(
                "MemoryConfig: dimensions must be greater than 0",
            ));
        }

        let embedding_model = self.embedding_model.ok_or_else(|| {
            MemoryError::invalid_input("MemoryConfig: embedding_model must be specified")
        })?;

        if embedding_model.trim().is_empty() {
            return Err(MemoryError::invalid_input(
                "MemoryConfig: embedding_model cannot be empty",
            ));
        }

        Ok(MemoryConfig {
            path: self.path,
            dimensions,
            embedding_model,
            max_recall_limit: self.max_recall_limit,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_config() {
        let config = MemoryConfig::builder("./test_mem")
            .dimensions(1536)
            .embedding_model("text-embedding-3-small")
            .build()
            .unwrap();

        assert_eq!(config.dimensions(), 1536);
        assert_eq!(config.embedding_model(), "text-embedding-3-small");
        assert_eq!(config.max_recall_limit(), 100);
    }

    #[test]
    fn test_missing_fields_fail() {
        assert!(MemoryConfig::builder("./test_mem").build().is_err());
        assert!(MemoryConfig::builder("./test_mem")
            .dimensions(1536)
            .build()
            .is_err());
        assert!(MemoryConfig::builder("./test_mem")
            .embedding_model("test")
            .build()
            .is_err());
        assert!(MemoryConfig::builder("./test_mem")
            .dimensions(0)
            .embedding_model("test")
            .build()
            .is_err());
    }
}
