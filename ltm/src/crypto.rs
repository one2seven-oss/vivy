use crate::error::{MemoryError, Result};

/// Contract for providing encryption keys per tenant for encrypted store integration.
pub trait KeyProvider: Send + Sync {
    /// Retrieve the 256-bit encryption key for the specified tenant.
    fn get_key(&self, tenant_id: &str) -> Result<Vec<u8>>;
}

/// Default key provider that fails with `ENCRYPTION_KEY_UNAVAILABLE` when key resolution is requested.
#[derive(Debug, Default, Clone, Copy)]
pub struct MissingKeyProvider;

impl KeyProvider for MissingKeyProvider {
    fn get_key(&self, tenant_id: &str) -> Result<Vec<u8>> {
        Err(MemoryError::encryption_key_unavailable(format!(
            "No KeyProvider registered to retrieve encryption key for tenant '{tenant_id}'"
        )))
    }
}

/// Development key provider supplying a deterministic key for local testing.
#[derive(Debug, Clone)]
pub struct NoOpDevKeyProvider {
    dev_key: Vec<u8>,
}

impl NoOpDevKeyProvider {
    pub fn new() -> Self {
        Self {
            dev_key: vec![0x42; 32],
        }
    }

    pub fn with_custom_key(key: Vec<u8>) -> Self {
        Self { dev_key: key }
    }
}

impl Default for NoOpDevKeyProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyProvider for NoOpDevKeyProvider {
    fn get_key(&self, _tenant_id: &str) -> Result<Vec<u8>> {
        Ok(self.dev_key.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorCode;

    #[test]
    fn test_missing_key_provider_returns_correct_error_code() {
        let provider = MissingKeyProvider;
        let err = provider.get_key("tenant-alpha").unwrap_err();
        assert_eq!(err.code(), ErrorCode::EncryptionKeyUnavailable);
        assert!(err.to_string().contains("tenant-alpha"));
    }

    #[test]
    fn test_noop_dev_key_provider() {
        let provider = NoOpDevKeyProvider::new();
        let key = provider.get_key("tenant-beta").unwrap();
        assert_eq!(key.len(), 32);
        assert_eq!(key[0], 0x42);
    }
}
