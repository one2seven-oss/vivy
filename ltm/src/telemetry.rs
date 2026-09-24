use serde::{Deserialize, Serialize};

/// Sanitized operational telemetry event for observability without data leakage.
///
/// **Non-Negotiable Requirement**: Raw memory contents, query strings, float embeddings,
/// and encryption key bytes MUST NEVER be included in telemetry events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetryRecord {
    pub operation_id: Option<String>,
    pub tenant_id: String,
    pub namespace: String,
    pub operation_kind: String,
    pub duration_us: u64,
    pub success: bool,
    pub error_code: Option<String>,
    pub records_count: usize,
    pub bytes_processed: usize,
}

impl TelemetryRecord {
    pub fn new(
        tenant_id: impl Into<String>,
        namespace: impl Into<String>,
        operation_kind: impl Into<String>,
    ) -> Self {
        Self {
            operation_id: None,
            tenant_id: tenant_id.into(),
            namespace: namespace.into(),
            operation_kind: operation_kind.into(),
            duration_us: 0,
            success: true,
            error_code: None,
            records_count: 0,
            bytes_processed: 0,
        }
    }

    pub fn with_operation_id(mut self, op_id: Option<String>) -> Self {
        self.operation_id = op_id;
        self
    }

    pub fn with_duration_us(mut self, duration_us: u64) -> Self {
        self.duration_us = duration_us;
        self
    }

    pub fn with_outcome(mut self, success: bool, error_code: Option<String>) -> Self {
        self.success = success;
        self.error_code = error_code;
        self
    }

    pub fn with_metrics(mut self, records_count: usize, bytes_processed: usize) -> Self {
        self.records_count = records_count;
        self.bytes_processed = bytes_processed;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_serialization_excludes_raw_content_and_embeddings() {
        let record = TelemetryRecord::new("tenant-acme", "support-chat", "remember")
            .with_operation_id(Some("op-123".into()))
            .with_duration_us(420)
            .with_outcome(true, None)
            .with_metrics(1, 1024);

        let json = serde_json::to_string(&record).unwrap();

        // Assert JSON contains telemetry metadata fields
        assert!(json.contains("tenant-acme"));
        assert!(json.contains("support-chat"));
        assert!(json.contains("remember"));
        assert!(json.contains("op-123"));

        // Assert forbidden confidential fields (content, embeddings, keys) do not exist
        assert!(!json.contains("content"));
        assert!(!json.contains("embedding"));
        assert!(!json.contains("secret"));
        assert!(!json.contains("key"));
    }
}
