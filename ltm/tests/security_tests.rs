use vivy_memory::*;

#[test]
fn test_missing_key_provider_returns_encryption_key_unavailable() {
    let missing_provider = MissingKeyProvider;
    let err = missing_provider.get_key("acme-corp").unwrap_err();

    assert_eq!(err.code(), ErrorCode::EncryptionKeyUnavailable);
    assert_eq!(err.code().to_string(), "ENCRYPTION_KEY_UNAVAILABLE");
    assert!(err.to_string().contains("acme-corp"));
}

#[test]
fn test_noop_dev_key_provider_supplies_deterministic_key() {
    let dev_provider = NoOpDevKeyProvider::new();
    let key_bytes = dev_provider.get_key("acme-corp").unwrap();

    assert_eq!(key_bytes.len(), 32);
    assert_eq!(key_bytes, vec![0x42; 32]);
}

#[test]
fn test_telemetry_event_sanitization_guarantees_no_payload_leakage() {
    let telemetry = TelemetryRecord::new("tenant-secret", "confidential-ns", "recall")
        .with_operation_id(Some("op-sec-999".into()))
        .with_duration_us(1500)
        .with_outcome(true, None)
        .with_metrics(5, 4096);

    let serialized_json = serde_json::to_string(&telemetry).unwrap();

    // Verify metadata telemetry fields are present
    assert!(serialized_json.contains("tenant-secret"));
    assert!(serialized_json.contains("confidential-ns"));
    assert!(serialized_json.contains("recall"));

    // Verify sensitive data fields (content, embedding, key) are never present
    assert!(!serialized_json.contains("content"));
    assert!(!serialized_json.contains("embedding"));
    assert!(!serialized_json.contains("key"));
}

#[test]
fn test_error_formatting_redacts_memory_content_and_keys() {
    let _scope = MemoryScope::new("acme-tenant", "support-ns").unwrap();
    let err = MemoryError::invalid_scope("invalid tenant formatting");

    let err_msg = format!("{}", err);
    assert!(err_msg.contains("INVALID_SCOPE"));
    assert!(err_msg.contains("invalid tenant formatting"));

    // Verify formatted error string contains no raw memory content or key primitives
    assert!(!err_msg.contains("content="));
    assert!(!err_msg.contains("embedding="));
    assert!(!err_msg.contains("key_bytes="));
}
