use crate::error::{MemoryError, Result};
use serde::{Deserialize, Serialize};

/// Strongly-typed, validated scope partition for multi-tenant isolation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryScope {
    tenant_id: String,
    namespace: String,
    agent_id: Option<String>,
    user_id: Option<String>,
}

impl MemoryScope {
    pub fn new(tenant_id: impl Into<String>, namespace: impl Into<String>) -> Result<Self> {
        let tenant_id = tenant_id.into().trim().to_string();
        let namespace = namespace.into().trim().to_string();

        if tenant_id.is_empty() {
            return Err(MemoryError::invalid_scope("tenant_id must not be empty"));
        }
        if namespace.is_empty() {
            return Err(MemoryError::invalid_scope("namespace must not be empty"));
        }

        Ok(Self {
            tenant_id,
            namespace,
            agent_id: None,
            user_id: None,
        })
    }

    pub fn with_agent(mut self, agent_id: impl Into<String>) -> Result<Self> {
        let agent = agent_id.into().trim().to_string();
        if agent.is_empty() {
            return Err(MemoryError::invalid_scope("agent_id cannot be empty string"));
        }
        self.agent_id = Some(agent);
        Ok(self)
    }

    pub fn with_user(mut self, user_id: impl Into<String>) -> Result<Self> {
        let user = user_id.into().trim().to_string();
        if user.is_empty() {
            return Err(MemoryError::invalid_scope("user_id cannot be empty string"));
        }
        self.user_id = Some(user);
        Ok(self)
    }

    pub fn tenant_id(&self) -> &str {
        &self.tenant_id
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn agent_id(&self) -> Option<&str> {
        self.agent_id.as_deref()
    }

    pub fn user_id(&self) -> Option<&str> {
        self.user_id.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_scope() {
        let scope = MemoryScope::new("acme", "support")
            .unwrap()
            .with_agent("agent-007")
            .unwrap()
            .with_user("user-42")
            .unwrap();
        assert_eq!(scope.tenant_id(), "acme");
        assert_eq!(scope.namespace(), "support");
        assert_eq!(scope.agent_id(), Some("agent-007"));
        assert_eq!(scope.user_id(), Some("user-42"));
    }

    #[test]
    fn test_empty_scope_rejected() {
        assert!(MemoryScope::new("", "support").is_err());
        assert!(MemoryScope::new("   ", "support").is_err());
        assert!(MemoryScope::new("acme", "").is_err());
        assert!(MemoryScope::new("acme", "   ").is_err());
    }

    #[test]
    fn test_empty_optional_fields_rejected() {
        let scope = MemoryScope::new("acme", "support").unwrap();
        assert!(scope.with_agent("").is_err());
        let scope = MemoryScope::new("acme", "support").unwrap();
        assert!(scope.with_user("  ").is_err());
    }
}
