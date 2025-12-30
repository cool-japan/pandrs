//! Session Management
//!
//! This module provides session management for authenticated users,
//! including session creation, validation, refresh, and expiration.

use crate::multitenancy::TenantId;
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};

/// User session information
#[derive(Debug, Clone)]
pub struct Session {
    /// Unique session identifier
    pub session_id: String,
    /// Associated user ID
    pub user_id: String,
    /// Associated tenant ID
    pub tenant_id: TenantId,
    /// Session creation time
    pub created_at: SystemTime,
    /// Last activity time
    pub last_activity: Instant,
    /// Session timeout duration
    pub timeout: Duration,
    /// Session data/attributes
    pub attributes: HashMap<String, String>,
    /// IP address (if available)
    pub ip_address: Option<String>,
    /// User agent (if available)
    pub user_agent: Option<String>,
    /// Whether the session is active
    pub active: bool,
}

impl Session {
    /// Create a new session
    pub fn new(user_id: impl Into<String>, tenant_id: impl Into<String>) -> Self {
        Session {
            session_id: generate_session_id(),
            user_id: user_id.into(),
            tenant_id: tenant_id.into(),
            created_at: SystemTime::now(),
            last_activity: Instant::now(),
            timeout: Duration::from_secs(3600), // Default 1 hour
            attributes: HashMap::new(),
            ip_address: None,
            user_agent: None,
            active: true,
        }
    }

    /// Set session timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set IP address
    pub fn with_ip_address(mut self, ip: impl Into<String>) -> Self {
        self.ip_address = Some(ip.into());
        self
    }

    /// Set user agent
    pub fn with_user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = Some(ua.into());
        self
    }

    /// Set a session attribute
    pub fn set_attribute(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.attributes.insert(key.into(), value.into());
    }

    /// Get a session attribute
    pub fn get_attribute(&self, key: &str) -> Option<&String> {
        self.attributes.get(key)
    }

    /// Remove a session attribute
    pub fn remove_attribute(&mut self, key: &str) -> Option<String> {
        self.attributes.remove(key)
    }

    /// Check if the session has expired
    pub fn is_expired(&self) -> bool {
        !self.active || self.last_activity.elapsed() > self.timeout
    }

    /// Refresh the session (update last activity time)
    pub fn refresh(&mut self) {
        self.last_activity = Instant::now();
    }

    /// Get remaining time before expiration
    pub fn time_remaining(&self) -> Duration {
        if self.is_expired() {
            Duration::ZERO
        } else {
            self.timeout.saturating_sub(self.last_activity.elapsed())
        }
    }

    /// Invalidate the session
    pub fn invalidate(&mut self) {
        self.active = false;
    }

    /// Get session duration
    pub fn duration(&self) -> Duration {
        self.created_at.elapsed().unwrap_or(Duration::ZERO)
    }
}

/// Session store for managing multiple sessions
#[derive(Debug)]
pub struct SessionStore {
    /// Active sessions by ID
    sessions: HashMap<String, Session>,
    /// Sessions by user ID
    user_sessions: HashMap<String, Vec<String>>,
    /// Maximum sessions per user
    max_sessions_per_user: usize,
    /// Default session timeout
    default_timeout: Duration,
    /// Whether to allow concurrent sessions
    allow_concurrent: bool,
}

impl SessionStore {
    /// Create a new session store
    pub fn new() -> Self {
        SessionStore {
            sessions: HashMap::new(),
            user_sessions: HashMap::new(),
            max_sessions_per_user: 5,
            default_timeout: Duration::from_secs(3600),
            allow_concurrent: true,
        }
    }

    /// Set maximum sessions per user
    pub fn with_max_sessions(mut self, max: usize) -> Self {
        self.max_sessions_per_user = max;
        self
    }

    /// Set default session timeout
    pub fn with_default_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    /// Disable concurrent sessions
    pub fn without_concurrent_sessions(mut self) -> Self {
        self.allow_concurrent = false;
        self
    }

    /// Create a new session for a user
    pub fn create_session(&mut self, user_id: &str, tenant_id: &str) -> Session {
        // If concurrent sessions not allowed, invalidate existing sessions
        if !self.allow_concurrent {
            self.invalidate_user_sessions(user_id);
        }

        // Check if user has reached max sessions
        if let Some(session_ids) = self.user_sessions.get(user_id) {
            if session_ids.len() >= self.max_sessions_per_user {
                // Remove oldest session
                if let Some(oldest_id) = session_ids.first().cloned() {
                    self.remove_session(&oldest_id);
                }
            }
        }

        let session = Session::new(user_id, tenant_id).with_timeout(self.default_timeout);

        let session_id = session.session_id.clone();

        self.sessions.insert(session_id.clone(), session.clone());

        self.user_sessions
            .entry(user_id.to_string())
            .or_insert_with(Vec::new)
            .push(session_id);

        session
    }

    /// Get a session by ID
    pub fn get_session(&self, session_id: &str) -> Option<&Session> {
        self.sessions.get(session_id)
    }

    /// Get a mutable reference to a session
    pub fn get_session_mut(&mut self, session_id: &str) -> Option<&mut Session> {
        self.sessions.get_mut(session_id)
    }

    /// Validate and refresh a session
    pub fn validate_session(&mut self, session_id: &str) -> Option<&Session> {
        if let Some(session) = self.sessions.get_mut(session_id) {
            if session.is_expired() {
                return None;
            }
            session.refresh();
            return self.sessions.get(session_id);
        }
        None
    }

    /// Remove a session
    pub fn remove_session(&mut self, session_id: &str) -> Option<Session> {
        if let Some(session) = self.sessions.remove(session_id) {
            // Remove from user_sessions
            if let Some(session_ids) = self.user_sessions.get_mut(&session.user_id) {
                session_ids.retain(|id| id != session_id);
            }
            return Some(session);
        }
        None
    }

    /// Invalidate all sessions for a user
    pub fn invalidate_user_sessions(&mut self, user_id: &str) {
        if let Some(session_ids) = self.user_sessions.get(user_id) {
            for session_id in session_ids.clone() {
                if let Some(session) = self.sessions.get_mut(&session_id) {
                    session.invalidate();
                }
            }
        }
    }

    /// Get all sessions for a user
    pub fn get_user_sessions(&self, user_id: &str) -> Vec<&Session> {
        self.user_sessions
            .get(user_id)
            .map(|session_ids| {
                session_ids
                    .iter()
                    .filter_map(|id| self.sessions.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get active session count for a user
    pub fn get_active_session_count(&self, user_id: &str) -> usize {
        self.get_user_sessions(user_id)
            .iter()
            .filter(|s| !s.is_expired())
            .count()
    }

    /// Clean up expired sessions
    pub fn cleanup_expired(&mut self) {
        let expired_ids: Vec<String> = self
            .sessions
            .iter()
            .filter(|(_, session)| session.is_expired())
            .map(|(id, _)| id.clone())
            .collect();

        for session_id in expired_ids {
            self.remove_session(&session_id);
        }
    }

    /// Get total session count
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Get active session count
    pub fn active_session_count(&self) -> usize {
        self.sessions.values().filter(|s| !s.is_expired()).count()
    }
}

impl Default for SessionStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Session-based authentication context
#[derive(Debug, Clone)]
pub struct SessionContext {
    /// Current session
    pub session: Session,
    /// Authenticated user ID
    pub user_id: String,
    /// Tenant ID
    pub tenant_id: TenantId,
    /// Request-specific data
    pub request_data: HashMap<String, String>,
}

impl SessionContext {
    /// Create from a session
    pub fn from_session(session: Session) -> Self {
        SessionContext {
            user_id: session.user_id.clone(),
            tenant_id: session.tenant_id.clone(),
            session,
            request_data: HashMap::new(),
        }
    }

    /// Set request data
    pub fn set_request_data(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.request_data.insert(key.into(), value.into());
    }

    /// Get request data
    pub fn get_request_data(&self, key: &str) -> Option<&String> {
        self.request_data.get(key)
    }
}

// Helper functions

/// Generate a unique session ID
fn generate_session_id() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    format!(
        "sess_{}",
        bytes
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_creation() {
        let session = Session::new("user1", "tenant_a");

        assert!(session.session_id.starts_with("sess_"));
        assert_eq!(session.user_id, "user1");
        assert_eq!(session.tenant_id, "tenant_a");
        assert!(session.active);
        assert!(!session.is_expired());
    }

    #[test]
    fn test_session_expiration() {
        let session = Session::new("user1", "tenant_a").with_timeout(Duration::from_millis(1));

        // Wait for session to expire
        std::thread::sleep(Duration::from_millis(10));

        assert!(session.is_expired());
    }

    #[test]
    fn test_session_refresh() {
        let mut session = Session::new("user1", "tenant_a").with_timeout(Duration::from_secs(1));

        // Wait a bit
        std::thread::sleep(Duration::from_millis(100));

        let time_before_refresh = session.time_remaining();

        // Refresh session
        session.refresh();

        let time_after_refresh = session.time_remaining();

        // Time remaining should be reset
        assert!(time_after_refresh > time_before_refresh);
    }

    #[test]
    fn test_session_attributes() {
        let mut session = Session::new("user1", "tenant_a");

        session.set_attribute("theme", "dark");
        session.set_attribute("language", "en");

        assert_eq!(session.get_attribute("theme"), Some(&"dark".to_string()));
        assert_eq!(session.get_attribute("language"), Some(&"en".to_string()));
        assert_eq!(session.get_attribute("missing"), None);

        // Remove attribute
        let removed = session.remove_attribute("theme");
        assert_eq!(removed, Some("dark".to_string()));
        assert_eq!(session.get_attribute("theme"), None);
    }

    #[test]
    fn test_session_store() {
        let mut store = SessionStore::new();

        // Create session
        let session = store.create_session("user1", "tenant_a");
        let session_id = session.session_id.clone();

        // Get session
        assert!(store.get_session(&session_id).is_some());

        // Validate session
        assert!(store.validate_session(&session_id).is_some());

        // Remove session
        let removed = store.remove_session(&session_id);
        assert!(removed.is_some());
        assert!(store.get_session(&session_id).is_none());
    }

    #[test]
    fn test_session_store_max_sessions() {
        let mut store = SessionStore::new().with_max_sessions(2);

        // Create 3 sessions for the same user
        let s1 = store.create_session("user1", "tenant_a");
        let s2 = store.create_session("user1", "tenant_a");
        let s3 = store.create_session("user1", "tenant_a");

        // First session should be removed
        assert!(store.get_session(&s1.session_id).is_none());
        assert!(store.get_session(&s2.session_id).is_some());
        assert!(store.get_session(&s3.session_id).is_some());
    }

    #[test]
    fn test_session_store_no_concurrent() {
        let mut store = SessionStore::new().without_concurrent_sessions();

        // Create first session
        let s1 = store.create_session("user1", "tenant_a");

        // Create second session (should invalidate first)
        let s2 = store.create_session("user1", "tenant_a");

        // First session should be invalidated
        let session1 = store.get_session(&s1.session_id);
        assert!(session1.is_none() || !session1.unwrap().active);

        // Second session should be active
        assert!(store.get_session(&s2.session_id).is_some());
    }

    #[test]
    fn test_session_store_cleanup() {
        let mut store = SessionStore::new().with_default_timeout(Duration::from_millis(1));

        // Create sessions
        store.create_session("user1", "tenant_a");
        store.create_session("user2", "tenant_b");

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(10));

        // Cleanup
        store.cleanup_expired();

        assert_eq!(store.session_count(), 0);
    }

    #[test]
    fn test_user_sessions() {
        let mut store = SessionStore::new();

        store.create_session("user1", "tenant_a");
        store.create_session("user1", "tenant_a");
        store.create_session("user2", "tenant_b");

        assert_eq!(store.get_user_sessions("user1").len(), 2);
        assert_eq!(store.get_user_sessions("user2").len(), 1);
        assert_eq!(store.get_active_session_count("user1"), 2);
    }

    #[test]
    fn test_session_context() {
        let session = Session::new("user1", "tenant_a");
        let mut ctx = SessionContext::from_session(session);

        assert_eq!(ctx.user_id, "user1");
        assert_eq!(ctx.tenant_id, "tenant_a");

        ctx.set_request_data("correlation_id", "abc123");
        assert_eq!(
            ctx.get_request_data("correlation_id"),
            Some(&"abc123".to_string())
        );
    }
}
