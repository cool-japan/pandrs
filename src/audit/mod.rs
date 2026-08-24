//! Audit logging module for DataFrame operations
//!
//! This module provides comprehensive audit logging capabilities for tracking
//! all operations performed on DataFrames, enabling:
//!
//! - Compliance tracking and reporting
//! - Operation history for debugging
//! - Performance monitoring
//! - Security audit trails
//! - Data access logging
//!
//! # Quick Start
//!
//! ```rust
//! use pandrs::audit::{AuditLogger, AuditConfig, LogLevel};
//!
//! // Create an audit logger
//! let mut logger = AuditLogger::new(AuditConfig::default());
//!
//! // Log an operation
//! logger.log_operation("select", "DataFrame", "Selecting columns a, b");
//!
//! // Log with context
//! logger.log_with_context("filter", "DataFrame", "Filtering rows", |ctx| {
//!     ctx.set("condition", "value > 10");
//!     ctx.set("rows_before", "1000");
//!     ctx.set("rows_after", "500");
//! });
//! ```
//!
//! # Configuration
//!
//! ```rust
//! use pandrs::audit::{AuditLogger, AuditConfig, LogLevel, LogDestination};
//!
//! let config = AuditConfig::builder()
//!     .level(LogLevel::Info)
//!     .destination(LogDestination::Memory)
//!     .max_entries(10000)
//!     .include_timestamps(true)
//!     .include_user(true)
//!     .build();
//!
//! let logger = AuditLogger::new(config);
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt::{Debug, Display};
use std::io::Write;
use std::sync::{Arc, Mutex, RwLock};

/// Log level for audit entries
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LogLevel {
    /// Trace level - very detailed logging
    Trace,
    /// Debug level - debugging information
    Debug,
    /// Info level - general information
    Info,
    /// Warning level - warnings
    Warn,
    /// Error level - errors
    Error,
}

impl Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "TRACE"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
        }
    }
}

impl Default for LogLevel {
    fn default() -> Self {
        LogLevel::Info
    }
}

/// Destination for audit logs
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogDestination {
    /// Log to memory (for testing or in-memory analysis)
    Memory,
    /// Log to standard output
    Stdout,
    /// Log to standard error
    Stderr,
    /// Log to a file
    File(String),
    /// Log to multiple destinations
    Multi(Vec<LogDestination>),
}

impl Default for LogDestination {
    fn default() -> Self {
        LogDestination::Memory
    }
}

/// Category of audit event
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventCategory {
    /// Data access event
    DataAccess,
    /// Data modification event
    DataModification,
    /// Schema change event
    SchemaChange,
    /// Query execution event
    QueryExecution,
    /// I/O operation event
    IoOperation,
    /// Security-related event
    Security,
    /// System event
    System,
    /// Custom category
    Custom(String),
}

impl Display for EventCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventCategory::DataAccess => write!(f, "DATA_ACCESS"),
            EventCategory::DataModification => write!(f, "DATA_MODIFICATION"),
            EventCategory::SchemaChange => write!(f, "SCHEMA_CHANGE"),
            EventCategory::QueryExecution => write!(f, "QUERY_EXECUTION"),
            EventCategory::IoOperation => write!(f, "IO_OPERATION"),
            EventCategory::Security => write!(f, "SECURITY"),
            EventCategory::System => write!(f, "SYSTEM"),
            EventCategory::Custom(name) => write!(f, "CUSTOM:{}", name),
        }
    }
}

/// A single audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique ID for this entry
    pub id: String,
    /// Timestamp of the event
    pub timestamp: DateTime<Utc>,
    /// Log level
    pub level: LogLevel,
    /// Event category
    pub category: EventCategory,
    /// Operation name
    pub operation: String,
    /// Target object (e.g., DataFrame name)
    pub target: String,
    /// Human-readable message
    pub message: String,
    /// User who performed the operation
    pub user: Option<String>,
    /// Session ID
    pub session_id: Option<String>,
    /// Duration in milliseconds
    pub duration_ms: Option<u64>,
    /// Additional context/metadata
    pub context: HashMap<String, String>,
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Hash of the previous entry in the tamper-evident chain (hex). Empty for
    /// the genesis entry. Populated by the logger at append time.
    #[serde(default)]
    pub prev_hash: String,
    /// HMAC chain hash of this entry (hex). Populated by the logger.
    #[serde(default)]
    pub entry_hash: String,
}

impl AuditEntry {
    /// Creates a new audit entry
    pub fn new(
        level: LogLevel,
        category: EventCategory,
        operation: &str,
        target: &str,
        message: &str,
    ) -> Self {
        AuditEntry {
            id: generate_entry_id(),
            timestamp: Utc::now(),
            level,
            category,
            operation: operation.to_string(),
            target: target.to_string(),
            message: message.to_string(),
            user: None,
            session_id: None,
            duration_ms: None,
            context: HashMap::new(),
            success: true,
            error: None,
            prev_hash: String::new(),
            entry_hash: String::new(),
        }
    }

    /// Sets the user
    pub fn with_user(mut self, user: &str) -> Self {
        self.user = Some(user.to_string());
        self
    }

    /// Sets the session ID
    pub fn with_session(mut self, session_id: &str) -> Self {
        self.session_id = Some(session_id.to_string());
        self
    }

    /// Sets the duration
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }

    /// Adds context
    pub fn with_context(mut self, key: &str, value: &str) -> Self {
        self.context.insert(key.to_string(), value.to_string());
        self
    }

    /// Marks as failed with an error
    pub fn with_error(mut self, error: &str) -> Self {
        self.success = false;
        self.error = Some(error.to_string());
        self
    }

    /// Formats the entry as a string
    pub fn format(&self) -> String {
        let duration_str = self
            .duration_ms
            .map(|d| format!(" ({}ms)", d))
            .unwrap_or_default();

        let user_str = self
            .user
            .as_ref()
            .map(|u| format!(" [user: {}]", u))
            .unwrap_or_default();

        let status = if self.success { "OK" } else { "FAILED" };

        format!(
            "{} [{}] [{}] {}/{}: {}{}{}[{}]",
            self.timestamp.format("%Y-%m-%d %H:%M:%S%.3f"),
            self.level,
            self.category,
            self.operation,
            self.target,
            self.message,
            duration_str,
            user_str,
            status
        )
    }

    /// Converts to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// Generate a unique entry ID.
///
/// Uses 128 bits from the CSPRNG formatted as a version-4 UUID string. The
/// previous implementation used the wall-clock nanosecond count, which is
/// predictable and collides for entries created in the same nanosecond (or when
/// the clock is not monotonic). This also removes a `.expect` from the logging
/// path.
fn generate_entry_id() -> String {
    use scirs2_core::random::Rng;
    let mut bytes = [0u8; 16];
    scirs2_core::random::rng().fill_bytes(&mut bytes);
    // Set version (4) and variant (RFC 4122) bits.
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let h: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &h[0..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..32]
    )
}

/// Redact string and numeric literals from a query, preserving structure.
/// Quoted strings become `'?'` and numbers become `?`.
fn redact_query(query: &str) -> String {
    let mut out = String::with_capacity(query.len());
    let mut chars = query.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\'' | '"' => {
                let quote = c;
                out.push(quote);
                out.push('?');
                out.push(quote);
                // Consume through the closing quote.
                while let Some(n) = chars.next() {
                    if n == quote {
                        break;
                    }
                }
            }
            '0'..='9' => {
                out.push('?');
                while let Some(&n) = chars.peek() {
                    if n.is_ascii_digit() || n == '.' {
                        chars.next();
                    } else {
                        break;
                    }
                }
            }
            _ => out.push(c),
        }
    }
    out
}

/// Stable one-way fingerprint (truncated SHA-256 hex) of a query, so identical
/// queries correlate without storing their literal (possibly sensitive) text.
fn query_fingerprint(query: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(query.as_bytes());
    digest
        .iter()
        .take(8)
        .map(|b| format!("{:02x}", b))
        .collect()
}

/// HMAC-SHA256 over `message` with `key`, returned as a lowercase hex string.
/// Used to build the tamper-evident audit chain.
fn hmac_sha256_hex(key: &[u8], message: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    const BLOCK_SIZE: usize = 64;

    let key_bytes: Vec<u8> = if key.len() > BLOCK_SIZE {
        Sha256::digest(key).to_vec()
    } else {
        key.to_vec()
    };
    let mut key_padded = [0u8; BLOCK_SIZE];
    key_padded[..key_bytes.len()].copy_from_slice(&key_bytes);

    let mut ipad = vec![0x36u8; BLOCK_SIZE];
    let mut opad = vec![0x5cu8; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        ipad[i] ^= key_padded[i];
        opad[i] ^= key_padded[i];
    }

    let mut inner = Sha256::new();
    inner.update(&ipad);
    inner.update(message);
    let inner_hash = inner.finalize();

    let mut outer = Sha256::new();
    outer.update(&opad);
    outer.update(inner_hash);
    outer
        .finalize()
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}

/// Canonical, hash-stable serialization of an entry's content (everything
/// except the chain fields themselves).
fn canonical_payload(entry: &AuditEntry) -> String {
    let mut ctx: Vec<(&String, &String)> = entry.context.iter().collect();
    ctx.sort_by(|a, b| a.0.cmp(b.0));
    format!(
        "{}|{}|{}|{}|{}|{}|{}|{:?}|{:?}|{}|{:?}|{:?}",
        entry.id,
        entry.timestamp.to_rfc3339(),
        entry.level,
        entry.category,
        entry.operation,
        entry.target,
        entry.message,
        entry.user,
        entry.session_id,
        entry.success,
        entry.error,
        ctx,
    )
}

/// Configuration for the audit logger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Minimum log level to record
    pub level: LogLevel,
    /// Where to send logs
    pub destination: LogDestination,
    /// Maximum number of entries to keep in memory
    pub max_entries: usize,
    /// Whether to include timestamps
    pub include_timestamps: bool,
    /// Whether to include user information
    pub include_user: bool,
    /// Default user name
    pub default_user: Option<String>,
    /// Session ID
    pub session_id: Option<String>,
    /// Categories to include (empty = all)
    pub include_categories: Vec<EventCategory>,
    /// Categories to exclude
    pub exclude_categories: Vec<EventCategory>,
    /// Whether to log in JSON format
    pub json_format: bool,
}

impl Default for AuditConfig {
    fn default() -> Self {
        AuditConfig {
            level: LogLevel::Info,
            destination: LogDestination::Memory,
            max_entries: 10000,
            include_timestamps: true,
            include_user: true,
            default_user: None,
            session_id: None,
            include_categories: Vec::new(),
            exclude_categories: Vec::new(),
            json_format: false,
        }
    }
}

/// Builder for AuditConfig
pub struct AuditConfigBuilder {
    config: AuditConfig,
}

impl AuditConfigBuilder {
    /// Creates a new builder
    pub fn new() -> Self {
        AuditConfigBuilder {
            config: AuditConfig::default(),
        }
    }

    /// Sets the log level
    pub fn level(mut self, level: LogLevel) -> Self {
        self.config.level = level;
        self
    }

    /// Sets the destination
    pub fn destination(mut self, destination: LogDestination) -> Self {
        self.config.destination = destination;
        self
    }

    /// Sets the maximum entries
    pub fn max_entries(mut self, max: usize) -> Self {
        self.config.max_entries = max;
        self
    }

    /// Sets whether to include timestamps
    pub fn include_timestamps(mut self, include: bool) -> Self {
        self.config.include_timestamps = include;
        self
    }

    /// Sets whether to include user
    pub fn include_user(mut self, include: bool) -> Self {
        self.config.include_user = include;
        self
    }

    /// Sets the default user
    pub fn default_user(mut self, user: &str) -> Self {
        self.config.default_user = Some(user.to_string());
        self
    }

    /// Sets the session ID
    pub fn session_id(mut self, session_id: &str) -> Self {
        self.config.session_id = Some(session_id.to_string());
        self
    }

    /// Includes a category
    pub fn include_category(mut self, category: EventCategory) -> Self {
        self.config.include_categories.push(category);
        self
    }

    /// Excludes a category
    pub fn exclude_category(mut self, category: EventCategory) -> Self {
        self.config.exclude_categories.push(category);
        self
    }

    /// Sets JSON format
    pub fn json_format(mut self, json: bool) -> Self {
        self.config.json_format = json;
        self
    }

    /// Builds the config
    pub fn build(self) -> AuditConfig {
        self.config
    }
}

impl Default for AuditConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AuditConfig {
    /// Creates a builder
    pub fn builder() -> AuditConfigBuilder {
        AuditConfigBuilder::new()
    }
}

/// Context builder for adding metadata to log entries
pub struct LogContext {
    context: HashMap<String, String>,
}

impl LogContext {
    /// Creates a new context
    pub fn new() -> Self {
        LogContext {
            context: HashMap::new(),
        }
    }

    /// Sets a context value
    pub fn set(&mut self, key: &str, value: &str) {
        self.context.insert(key.to_string(), value.to_string());
    }

    /// Gets all context values
    pub fn into_map(self) -> HashMap<String, String> {
        self.context
    }
}

impl Default for LogContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Main audit logger
#[derive(Debug)]
pub struct AuditLogger {
    config: AuditConfig,
    entries: VecDeque<AuditEntry>,
    file_handle: Option<Arc<Mutex<std::fs::File>>>,
    /// Per-logger HMAC key for the tamper-evident chain (random at creation).
    chain_key: Vec<u8>,
    /// Hash of the most recently appended entry (chain head).
    last_hash: String,
    /// Number of entries evicted by the ring buffer (evidence-of-drop marker).
    dropped_entries: u64,
    /// True when a File destination could not be opened; entries are still kept
    /// in memory and mirrored to stderr rather than silently vanishing.
    degraded: bool,
}

impl AuditLogger {
    /// Creates a new audit logger.
    ///
    /// If a `File` destination cannot be opened, the logger degrades to
    /// in-memory + stderr (with a one-time warning) rather than silently
    /// dropping every entry. Use [`AuditLogger::try_new`] to receive the open
    /// error instead.
    pub fn new(config: AuditConfig) -> Self {
        match Self::try_new(config) {
            Ok(logger) => logger,
            Err((config, err)) => {
                eprintln!(
                    "audit: failed to open log file ({}); degrading to in-memory+stderr",
                    err
                );
                AuditLogger {
                    chain_key: Self::new_chain_key(),
                    entries: VecDeque::new(),
                    file_handle: None,
                    last_hash: String::new(),
                    dropped_entries: 0,
                    degraded: true,
                    config,
                }
            }
        }
    }

    /// Fallible constructor: propagates a `File` open failure (returning the
    /// config back so the caller can fall back) instead of swallowing it.
    #[allow(clippy::result_large_err)]
    pub fn try_new(
        config: AuditConfig,
    ) -> std::result::Result<Self, (AuditConfig, std::io::Error)> {
        let file_handle = match &config.destination {
            LogDestination::File(path) => {
                match std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                {
                    Ok(f) => Some(Arc::new(Mutex::new(f))),
                    Err(e) => return Err((config, e)),
                }
            }
            _ => None,
        };

        Ok(AuditLogger {
            chain_key: Self::new_chain_key(),
            config,
            entries: VecDeque::new(),
            file_handle,
            last_hash: String::new(),
            dropped_entries: 0,
            degraded: false,
        })
    }

    /// Generate a fresh random HMAC chain key.
    fn new_chain_key() -> Vec<u8> {
        use scirs2_core::random::Rng;
        let mut key = vec![0u8; 32];
        scirs2_core::random::rng().fill_bytes(&mut key);
        key
    }

    /// Number of entries evicted by the ring buffer since creation.
    pub fn dropped_entries(&self) -> u64 {
        self.dropped_entries
    }

    /// Verify the tamper-evident hash chain over the retained entries.
    ///
    /// Returns `Ok(())` when every retained entry's `entry_hash` matches a
    /// recomputation over its content and recorded `prev_hash`, and consecutive
    /// entries are linked (`entry[i].prev_hash == entry[i-1].entry_hash`). Any
    /// in-place mutation, reordering, or deletion of a retained entry breaks the
    /// check.
    pub fn verify_chain(&self) -> std::result::Result<(), String> {
        let mut prev: Option<&str> = None;
        for (i, entry) in self.entries.iter().enumerate() {
            let recomputed = hmac_sha256_hex(
                &self.chain_key,
                format!("{}{}", canonical_payload(entry), entry.prev_hash).as_bytes(),
            );
            if recomputed != entry.entry_hash {
                return Err(format!("entry {} hash mismatch (tampered content)", i));
            }
            if let Some(prev_hash) = prev {
                if entry.prev_hash != prev_hash {
                    return Err(format!("entry {} chain linkage broken", i));
                }
            }
            prev = Some(&entry.entry_hash);
        }
        Ok(())
    }

    /// Checks if a category should be logged
    fn should_log_category(&self, category: &EventCategory) -> bool {
        // If exclude list contains the category, don't log
        if self.config.exclude_categories.contains(category) {
            return false;
        }

        // If include list is empty, log all (except excluded)
        if self.config.include_categories.is_empty() {
            return true;
        }

        // Otherwise, only log if in include list
        self.config.include_categories.contains(category)
    }

    /// Logs an entry
    pub fn log(&mut self, entry: AuditEntry) {
        // Check level
        if entry.level < self.config.level {
            return;
        }

        // Check category
        if !self.should_log_category(&entry.category) {
            return;
        }

        // Apply default user if needed
        let mut entry = entry;
        if entry.user.is_none() && self.config.include_user {
            entry.user = self.config.default_user.clone();
        }

        // Apply session ID if needed
        if entry.session_id.is_none() {
            entry.session_id = self.config.session_id.clone();
        }

        // Extend the tamper-evident chain: link to the previous entry's hash
        // and compute this entry's HMAC over (content || prev_hash).
        entry.prev_hash = self.last_hash.clone();
        entry.entry_hash = hmac_sha256_hex(
            &self.chain_key,
            format!("{}{}", canonical_payload(&entry), entry.prev_hash).as_bytes(),
        );
        self.last_hash = entry.entry_hash.clone();

        // Security entries are durability-critical: fsync them to the file.
        let is_security = entry.category == EventCategory::Security;

        // Write to destination(s). In degraded mode, mirror to stderr so
        // entries are visible even though the file could not be opened.
        self.write_entry(&entry, is_security);
        if self.degraded {
            self.write_to_destination(&LogDestination::Stderr, &self.format_output(&entry), false);
        }

        // Store in memory
        self.entries.push_back(entry);

        // Enforce max entries, counting evictions as an evidence-of-drop marker.
        while self.entries.len() > self.config.max_entries {
            self.entries.pop_front();
            self.dropped_entries = self.dropped_entries.saturating_add(1);
        }
    }

    /// Render an entry to its output string (JSON or human-readable).
    fn format_output(&self, entry: &AuditEntry) -> String {
        if self.config.json_format {
            entry.to_json().unwrap_or_else(|_| entry.format())
        } else {
            entry.format()
        }
    }

    /// Writes an entry to the configured destination
    fn write_entry(&self, entry: &AuditEntry, fsync: bool) {
        let output = self.format_output(entry);
        self.write_to_destination(&self.config.destination, &output, fsync);
    }

    /// Writes to a specific destination
    fn write_to_destination(&self, dest: &LogDestination, output: &str, fsync: bool) {
        match dest {
            LogDestination::Memory => {
                // Already stored in entries
            }
            LogDestination::Stdout => {
                println!("{}", output);
            }
            LogDestination::Stderr => {
                eprintln!("{}", output);
            }
            LogDestination::File(_) => {
                if let Some(ref file) = self.file_handle {
                    if let Ok(mut f) = file.lock() {
                        let _ = writeln!(f, "{}", output);
                        if fsync {
                            // Durability for security-critical entries.
                            let _ = f.flush();
                            let _ = f.sync_all();
                        }
                    }
                }
            }
            LogDestination::Multi(destinations) => {
                for d in destinations {
                    self.write_to_destination(d, output, fsync);
                }
            }
        }
    }

    /// Logs an operation
    pub fn log_operation(&mut self, operation: &str, target: &str, message: &str) {
        let entry = AuditEntry::new(
            LogLevel::Info,
            EventCategory::DataModification,
            operation,
            target,
            message,
        );
        self.log(entry);
    }

    /// Logs with custom context
    pub fn log_with_context<F>(&mut self, operation: &str, target: &str, message: &str, f: F)
    where
        F: FnOnce(&mut LogContext),
    {
        let mut ctx = LogContext::new();
        f(&mut ctx);

        let mut entry = AuditEntry::new(
            LogLevel::Info,
            EventCategory::DataModification,
            operation,
            target,
            message,
        );
        entry.context = ctx.into_map();
        self.log(entry);
    }

    /// Logs a data access event
    pub fn log_data_access(&mut self, target: &str, message: &str) {
        let entry = AuditEntry::new(
            LogLevel::Info,
            EventCategory::DataAccess,
            "access",
            target,
            message,
        );
        self.log(entry);
    }

    /// Logs a schema change event
    pub fn log_schema_change(&mut self, target: &str, message: &str) {
        let entry = AuditEntry::new(
            LogLevel::Info,
            EventCategory::SchemaChange,
            "schema_change",
            target,
            message,
        );
        self.log(entry);
    }

    /// Logs a query execution event.
    ///
    /// The query is redacted (string and numeric literals masked) before being
    /// stored so that PII / secrets embedded in literals do not leak into audit
    /// exports. A stable one-way fingerprint of the original query is attached
    /// so identical queries can still be correlated.
    pub fn log_query(&mut self, query: &str, duration_ms: u64) {
        let entry = AuditEntry::new(
            LogLevel::Info,
            EventCategory::QueryExecution,
            "query",
            "database",
            &redact_query(query),
        )
        .with_duration(duration_ms)
        .with_context("query_fingerprint", &query_fingerprint(query));
        self.log(entry);
    }

    /// Logs an I/O operation event
    pub fn log_io(&mut self, operation: &str, path: &str, message: &str) {
        let entry = AuditEntry::new(
            LogLevel::Info,
            EventCategory::IoOperation,
            operation,
            path,
            message,
        );
        self.log(entry);
    }

    /// Logs a security event
    pub fn log_security(&mut self, operation: &str, message: &str) {
        let entry = AuditEntry::new(
            LogLevel::Warn,
            EventCategory::Security,
            operation,
            "security",
            message,
        );
        self.log(entry);
    }

    /// Logs an error
    pub fn log_error(&mut self, operation: &str, target: &str, error: &str) {
        let entry = AuditEntry::new(
            LogLevel::Error,
            EventCategory::System,
            operation,
            target,
            error,
        )
        .with_error(error);
        self.log(entry);
    }

    /// Gets all entries
    pub fn entries(&self) -> &VecDeque<AuditEntry> {
        &self.entries
    }

    /// Gets entries by category
    pub fn entries_by_category(&self, category: &EventCategory) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| &e.category == category)
            .collect()
    }

    /// Gets entries by level
    pub fn entries_by_level(&self, level: LogLevel) -> Vec<&AuditEntry> {
        self.entries.iter().filter(|e| e.level >= level).collect()
    }

    /// Gets entries by operation
    pub fn entries_by_operation(&self, operation: &str) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.operation == operation)
            .collect()
    }

    /// Gets entries in a time range
    pub fn entries_in_range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .collect()
    }

    /// Gets failed entries
    pub fn failed_entries(&self) -> Vec<&AuditEntry> {
        self.entries.iter().filter(|e| !e.success).collect()
    }

    /// Clears all entries
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Gets statistics about logged entries
    pub fn stats(&self) -> AuditStats {
        let by_level: HashMap<String, usize> = self
            .entries
            .iter()
            .map(|e| e.level.to_string())
            .fold(HashMap::new(), |mut acc, level| {
                *acc.entry(level).or_insert(0) += 1;
                acc
            });

        let by_category: HashMap<String, usize> = self
            .entries
            .iter()
            .map(|e| e.category.to_string())
            .fold(HashMap::new(), |mut acc, cat| {
                *acc.entry(cat).or_insert(0) += 1;
                acc
            });

        let failed_count = self.entries.iter().filter(|e| !e.success).count();

        AuditStats {
            total_entries: self.entries.len(),
            by_level,
            by_category,
            failed_count,
        }
    }

    /// Exports entries to JSON
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.entries.iter().collect::<Vec<_>>())
    }
}

impl Default for AuditLogger {
    fn default() -> Self {
        Self::new(AuditConfig::default())
    }
}

/// Statistics about audit log entries
#[derive(Debug, Clone)]
pub struct AuditStats {
    /// Total number of entries
    pub total_entries: usize,
    /// Count by log level
    pub by_level: HashMap<String, usize>,
    /// Count by category
    pub by_category: HashMap<String, usize>,
    /// Number of failed operations
    pub failed_count: usize,
}

/// Thread-safe shared audit logger
#[derive(Debug, Clone)]
pub struct SharedAuditLogger {
    inner: Arc<RwLock<AuditLogger>>,
}

impl SharedAuditLogger {
    /// Creates a new shared audit logger
    pub fn new(config: AuditConfig) -> Self {
        SharedAuditLogger {
            inner: Arc::new(RwLock::new(AuditLogger::new(config))),
        }
    }

    /// Logs an entry. Recovers a poisoned lock (a panic in another thread while
    /// holding it must not permanently disable auditing — that would be an easy
    /// way to blind the audit trail).
    pub fn log(&self, entry: AuditEntry) {
        let mut logger = self.inner.write().unwrap_or_else(|p| p.into_inner());
        logger.log(entry);
    }

    /// Logs an operation
    pub fn log_operation(&self, operation: &str, target: &str, message: &str) {
        let mut logger = self.inner.write().unwrap_or_else(|p| p.into_inner());
        logger.log_operation(operation, target, message);
    }

    /// Verify the underlying chain integrity.
    pub fn verify_chain(&self) -> std::result::Result<(), String> {
        let logger = self.inner.read().unwrap_or_else(|p| p.into_inner());
        logger.verify_chain()
    }

    /// Gets stats
    pub fn stats(&self) -> Option<AuditStats> {
        let logger = self.inner.read().unwrap_or_else(|p| p.into_inner());
        Some(logger.stats())
    }

    /// Exports to JSON
    pub fn export_json(&self) -> Option<String> {
        let logger = self.inner.read().unwrap_or_else(|p| p.into_inner());
        logger.export_json().ok()
    }
}

impl Default for SharedAuditLogger {
    fn default() -> Self {
        Self::new(AuditConfig::default())
    }
}

/// Global audit logger instance
static GLOBAL_LOGGER: std::sync::OnceLock<SharedAuditLogger> = std::sync::OnceLock::new();

/// Initializes the global audit logger
pub fn init_global_logger(config: AuditConfig) {
    let _ = GLOBAL_LOGGER.set(SharedAuditLogger::new(config));
}

/// Gets the global audit logger
pub fn global_logger() -> Option<&'static SharedAuditLogger> {
    GLOBAL_LOGGER.get()
}

/// Logs to the global logger
pub fn log_global(entry: AuditEntry) {
    if let Some(logger) = global_logger() {
        logger.log(entry);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_entry_creation() {
        let entry = AuditEntry::new(
            LogLevel::Info,
            EventCategory::DataModification,
            "select",
            "df",
            "Selected columns",
        );

        assert_eq!(entry.operation, "select");
        assert_eq!(entry.target, "df");
        assert!(entry.success);
    }

    #[test]
    fn test_audit_entry_with_error() {
        let entry = AuditEntry::new(
            LogLevel::Error,
            EventCategory::System,
            "load",
            "file.csv",
            "Failed to load",
        )
        .with_error("File not found");

        assert!(!entry.success);
        assert_eq!(entry.error, Some("File not found".to_string()));
    }

    #[test]
    fn test_audit_logger() {
        let mut logger = AuditLogger::new(AuditConfig::default());

        logger.log_operation("select", "df", "Selected columns a, b");
        logger.log_operation("filter", "df", "Filtered rows");

        assert_eq!(logger.entries().len(), 2);
    }

    #[test]
    fn test_log_with_context() {
        let mut logger = AuditLogger::new(AuditConfig::default());

        logger.log_with_context("filter", "df", "Filtered rows", |ctx| {
            ctx.set("condition", "value > 10");
            ctx.set("rows_before", "1000");
        });

        let entry = logger.entries().back().expect("operation should succeed");
        assert_eq!(
            entry.context.get("condition"),
            Some(&"value > 10".to_string())
        );
    }

    #[test]
    fn test_logger_stats() {
        let mut logger = AuditLogger::new(AuditConfig::default());

        logger.log_operation("select", "df", "Selected");
        logger.log_data_access("df", "Accessed");
        logger.log_error("load", "file", "Failed");

        let stats = logger.stats();
        assert_eq!(stats.total_entries, 3);
        assert_eq!(stats.failed_count, 1);
    }

    #[test]
    fn test_entries_by_category() {
        let mut logger = AuditLogger::new(AuditConfig::default());

        logger.log_operation("select", "df", "Selected");
        logger.log_data_access("df", "Accessed");
        logger.log_operation("filter", "df", "Filtered");

        let data_mods = logger.entries_by_category(&EventCategory::DataModification);
        assert_eq!(data_mods.len(), 2);
    }

    #[test]
    fn test_config_builder() {
        let config = AuditConfig::builder()
            .level(LogLevel::Debug)
            .max_entries(5000)
            .default_user("test_user")
            .build();

        assert_eq!(config.level, LogLevel::Debug);
        assert_eq!(config.max_entries, 5000);
        assert_eq!(config.default_user, Some("test_user".to_string()));
    }

    #[test]
    fn test_shared_logger() {
        let logger = SharedAuditLogger::new(AuditConfig::default());

        logger.log_operation("select", "df", "Selected");

        let stats = logger.stats().expect("operation should succeed");
        assert_eq!(stats.total_entries, 1);
    }

    #[test]
    fn test_export_json() {
        let mut logger = AuditLogger::new(AuditConfig::default());
        logger.log_operation("select", "df", "Selected");

        let json = logger.export_json();
        assert!(json.is_ok());
        assert!(json.expect("operation should succeed").contains("select"));
    }
}
