//! Multi-Tenancy Support Module
//!
//! This module provides data isolation and multi-tenancy support for PandRS,
//! enabling secure separation of data between different tenants or users.
//!
//! # Features
//!
//! - Tenant-scoped DataFrames with automatic isolation
//! - Role-based access control (RBAC)
//! - Resource quotas per tenant
//! - Audit trails for tenant operations
//! - Cross-tenant query prevention
//!
//! # Example
//!
//! ```ignore
//! use pandrs::multitenancy::{TenantManager, TenantConfig, Permission};
//!
//! // Create tenant manager
//! let mut manager = TenantManager::new();
//!
//! // Register tenants
//! let config = TenantConfig::new("tenant_a")
//!     .with_max_rows(1_000_000)
//!     .with_permission(Permission::Read)
//!     .with_permission(Permission::Write);
//! manager.register_tenant(config)?;
//!
//! // Store data for a tenant
//! let df = DataFrame::new();
//! manager.store_dataframe("tenant_a", "sales_data", df)?;
//!
//! // Retrieve data (isolated per tenant)
//! let sales = manager.get_dataframe("tenant_a", "sales_data")?;
//! ```

use crate::audit::{AuditEntry, EventCategory, LogLevel, SharedAuditLogger};
use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Rough per-cell memory estimate (bytes) used only for quota accounting.
///
/// The tenant layer does not have cheap access to a `DataFrame`'s true heap
/// footprint, so `TenantUsage::estimated_memory` and the `max_memory_bytes`
/// quota are computed from `rows * cols * ESTIMATED_BYTES_PER_CELL`. This is an
/// **estimate**, deliberately conservative (an 8-byte `f64`/pointer plus
/// overhead), never presented as a measured value.
const ESTIMATED_BYTES_PER_CELL: usize = 16;

/// Tenant identifier type
pub type TenantId = String;

/// Dataset identifier type
pub type DatasetId = String;

/// Permission types for tenant access control
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Permission {
    /// Read data
    Read,
    /// Write/modify data
    Write,
    /// Delete data
    Delete,
    /// Create new datasets
    Create,
    /// Share data with other tenants
    Share,
    /// Administrative operations
    Admin,
}

/// Resource quota configuration
#[derive(Debug, Clone)]
pub struct ResourceQuota {
    /// Maximum number of rows across all datasets
    pub max_total_rows: Option<usize>,
    /// Maximum number of datasets
    pub max_datasets: Option<usize>,
    /// Maximum memory usage in bytes
    pub max_memory_bytes: Option<usize>,
    /// Maximum number of columns per dataset
    pub max_columns_per_dataset: Option<usize>,
    /// Maximum query execution time
    pub max_query_time: Option<Duration>,
}

impl Default for ResourceQuota {
    fn default() -> Self {
        ResourceQuota {
            max_total_rows: Some(10_000_000),
            max_datasets: Some(100),
            max_memory_bytes: Some(1024 * 1024 * 1024), // 1GB
            max_columns_per_dataset: Some(1000),
            max_query_time: Some(Duration::from_secs(300)),
        }
    }
}

impl ResourceQuota {
    /// Create unlimited quota
    pub fn unlimited() -> Self {
        ResourceQuota {
            max_total_rows: None,
            max_datasets: None,
            max_memory_bytes: None,
            max_columns_per_dataset: None,
            max_query_time: None,
        }
    }
}

/// Tenant configuration
#[derive(Debug, Clone)]
pub struct TenantConfig {
    /// Unique tenant identifier
    pub id: TenantId,
    /// Display name
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Permissions granted to this tenant
    pub permissions: HashSet<Permission>,
    /// Resource quotas
    pub quota: ResourceQuota,
    /// Whether the tenant is active
    pub active: bool,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Tags for categorization
    pub tags: HashMap<String, String>,
}

impl TenantConfig {
    /// Create a new tenant configuration
    pub fn new(id: impl Into<String>) -> Self {
        let id = id.into();
        TenantConfig {
            id: id.clone(),
            name: id,
            description: None,
            permissions: HashSet::new(),
            quota: ResourceQuota::default(),
            active: true,
            created_at: SystemTime::now(),
            tags: HashMap::new(),
        }
    }

    /// Set display name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a permission
    pub fn with_permission(mut self, perm: Permission) -> Self {
        self.permissions.insert(perm);
        self
    }

    /// Set all permissions
    pub fn with_permissions(mut self, perms: HashSet<Permission>) -> Self {
        self.permissions = perms;
        self
    }

    /// Set resource quota
    pub fn with_quota(mut self, quota: ResourceQuota) -> Self {
        self.quota = quota;
        self
    }

    /// Set max rows quota
    pub fn with_max_rows(mut self, max: usize) -> Self {
        self.quota.max_total_rows = Some(max);
        self
    }

    /// Set max datasets quota
    pub fn with_max_datasets(mut self, max: usize) -> Self {
        self.quota.max_datasets = Some(max);
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Create a default configuration with read/write permissions
    pub fn default_rw(id: impl Into<String>) -> Self {
        Self::new(id)
            .with_permission(Permission::Read)
            .with_permission(Permission::Write)
            .with_permission(Permission::Create)
    }
}

/// Usage statistics for a tenant
#[derive(Debug, Clone, Default)]
pub struct TenantUsage {
    /// Number of datasets
    pub dataset_count: usize,
    /// Total row count across all datasets
    pub total_rows: usize,
    /// Estimated memory usage in bytes
    pub estimated_memory: usize,
    /// Number of read operations
    pub read_operations: u64,
    /// Number of write operations
    pub write_operations: u64,
    /// Last access time
    pub last_access: Option<Instant>,
}

/// Audit log entry for tenant operations
#[derive(Debug, Clone)]
pub struct TenantAuditEntry {
    /// Timestamp of the operation
    pub timestamp: SystemTime,
    /// Tenant that performed the operation
    pub tenant_id: TenantId,
    /// Type of operation
    pub operation: TenantOperation,
    /// Target dataset (if applicable)
    pub dataset_id: Option<DatasetId>,
    /// Whether the operation succeeded
    pub success: bool,
    /// Error message (if failed)
    pub error_message: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Types of tenant operations for auditing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TenantOperation {
    /// Created a new dataset
    CreateDataset,
    /// Read a dataset
    ReadDataset,
    /// Updated/modified a dataset
    UpdateDataset,
    /// Deleted a dataset
    DeleteDataset,
    /// Shared a dataset with another tenant
    ShareDataset,
    /// Query executed
    Query,
    /// Schema modification
    SchemaChange,
    /// Tenant configuration change
    ConfigChange,
}

/// Dataset metadata for tenant storage
#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    /// Dataset identifier
    pub id: DatasetId,
    /// Owner tenant
    pub owner: TenantId,
    /// Creation time
    pub created_at: SystemTime,
    /// Last modified time
    pub modified_at: SystemTime,
    /// Row count
    pub row_count: usize,
    /// Column count
    pub column_count: usize,
    /// Column names
    pub columns: Vec<String>,
    /// Tags/labels
    pub tags: HashMap<String, String>,
    /// Tenants with shared access
    pub shared_with: HashSet<TenantId>,
}

/// Tenant data store
#[derive(Debug)]
struct TenantStore {
    /// DataFrames stored by this tenant
    datasets: HashMap<DatasetId, Arc<RwLock<DataFrame>>>,
    /// Metadata for each dataset
    metadata: HashMap<DatasetId, DatasetMetadata>,
    /// Usage statistics
    usage: TenantUsage,
}

impl TenantStore {
    fn new() -> Self {
        TenantStore {
            datasets: HashMap::new(),
            metadata: HashMap::new(),
            usage: TenantUsage::default(),
        }
    }
}

/// Multi-tenant data manager
#[derive(Debug)]
pub struct TenantManager {
    /// Registered tenants
    tenants: HashMap<TenantId, TenantConfig>,
    /// Data stores per tenant
    stores: HashMap<TenantId, TenantStore>,
    /// Audit log
    audit_log: Vec<TenantAuditEntry>,
    /// Maximum audit log entries to keep
    max_audit_entries: usize,
    /// Whether to enforce quotas
    enforce_quotas: bool,
    /// Optional shared security audit sink. When set, permission/active denials
    /// on data paths are emitted as `Security` audit entries (the internal
    /// `audit_log` only records tenant operations, historically successes only).
    audit: Option<SharedAuditLogger>,
}

impl TenantManager {
    /// Create a new tenant manager
    pub fn new() -> Self {
        TenantManager {
            tenants: HashMap::new(),
            stores: HashMap::new(),
            audit_log: Vec::new(),
            max_audit_entries: 10000,
            enforce_quotas: true,
            audit: None,
        }
    }

    /// Set maximum audit log entries
    pub fn with_max_audit_entries(mut self, max: usize) -> Self {
        self.max_audit_entries = max;
        self
    }

    /// Enable or disable quota enforcement
    pub fn with_quota_enforcement(mut self, enforce: bool) -> Self {
        self.enforce_quotas = enforce;
        self
    }

    /// Attach a shared security audit logger. Data-path denials (inactive
    /// tenant or missing permission) are emitted as `Security` audit entries.
    pub fn with_audit_logger(mut self, logger: SharedAuditLogger) -> Self {
        self.audit = Some(logger);
        self
    }

    /// Emit a `Security` audit entry for a denied data-path operation. Uses
    /// interior mutability of [`SharedAuditLogger`] so it works from `&self`
    /// (read-path) methods too. Best-effort: never blocks the caller.
    fn audit_denied(&self, tenant_id: &str, operation: &str, reason: &str) {
        if let Some(ref logger) = self.audit {
            let message = format!("tenant '{}' denied {}: {}", tenant_id, operation, reason);
            let entry = AuditEntry::new(
                LogLevel::Warn,
                EventCategory::Security,
                operation,
                tenant_id,
                &message,
            )
            .with_user(tenant_id)
            .with_error(reason);
            logger.log(entry);
        }
    }

    /// Single authorization chokepoint for every tenant data path.
    ///
    /// A request is authorized only when the tenant exists, is **active**, and
    /// holds `permission`. Any failure emits a `Security` denial (when a logger
    /// is attached) and returns an error. Routing all six operations through
    /// this helper is what keeps `active`-gating and permission checks from
    /// drifting apart (the previous inline checks omitted the `active` gate
    /// entirely, so a deactivated tenant retained full access).
    fn authorize(&self, tenant_id: &str, permission: Permission, operation: &str) -> Result<()> {
        let config = self.tenants.get(tenant_id).ok_or_else(|| {
            self.audit_denied(tenant_id, operation, "unknown tenant");
            Error::InvalidInput(format!("Tenant '{}' not found", tenant_id))
        })?;

        if !config.active {
            self.audit_denied(tenant_id, operation, "tenant is not active");
            return Err(Error::InvalidOperation(format!(
                "Tenant '{}' is not active",
                tenant_id
            )));
        }

        if !config.permissions.contains(&permission) {
            self.audit_denied(tenant_id, operation, "missing permission");
            return Err(Error::InvalidOperation(format!(
                "Tenant '{}' does not have {:?} permission",
                tenant_id, permission
            )));
        }

        Ok(())
    }

    /// Register a new tenant
    pub fn register_tenant(&mut self, config: TenantConfig) -> Result<()> {
        if self.tenants.contains_key(&config.id) {
            return Err(Error::InvalidInput(format!(
                "Tenant '{}' already exists",
                config.id
            )));
        }

        let tenant_id = config.id.clone();
        self.tenants.insert(tenant_id.clone(), config);
        self.stores.insert(tenant_id, TenantStore::new());

        Ok(())
    }

    /// Remove a tenant and all their data
    pub fn remove_tenant(&mut self, tenant_id: &str) -> Result<()> {
        if !self.tenants.contains_key(tenant_id) {
            return Err(Error::InvalidInput(format!(
                "Tenant '{}' not found",
                tenant_id
            )));
        }

        self.tenants.remove(tenant_id);
        self.stores.remove(tenant_id);

        self.log_operation(
            tenant_id.to_string(),
            TenantOperation::ConfigChange,
            None,
            true,
            None,
        );

        Ok(())
    }

    /// Get tenant configuration
    pub fn get_tenant(&self, tenant_id: &str) -> Option<&TenantConfig> {
        self.tenants.get(tenant_id)
    }

    /// Update tenant configuration
    pub fn update_tenant(&mut self, config: TenantConfig) -> Result<()> {
        if !self.tenants.contains_key(&config.id) {
            return Err(Error::InvalidInput(format!(
                "Tenant '{}' not found",
                config.id
            )));
        }

        let tenant_id = config.id.clone();
        self.tenants.insert(tenant_id.clone(), config);

        self.log_operation(tenant_id, TenantOperation::ConfigChange, None, true, None);

        Ok(())
    }

    /// List all tenant IDs
    pub fn list_tenants(&self) -> Vec<&TenantId> {
        self.tenants.keys().collect()
    }

    /// Check if tenant has a specific permission
    pub fn has_permission(&self, tenant_id: &str, permission: Permission) -> bool {
        self.tenants
            .get(tenant_id)
            .map(|t| t.active && t.permissions.contains(&permission))
            .unwrap_or(false)
    }

    /// Store a DataFrame for a tenant
    pub fn store_dataframe(
        &mut self,
        tenant_id: &str,
        dataset_id: &str,
        df: DataFrame,
    ) -> Result<()> {
        // An overwrite of an existing dataset is an update (needs Write); a new
        // dataset is a create (needs Create). Determine this before authorizing
        // so the correct permission — and the active gate — are enforced.
        let is_update = self
            .stores
            .get(tenant_id)
            .map(|s| s.datasets.contains_key(dataset_id))
            .unwrap_or(false);

        let (operation, permission) = if is_update {
            (TenantOperation::UpdateDataset, Permission::Write)
        } else {
            (TenantOperation::CreateDataset, Permission::Create)
        };

        self.authorize(tenant_id, permission, "store_dataframe")?;

        // Check quotas (accounts for an in-place overwrite so replaced rows and
        // memory are not charged twice against the quota).
        if self.enforce_quotas {
            self.check_quotas(tenant_id, dataset_id, &df)?;
        }

        let store = self
            .stores
            .get_mut(tenant_id)
            .ok_or_else(|| Error::InvalidOperation("Tenant store not found".to_string()))?;

        // Create metadata
        let column_names = df.column_names();
        let row_count = df.row_count();
        let col_count = column_names.len();

        let metadata = DatasetMetadata {
            id: dataset_id.to_string(),
            owner: tenant_id.to_string(),
            created_at: SystemTime::now(),
            modified_at: SystemTime::now(),
            row_count,
            column_count: col_count,
            columns: column_names.to_vec(),
            tags: HashMap::new(),
            shared_with: HashSet::new(),
        };

        // Update usage stats. On an overwrite, back out the previous dataset's
        // rows and estimated memory before adding the new figures so neither is
        // double-counted; on a create, bump the dataset count.
        let new_memory = row_count
            .saturating_mul(col_count)
            .saturating_mul(ESTIMATED_BYTES_PER_CELL);
        if let Some(old_meta) = store.metadata.get(dataset_id) {
            let old_memory = old_meta
                .row_count
                .saturating_mul(old_meta.column_count)
                .saturating_mul(ESTIMATED_BYTES_PER_CELL);
            store.usage.total_rows = store.usage.total_rows.saturating_sub(old_meta.row_count);
            store.usage.estimated_memory = store.usage.estimated_memory.saturating_sub(old_memory);
        } else {
            store.usage.dataset_count += 1;
        }
        store.usage.total_rows += row_count;
        store.usage.estimated_memory = store.usage.estimated_memory.saturating_add(new_memory);
        store.usage.write_operations += 1;
        store.usage.last_access = Some(Instant::now());

        // Store the data
        store
            .datasets
            .insert(dataset_id.to_string(), Arc::new(RwLock::new(df)));
        store.metadata.insert(dataset_id.to_string(), metadata);

        self.log_operation(
            tenant_id.to_string(),
            operation,
            Some(dataset_id.to_string()),
            true,
            None,
        );

        Ok(())
    }

    /// Get a DataFrame for a tenant (cloned).
    ///
    /// Resolves the tenant's own datasets first, then datasets that another
    /// *active* tenant has shared with this one (read-only). A dataset that is
    /// neither owned nor shared returns the **same** "not found" error as a
    /// non-existent one, so the shared path cannot be turned into a cross-tenant
    /// dataset-enumeration oracle.
    pub fn get_dataframe(&mut self, tenant_id: &str, dataset_id: &str) -> Result<DataFrame> {
        self.authorize(tenant_id, Permission::Read, "get_dataframe")?;

        // Fast path: the tenant owns the dataset.
        let owns = self
            .stores
            .get(tenant_id)
            .map(|s| s.datasets.contains_key(dataset_id))
            .unwrap_or(false);

        if owns {
            let store = self
                .stores
                .get_mut(tenant_id)
                .ok_or_else(|| Error::InvalidOperation("Tenant store not found".to_string()))?;

            let df_lock = store.datasets.get(dataset_id).ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Dataset '{}' not found for tenant '{}'",
                    dataset_id, tenant_id
                ))
            })?;

            let df = df_lock
                .read()
                .map_err(|_| Error::InvalidOperation("Failed to acquire read lock".to_string()))?
                .clone();

            store.usage.read_operations += 1;
            store.usage.last_access = Some(Instant::now());

            self.log_operation(
                tenant_id.to_string(),
                TenantOperation::ReadDataset,
                Some(dataset_id.to_string()),
                true,
                None,
            );

            return Ok(df);
        }

        // Shared path: read-only access to a dataset shared by an active owner.
        if let Some(df) = self.resolve_shared_dataframe(tenant_id, dataset_id)? {
            self.log_operation(
                tenant_id.to_string(),
                TenantOperation::ReadDataset,
                Some(dataset_id.to_string()),
                true,
                None,
            );
            return Ok(df);
        }

        Err(Error::InvalidInput(format!(
            "Dataset '{}' not found for tenant '{}'",
            dataset_id, tenant_id
        )))
    }

    /// Resolve a dataset shared with `requester` by another owner tenant.
    ///
    /// Returns the cloned DataFrame (read-only) when some *active* owner tenant
    /// holds `dataset_id` and has shared it with `requester`; `Ok(None)` when no
    /// such share exists. The caller maps `None` to the same "not found" error a
    /// missing dataset yields, so "exists but not shared with you" is
    /// indistinguishable from "does not exist". Sharing never grants write or
    /// delete — only this read path consults `shared_with`.
    fn resolve_shared_dataframe(
        &self,
        requester: &str,
        dataset_id: &str,
    ) -> Result<Option<DataFrame>> {
        for (owner_id, store) in &self.stores {
            if owner_id == requester {
                continue;
            }
            let Some(metadata) = store.metadata.get(dataset_id) else {
                continue;
            };
            if !metadata.shared_with.contains(requester) {
                continue;
            }
            // A deactivated owner's shared data must not remain readable.
            let owner_active = self
                .tenants
                .get(owner_id)
                .map(|t| t.active)
                .unwrap_or(false);
            if !owner_active {
                continue;
            }
            if let Some(df_lock) = store.datasets.get(dataset_id) {
                let df = df_lock
                    .read()
                    .map_err(|_| {
                        Error::InvalidOperation("Failed to acquire read lock".to_string())
                    })?
                    .clone();
                return Ok(Some(df));
            }
        }
        Ok(None)
    }

    /// Delete a dataset for a tenant
    pub fn delete_dataframe(&mut self, tenant_id: &str, dataset_id: &str) -> Result<()> {
        self.authorize(tenant_id, Permission::Delete, "delete_dataframe")?;

        let store = self
            .stores
            .get_mut(tenant_id)
            .ok_or_else(|| Error::InvalidOperation("Tenant store not found".to_string()))?;

        if let Some(metadata) = store.metadata.remove(dataset_id) {
            store.datasets.remove(dataset_id);
            store.usage.dataset_count = store.usage.dataset_count.saturating_sub(1);
            store.usage.total_rows = store.usage.total_rows.saturating_sub(metadata.row_count);
            let freed = metadata
                .row_count
                .saturating_mul(metadata.column_count)
                .saturating_mul(ESTIMATED_BYTES_PER_CELL);
            store.usage.estimated_memory = store.usage.estimated_memory.saturating_sub(freed);
        }

        self.log_operation(
            tenant_id.to_string(),
            TenantOperation::DeleteDataset,
            Some(dataset_id.to_string()),
            true,
            None,
        );

        Ok(())
    }

    /// List datasets for a tenant.
    ///
    /// Requires the tenant to be active and hold `Read` permission — dataset
    /// metadata (ids, columns, row counts, share lists) is itself sensitive and
    /// was previously returned to any caller with no check at all.
    pub fn list_datasets(&self, tenant_id: &str) -> Result<Vec<&DatasetMetadata>> {
        self.authorize(tenant_id, Permission::Read, "list_datasets")?;

        let store = self
            .stores
            .get(tenant_id)
            .ok_or_else(|| Error::InvalidOperation("Tenant store not found".to_string()))?;

        Ok(store.metadata.values().collect())
    }

    /// Get dataset metadata (requires active tenant + `Read` permission).
    pub fn get_dataset_metadata(
        &self,
        tenant_id: &str,
        dataset_id: &str,
    ) -> Result<&DatasetMetadata> {
        self.authorize(tenant_id, Permission::Read, "get_dataset_metadata")?;

        let store = self
            .stores
            .get(tenant_id)
            .ok_or_else(|| Error::InvalidOperation("Tenant store not found".to_string()))?;

        store.metadata.get(dataset_id).ok_or_else(|| {
            Error::InvalidInput(format!(
                "Dataset '{}' not found for tenant '{}'",
                dataset_id, tenant_id
            ))
        })
    }

    /// Share a dataset with another tenant
    pub fn share_dataset(
        &mut self,
        owner_tenant: &str,
        dataset_id: &str,
        target_tenant: &str,
    ) -> Result<()> {
        // Owner must be active and hold Share permission.
        self.authorize(owner_tenant, Permission::Share, "share_dataset")?;

        // Check target tenant exists
        if !self.tenants.contains_key(target_tenant) {
            return Err(Error::InvalidInput(format!(
                "Target tenant '{}' not found",
                target_tenant
            )));
        }

        // Update metadata
        let store = self
            .stores
            .get_mut(owner_tenant)
            .ok_or_else(|| Error::InvalidOperation("Tenant store not found".to_string()))?;

        let metadata = store
            .metadata
            .get_mut(dataset_id)
            .ok_or_else(|| Error::InvalidInput(format!("Dataset '{}' not found", dataset_id)))?;

        metadata.shared_with.insert(target_tenant.to_string());

        self.log_operation(
            owner_tenant.to_string(),
            TenantOperation::ShareDataset,
            Some(dataset_id.to_string()),
            true,
            None,
        );

        Ok(())
    }

    /// Get tenant usage statistics
    pub fn get_usage(&self, tenant_id: &str) -> Result<&TenantUsage> {
        let store = self
            .stores
            .get(tenant_id)
            .ok_or_else(|| Error::InvalidInput(format!("Tenant '{}' not found", tenant_id)))?;

        Ok(&store.usage)
    }

    /// Get audit log for a tenant
    pub fn get_audit_log(&self, tenant_id: Option<&str>) -> Vec<&TenantAuditEntry> {
        self.audit_log
            .iter()
            .filter(|entry| tenant_id.map(|id| entry.tenant_id == id).unwrap_or(true))
            .collect()
    }

    /// Check resource quotas for a prospective store of `df` into `dataset_id`.
    ///
    /// An overwrite of an existing dataset backs out that dataset's current
    /// rows/memory before adding the new figures, so replacing data never
    /// double-counts against the quota (previously an update at the row cap was
    /// falsely rejected) and does not consume a fresh dataset slot.
    ///
    /// `max_memory_bytes` is enforced against an **estimate**
    /// (`rows * cols * ESTIMATED_BYTES_PER_CELL`), never a measured footprint.
    /// `max_query_time` is not enforced here: there is no query execution at the
    /// storage layer — callers enforce it via
    /// [`IsolationContext::check_time_limit`].
    fn check_quotas(&self, tenant_id: &str, dataset_id: &str, df: &DataFrame) -> Result<()> {
        let config = self
            .tenants
            .get(tenant_id)
            .ok_or_else(|| Error::InvalidInput(format!("Tenant '{}' not found", tenant_id)))?;

        let store = self
            .stores
            .get(tenant_id)
            .ok_or_else(|| Error::InvalidOperation("Tenant store not found".to_string()))?;

        let new_rows = df.row_count();
        let new_cols = df.column_names().len();

        // Existing footprint of the dataset being overwritten (0 for a create).
        let (old_rows, old_memory) = store
            .metadata
            .get(dataset_id)
            .map(|m| {
                (
                    m.row_count,
                    m.row_count
                        .saturating_mul(m.column_count)
                        .saturating_mul(ESTIMATED_BYTES_PER_CELL),
                )
            })
            .unwrap_or((0, 0));
        let is_update = store.datasets.contains_key(dataset_id);

        // Max datasets — only a *new* dataset consumes a slot.
        if !is_update {
            if let Some(max) = config.quota.max_datasets {
                if store.usage.dataset_count >= max {
                    return Err(Error::InvalidOperation(format!(
                        "Dataset quota exceeded: max {} datasets allowed",
                        max
                    )));
                }
            }
        }

        // Max rows — project by replacing the old dataset's rows.
        if let Some(max) = config.quota.max_total_rows {
            let projected = store.usage.total_rows.saturating_sub(old_rows) + new_rows;
            if projected > max {
                return Err(Error::InvalidOperation(format!(
                    "Row quota exceeded: max {} total rows allowed",
                    max
                )));
            }
        }

        // Max columns per dataset.
        if let Some(max) = config.quota.max_columns_per_dataset {
            if new_cols > max {
                return Err(Error::InvalidOperation(format!(
                    "Column quota exceeded: max {} columns per dataset",
                    max
                )));
            }
        }

        // Max memory (estimate) — project by replacing the old dataset's memory.
        if let Some(max) = config.quota.max_memory_bytes {
            let new_memory = new_rows
                .saturating_mul(new_cols)
                .saturating_mul(ESTIMATED_BYTES_PER_CELL);
            let projected = store.usage.estimated_memory.saturating_sub(old_memory) + new_memory;
            if projected > max {
                return Err(Error::InvalidOperation(format!(
                    "Memory quota exceeded: max {} bytes allowed (estimated)",
                    max
                )));
            }
        }

        Ok(())
    }

    /// Log an operation to the audit trail
    fn log_operation(
        &mut self,
        tenant_id: TenantId,
        operation: TenantOperation,
        dataset_id: Option<DatasetId>,
        success: bool,
        error_message: Option<String>,
    ) {
        let entry = TenantAuditEntry {
            timestamp: SystemTime::now(),
            tenant_id,
            operation,
            dataset_id,
            success,
            error_message,
            metadata: HashMap::new(),
        };

        self.audit_log.push(entry);

        // Trim audit log if needed
        if self.audit_log.len() > self.max_audit_entries {
            let excess = self.audit_log.len() - self.max_audit_entries;
            self.audit_log.drain(0..excess);
        }
    }
}

impl Default for TenantManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe tenant manager
pub type SharedTenantManager = Arc<RwLock<TenantManager>>;

/// Create a new shared tenant manager
pub fn create_shared_manager() -> SharedTenantManager {
    Arc::new(RwLock::new(TenantManager::new()))
}

/// Generate a CSPRNG-backed session id (128 bits, hex) for an isolation context.
fn generate_session_id() -> String {
    use scirs2_core::random::Rng;
    let mut bytes = [0u8; 16];
    scirs2_core::random::rng().fill_bytes(&mut bytes);
    format!(
        "session_{}",
        bytes
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>()
    )
}

/// Isolation context for tenant-scoped operations
#[derive(Debug, Clone)]
pub struct IsolationContext {
    /// Current tenant ID
    pub tenant_id: TenantId,
    /// Session ID
    pub session_id: String,
    /// Start time
    pub start_time: Instant,
    /// Maximum execution time
    pub max_execution_time: Option<Duration>,
}

impl IsolationContext {
    /// Create a new isolation context.
    ///
    /// The session id is drawn from the CSPRNG (128 bits, hex). The previous
    /// millisecond-timestamp id was predictable and collided for contexts
    /// created in the same millisecond.
    pub fn new(tenant_id: impl Into<String>) -> Self {
        IsolationContext {
            tenant_id: tenant_id.into(),
            session_id: generate_session_id(),
            start_time: Instant::now(),
            max_execution_time: None,
        }
    }

    /// Check if execution time limit is exceeded
    pub fn check_time_limit(&self) -> Result<()> {
        if let Some(max_time) = self.max_execution_time {
            if self.start_time.elapsed() > max_time {
                return Err(Error::InvalidOperation(
                    "Query execution time limit exceeded".to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::Series;

    fn create_test_df() -> DataFrame {
        let mut df = DataFrame::new();
        let x = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("x".to_string()))
            .expect("operation should succeed");
        let y = Series::new(vec![10.0, 20.0, 30.0, 40.0, 50.0], Some("y".to_string()))
            .expect("operation should succeed");
        df.add_column("x".to_string(), x)
            .expect("operation should succeed");
        df.add_column("y".to_string(), y)
            .expect("operation should succeed");
        df
    }

    #[test]
    fn test_tenant_registration() {
        let mut manager = TenantManager::new();

        let config = TenantConfig::default_rw("tenant_a");
        manager
            .register_tenant(config)
            .expect("operation should succeed");

        assert!(manager.get_tenant("tenant_a").is_some());
        assert!(manager.get_tenant("tenant_b").is_none());
    }

    #[test]
    fn test_data_isolation() {
        let mut manager = TenantManager::new();

        // Register two tenants
        manager
            .register_tenant(TenantConfig::default_rw("tenant_a"))
            .expect("operation should succeed");
        manager
            .register_tenant(TenantConfig::default_rw("tenant_b"))
            .expect("operation should succeed");

        // Store data for tenant_a
        let df = create_test_df();
        manager
            .store_dataframe("tenant_a", "data", df)
            .expect("operation should succeed");

        // tenant_a can access their data
        assert!(manager.get_dataframe("tenant_a", "data").is_ok());

        // tenant_b cannot access tenant_a's data
        assert!(manager.get_dataframe("tenant_b", "data").is_err());
    }

    #[test]
    fn test_permission_enforcement() {
        let mut manager = TenantManager::new();

        // Create read-only tenant
        let config = TenantConfig::new("readonly").with_permission(Permission::Read);
        manager
            .register_tenant(config)
            .expect("operation should succeed");

        // Cannot create data without Create permission
        let df = create_test_df();
        assert!(manager.store_dataframe("readonly", "data", df).is_err());
    }

    #[test]
    fn test_quota_enforcement() {
        let mut manager = TenantManager::new();

        let config = TenantConfig::default_rw("limited").with_max_rows(8); // Max 8 rows (5 + 5 = 10 would exceed)
        manager
            .register_tenant(config)
            .expect("operation should succeed");

        // First dataset with 5 rows should succeed
        let df = create_test_df();
        manager
            .store_dataframe("limited", "data1", df)
            .expect("operation should succeed");

        // Second dataset would exceed quota (5 + 5 > 8)
        let df2 = create_test_df();
        let result = manager.store_dataframe("limited", "data2", df2);
        assert!(result.is_err(), "Should fail due to quota exceeded");
    }

    #[test]
    fn test_usage_tracking() {
        let mut manager = TenantManager::new();

        manager
            .register_tenant(TenantConfig::default_rw("tenant_a"))
            .expect("operation should succeed");

        let df = create_test_df();
        manager
            .store_dataframe("tenant_a", "data", df)
            .expect("operation should succeed");

        let usage = manager
            .get_usage("tenant_a")
            .expect("operation should succeed");
        assert_eq!(usage.dataset_count, 1);
        assert_eq!(usage.total_rows, 5);
        assert_eq!(usage.write_operations, 1);
    }

    #[test]
    fn test_audit_log() {
        let mut manager = TenantManager::new();

        manager
            .register_tenant(TenantConfig::default_rw("tenant_a"))
            .expect("operation should succeed");

        let df = create_test_df();
        manager
            .store_dataframe("tenant_a", "data", df)
            .expect("operation should succeed");
        let _ = manager.get_dataframe("tenant_a", "data");

        let audit = manager.get_audit_log(Some("tenant_a"));
        assert!(audit.len() >= 2); // At least create and read operations
    }

    #[test]
    fn test_dataset_sharing() {
        let mut manager = TenantManager::new();

        // Create tenant with share permission
        let config_a = TenantConfig::default_rw("tenant_a").with_permission(Permission::Share);
        manager
            .register_tenant(config_a)
            .expect("operation should succeed");
        manager
            .register_tenant(TenantConfig::default_rw("tenant_b"))
            .expect("operation should succeed");

        // Store data
        let df = create_test_df();
        manager
            .store_dataframe("tenant_a", "data", df)
            .expect("operation should succeed");

        // Share with tenant_b
        manager
            .share_dataset("tenant_a", "data", "tenant_b")
            .expect("operation should succeed");

        // Check metadata
        let metadata = manager
            .get_dataset_metadata("tenant_a", "data")
            .expect("operation should succeed");
        assert!(metadata.shared_with.contains("tenant_b"));
    }

    #[test]
    fn test_list_datasets() {
        let mut manager = TenantManager::new();

        manager
            .register_tenant(TenantConfig::default_rw("tenant_a"))
            .expect("operation should succeed");

        let df1 = create_test_df();
        let df2 = create_test_df();
        manager
            .store_dataframe("tenant_a", "data1", df1)
            .expect("operation should succeed");
        manager
            .store_dataframe("tenant_a", "data2", df2)
            .expect("operation should succeed");

        let datasets = manager
            .list_datasets("tenant_a")
            .expect("operation should succeed");
        assert_eq!(datasets.len(), 2);
    }

    #[test]
    fn test_delete_dataset() {
        let mut manager = TenantManager::new();

        let config = TenantConfig::default_rw("tenant_a").with_permission(Permission::Delete);
        manager
            .register_tenant(config)
            .expect("operation should succeed");

        let df = create_test_df();
        manager
            .store_dataframe("tenant_a", "data", df)
            .expect("operation should succeed");

        assert!(manager.get_dataframe("tenant_a", "data").is_ok());
        manager
            .delete_dataframe("tenant_a", "data")
            .expect("operation should succeed");
        assert!(manager.get_dataframe("tenant_a", "data").is_err());
    }

    #[test]
    fn test_isolation_context() {
        let ctx = IsolationContext::new("tenant_a");
        assert_eq!(ctx.tenant_id, "tenant_a");
        assert!(ctx.check_time_limit().is_ok());
    }
}
