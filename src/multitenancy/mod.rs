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

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

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
        // Check tenant exists
        let config = self
            .tenants
            .get(tenant_id)
            .ok_or_else(|| Error::InvalidInput(format!("Tenant '{}' not found", tenant_id)))?;

        // Check permissions
        let operation = if self
            .stores
            .get(tenant_id)
            .map(|s| s.datasets.contains_key(dataset_id))
            .unwrap_or(false)
        {
            if !config.permissions.contains(&Permission::Write) {
                return Err(Error::InvalidOperation(format!(
                    "Tenant '{}' does not have write permission",
                    tenant_id
                )));
            }
            TenantOperation::UpdateDataset
        } else {
            if !config.permissions.contains(&Permission::Create) {
                return Err(Error::InvalidOperation(format!(
                    "Tenant '{}' does not have create permission",
                    tenant_id
                )));
            }
            TenantOperation::CreateDataset
        };

        // Check quotas
        if self.enforce_quotas {
            self.check_quotas(tenant_id, &df)?;
        }

        let store = self
            .stores
            .get_mut(tenant_id)
            .ok_or_else(|| Error::InvalidInput(format!("Tenant store not found")))?;

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
            columns: column_names,
            tags: HashMap::new(),
            shared_with: HashSet::new(),
        };

        // Update usage stats
        if let Some(old_meta) = store.metadata.get(dataset_id) {
            store.usage.total_rows = store.usage.total_rows.saturating_sub(old_meta.row_count);
        } else {
            store.usage.dataset_count += 1;
        }
        store.usage.total_rows += row_count;
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

    /// Get a DataFrame for a tenant (cloned)
    pub fn get_dataframe(&mut self, tenant_id: &str, dataset_id: &str) -> Result<DataFrame> {
        // Check tenant exists and has permission
        let config = self
            .tenants
            .get(tenant_id)
            .ok_or_else(|| Error::InvalidInput(format!("Tenant '{}' not found", tenant_id)))?;

        if !config.permissions.contains(&Permission::Read) {
            return Err(Error::InvalidOperation(format!(
                "Tenant '{}' does not have read permission",
                tenant_id
            )));
        }

        let store = self
            .stores
            .get_mut(tenant_id)
            .ok_or_else(|| Error::InvalidInput(format!("Tenant store not found")))?;

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

        // Update usage
        store.usage.read_operations += 1;
        store.usage.last_access = Some(Instant::now());

        self.log_operation(
            tenant_id.to_string(),
            TenantOperation::ReadDataset,
            Some(dataset_id.to_string()),
            true,
            None,
        );

        Ok(df)
    }

    /// Delete a dataset for a tenant
    pub fn delete_dataframe(&mut self, tenant_id: &str, dataset_id: &str) -> Result<()> {
        // Check tenant exists and has permission
        let config = self
            .tenants
            .get(tenant_id)
            .ok_or_else(|| Error::InvalidInput(format!("Tenant '{}' not found", tenant_id)))?;

        if !config.permissions.contains(&Permission::Delete) {
            return Err(Error::InvalidOperation(format!(
                "Tenant '{}' does not have delete permission",
                tenant_id
            )));
        }

        let store = self
            .stores
            .get_mut(tenant_id)
            .ok_or_else(|| Error::InvalidInput(format!("Tenant store not found")))?;

        if let Some(metadata) = store.metadata.remove(dataset_id) {
            store.datasets.remove(dataset_id);
            store.usage.dataset_count = store.usage.dataset_count.saturating_sub(1);
            store.usage.total_rows = store.usage.total_rows.saturating_sub(metadata.row_count);
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

    /// List datasets for a tenant
    pub fn list_datasets(&self, tenant_id: &str) -> Result<Vec<&DatasetMetadata>> {
        let store = self
            .stores
            .get(tenant_id)
            .ok_or_else(|| Error::InvalidInput(format!("Tenant '{}' not found", tenant_id)))?;

        Ok(store.metadata.values().collect())
    }

    /// Get dataset metadata
    pub fn get_dataset_metadata(
        &self,
        tenant_id: &str,
        dataset_id: &str,
    ) -> Result<&DatasetMetadata> {
        let store = self
            .stores
            .get(tenant_id)
            .ok_or_else(|| Error::InvalidInput(format!("Tenant '{}' not found", tenant_id)))?;

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
        // Check owner has share permission
        let owner_config = self
            .tenants
            .get(owner_tenant)
            .ok_or_else(|| Error::InvalidInput(format!("Tenant '{}' not found", owner_tenant)))?;

        if !owner_config.permissions.contains(&Permission::Share) {
            return Err(Error::InvalidOperation(format!(
                "Tenant '{}' does not have share permission",
                owner_tenant
            )));
        }

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
            .ok_or_else(|| Error::InvalidInput(format!("Tenant store not found")))?;

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

    /// Check resource quotas
    fn check_quotas(&self, tenant_id: &str, df: &DataFrame) -> Result<()> {
        let config = self
            .tenants
            .get(tenant_id)
            .ok_or_else(|| Error::InvalidInput(format!("Tenant '{}' not found", tenant_id)))?;

        let store = self
            .stores
            .get(tenant_id)
            .ok_or_else(|| Error::InvalidInput(format!("Tenant store not found")))?;

        let new_rows = df.row_count();
        let new_cols = df.column_names().len();

        // Check max datasets
        if let Some(max) = config.quota.max_datasets {
            if store.usage.dataset_count >= max {
                return Err(Error::InvalidOperation(format!(
                    "Dataset quota exceeded: max {} datasets allowed",
                    max
                )));
            }
        }

        // Check max rows
        if let Some(max) = config.quota.max_total_rows {
            if store.usage.total_rows + new_rows > max {
                return Err(Error::InvalidOperation(format!(
                    "Row quota exceeded: max {} total rows allowed",
                    max
                )));
            }
        }

        // Check max columns
        if let Some(max) = config.quota.max_columns_per_dataset {
            if new_cols > max {
                return Err(Error::InvalidOperation(format!(
                    "Column quota exceeded: max {} columns per dataset",
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
    /// Create a new isolation context
    pub fn new(tenant_id: impl Into<String>) -> Self {
        IsolationContext {
            tenant_id: tenant_id.into(),
            session_id: format!(
                "session_{}",
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis()
            ),
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
        let x = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("x".to_string())).unwrap();
        let y = Series::new(vec![10.0, 20.0, 30.0, 40.0, 50.0], Some("y".to_string())).unwrap();
        df.add_column("x".to_string(), x).unwrap();
        df.add_column("y".to_string(), y).unwrap();
        df
    }

    #[test]
    fn test_tenant_registration() {
        let mut manager = TenantManager::new();

        let config = TenantConfig::default_rw("tenant_a");
        manager.register_tenant(config).unwrap();

        assert!(manager.get_tenant("tenant_a").is_some());
        assert!(manager.get_tenant("tenant_b").is_none());
    }

    #[test]
    fn test_data_isolation() {
        let mut manager = TenantManager::new();

        // Register two tenants
        manager
            .register_tenant(TenantConfig::default_rw("tenant_a"))
            .unwrap();
        manager
            .register_tenant(TenantConfig::default_rw("tenant_b"))
            .unwrap();

        // Store data for tenant_a
        let df = create_test_df();
        manager.store_dataframe("tenant_a", "data", df).unwrap();

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
        manager.register_tenant(config).unwrap();

        // Cannot create data without Create permission
        let df = create_test_df();
        assert!(manager.store_dataframe("readonly", "data", df).is_err());
    }

    #[test]
    fn test_quota_enforcement() {
        let mut manager = TenantManager::new();

        let config = TenantConfig::default_rw("limited").with_max_rows(8); // Max 8 rows (5 + 5 = 10 would exceed)
        manager.register_tenant(config).unwrap();

        // First dataset with 5 rows should succeed
        let df = create_test_df();
        manager.store_dataframe("limited", "data1", df).unwrap();

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
            .unwrap();

        let df = create_test_df();
        manager.store_dataframe("tenant_a", "data", df).unwrap();

        let usage = manager.get_usage("tenant_a").unwrap();
        assert_eq!(usage.dataset_count, 1);
        assert_eq!(usage.total_rows, 5);
        assert_eq!(usage.write_operations, 1);
    }

    #[test]
    fn test_audit_log() {
        let mut manager = TenantManager::new();

        manager
            .register_tenant(TenantConfig::default_rw("tenant_a"))
            .unwrap();

        let df = create_test_df();
        manager.store_dataframe("tenant_a", "data", df).unwrap();
        let _ = manager.get_dataframe("tenant_a", "data");

        let audit = manager.get_audit_log(Some("tenant_a"));
        assert!(audit.len() >= 2); // At least create and read operations
    }

    #[test]
    fn test_dataset_sharing() {
        let mut manager = TenantManager::new();

        // Create tenant with share permission
        let config_a = TenantConfig::default_rw("tenant_a").with_permission(Permission::Share);
        manager.register_tenant(config_a).unwrap();
        manager
            .register_tenant(TenantConfig::default_rw("tenant_b"))
            .unwrap();

        // Store data
        let df = create_test_df();
        manager.store_dataframe("tenant_a", "data", df).unwrap();

        // Share with tenant_b
        manager
            .share_dataset("tenant_a", "data", "tenant_b")
            .unwrap();

        // Check metadata
        let metadata = manager.get_dataset_metadata("tenant_a", "data").unwrap();
        assert!(metadata.shared_with.contains("tenant_b"));
    }

    #[test]
    fn test_list_datasets() {
        let mut manager = TenantManager::new();

        manager
            .register_tenant(TenantConfig::default_rw("tenant_a"))
            .unwrap();

        let df1 = create_test_df();
        let df2 = create_test_df();
        manager.store_dataframe("tenant_a", "data1", df1).unwrap();
        manager.store_dataframe("tenant_a", "data2", df2).unwrap();

        let datasets = manager.list_datasets("tenant_a").unwrap();
        assert_eq!(datasets.len(), 2);
    }

    #[test]
    fn test_delete_dataset() {
        let mut manager = TenantManager::new();

        let config = TenantConfig::default_rw("tenant_a").with_permission(Permission::Delete);
        manager.register_tenant(config).unwrap();

        let df = create_test_df();
        manager.store_dataframe("tenant_a", "data", df).unwrap();

        assert!(manager.get_dataframe("tenant_a", "data").is_ok());
        manager.delete_dataframe("tenant_a", "data").unwrap();
        assert!(manager.get_dataframe("tenant_a", "data").is_err());
    }

    #[test]
    fn test_isolation_context() {
        let ctx = IsolationContext::new("tenant_a");
        assert_eq!(ctx.tenant_id, "tenant_a");
        assert!(ctx.check_time_limit().is_ok());
    }
}
