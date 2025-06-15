//! Backward Compatibility Layer and Migration Utilities for PandRS
//!
//! This module provides comprehensive migration utilities for maintaining
//! backward compatibility while transitioning to new trait-based architecture
//! and unified memory management systems.

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, SystemTime};

/// Version information for migration tracking
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Version {
    /// Major version number
    pub major: u32,
    /// Minor version number
    pub minor: u32,
    /// Patch version number
    pub patch: u32,
    /// Pre-release identifier
    pub pre_release: Option<String>,
}

impl Version {
    /// Create a new version
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            pre_release: None,
        }
    }

    /// Create version with pre-release identifier
    pub fn new_pre_release(major: u32, minor: u32, patch: u32, pre_release: String) -> Self {
        Self {
            major,
            minor,
            patch,
            pre_release: Some(pre_release),
        }
    }

    /// Check if this version is compatible with another version
    pub fn is_compatible_with(&self, other: &Version) -> bool {
        // Major version must match for compatibility
        self.major == other.major
    }

    /// Check if this version requires migration from another version
    pub fn requires_migration_from(&self, other: &Version) -> bool {
        self > other && !self.is_compatible_with(other)
    }

    /// Parse version from string (e.g., "0.1.0-alpha.4")
    pub fn parse(version_str: &str) -> Result<Self> {
        let parts: Vec<&str> = version_str.split('-').collect();
        let version_part = parts[0];
        let pre_release = if parts.len() > 1 {
            Some(parts[1..].join("-"))
        } else {
            None
        };

        let version_nums: Vec<&str> = version_part.split('.').collect();
        if version_nums.len() != 3 {
            return Err(Error::InvalidInput(format!(
                "Invalid version format: {}",
                version_str
            )));
        }

        let major = version_nums[0]
            .parse::<u32>()
            .map_err(|_| Error::InvalidInput("Invalid major version".to_string()))?;
        let minor = version_nums[1]
            .parse::<u32>()
            .map_err(|_| Error::InvalidInput("Invalid minor version".to_string()))?;
        let patch = version_nums[2]
            .parse::<u32>()
            .map_err(|_| Error::InvalidInput("Invalid patch version".to_string()))?;

        Ok(Self {
            major,
            minor,
            patch,
            pre_release,
        })
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref pre) = self.pre_release {
            write!(f, "{}.{}.{}-{}", self.major, self.minor, self.patch, pre)
        } else {
            write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
        }
    }
}

/// Migration plan for transitioning between versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPlan {
    /// Source version
    pub from_version: Version,
    /// Target version
    pub to_version: Version,
    /// Migration steps
    pub steps: Vec<MigrationStep>,
    /// Estimated duration
    pub estimated_duration: Duration,
    /// Risk assessment
    pub risk_level: MigrationRiskLevel,
    /// Rollback plan
    pub rollback_plan: Option<RollbackPlan>,
}

/// Individual migration step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStep {
    /// Step identifier
    pub id: String,
    /// Step description
    pub description: String,
    /// Step type
    pub step_type: MigrationStepType,
    /// Dependencies on other steps
    pub dependencies: Vec<String>,
    /// Validation criteria
    pub validation: Vec<ValidationCriteria>,
    /// Rollback action
    pub rollback_action: Option<String>,
    /// Estimated time for this step
    pub estimated_time: Duration,
}

/// Migration step types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationStepType {
    DataStructureUpdate,
    APISignatureChange,
    StorageFormatMigration,
    ConfigurationUpdate,
    DependencyUpdate,
    Custom,
}

/// Migration risk levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MigrationRiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Validation criteria for migration steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCriteria {
    /// Criteria description
    pub description: String,
    /// Validation type
    pub validation_type: ValidationType,
    /// Expected outcome
    pub expected_outcome: String,
}

/// Validation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationType {
    DataIntegrity,
    PerformanceRegression,
    APICompatibility,
    FunctionalEquivalence,
    Custom,
}

/// Rollback plan for failed migrations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackPlan {
    /// Rollback steps
    pub steps: Vec<RollbackStep>,
    /// Data backup strategy
    pub backup_strategy: BackupStrategy,
    /// Recovery time objective (RTO)
    pub recovery_time_objective: Duration,
    /// Recovery point objective (RPO)
    pub recovery_point_objective: Duration,
}

/// Individual rollback step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackStep {
    /// Step description
    pub description: String,
    /// Action to perform
    pub action: String,
    /// Verification method
    pub verification: String,
}

/// Backup strategy for migration safety
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackupStrategy {
    FullBackup,
    IncrementalBackup,
    SnapshotBased,
    CopyOnWrite,
    NoBackup,
}

/// Migration execution context
#[derive(Debug)]
pub struct MigrationContext {
    /// Current migration plan
    pub plan: MigrationPlan,
    /// Execution state
    pub state: MigrationState,
    /// Progress tracking
    pub progress: MigrationProgress,
    /// Error handling
    pub error_handler: MigrationErrorHandler,
    /// Backup manager
    pub backup_manager: Option<BackupManager>,
}

/// Migration execution state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MigrationState {
    NotStarted,
    InProgress,
    Completed,
    Failed,
    RolledBack,
    Paused,
}

/// Migration progress tracking
#[derive(Debug, Clone)]
pub struct MigrationProgress {
    /// Current step index
    pub current_step: usize,
    /// Steps completed
    pub completed_steps: Vec<String>,
    /// Failed steps
    pub failed_steps: Vec<(String, String)>, // (step_id, error_message)
    /// Start time
    pub start_time: Option<SystemTime>,
    /// End time
    pub end_time: Option<SystemTime>,
    /// Progress percentage (0.0 to 1.0)
    pub progress_percentage: f64,
}

impl MigrationProgress {
    pub fn new() -> Self {
        Self {
            current_step: 0,
            completed_steps: Vec::new(),
            failed_steps: Vec::new(),
            start_time: None,
            end_time: None,
            progress_percentage: 0.0,
        }
    }

    pub fn start(&mut self) {
        self.start_time = Some(SystemTime::now());
    }

    pub fn complete_step(&mut self, step_id: String, total_steps: usize) {
        self.completed_steps.push(step_id);
        self.current_step += 1;
        self.progress_percentage = self.completed_steps.len() as f64 / total_steps as f64;
    }

    pub fn fail_step(&mut self, step_id: String, error_message: String) {
        self.failed_steps.push((step_id, error_message));
    }

    pub fn finish(&mut self) {
        self.end_time = Some(SystemTime::now());
    }

    pub fn duration(&self) -> Option<Duration> {
        match (self.start_time, self.end_time) {
            (Some(start), Some(end)) => end.duration_since(start).ok(),
            _ => None,
        }
    }
}

/// Migration error handler
#[derive(Debug)]
pub struct MigrationErrorHandler {
    /// Error handling strategy
    pub strategy: ErrorHandlingStrategy,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Retry delay
    pub retry_delay: Duration,
}

/// Error handling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorHandlingStrategy {
    FailFast,
    RetryWithBackoff,
    SkipAndContinue,
    PauseForManualIntervention,
}

/// Backup manager for migration safety
#[derive(Debug)]
pub struct BackupManager {
    /// Backup strategy
    pub strategy: BackupStrategy,
    /// Backup location
    pub backup_path: std::path::PathBuf,
    /// Backup metadata
    pub metadata: HashMap<String, String>,
}

impl BackupManager {
    pub fn new(strategy: BackupStrategy, backup_path: std::path::PathBuf) -> Self {
        Self {
            strategy,
            backup_path,
            metadata: HashMap::new(),
        }
    }

    /// Create backup before migration
    pub fn create_backup(&mut self, data_path: &Path) -> Result<String> {
        let backup_id = format!(
            "backup_{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or(Duration::from_secs(0))
                .as_secs()
        );

        // Create backup directory
        let backup_dir = self.backup_path.join(&backup_id);
        std::fs::create_dir_all(&backup_dir)
            .map_err(|e| Error::IoError(format!("Failed to create backup directory: {}", e)))?;

        // Implement backup logic based on strategy
        match self.strategy {
            BackupStrategy::FullBackup => {
                self.perform_full_backup(data_path, &backup_dir)?;
            }
            BackupStrategy::IncrementalBackup => {
                self.perform_incremental_backup(data_path, &backup_dir)?;
            }
            BackupStrategy::SnapshotBased => {
                self.perform_snapshot_backup(data_path, &backup_dir)?;
            }
            BackupStrategy::CopyOnWrite => {
                self.perform_cow_backup(data_path, &backup_dir)?;
            }
            BackupStrategy::NoBackup => {
                // No backup required
            }
        }

        // Store backup metadata
        self.metadata
            .insert(backup_id.clone(), backup_dir.to_string_lossy().to_string());

        Ok(backup_id)
    }

    /// Restore from backup
    pub fn restore_backup(&self, backup_id: &str, target_path: &Path) -> Result<()> {
        if let Some(backup_path) = self.metadata.get(backup_id) {
            let backup_dir = std::path::Path::new(backup_path);
            if backup_dir.exists() {
                self.perform_restore(backup_dir, target_path)?;
                Ok(())
            } else {
                Err(Error::IoError(format!(
                    "Backup directory not found: {}",
                    backup_path
                )))
            }
        } else {
            Err(Error::InvalidInput(format!(
                "Backup ID not found: {}",
                backup_id
            )))
        }
    }

    // Backup implementation methods (simplified for demonstration)
    fn perform_full_backup(&self, source: &Path, target: &Path) -> Result<()> {
        if source.is_file() {
            let target_file = target.join(source.file_name().unwrap_or_default());
            std::fs::copy(source, target_file)
                .map_err(|e| Error::IoError(format!("Backup copy failed: {}", e)))?;
        }
        Ok(())
    }

    fn perform_incremental_backup(&self, _source: &Path, _target: &Path) -> Result<()> {
        Ok(())
    }

    fn perform_snapshot_backup(&self, _source: &Path, _target: &Path) -> Result<()> {
        Ok(())
    }

    fn perform_cow_backup(&self, _source: &Path, _target: &Path) -> Result<()> {
        Ok(())
    }

    fn perform_restore(&self, _backup_dir: &Path, _target: &Path) -> Result<()> {
        Ok(())
    }
}

/// Migration executor for running migration plans
pub struct MigrationExecutor {
    /// Migration context
    context: MigrationContext,
    /// Migration validators
    validators: Vec<Box<dyn MigrationValidator>>,
}

impl MigrationExecutor {
    /// Create new migration executor
    pub fn new(plan: MigrationPlan) -> Self {
        let error_handler = MigrationErrorHandler {
            strategy: ErrorHandlingStrategy::FailFast,
            max_retries: 3,
            retry_delay: Duration::from_secs(5),
        };

        let context = MigrationContext {
            plan,
            state: MigrationState::NotStarted,
            progress: MigrationProgress::new(),
            error_handler,
            backup_manager: None,
        };

        Self {
            context,
            validators: Vec::new(),
        }
    }

    /// Execute migration plan
    pub fn execute(&mut self) -> Result<MigrationResult> {
        self.context.state = MigrationState::InProgress;
        self.context.progress.start();

        // Execute migration steps
        let total_steps = self.context.plan.steps.len();
        let mut step_results = Vec::new();

        let steps = self.context.plan.steps.clone();
        for (index, step) in steps.iter().enumerate() {
            self.context.progress.current_step = index;

            match self.execute_step(step) {
                Ok(result) => {
                    self.context
                        .progress
                        .complete_step(step.id.clone(), total_steps);
                    step_results.push(result);
                }
                Err(error) => {
                    self.context
                        .progress
                        .fail_step(step.id.clone(), error.to_string());
                    return Ok(MigrationResult {
                        success: false,
                        step_results: Vec::new(),
                        duration: self.context.progress.duration(),
                        backup_id: None,
                        errors: vec![error.to_string()],
                    });
                }
            }
        }

        // Complete migration
        self.context.state = MigrationState::Completed;
        self.context.progress.finish();

        Ok(MigrationResult {
            success: true,
            step_results,
            duration: self.context.progress.duration(),
            backup_id: None,
            errors: Vec::new(),
        })
    }

    /// Execute individual migration step
    fn execute_step(&mut self, step: &MigrationStep) -> Result<StepResult> {
        let start_time = SystemTime::now();

        // Execute step based on type (placeholder implementations)
        let result: Result<()> = match step.step_type {
            MigrationStepType::DataStructureUpdate => Ok(()),
            MigrationStepType::APISignatureChange => Ok(()),
            MigrationStepType::StorageFormatMigration => Ok(()),
            MigrationStepType::ConfigurationUpdate => Ok(()),
            MigrationStepType::DependencyUpdate => Ok(()),
            MigrationStepType::Custom => Ok(()),
        };

        let end_time = SystemTime::now();
        let duration = end_time
            .duration_since(start_time)
            .unwrap_or(Duration::from_secs(0));

        match result {
            Ok(()) => Ok(StepResult {
                step_id: step.id.clone(),
                success: true,
                duration,
                error_message: None,
                rollback_required: false,
            }),
            Err(error) => Ok(StepResult {
                step_id: step.id.clone(),
                success: false,
                duration,
                error_message: Some(error.to_string()),
                rollback_required: step.rollback_action.is_some(),
            }),
        }
    }
}

/// Migration result
#[derive(Debug, Clone)]
pub struct MigrationResult {
    /// Whether migration succeeded
    pub success: bool,
    /// Results of individual steps
    pub step_results: Vec<StepResult>,
    /// Total migration duration
    pub duration: Option<Duration>,
    /// Backup ID (if backup was created)
    pub backup_id: Option<String>,
    /// Any errors that occurred
    pub errors: Vec<String>,
}

/// Individual step result
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Step identifier
    pub step_id: String,
    /// Whether step succeeded
    pub success: bool,
    /// Step execution duration
    pub duration: Duration,
    /// Error message if step failed
    pub error_message: Option<String>,
    /// Whether rollback is required
    pub rollback_required: bool,
}

/// Trait for migration validators
pub trait MigrationValidator: Send + Sync {
    /// Validate migration step
    fn validate_step(&self, step: &MigrationStep, result: &StepResult) -> Result<()>;

    /// Validate overall migration
    fn validate_migration(&self, result: &MigrationResult) -> Result<()>;
}

/// Backward compatibility layer
pub struct BackwardCompatibilityLayer {
    /// Supported legacy versions
    pub supported_versions: Vec<Version>,
    /// Migration plan cache
    pub migration_plans: HashMap<(Version, Version), MigrationPlan>,
}

impl BackwardCompatibilityLayer {
    /// Create new compatibility layer
    pub fn new() -> Self {
        Self {
            supported_versions: Vec::new(),
            migration_plans: HashMap::new(),
        }
    }

    /// Add supported legacy version
    pub fn add_supported_version(mut self, version: Version) -> Self {
        self.supported_versions.push(version);
        self
    }

    /// Check if version is supported
    pub fn is_version_supported(&self, version: &Version) -> bool {
        self.supported_versions.contains(version)
    }
}

/// Migration plan builder
pub struct MigrationPlanBuilder {
    from_version: Option<Version>,
    to_version: Option<Version>,
    steps: Vec<MigrationStep>,
    risk_level: MigrationRiskLevel,
}

impl MigrationPlanBuilder {
    /// Create new migration plan builder
    pub fn new() -> Self {
        Self {
            from_version: None,
            to_version: None,
            steps: Vec::new(),
            risk_level: MigrationRiskLevel::Low,
        }
    }

    /// Set source version
    pub fn from_version(mut self, version: Version) -> Self {
        self.from_version = Some(version);
        self
    }

    /// Set target version
    pub fn to_version(mut self, version: Version) -> Self {
        self.to_version = Some(version);
        self
    }

    /// Add migration step
    pub fn add_step(mut self, step: MigrationStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Set risk level
    pub fn risk_level(mut self, risk: MigrationRiskLevel) -> Self {
        self.risk_level = risk;
        self
    }

    /// Build migration plan
    pub fn build(self) -> Result<MigrationPlan> {
        let from_version = self
            .from_version
            .ok_or_else(|| Error::InvalidInput("From version not specified".to_string()))?;
        let to_version = self
            .to_version
            .ok_or_else(|| Error::InvalidInput("To version not specified".to_string()))?;

        let estimated_duration = self
            .steps
            .iter()
            .map(|s| s.estimated_time)
            .fold(Duration::from_secs(0), |acc, d| acc + d);

        Ok(MigrationPlan {
            from_version,
            to_version,
            steps: self.steps,
            estimated_duration,
            risk_level: self.risk_level,
            rollback_plan: None,
        })
    }
}

/// Current version constant
pub const CURRENT_VERSION: &str = "0.1.0-alpha.4";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parsing() {
        let version = Version::parse("0.1.0-alpha.4").unwrap();
        assert_eq!(version.major, 0);
        assert_eq!(version.minor, 1);
        assert_eq!(version.patch, 0);
        assert_eq!(version.pre_release, Some("alpha.4".to_string()));
    }

    #[test]
    fn test_version_compatibility() {
        let v1 = Version::new(1, 0, 0);
        let v1_1 = Version::new(1, 1, 0);
        let v2 = Version::new(2, 0, 0);

        assert!(v1.is_compatible_with(&v1_1));
        assert!(!v1.is_compatible_with(&v2));
        assert!(v2.requires_migration_from(&v1));
    }

    #[test]
    fn test_migration_plan_builder() {
        let from_version = Version::new_pre_release(0, 1, 0, "alpha.3".to_string());
        let to_version = Version::new_pre_release(0, 1, 0, "alpha.4".to_string());

        let step = MigrationStep {
            id: "update_traits".to_string(),
            description: "Update to new trait system".to_string(),
            step_type: MigrationStepType::APISignatureChange,
            dependencies: Vec::new(),
            validation: Vec::new(),
            rollback_action: None,
            estimated_time: Duration::from_secs(300),
        };

        let plan = MigrationPlanBuilder::new()
            .from_version(from_version.clone())
            .to_version(to_version.clone())
            .add_step(step)
            .risk_level(MigrationRiskLevel::Medium)
            .build()
            .unwrap();

        assert_eq!(plan.from_version, from_version);
        assert_eq!(plan.to_version, to_version);
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.risk_level, MigrationRiskLevel::Medium);
    }
}
