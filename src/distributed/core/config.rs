//! # Distributed Processing Configuration
//!
//! Provides configuration options for distributed processing in PandRS.

use std::collections::HashMap;

/// Execution engine types supported by PandRS
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutorType {
    /// Local execution using DataFusion
    DataFusion,
    /// Distributed execution using Ballista
    Ballista,
}

impl Default for ExecutorType {
    fn default() -> Self {
        Self::DataFusion
    }
}

impl std::fmt::Display for ExecutorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DataFusion => write!(f, "datafusion"),
            Self::Ballista => write!(f, "ballista"),
        }
    }
}

impl std::str::FromStr for ExecutorType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "datafusion" => Ok(Self::DataFusion),
            "ballista" => Ok(Self::Ballista),
            _ => Err(format!("Unknown executor type: {}", s)),
        }
    }
}

/// Configuration for distributed processing
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Type of executor to use
    executor_type: ExecutorType,
    /// Number of threads to use for local execution
    concurrency: usize,
    /// Memory limit in bytes
    memory_limit: Option<usize>,
    /// Whether to skip schema validation
    skip_validation: bool,
    /// Whether to enable query optimization
    enable_optimization: bool,
    /// Optimizer rule configurations
    optimizer_rules: HashMap<String, String>,
    /// Additional configuration options
    options: HashMap<String, String>,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        let mut optimizer_rules = HashMap::new();

        // Default optimizer rule settings
        optimizer_rules.insert("filter_pushdown".to_string(), "true".to_string());
        optimizer_rules.insert("join_reordering".to_string(), "true".to_string());
        optimizer_rules.insert("predicate_pushdown".to_string(), "true".to_string());
        optimizer_rules.insert("projection_pushdown".to_string(), "true".to_string());
        optimizer_rules.insert("skip_failed_rules".to_string(), "true".to_string());
        optimizer_rules.insert(
            "enable_round_robin_repartition".to_string(),
            "true".to_string(),
        );
        optimizer_rules.insert("prefer_hash_join".to_string(), "true".to_string());

        Self {
            executor_type: ExecutorType::default(),
            concurrency: num_cpus::get(),
            memory_limit: None,
            skip_validation: false,
            enable_optimization: true,
            optimizer_rules,
            options: HashMap::new(),
        }
    }
}

impl DistributedConfig {
    /// Creates a new configuration with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the executor type
    ///
    /// # Arguments
    ///
    /// * `executor` - The executor type to use
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_executor(mut self, executor: impl Into<String>) -> Self {
        if let Ok(exec_type) = executor.into().parse() {
            self.executor_type = exec_type;
        }
        self
    }

    /// Sets the executor type directly
    ///
    /// # Arguments
    ///
    /// * `executor_type` - The executor type to use
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_executor_type(mut self, executor_type: ExecutorType) -> Self {
        self.executor_type = executor_type;
        self
    }

    /// Sets the number of threads to use for local execution
    ///
    /// # Arguments
    ///
    /// * `concurrency` - The number of threads to use
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Sets the memory limit
    ///
    /// # Arguments
    ///
    /// * `limit` - The memory limit in bytes
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = Some(limit);
        self
    }

    /// Sets the memory limit from a string representation
    ///
    /// # Arguments
    ///
    /// * `limit` - The memory limit as a string (e.g., "4GB", "512MB")
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_memory_limit_str(mut self, limit: impl AsRef<str>) -> Self {
        let limit = limit.as_ref();
        if let Some(bytes) = parse_memory_size(limit) {
            self.memory_limit = Some(bytes);
        }
        self
    }

    /// Sets an additional configuration option
    ///
    /// # Arguments
    ///
    /// * `key` - The option key
    /// * `value` - The option value
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.insert(key.into(), value.into());
        self
    }

    /// Gets the executor type
    pub fn executor_type(&self) -> ExecutorType {
        self.executor_type
    }

    /// Gets the concurrency level
    pub fn concurrency(&self) -> usize {
        self.concurrency
    }

    /// Gets the memory limit
    pub fn memory_limit(&self) -> Option<usize> {
        self.memory_limit
    }

    /// Sets whether to skip schema validation
    ///
    /// # Arguments
    ///
    /// * `skip` - Whether to skip schema validation
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_skip_validation(mut self, skip: bool) -> Self {
        self.skip_validation = skip;
        self
    }

    /// Gets an option value
    pub fn option(&self, key: &str) -> Option<&String> {
        self.options.get(key)
    }

    /// Gets all options
    pub fn options(&self) -> &HashMap<String, String> {
        &self.options
    }

    /// Gets whether to skip schema validation
    pub fn skip_validation(&self) -> bool {
        self.skip_validation
    }

    /// Sets whether to enable query optimization
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to enable query optimization
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_optimization(mut self, enable: bool) -> Self {
        self.enable_optimization = enable;
        self
    }

    /// Gets whether query optimization is enabled
    pub fn enable_optimization(&self) -> bool {
        self.enable_optimization
    }

    /// Sets an optimizer rule
    ///
    /// # Arguments
    ///
    /// * `rule` - The name of the rule
    /// * `enable` - Whether to enable the rule
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_optimizer_rule(mut self, rule: impl Into<String>, enable: bool) -> Self {
        self.optimizer_rules.insert(rule.into(), enable.to_string());
        self
    }

    /// Gets the value of an optimizer rule
    pub fn optimizer_rule(&self, rule: &str) -> Option<bool> {
        self.optimizer_rules.get(rule).and_then(|v| v.parse().ok())
    }

    /// Gets all optimizer rules
    pub fn optimizer_rules(&self) -> &HashMap<String, String> {
        &self.optimizer_rules
    }
}

/// Configuration specific to Ballista distributed processing
#[derive(Debug, Clone)]
pub struct BallistaConfig {
    /// Base distributed configuration
    base_config: DistributedConfig,
    /// Scheduler endpoint
    scheduler: Option<String>,
    /// Number of executors to use
    num_executors: Option<usize>,
}

impl Default for BallistaConfig {
    fn default() -> Self {
        Self {
            base_config: DistributedConfig::default().with_executor_type(ExecutorType::Ballista),
            scheduler: None,
            num_executors: None,
        }
    }
}

impl BallistaConfig {
    /// Creates a new Ballista configuration with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the scheduler endpoint
    ///
    /// # Arguments
    ///
    /// * `scheduler` - The scheduler endpoint (e.g., "localhost:50050")
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_scheduler(mut self, scheduler: impl Into<String>) -> Self {
        self.scheduler = Some(scheduler.into());
        self
    }

    /// Sets the number of executors
    ///
    /// # Arguments
    ///
    /// * `num_executors` - The number of executors to use
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_num_executors(mut self, num_executors: usize) -> Self {
        self.num_executors = Some(num_executors);
        self
    }

    /// Sets the concurrency level (thread count per executor)
    ///
    /// # Arguments
    ///
    /// * `concurrency` - The number of threads per executor
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.base_config = self.base_config.with_concurrency(concurrency);
        self
    }

    /// Sets the memory limit
    ///
    /// # Arguments
    ///
    /// * `limit` - The memory limit in bytes
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_memory_limit(mut self, limit: usize) -> Self {
        self.base_config = self.base_config.with_memory_limit(limit);
        self
    }

    /// Sets the memory limit from a string representation
    ///
    /// # Arguments
    ///
    /// * `limit` - The memory limit as a string (e.g., "4GB", "512MB")
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_memory_limit_str(mut self, limit: impl AsRef<str>) -> Self {
        self.base_config = self.base_config.with_memory_limit_str(limit);
        self
    }

    /// Sets an additional configuration option
    ///
    /// # Arguments
    ///
    /// * `key` - The option key
    /// * `value` - The option value
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.base_config = self.base_config.with_option(key, value);
        self
    }

    /// Gets the base distributed configuration
    pub fn base_config(&self) -> &DistributedConfig {
        &self.base_config
    }

    /// Gets the scheduler endpoint
    pub fn scheduler(&self) -> Option<&String> {
        self.scheduler.as_ref()
    }

    /// Gets the number of executors
    pub fn num_executors(&self) -> Option<usize> {
        self.num_executors
    }

    /// Converts to a regular distributed configuration
    pub fn to_distributed_config(&self) -> DistributedConfig {
        let mut config = self.base_config.clone();
        if let Some(scheduler) = &self.scheduler {
            config = config.with_option("scheduler", scheduler);
        }
        if let Some(num_executors) = self.num_executors {
            config = config.with_option("num_executors", num_executors.to_string());
        }
        config
    }
}

impl From<BallistaConfig> for DistributedConfig {
    fn from(config: BallistaConfig) -> Self {
        config.to_distributed_config()
    }
}

// Helper function to parse memory size from string
fn parse_memory_size(s: &str) -> Option<usize> {
    let s = s.trim().to_uppercase();

    // Match patterns like "4GB", "512MB", etc.
    let re = regex::Regex::new(r"^(\d+)\s*([KMGT]B)?$").ok()?;
    let caps = re.captures(&s)?;

    let num: usize = caps.get(1)?.as_str().parse().ok()?;
    let unit = caps.get(2).map_or("B", |m| m.as_str());

    // Convert to bytes
    match unit {
        "B" => Some(num),
        "KB" => Some(num * 1024),
        "MB" => Some(num * 1024 * 1024),
        "GB" => Some(num * 1024 * 1024 * 1024),
        "TB" => Some(num * 1024 * 1024 * 1024 * 1024),
        _ => None,
    }
}
