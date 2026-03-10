use crate::core::error::Result;
use crate::dataframe::DataFrame;
use std::collections::HashMap;

/// Metadata for a plugin
#[derive(Debug, Clone)]
pub struct PluginMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub plugin_type: PluginType,
    pub capabilities: Vec<String>,
}

/// The type of plugin
#[derive(Debug, Clone, PartialEq)]
pub enum PluginType {
    DataSource,
    DataSink,
    Transform,
    Aggregator,
    Validator,
    Connector,
}

impl std::fmt::Display for PluginType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PluginType::DataSource => write!(f, "DataSource"),
            PluginType::DataSink => write!(f, "DataSink"),
            PluginType::Transform => write!(f, "Transform"),
            PluginType::Aggregator => write!(f, "Aggregator"),
            PluginType::Validator => write!(f, "Validator"),
            PluginType::Connector => write!(f, "Connector"),
        }
    }
}

/// Data source plugin - provides DataFrames from some source
pub trait DataSourcePlugin: Send + Sync {
    fn metadata(&self) -> &PluginMetadata;
    fn read(&self, options: &HashMap<String, String>) -> Result<DataFrame>;
    fn supports_streaming(&self) -> bool {
        false
    }
}

/// Data sink plugin - writes DataFrames to some destination
pub trait DataSinkPlugin: Send + Sync {
    fn metadata(&self) -> &PluginMetadata;
    fn write(&self, df: &DataFrame, options: &HashMap<String, String>) -> Result<()>;
    fn supports_append(&self) -> bool {
        false
    }
}

/// Transform plugin - transforms a DataFrame into another DataFrame
pub trait TransformPlugin: Send + Sync {
    fn metadata(&self) -> &PluginMetadata;
    fn transform(&self, df: DataFrame, options: &HashMap<String, String>) -> Result<DataFrame>;
}

/// Aggregator plugin - aggregates a DataFrame
pub trait AggregatorPlugin: Send + Sync {
    fn metadata(&self) -> &PluginMetadata;
    fn aggregate(&self, df: &DataFrame, options: &HashMap<String, String>) -> Result<DataFrame>;
}

/// Severity of a validation issue
#[derive(Debug, Clone, PartialEq)]
pub enum IssueSeverity {
    Error,
    Warning,
    Info,
}

impl std::fmt::Display for IssueSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IssueSeverity::Error => write!(f, "Error"),
            IssueSeverity::Warning => write!(f, "Warning"),
            IssueSeverity::Info => write!(f, "Info"),
        }
    }
}

/// A single validation issue found in a DataFrame
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    pub severity: IssueSeverity,
    pub message: String,
    pub column: Option<String>,
    pub row: Option<usize>,
}

impl ValidationIssue {
    pub fn error(message: impl Into<String>) -> Self {
        ValidationIssue {
            severity: IssueSeverity::Error,
            message: message.into(),
            column: None,
            row: None,
        }
    }

    pub fn warning(message: impl Into<String>) -> Self {
        ValidationIssue {
            severity: IssueSeverity::Warning,
            message: message.into(),
            column: None,
            row: None,
        }
    }

    pub fn info(message: impl Into<String>) -> Self {
        ValidationIssue {
            severity: IssueSeverity::Info,
            message: message.into(),
            column: None,
            row: None,
        }
    }

    pub fn with_column(mut self, column: impl Into<String>) -> Self {
        self.column = Some(column.into());
        self
    }

    pub fn with_row(mut self, row: usize) -> Self {
        self.row = Some(row);
        self
    }
}

/// Validator plugin - validates a DataFrame
pub trait ValidatorPlugin: Send + Sync {
    fn metadata(&self) -> &PluginMetadata;
    fn validate(
        &self,
        df: &DataFrame,
        options: &HashMap<String, String>,
    ) -> Result<Vec<ValidationIssue>>;
}
