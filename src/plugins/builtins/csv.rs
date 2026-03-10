use std::collections::HashMap;
use std::sync::Arc;

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::io::csv::{read_csv, write_csv};

use crate::plugins::traits::{DataSinkPlugin, DataSourcePlugin, PluginMetadata, PluginType};

/// Built-in CSV data source plugin.
///
/// Options:
/// - `path`: Path to the CSV file (required)
/// - `has_header`: "true" or "false" (default: "true")
/// - `delimiter`: single character delimiter (currently passed to underlying reader; default: ",")
pub struct CsvSourcePlugin {
    metadata: PluginMetadata,
}

impl CsvSourcePlugin {
    pub fn new() -> Self {
        CsvSourcePlugin {
            metadata: PluginMetadata {
                name: "csv_source".to_string(),
                version: "1.0.0".to_string(),
                description: "Read DataFrames from CSV files".to_string(),
                author: "PandRS".to_string(),
                plugin_type: PluginType::DataSource,
                capabilities: vec![
                    "read".to_string(),
                    "header".to_string(),
                    "delimiter".to_string(),
                ],
            },
        }
    }

    pub fn arc() -> Arc<Self> {
        Arc::new(Self::new())
    }
}

impl Default for CsvSourcePlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl DataSourcePlugin for CsvSourcePlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn read(&self, options: &HashMap<String, String>) -> Result<DataFrame> {
        let path = options.get("path").ok_or_else(|| {
            Error::InvalidInput("csv_source: 'path' option is required".to_string())
        })?;

        let has_header = options
            .get("has_header")
            .map(|v| v.to_lowercase() != "false")
            .unwrap_or(true);

        read_csv(path, has_header)
    }
}

/// Built-in CSV data sink plugin.
///
/// Options:
/// - `path`: Path to write the CSV file (required)
pub struct CsvSinkPlugin {
    metadata: PluginMetadata,
}

impl CsvSinkPlugin {
    pub fn new() -> Self {
        CsvSinkPlugin {
            metadata: PluginMetadata {
                name: "csv_sink".to_string(),
                version: "1.0.0".to_string(),
                description: "Write DataFrames to CSV files".to_string(),
                author: "PandRS".to_string(),
                plugin_type: PluginType::DataSink,
                capabilities: vec!["write".to_string()],
            },
        }
    }

    pub fn arc() -> Arc<Self> {
        Arc::new(Self::new())
    }
}

impl Default for CsvSinkPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl DataSinkPlugin for CsvSinkPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn write(&self, df: &DataFrame, options: &HashMap<String, String>) -> Result<()> {
        let path = options.get("path").ok_or_else(|| {
            Error::InvalidInput("csv_sink: 'path' option is required".to_string())
        })?;

        write_csv(df, path)
    }
}
