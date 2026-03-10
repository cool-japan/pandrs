use std::collections::HashMap;
use std::sync::Arc;

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::io::json::JsonOrient;
use crate::io::json::{read_json, write_json};

use crate::plugins::traits::{DataSinkPlugin, DataSourcePlugin, PluginMetadata, PluginType};

/// Built-in JSON data source plugin.
///
/// Options:
/// - `path`: Path to the JSON file (required)
/// - `orient`: "records" (default) or "columns"
pub struct JsonSourcePlugin {
    metadata: PluginMetadata,
}

impl JsonSourcePlugin {
    pub fn new() -> Self {
        JsonSourcePlugin {
            metadata: PluginMetadata {
                name: "json_source".to_string(),
                version: "1.0.0".to_string(),
                description: "Read DataFrames from JSON files".to_string(),
                author: "PandRS".to_string(),
                plugin_type: PluginType::DataSource,
                capabilities: vec![
                    "read".to_string(),
                    "records".to_string(),
                    "columns".to_string(),
                ],
            },
        }
    }

    pub fn arc() -> Arc<Self> {
        Arc::new(Self::new())
    }
}

impl Default for JsonSourcePlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl DataSourcePlugin for JsonSourcePlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn read(&self, options: &HashMap<String, String>) -> Result<DataFrame> {
        let path = options.get("path").ok_or_else(|| {
            Error::InvalidInput("json_source: 'path' option is required".to_string())
        })?;

        // The orient option is noted but read_json auto-detects based on the JSON shape.
        // We pass it through for documentation purposes.
        let _orient = options
            .get("orient")
            .map(|s| s.as_str())
            .unwrap_or("records");

        read_json(path)
    }
}

/// Built-in JSON data sink plugin.
///
/// Options:
/// - `path`: Path to write the JSON file (required)
/// - `orient`: "records" (default) or "columns"
pub struct JsonSinkPlugin {
    metadata: PluginMetadata,
}

impl JsonSinkPlugin {
    pub fn new() -> Self {
        JsonSinkPlugin {
            metadata: PluginMetadata {
                name: "json_sink".to_string(),
                version: "1.0.0".to_string(),
                description: "Write DataFrames to JSON files".to_string(),
                author: "PandRS".to_string(),
                plugin_type: PluginType::DataSink,
                capabilities: vec![
                    "write".to_string(),
                    "records".to_string(),
                    "columns".to_string(),
                ],
            },
        }
    }

    pub fn arc() -> Arc<Self> {
        Arc::new(Self::new())
    }
}

impl Default for JsonSinkPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl DataSinkPlugin for JsonSinkPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn write(&self, df: &DataFrame, options: &HashMap<String, String>) -> Result<()> {
        let path = options.get("path").ok_or_else(|| {
            Error::InvalidInput("json_sink: 'path' option is required".to_string())
        })?;

        let orient = match options
            .get("orient")
            .map(|s| s.as_str())
            .unwrap_or("records")
        {
            "columns" => JsonOrient::Columns,
            _ => JsonOrient::Records,
        };

        write_json(df, path, orient)
    }
}
