use std::collections::HashMap;
use std::sync::Arc;

use crate::core::error::{Error, Result};

use super::traits::{
    AggregatorPlugin, DataSinkPlugin, DataSourcePlugin, PluginMetadata, PluginType,
    TransformPlugin, ValidatorPlugin,
};

/// Central registry for all plugins
pub struct PluginRegistry {
    data_sources: HashMap<String, Arc<dyn DataSourcePlugin>>,
    data_sinks: HashMap<String, Arc<dyn DataSinkPlugin>>,
    transforms: HashMap<String, Arc<dyn TransformPlugin>>,
    aggregators: HashMap<String, Arc<dyn AggregatorPlugin>>,
    validators: HashMap<String, Arc<dyn ValidatorPlugin>>,
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginRegistry {
    /// Create a new empty plugin registry
    pub fn new() -> Self {
        PluginRegistry {
            data_sources: HashMap::new(),
            data_sinks: HashMap::new(),
            transforms: HashMap::new(),
            aggregators: HashMap::new(),
            validators: HashMap::new(),
        }
    }

    /// Register a data source plugin
    pub fn register_source(&mut self, plugin: Arc<dyn DataSourcePlugin>) -> Result<()> {
        let name = plugin.metadata().name.clone();
        if self.data_sources.contains_key(&name) {
            return Err(Error::InvalidOperation(format!(
                "DataSource plugin '{}' is already registered",
                name
            )));
        }
        self.data_sources.insert(name, plugin);
        Ok(())
    }

    /// Register a data sink plugin
    pub fn register_sink(&mut self, plugin: Arc<dyn DataSinkPlugin>) -> Result<()> {
        let name = plugin.metadata().name.clone();
        if self.data_sinks.contains_key(&name) {
            return Err(Error::InvalidOperation(format!(
                "DataSink plugin '{}' is already registered",
                name
            )));
        }
        self.data_sinks.insert(name, plugin);
        Ok(())
    }

    /// Register a transform plugin
    pub fn register_transform(&mut self, plugin: Arc<dyn TransformPlugin>) -> Result<()> {
        let name = plugin.metadata().name.clone();
        if self.transforms.contains_key(&name) {
            return Err(Error::InvalidOperation(format!(
                "Transform plugin '{}' is already registered",
                name
            )));
        }
        self.transforms.insert(name, plugin);
        Ok(())
    }

    /// Register an aggregator plugin
    pub fn register_aggregator(&mut self, plugin: Arc<dyn AggregatorPlugin>) -> Result<()> {
        let name = plugin.metadata().name.clone();
        if self.aggregators.contains_key(&name) {
            return Err(Error::InvalidOperation(format!(
                "Aggregator plugin '{}' is already registered",
                name
            )));
        }
        self.aggregators.insert(name, plugin);
        Ok(())
    }

    /// Register a validator plugin
    pub fn register_validator(&mut self, plugin: Arc<dyn ValidatorPlugin>) -> Result<()> {
        let name = plugin.metadata().name.clone();
        if self.validators.contains_key(&name) {
            return Err(Error::InvalidOperation(format!(
                "Validator plugin '{}' is already registered",
                name
            )));
        }
        self.validators.insert(name, plugin);
        Ok(())
    }

    /// Get a data source plugin by name
    pub fn get_source(&self, name: &str) -> Option<Arc<dyn DataSourcePlugin>> {
        self.data_sources.get(name).cloned()
    }

    /// Get a data sink plugin by name
    pub fn get_sink(&self, name: &str) -> Option<Arc<dyn DataSinkPlugin>> {
        self.data_sinks.get(name).cloned()
    }

    /// Get a transform plugin by name
    pub fn get_transform(&self, name: &str) -> Option<Arc<dyn TransformPlugin>> {
        self.transforms.get(name).cloned()
    }

    /// Get an aggregator plugin by name
    pub fn get_aggregator(&self, name: &str) -> Option<Arc<dyn AggregatorPlugin>> {
        self.aggregators.get(name).cloned()
    }

    /// Get a validator plugin by name
    pub fn get_validator(&self, name: &str) -> Option<Arc<dyn ValidatorPlugin>> {
        self.validators.get(name).cloned()
    }

    /// List all registered plugins (metadata)
    pub fn list_plugins(&self) -> Vec<&PluginMetadata> {
        let mut result = Vec::new();
        for p in self.data_sources.values() {
            result.push(p.metadata());
        }
        for p in self.data_sinks.values() {
            result.push(p.metadata());
        }
        for p in self.transforms.values() {
            result.push(p.metadata());
        }
        for p in self.aggregators.values() {
            result.push(p.metadata());
        }
        for p in self.validators.values() {
            result.push(p.metadata());
        }
        result
    }

    /// List plugins filtered by type
    pub fn list_by_type(&self, plugin_type: &PluginType) -> Vec<&PluginMetadata> {
        self.list_plugins()
            .into_iter()
            .filter(|m| &m.plugin_type == plugin_type)
            .collect()
    }

    /// Check if a source plugin is registered
    pub fn has_source(&self, name: &str) -> bool {
        self.data_sources.contains_key(name)
    }

    /// Check if a transform plugin is registered
    pub fn has_transform(&self, name: &str) -> bool {
        self.transforms.contains_key(name)
    }

    /// Check if a sink plugin is registered
    pub fn has_sink(&self, name: &str) -> bool {
        self.data_sinks.contains_key(name)
    }

    /// Total number of registered plugins
    pub fn plugin_count(&self) -> usize {
        self.data_sources.len()
            + self.data_sinks.len()
            + self.transforms.len()
            + self.aggregators.len()
            + self.validators.len()
    }
}
