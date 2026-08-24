use std::collections::HashMap;
use std::sync::Arc;

use crate::core::error::{Error, Result};

use super::traits::{
    AggregatorPlugin, DataSinkPlugin, DataSourcePlugin, PluginMetadata, PluginType,
    TransformPlugin, ValidatorPlugin,
};

/// Central registry for all plugins.
///
/// Plugins are plain in-process trait objects (`Arc<dyn ...Plugin>`)
/// registered into `HashMap`s here -- there is no dynamic `.so`/`.dll`
/// loading anywhere in this module (no `libloading` or equivalent). A
/// "plugin" is Rust code linked into the same binary as `pandrs`; the value
/// this module adds is a uniform, name-addressable registry and pipeline
/// composition (see [`super::pipeline::PluginPipeline`]), not runtime
/// extensibility across process/binary boundaries.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugins::builtins::{CsvSinkPlugin, CsvSourcePlugin, FilterTransformPlugin};

    #[test]
    fn test_unregister_removes_and_returns_the_plugin() {
        let mut registry = PluginRegistry::new();
        registry
            .register_source(CsvSourcePlugin::arc())
            .expect("register");
        assert!(registry.has_source("csv_source"));

        let removed = registry.unregister_source("csv_source");
        assert!(removed.is_some());
        assert!(!registry.has_source("csv_source"));
        assert_eq!(registry.plugin_count(), 0);

        // Unregistering something never registered is a harmless `None`.
        assert!(registry.unregister_source("csv_source").is_none());
    }

    #[test]
    fn test_unregister_then_register_allows_reuse_of_the_name() {
        let mut registry = PluginRegistry::new();
        registry
            .register_transform(FilterTransformPlugin::arc())
            .expect("register");

        // Without unregistering first, re-registering the same name errors.
        assert!(registry
            .register_transform(FilterTransformPlugin::arc())
            .is_err());

        registry.unregister_transform("filter");
        // Now it can be registered again -- this is the "test pollution,
        // no hot-swap" gap the finding called out.
        assert!(registry
            .register_transform(FilterTransformPlugin::arc())
            .is_ok());
    }

    #[test]
    fn test_replace_swaps_the_plugin_without_erroring() {
        let mut registry = PluginRegistry::new();
        registry
            .register_source(CsvSourcePlugin::arc())
            .expect("register");

        // `register_source` would error here; `replace_source` hot-swaps.
        let previous = registry.replace_source(CsvSourcePlugin::arc());
        assert!(previous.is_some());
        assert!(registry.has_source("csv_source"));
        assert_eq!(registry.plugin_count(), 1);
    }

    #[test]
    fn test_clear_empties_every_plugin_kind() {
        let mut registry = PluginRegistry::new();
        registry
            .register_source(CsvSourcePlugin::arc())
            .expect("source");
        registry.register_sink(CsvSinkPlugin::arc()).expect("sink");
        registry
            .register_transform(FilterTransformPlugin::arc())
            .expect("transform");
        assert_eq!(registry.plugin_count(), 3);

        registry.clear();

        assert_eq!(registry.plugin_count(), 0);
        assert!(registry.list_plugins().is_empty());
        assert!(!registry.has_source("csv_source"));
        assert!(!registry.has_sink("csv_sink"));
        assert!(!registry.has_transform("filter"));
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

    /// Check if an aggregator plugin is registered
    pub fn has_aggregator(&self, name: &str) -> bool {
        self.aggregators.contains_key(name)
    }

    /// Check if a validator plugin is registered
    pub fn has_validator(&self, name: &str) -> bool {
        self.validators.contains_key(name)
    }

    /// Total number of registered plugins
    pub fn plugin_count(&self) -> usize {
        self.data_sources.len()
            + self.data_sinks.len()
            + self.transforms.len()
            + self.aggregators.len()
            + self.validators.len()
    }

    // -- Unregister / replace / clear ---------------------------------
    //
    // The `register_*` methods above error on a name collision, which is
    // right for normal registration but leaves no way to (a) remove a
    // plugin, (b) intentionally swap one implementation for another under
    // the same name (a hot-swap), or (c) reset a registry between tests
    // that each want their own clean set of registered names. These fill
    // that gap.

    /// Unregister (remove) a data source plugin by name, returning it if
    /// one was registered.
    pub fn unregister_source(&mut self, name: &str) -> Option<Arc<dyn DataSourcePlugin>> {
        self.data_sources.remove(name)
    }

    /// Unregister (remove) a data sink plugin by name, returning it if one
    /// was registered.
    pub fn unregister_sink(&mut self, name: &str) -> Option<Arc<dyn DataSinkPlugin>> {
        self.data_sinks.remove(name)
    }

    /// Unregister (remove) a transform plugin by name, returning it if one
    /// was registered.
    pub fn unregister_transform(&mut self, name: &str) -> Option<Arc<dyn TransformPlugin>> {
        self.transforms.remove(name)
    }

    /// Unregister (remove) an aggregator plugin by name, returning it if
    /// one was registered.
    pub fn unregister_aggregator(&mut self, name: &str) -> Option<Arc<dyn AggregatorPlugin>> {
        self.aggregators.remove(name)
    }

    /// Unregister (remove) a validator plugin by name, returning it if one
    /// was registered.
    pub fn unregister_validator(&mut self, name: &str) -> Option<Arc<dyn ValidatorPlugin>> {
        self.validators.remove(name)
    }

    /// Register a data source plugin, replacing any plugin already
    /// registered under the same name (unlike [`Self::register_source`],
    /// which errors on a name collision). Returns the plugin that was
    /// previously registered under that name, if any.
    pub fn replace_source(
        &mut self,
        plugin: Arc<dyn DataSourcePlugin>,
    ) -> Option<Arc<dyn DataSourcePlugin>> {
        let name = plugin.metadata().name.clone();
        self.data_sources.insert(name, plugin)
    }

    /// Like [`Self::replace_source`], for data sink plugins.
    pub fn replace_sink(
        &mut self,
        plugin: Arc<dyn DataSinkPlugin>,
    ) -> Option<Arc<dyn DataSinkPlugin>> {
        let name = plugin.metadata().name.clone();
        self.data_sinks.insert(name, plugin)
    }

    /// Like [`Self::replace_source`], for transform plugins.
    pub fn replace_transform(
        &mut self,
        plugin: Arc<dyn TransformPlugin>,
    ) -> Option<Arc<dyn TransformPlugin>> {
        let name = plugin.metadata().name.clone();
        self.transforms.insert(name, plugin)
    }

    /// Like [`Self::replace_source`], for aggregator plugins.
    pub fn replace_aggregator(
        &mut self,
        plugin: Arc<dyn AggregatorPlugin>,
    ) -> Option<Arc<dyn AggregatorPlugin>> {
        let name = plugin.metadata().name.clone();
        self.aggregators.insert(name, plugin)
    }

    /// Like [`Self::replace_source`], for validator plugins.
    pub fn replace_validator(
        &mut self,
        plugin: Arc<dyn ValidatorPlugin>,
    ) -> Option<Arc<dyn ValidatorPlugin>> {
        let name = plugin.metadata().name.clone();
        self.validators.insert(name, plugin)
    }

    /// Remove every registered plugin of every kind, returning the registry
    /// to the same empty state as [`Self::new`]. Useful for test isolation
    /// (a shared registry otherwise accumulates plugins test-to-test with
    /// no way to reset) and for a full hot-swap/reload of a local registry.
    pub fn clear(&mut self) {
        self.data_sources.clear();
        self.data_sinks.clear();
        self.transforms.clear();
        self.aggregators.clear();
        self.validators.clear();
    }
}
