use std::sync::{Arc, RwLock};

use lazy_static::lazy_static;

use crate::core::error::{Error, Result};

use super::builtins::{
    CsvSinkPlugin, CsvSourcePlugin, FillNaPlugin, FilterTransformPlugin, JsonSinkPlugin,
    JsonSourcePlugin, NormalizePlugin, SelectColumnsPlugin,
};
use super::registry::PluginRegistry;
use super::traits::{
    AggregatorPlugin, DataSinkPlugin, DataSourcePlugin, TransformPlugin, ValidatorPlugin,
};

lazy_static! {
    static ref GLOBAL_REGISTRY: RwLock<PluginRegistry> = RwLock::new(PluginRegistry::new());
}

/// Register all built-in plugins into the global registry.
///
/// This is idempotent: if a plugin is already registered it is silently skipped.
pub fn register_builtin_plugins() -> Result<()> {
    let mut registry = GLOBAL_REGISTRY.write().map_err(|_| Error::LockPoisoned {
        context: "global registry write lock".to_string(),
    })?;

    // Register sources (ignore "already registered" errors for idempotence)
    if !registry.has_source("csv_source") {
        registry.register_source(CsvSourcePlugin::arc())?;
    }
    if !registry.has_source("json_source") {
        registry.register_source(JsonSourcePlugin::arc())?;
    }

    // Register sinks
    if !registry.has_sink("csv_sink") {
        registry.register_sink(CsvSinkPlugin::arc())?;
    }
    if !registry.has_sink("json_sink") {
        registry.register_sink(JsonSinkPlugin::arc())?;
    }

    // Register transforms
    if !registry.has_transform("filter") {
        registry.register_transform(FilterTransformPlugin::arc())?;
    }
    if !registry.has_transform("select_columns") {
        registry.register_transform(SelectColumnsPlugin::arc())?;
    }
    if !registry.has_transform("normalize") {
        registry.register_transform(NormalizePlugin::arc())?;
    }
    if !registry.has_transform("fill_na") {
        registry.register_transform(FillNaPlugin::arc())?;
    }

    Ok(())
}

/// Obtain a read-only snapshot of the global registry wrapped in an Arc.
///
/// The returned Arc holds a clone of the current registry state so callers
/// can build pipelines without holding the global lock.
pub fn global_registry() -> Result<Arc<PluginRegistry>> {
    // We can't return &'static data easily without poisoning issues, so we clone.
    // This is intentional: pipelines are short-lived and the clone is cheap
    // (only Arc pointers are cloned, not plugin data).
    let registry = GLOBAL_REGISTRY.read().map_err(|_| Error::LockPoisoned {
        context: "global registry read lock".to_string(),
    })?;

    // Build a fresh PluginRegistry and re-populate from the global one.
    // Since we cannot clone PluginRegistry directly (dyn traits aren't Clone),
    // we instead return the internal state via a newtype wrapper.
    // As a practical approach we use a separate Arc-wrapped copy mechanism:
    drop(registry);

    // Alternative: expose a method on PluginRegistry to snapshot itself.
    // For now we use the simpler approach of returning an Arc to a new registry
    // that shares the same plugin Arcs.
    with_global_registry(|r| {
        let mut snapshot = PluginRegistry::new();
        // Re-register everything from the global
        for meta in r.list_plugins() {
            // We can only re-add plugins whose Arc we still hold.
            // The registry stores Arcs, so list_by_type + get_* works.
            let _ = meta; // handled below by type
        }
        // Populate by plugin type
        for name in r
            .list_plugins()
            .iter()
            .map(|m| m.name.clone())
            .collect::<Vec<_>>()
        {
            if let Some(p) = r.get_source(&name) {
                let _ = snapshot.register_source(p);
            }
            if let Some(p) = r.get_sink(&name) {
                let _ = snapshot.register_sink(p);
            }
            if let Some(p) = r.get_transform(&name) {
                let _ = snapshot.register_transform(p);
            }
            if let Some(p) = r.get_aggregator(&name) {
                let _ = snapshot.register_aggregator(p);
            }
            if let Some(p) = r.get_validator(&name) {
                let _ = snapshot.register_validator(p);
            }
        }
        Ok(Arc::new(snapshot))
    })
}

/// Execute a closure with a read reference to the global registry.
pub fn with_global_registry<F, T>(f: F) -> Result<T>
where
    F: FnOnce(&PluginRegistry) -> Result<T>,
{
    let registry = GLOBAL_REGISTRY.read().map_err(|_| Error::LockPoisoned {
        context: "global registry read lock".to_string(),
    })?;
    f(&registry)
}

/// Execute a closure with a mutable reference to the global registry.
pub fn with_global_registry_mut<F, T>(f: F) -> Result<T>
where
    F: FnOnce(&mut PluginRegistry) -> Result<T>,
{
    let mut registry = GLOBAL_REGISTRY.write().map_err(|_| Error::LockPoisoned {
        context: "global registry write lock".to_string(),
    })?;
    f(&mut registry)
}

/// Register a custom data source plugin into the global registry.
pub fn register_source(plugin: Arc<dyn DataSourcePlugin>) -> Result<()> {
    with_global_registry_mut(|r| r.register_source(plugin))
}

/// Register a custom data sink plugin into the global registry.
pub fn register_sink(plugin: Arc<dyn DataSinkPlugin>) -> Result<()> {
    with_global_registry_mut(|r| r.register_sink(plugin))
}

/// Register a custom transform plugin into the global registry.
pub fn register_transform(plugin: Arc<dyn TransformPlugin>) -> Result<()> {
    with_global_registry_mut(|r| r.register_transform(plugin))
}

/// Register a custom aggregator plugin into the global registry.
pub fn register_aggregator(plugin: Arc<dyn AggregatorPlugin>) -> Result<()> {
    with_global_registry_mut(|r| r.register_aggregator(plugin))
}

/// Register a custom validator plugin into the global registry.
pub fn register_validator(plugin: Arc<dyn ValidatorPlugin>) -> Result<()> {
    with_global_registry_mut(|r| r.register_validator(plugin))
}
