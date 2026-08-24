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
/// The returned Arc holds a **new** `PluginRegistry` populated with clones
/// of the same plugin `Arc`s the global registry holds (cheap: only the
/// `Arc` pointers are cloned, not plugin data), so callers can build
/// pipelines without holding the global lock for the pipeline's lifetime.
///
/// # Non-reentrancy
///
/// This acquires `GLOBAL_REGISTRY`'s read lock directly, exactly once (it
/// does not call [`with_global_registry`], specifically so this function
/// alone can never self-deadlock). `std::sync::RwLock` is not reentrant in
/// general, though: do not call this function, [`with_global_registry`], or
/// any `register_*`/`unregister_*` helper below from inside a closure
/// passed to [`with_global_registry_mut`] on the same thread. That thread
/// already holds the write lock, and any nested lock attempt -- read or
/// write -- blocks waiting for a lock the same thread is already holding,
/// which can never be released.
pub fn global_registry() -> Result<Arc<PluginRegistry>> {
    let registry = GLOBAL_REGISTRY.read().map_err(|_| Error::LockPoisoned {
        context: "global registry read lock".to_string(),
    })?;

    let mut snapshot = PluginRegistry::new();
    for name in registry
        .list_plugins()
        .iter()
        .map(|m| m.name.clone())
        .collect::<Vec<_>>()
    {
        if let Some(p) = registry.get_source(&name) {
            let _ = snapshot.register_source(p);
        }
        if let Some(p) = registry.get_sink(&name) {
            let _ = snapshot.register_sink(p);
        }
        if let Some(p) = registry.get_transform(&name) {
            let _ = snapshot.register_transform(p);
        }
        if let Some(p) = registry.get_aggregator(&name) {
            let _ = snapshot.register_aggregator(p);
        }
        if let Some(p) = registry.get_validator(&name) {
            let _ = snapshot.register_validator(p);
        }
    }
    Ok(Arc::new(snapshot))
}

/// Execute a closure with a read reference to the global registry.
///
/// # Non-reentrancy
///
/// See [`global_registry`]'s note: do not call this, `global_registry`, or
/// `with_global_registry_mut` from inside `f` here or inside a closure
/// passed to `with_global_registry_mut` -- `GLOBAL_REGISTRY` is a plain
/// `RwLock` with no reentrancy tracking, so a nested lock attempt on the
/// same thread deadlocks against the lock that same thread already holds.
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
///
/// # Non-reentrancy
///
/// `f` must not call [`global_registry`], [`with_global_registry`], or this
/// function again (directly or transitively) on the same thread: this
/// function holds `GLOBAL_REGISTRY`'s write lock for the duration of `f`,
/// and `std::sync::RwLock` is not reentrant, so any nested lock attempt
/// blocks forever on a lock this thread already holds.
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

/// Unregister a data source plugin from the global registry by name.
pub fn unregister_source(name: &str) -> Result<Option<Arc<dyn DataSourcePlugin>>> {
    with_global_registry_mut(|r| Ok(r.unregister_source(name)))
}

/// Unregister a data sink plugin from the global registry by name.
pub fn unregister_sink(name: &str) -> Result<Option<Arc<dyn DataSinkPlugin>>> {
    with_global_registry_mut(|r| Ok(r.unregister_sink(name)))
}

/// Unregister a transform plugin from the global registry by name.
pub fn unregister_transform(name: &str) -> Result<Option<Arc<dyn TransformPlugin>>> {
    with_global_registry_mut(|r| Ok(r.unregister_transform(name)))
}

/// Unregister an aggregator plugin from the global registry by name.
pub fn unregister_aggregator(name: &str) -> Result<Option<Arc<dyn AggregatorPlugin>>> {
    with_global_registry_mut(|r| Ok(r.unregister_aggregator(name)))
}

/// Unregister a validator plugin from the global registry by name.
pub fn unregister_validator(name: &str) -> Result<Option<Arc<dyn ValidatorPlugin>>> {
    with_global_registry_mut(|r| Ok(r.unregister_validator(name)))
}

/// Replace (hot-swap) a data source plugin in the global registry, even if
/// one is already registered under the same name.
pub fn replace_source(
    plugin: Arc<dyn DataSourcePlugin>,
) -> Result<Option<Arc<dyn DataSourcePlugin>>> {
    with_global_registry_mut(|r| Ok(r.replace_source(plugin)))
}

/// Replace (hot-swap) a data sink plugin in the global registry, even if
/// one is already registered under the same name.
pub fn replace_sink(plugin: Arc<dyn DataSinkPlugin>) -> Result<Option<Arc<dyn DataSinkPlugin>>> {
    with_global_registry_mut(|r| Ok(r.replace_sink(plugin)))
}

/// Replace (hot-swap) a transform plugin in the global registry, even if
/// one is already registered under the same name.
pub fn replace_transform(
    plugin: Arc<dyn TransformPlugin>,
) -> Result<Option<Arc<dyn TransformPlugin>>> {
    with_global_registry_mut(|r| Ok(r.replace_transform(plugin)))
}

/// Replace (hot-swap) an aggregator plugin in the global registry, even if
/// one is already registered under the same name.
pub fn replace_aggregator(
    plugin: Arc<dyn AggregatorPlugin>,
) -> Result<Option<Arc<dyn AggregatorPlugin>>> {
    with_global_registry_mut(|r| Ok(r.replace_aggregator(plugin)))
}

/// Replace (hot-swap) a validator plugin in the global registry, even if
/// one is already registered under the same name.
pub fn replace_validator(
    plugin: Arc<dyn ValidatorPlugin>,
) -> Result<Option<Arc<dyn ValidatorPlugin>>> {
    with_global_registry_mut(|r| Ok(r.replace_validator(plugin)))
}

/// Remove every plugin from the global registry.
///
/// This mutates process-wide state: prefer a local [`PluginRegistry`] for
/// test isolation over calling this from a test, since a test process that
/// runs other tests in the same process (e.g. plain `cargo test`, unlike
/// `cargo nextest`'s one-process-per-test model) would see those tests'
/// global registrations disappear out from under them.
pub fn clear_global_registry() -> Result<()> {
    with_global_registry_mut(|r| {
        r.clear();
        Ok(())
    })
}
