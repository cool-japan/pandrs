/// Plugin system for custom data sources, sinks, transforms, validators, and aggregators.
///
/// # Overview
///
/// PandRS's plugin system allows users to extend the library with custom data sources
/// (reading from proprietary formats), data sinks (writing to custom destinations),
/// transforms (row/column manipulations), aggregators, and validators.
///
/// # In-process only
///
/// A "plugin" here is a Rust value implementing one of this module's traits
/// (e.g. [`traits::TransformPlugin`]), registered by name into a
/// [`registry::PluginRegistry`]. There is no dynamic loading of external
/// `.so`/`.dll`/`.dylib` files (no `libloading` or equivalent) -- every
/// plugin is compiled into the same binary as the rest of `pandrs`. This is
/// a real, deliberate scope: the registry gives you name-addressable
/// discovery and pipeline composition (see [`pipeline::PluginPipeline`])
/// across plugins written anywhere in your crate (or a crate you depend
/// on), not runtime extensibility across process/binary boundaries.
///
/// # Non-reentrancy of the global registry
///
/// [`global_registry`], [`with_global_registry`], and
/// [`with_global_registry_mut`] each acquire a lock on the process-wide
/// registry; none of them may be called (directly or transitively) from
/// inside a closure passed to `with_global_registry_mut` on the same
/// thread, or the nested lock attempt deadlocks against the lock that
/// thread already holds. See each function's doc for details.
///
/// # Quick Start
///
/// ```rust,no_run
/// use pandrs::plugins;
/// use std::collections::HashMap;
///
/// // Initialize built-in plugins
/// plugins::register_builtin_plugins().expect("failed to register built-ins");
///
/// // Obtain a registry snapshot and build a pipeline
/// let registry = plugins::global_registry().expect("failed to get registry");
///
/// let mut src_opts = HashMap::new();
/// src_opts.insert("path".to_string(), "data.csv".to_string());
///
/// let pipeline = plugins::PluginPipeline::new(registry)
///     .source("csv_source", src_opts);
///
/// // Execute
/// if let Ok(Some(df)) = pipeline.execute() {
///     println!("Loaded {} rows", df.row_count());
/// }
/// ```
pub mod builtins;
pub mod global;
pub mod pipeline;
pub mod registry;
pub mod traits;

// Re-export the most commonly used types
pub use global::{
    clear_global_registry, global_registry, register_aggregator, register_builtin_plugins,
    register_sink, register_source, register_transform, register_validator, replace_aggregator,
    replace_sink, replace_source, replace_transform, replace_validator, unregister_aggregator,
    unregister_sink, unregister_source, unregister_transform, unregister_validator,
    with_global_registry, with_global_registry_mut,
};
pub use pipeline::{PipelineStep, PluginPipeline};
pub use registry::PluginRegistry;
pub use traits::{
    AggregatorPlugin, DataSinkPlugin, DataSourcePlugin, IssueSeverity, PluginMetadata, PluginType,
    TransformPlugin, ValidationIssue, ValidatorPlugin,
};

// Re-export built-ins for convenience
pub use builtins::{
    CsvSinkPlugin, CsvSourcePlugin, FillNaPlugin, FilterTransformPlugin, JsonSinkPlugin,
    JsonSourcePlugin, NormalizePlugin, SelectColumnsPlugin,
};
