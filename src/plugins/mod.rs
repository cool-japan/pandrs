/// Plugin system for custom data sources, sinks, transforms, validators, and aggregators.
///
/// # Overview
///
/// PandRS's plugin system allows users to extend the library with custom data sources
/// (reading from proprietary formats), data sinks (writing to custom destinations),
/// transforms (row/column manipulations), aggregators, and validators.
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
    global_registry, register_aggregator, register_builtin_plugins, register_sink, register_source,
    register_transform, register_validator, with_global_registry, with_global_registry_mut,
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
