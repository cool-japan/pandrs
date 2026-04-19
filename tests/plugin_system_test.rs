#![allow(clippy::result_large_err)]
use std::collections::HashMap;
use std::sync::Arc;

use pandrs::plugins::{
    global_registry, register_builtin_plugins, with_global_registry, with_global_registry_mut,
    CsvSinkPlugin, CsvSourcePlugin, DataSinkPlugin, DataSourcePlugin, FillNaPlugin,
    FilterTransformPlugin, IssueSeverity, NormalizePlugin, PluginMetadata, PluginPipeline,
    PluginRegistry, PluginType, SelectColumnsPlugin, TransformPlugin, ValidationIssue,
    ValidatorPlugin,
};
use pandrs::{DataFrame, Series};

// ---------------------------------------------------------------------------
// Helper: write a small CSV to a temp file and return the path
// ---------------------------------------------------------------------------
fn write_temp_csv(content: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("pandrs_plugin_test_{}.csv", uuid_simple()));
    std::fs::write(&path, content).expect("write temp csv");
    path
}

fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(12345);
    format!("{:016x}", ns as u64 ^ std::process::id() as u64)
}

// ---------------------------------------------------------------------------
// Tests: registry
// ---------------------------------------------------------------------------

#[test]
fn test_registry_new_is_empty() {
    let registry = PluginRegistry::new();
    assert_eq!(registry.plugin_count(), 0);
    assert!(registry.list_plugins().is_empty());
}

#[test]
fn test_register_and_get_source() {
    let mut registry = PluginRegistry::new();
    let plugin = CsvSourcePlugin::arc();
    registry
        .register_source(plugin)
        .expect("register csv_source");

    assert!(registry.has_source("csv_source"));
    assert!(registry.get_source("csv_source").is_some());
    assert_eq!(registry.plugin_count(), 1);
}

#[test]
fn test_register_duplicate_source_returns_error() {
    let mut registry = PluginRegistry::new();
    registry
        .register_source(CsvSourcePlugin::arc())
        .expect("first");
    let result = registry.register_source(CsvSourcePlugin::arc());
    assert!(result.is_err(), "duplicate registration should fail");
}

#[test]
fn test_register_sink_and_transform() {
    let mut registry = PluginRegistry::new();
    registry.register_sink(CsvSinkPlugin::arc()).expect("sink");
    registry
        .register_transform(FilterTransformPlugin::arc())
        .expect("transform");
    registry
        .register_transform(SelectColumnsPlugin::arc())
        .expect("select");
    registry
        .register_transform(NormalizePlugin::arc())
        .expect("normalize");
    registry
        .register_transform(FillNaPlugin::arc())
        .expect("fill_na");

    assert_eq!(registry.plugin_count(), 5);
    assert!(registry.has_sink("csv_sink"));
    assert!(registry.has_transform("filter"));
    assert!(registry.has_transform("select_columns"));
    assert!(registry.has_transform("normalize"));
    assert!(registry.has_transform("fill_na"));
}

#[test]
fn test_list_plugins_returns_all() {
    let mut registry = PluginRegistry::new();
    registry
        .register_source(CsvSourcePlugin::arc())
        .expect("source");
    registry.register_sink(CsvSinkPlugin::arc()).expect("sink");
    registry
        .register_transform(FilterTransformPlugin::arc())
        .expect("transform");

    let plugins = registry.list_plugins();
    assert_eq!(plugins.len(), 3);
}

#[test]
fn test_list_by_type() {
    let mut registry = PluginRegistry::new();
    registry
        .register_source(CsvSourcePlugin::arc())
        .expect("src");
    registry
        .register_transform(FilterTransformPlugin::arc())
        .expect("filter");
    registry
        .register_transform(SelectColumnsPlugin::arc())
        .expect("select");

    let sources = registry.list_by_type(&PluginType::DataSource);
    assert_eq!(sources.len(), 1);

    let transforms = registry.list_by_type(&PluginType::Transform);
    assert_eq!(transforms.len(), 2);
}

// ---------------------------------------------------------------------------
// Tests: built-in CSV plugin
// ---------------------------------------------------------------------------

#[test]
fn test_csv_source_plugin_reads_file() {
    let content = "name,age\nAlice,30\nBob,25\n";
    let path = write_temp_csv(content);

    let mut opts = HashMap::new();
    opts.insert("path".to_string(), path.to_string_lossy().to_string());

    let plugin = CsvSourcePlugin::new();
    let df = plugin.read(&opts).expect("read csv");
    assert_eq!(df.row_count(), 2);
    assert!(df.contains_column("name"));
    assert!(df.contains_column("age"));

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_csv_source_plugin_requires_path() {
    let plugin = CsvSourcePlugin::new();
    let opts = HashMap::new();
    let result = plugin.read(&opts);
    assert!(result.is_err());
}

#[test]
fn test_csv_sink_plugin_writes_file() {
    let mut df = DataFrame::new();
    df.add_column(
        "x".to_string(),
        Series::new(
            vec!["1".to_string(), "2".to_string()],
            Some("x".to_string()),
        )
        .expect("series"),
    )
    .expect("add");

    let mut path = std::env::temp_dir();
    path.push(format!("pandrs_sink_test_{}.csv", uuid_simple()));

    let mut opts = HashMap::new();
    opts.insert("path".to_string(), path.to_string_lossy().to_string());

    let plugin = CsvSinkPlugin::new();
    plugin.write(&df, &opts).expect("write csv");

    assert!(path.exists());
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Tests: built-in transform plugins
// ---------------------------------------------------------------------------

#[test]
fn test_filter_transform_numeric_gt() {
    let content = "value\n10\n20\n5\n30\n";
    let path = write_temp_csv(content);

    let mut src_opts = HashMap::new();
    src_opts.insert("path".to_string(), path.to_string_lossy().to_string());

    let csv_plugin = CsvSourcePlugin::new();
    let df = csv_plugin.read(&src_opts).expect("read");

    let mut opts = HashMap::new();
    opts.insert("column".to_string(), "value".to_string());
    opts.insert("operator".to_string(), "gt".to_string());
    opts.insert("value".to_string(), "10".to_string());

    let filter = FilterTransformPlugin::new();
    let result = filter.transform(df, &opts).expect("filter");

    // Should contain rows with value > 10: 20 and 30
    assert_eq!(result.row_count(), 2);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_filter_transform_string_contains() {
    let content = "name\nAlice\nBob\nCharlie\n";
    let path = write_temp_csv(content);

    let mut src_opts = HashMap::new();
    src_opts.insert("path".to_string(), path.to_string_lossy().to_string());

    let df = CsvSourcePlugin::new().read(&src_opts).expect("read");

    let mut opts = HashMap::new();
    opts.insert("column".to_string(), "name".to_string());
    opts.insert("operator".to_string(), "contains".to_string());
    opts.insert("value".to_string(), "li".to_string());

    let filter = FilterTransformPlugin::new();
    let result = filter.transform(df, &opts).expect("filter");

    // "Alice" and "Charlie" contain "li"
    assert_eq!(result.row_count(), 2);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_select_columns_plugin() {
    let content = "a,b,c\n1,2,3\n4,5,6\n";
    let path = write_temp_csv(content);

    let mut src_opts = HashMap::new();
    src_opts.insert("path".to_string(), path.to_string_lossy().to_string());

    let df = CsvSourcePlugin::new().read(&src_opts).expect("read");

    let mut opts = HashMap::new();
    opts.insert("columns".to_string(), "a,c".to_string());

    let select = SelectColumnsPlugin::new();
    let result = select.transform(df, &opts).expect("select");

    assert_eq!(result.column_count(), 2);
    assert!(result.contains_column("a"));
    assert!(result.contains_column("c"));
    assert!(!result.contains_column("b"));

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_select_columns_drop() {
    let content = "a,b,c\n1,2,3\n";
    let path = write_temp_csv(content);

    let mut src_opts = HashMap::new();
    src_opts.insert("path".to_string(), path.to_string_lossy().to_string());

    let df = CsvSourcePlugin::new().read(&src_opts).expect("read");

    let mut opts = HashMap::new();
    opts.insert("drop_columns".to_string(), "b".to_string());

    let select = SelectColumnsPlugin::new();
    let result = select.transform(df, &opts).expect("drop");

    assert_eq!(result.column_count(), 2);
    assert!(!result.contains_column("b"));

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_normalize_plugin() {
    let mut df = DataFrame::new();
    df.add_column(
        "val".to_string(),
        Series::new(vec![0.0f64, 5.0, 10.0], Some("val".to_string())).expect("s"),
    )
    .expect("add");

    let opts = HashMap::new();
    let normalize = NormalizePlugin::new();
    let result = normalize.transform(df, &opts).expect("normalize");

    assert_eq!(result.row_count(), 3);
    let values = result.get_column_numeric_values("val").expect("get");
    // min should be ~0.0, max should be ~1.0
    assert!((values[0] - 0.0).abs() < 1e-9);
    assert!((values[2] - 1.0).abs() < 1e-9);
}

#[test]
fn test_fill_na_plugin() {
    let content = "name,score\nAlice,\nBob,90\nCarol,NA\n";
    let path = write_temp_csv(content);

    let mut src_opts = HashMap::new();
    src_opts.insert("path".to_string(), path.to_string_lossy().to_string());

    let df = CsvSourcePlugin::new().read(&src_opts).expect("read");

    let mut opts = HashMap::new();
    opts.insert("value".to_string(), "0".to_string());
    opts.insert("columns".to_string(), "score".to_string());

    let fill = FillNaPlugin::new();
    let result = fill.transform(df, &opts).expect("fill_na");

    let scores = result.get_column_string_values("score").expect("scores");
    assert_eq!(scores[0], "0"); // was empty
    assert_eq!(scores[1], "90"); // unchanged
    assert_eq!(scores[2], "0"); // was NA

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Tests: pipeline
// ---------------------------------------------------------------------------

#[test]
fn test_pipeline_source_only() {
    let content = "a,b\n1,2\n3,4\n";
    let path = write_temp_csv(content);

    let mut registry = PluginRegistry::new();
    registry
        .register_source(CsvSourcePlugin::arc())
        .expect("src");

    let registry = Arc::new(registry);

    let mut src_opts = HashMap::new();
    src_opts.insert("path".to_string(), path.to_string_lossy().to_string());

    let pipeline = PluginPipeline::new(registry).source("csv_source", src_opts);
    let result = pipeline.execute().expect("execute");
    assert!(result.is_some());
    let df = result.expect("some df");
    assert_eq!(df.row_count(), 2);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_pipeline_source_then_filter_transform() {
    let content = "score\n10\n50\n80\n20\n";
    let path = write_temp_csv(content);

    let mut registry = PluginRegistry::new();
    registry
        .register_source(CsvSourcePlugin::arc())
        .expect("src");
    registry
        .register_transform(FilterTransformPlugin::arc())
        .expect("filter");

    let registry = Arc::new(registry);

    let mut src_opts = HashMap::new();
    src_opts.insert("path".to_string(), path.to_string_lossy().to_string());

    let mut filter_opts = HashMap::new();
    filter_opts.insert("column".to_string(), "score".to_string());
    filter_opts.insert("operator".to_string(), "gt".to_string());
    filter_opts.insert("value".to_string(), "25".to_string());

    let pipeline = PluginPipeline::new(registry)
        .source("csv_source", src_opts)
        .transform("filter", filter_opts);

    let result = pipeline.execute().expect("execute");
    let df = result.expect("some df");
    // 50 and 80 pass the > 25 filter
    assert_eq!(df.row_count(), 2);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_pipeline_with_sink_returns_none() {
    let content = "x\n1\n2\n";
    let src_path = write_temp_csv(content);

    let mut sink_path = std::env::temp_dir();
    sink_path.push(format!("pandrs_sink_{}.csv", uuid_simple()));

    let mut registry = PluginRegistry::new();
    registry
        .register_source(CsvSourcePlugin::arc())
        .expect("src");
    registry.register_sink(CsvSinkPlugin::arc()).expect("sink");

    let registry = Arc::new(registry);

    let mut src_opts = HashMap::new();
    src_opts.insert("path".to_string(), src_path.to_string_lossy().to_string());

    let mut sink_opts = HashMap::new();
    sink_opts.insert("path".to_string(), sink_path.to_string_lossy().to_string());

    let pipeline = PluginPipeline::new(registry)
        .source("csv_source", src_opts)
        .sink("csv_sink", sink_opts);

    let result = pipeline.execute().expect("execute");
    assert!(
        result.is_none(),
        "pipeline ending with sink should return None"
    );
    assert!(sink_path.exists(), "sink should have written the file");

    std::fs::remove_file(&src_path).ok();
    std::fs::remove_file(&sink_path).ok();
}

#[test]
fn test_pipeline_missing_plugin_returns_error() {
    let mut registry = PluginRegistry::new();
    registry
        .register_source(CsvSourcePlugin::arc())
        .expect("src");

    let registry = Arc::new(registry);

    let mut src_opts = HashMap::new();
    let dummy_csv_path = std::env::temp_dir().join("dummy.csv");
    let fallback = dummy_csv_path.to_string_lossy().into_owned();
    src_opts.insert(
        "path".to_string(),
        dummy_csv_path.to_str().unwrap_or(&fallback).to_string(),
    );

    let pipeline = PluginPipeline::new(registry)
        .source("csv_source", src_opts)
        .transform("nonexistent_plugin", HashMap::new());

    // Should fail because the plugin is not registered
    let result = pipeline.execute();
    assert!(result.is_err());
}

#[test]
fn test_pipeline_empty_fails() {
    let registry = Arc::new(PluginRegistry::new());
    let pipeline = PluginPipeline::new(registry);
    let result = pipeline.execute();
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Tests: global registry
// ---------------------------------------------------------------------------

#[test]
fn test_global_register_builtin_plugins_is_idempotent() {
    register_builtin_plugins().expect("first call");
    register_builtin_plugins().expect("second call should be idempotent");
}

#[test]
fn test_global_registry_has_builtin_plugins() {
    register_builtin_plugins().expect("register");

    with_global_registry(|r| {
        assert!(r.has_source("csv_source"));
        assert!(r.has_source("json_source"));
        assert!(r.has_sink("csv_sink"));
        assert!(r.has_transform("filter"));
        assert!(r.has_transform("select_columns"));
        assert!(r.has_transform("normalize"));
        assert!(r.has_transform("fill_na"));
        Ok(())
    })
    .expect("with_global_registry");
}

#[test]
fn test_global_registry_snapshot_for_pipeline() {
    register_builtin_plugins().expect("register");

    let content = "col\n100\n200\n50\n";
    let path = write_temp_csv(content);

    let registry = global_registry().expect("get registry");

    let mut src_opts = HashMap::new();
    src_opts.insert("path".to_string(), path.to_string_lossy().to_string());

    let mut filter_opts = HashMap::new();
    filter_opts.insert("column".to_string(), "col".to_string());
    filter_opts.insert("operator".to_string(), "gte".to_string());
    filter_opts.insert("value".to_string(), "100".to_string());

    let pipeline = PluginPipeline::new(registry)
        .source("csv_source", src_opts)
        .transform("filter", filter_opts);

    let result = pipeline.execute().expect("execute");
    let df = result.expect("some df");
    assert_eq!(df.row_count(), 2); // 100 and 200

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Tests: plugin metadata
// ---------------------------------------------------------------------------

#[test]
fn test_plugin_metadata_fields() {
    let plugin = CsvSourcePlugin::new();
    let meta = plugin.metadata();
    assert_eq!(meta.name, "csv_source");
    assert_eq!(meta.plugin_type, PluginType::DataSource);
    assert!(!meta.version.is_empty());
    assert!(!meta.description.is_empty());
    assert!(!meta.author.is_empty());
}

#[test]
fn test_plugin_type_display() {
    assert_eq!(PluginType::DataSource.to_string(), "DataSource");
    assert_eq!(PluginType::Transform.to_string(), "Transform");
    assert_eq!(PluginType::DataSink.to_string(), "DataSink");
}
