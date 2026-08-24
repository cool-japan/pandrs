//! # DataFusion Execution Engine
//!
//! This module provides an implementation of the execution engine interface
//! using Apache Arrow DataFusion.

// DataFusion conversion utilities
pub mod conversion;

#[cfg(feature = "distributed")]
use std::collections::HashMap;
#[cfg(feature = "distributed")]
use std::sync::Arc;

#[cfg(feature = "distributed")]
use crate::distributed::core::config::DistributedConfig;
#[cfg(feature = "distributed")]
use crate::distributed::core::partition::PartitionSet;
#[cfg(feature = "distributed")]
use crate::distributed::execution::{
    AggregateExpr, ExecutionContext, ExecutionEngine, ExecutionMetrics, ExecutionPlan,
    ExecutionResult, JoinType, Operation,
};
#[cfg(feature = "distributed")]
use crate::error::{Error, Result};

/// DataFusion execution engine
#[cfg(feature = "distributed")]
pub struct DataFusionEngine {
    /// Whether the engine is initialized
    initialized: bool,
    /// Configuration
    config: Option<DistributedConfig>,
}

#[cfg(feature = "distributed")]
impl DataFusionEngine {
    /// Creates a new DataFusion engine
    pub fn new() -> Self {
        Self {
            initialized: false,
            config: None,
        }
    }
}

#[cfg(feature = "distributed")]
impl ExecutionEngine for DataFusionEngine {
    fn initialize(&mut self, config: &DistributedConfig) -> Result<()> {
        self.initialized = true;
        self.config = Some(config.clone());
        Ok(())
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }

    fn create_context(&self, config: &DistributedConfig) -> Result<Box<dyn ExecutionContext>> {
        if !self.initialized {
            return Err(Error::InvalidOperation(
                "Engine not initialized".to_string(),
            ));
        }

        // Use the fallible constructor so a configured memory limit that cannot
        // be honored surfaces as an error here (this method returns `Result`)
        // rather than silently downgrading to an unlimited runtime.
        let ctx = DataFusionContext::try_new(config)?;
        Ok(Box::new(ctx))
    }

    fn clone(&self) -> Box<dyn ExecutionEngine> {
        Box::new(Self {
            initialized: self.initialized,
            config: self.config.clone(),
        })
    }
}

/// DataFusion execution context
#[cfg(feature = "distributed")]
pub struct DataFusionContext {
    /// DataFusion context
    #[cfg(feature = "distributed")]
    context: datafusion::execution::context::SessionContext,
    /// Tokio runtime that drives DataFusion's async execution.
    ///
    /// DataFusion physical operators (e.g. `RepartitionExec`) require an active
    /// Tokio reactor; driving them with `futures::executor::block_on` panics
    /// with "there is no reactor running". We own a multi-thread runtime and
    /// drive every DataFusion future on it via [`Self::block_on`].
    ///
    /// It is built lazily on first use (runtime construction is fallible and
    /// `new`/`clone` are infallible) and shared (`Arc<OnceLock<..>>`) across
    /// clones so cloned contexts reuse the same thread pool.
    runtime: Arc<std::sync::OnceLock<Arc<tokio::runtime::Runtime>>>,
    /// Configuration
    config: DistributedConfig,
    /// Registered datasets
    registered_tables: HashMap<String, PartitionSet>,
    /// Execution metrics
    metrics: ExecutionMetrics,
}

#[cfg(feature = "distributed")]
impl DataFusionContext {
    /// Creates a new DataFusion context.
    ///
    /// This constructor is infallible. When a memory limit is configured but the
    /// memory-limited `RuntimeEnv` cannot be built, it logs a warning and falls
    /// back to an unlimited runtime — the configured limit is then NOT enforced.
    /// Callers that must treat that as a hard error (so the context never runs
    /// with a weaker limit than requested) should use [`Self::try_new`], which
    /// propagates the failure instead of silently downgrading.
    pub fn new(config: &DistributedConfig) -> Self {
        let df_config = Self::build_session_config(config);
        let runtime_env = Self::build_runtime_env(config).unwrap_or_else(|e| {
            // The memory-limited runtime could not be built. Rather than
            // silently discarding the configured limit, make the downgrade
            // visible: log a warning and fall back to the default (unlimited)
            // runtime. `try_new` propagates this error instead.
            log::warn!(
                "DataFusionContext::new: {}; falling back to an unlimited RuntimeEnv \
                 — the configured memory limit is NOT enforced. Construct via \
                 DataFusionContext::try_new to treat this as a hard error instead.",
                e
            );
            std::sync::Arc::new(datafusion::execution::runtime_env::RuntimeEnv::default())
        });
        Self::assemble(config, df_config, runtime_env)
    }

    /// Creates a new DataFusion context, honoring the configured memory limit
    /// strictly.
    ///
    /// Unlike [`Self::new`], if a memory limit is configured and the
    /// memory-limited `RuntimeEnv` cannot be built, the error is propagated
    /// rather than silently discarded, so the returned context never runs with
    /// a weaker memory limit than the caller requested.
    pub fn try_new(config: &DistributedConfig) -> Result<Self> {
        let df_config = Self::build_session_config(config);
        let runtime_env = Self::build_runtime_env(config)?;
        Ok(Self::assemble(config, df_config, runtime_env))
    }

    /// Builds the DataFusion `SessionConfig` from a [`DistributedConfig`],
    /// applying the concurrency setting and the optimizer-rule translation
    /// table. Infallible.
    fn build_session_config(
        config: &DistributedConfig,
    ) -> datafusion::execution::context::SessionConfig {
        // Create DataFusion configuration
        let mut df_config = datafusion::execution::context::SessionConfig::new();

        // Set concurrency
        df_config = df_config.with_target_partitions(config.concurrency());

        // --- Optimizer rule translation table ---
        //
        // DistributedConfig rule names are mapped to typed DataFusion 53.1.0
        // `SessionConfig.options_mut().optimizer.*` fields.  Using typed field
        // access (not the string-key `.set()` form) avoids the internal
        // `.unwrap()` that `.set()` performs on unknown keys.
        //
        // Translation table:
        //   "skip_failed_rules"              → optimizer.skip_failed_rules
        //   "enable_round_robin_repartition" → optimizer.enable_round_robin_repartition
        //   "prefer_hash_join"               → optimizer.prefer_hash_join
        //   "join_reordering"                → optimizer.top_down_join_key_reordering
        //       (closest analog; controls top-down join key reordering)
        //   "filter_pushdown"   \
        //   "predicate_pushdown" > always-on in DataFusion 53 — no config toggle;
        //   "projection_pushdown"/  unmapped ≠ disabled.
        //
        // Global disable: enable_optimization() == false sets max_passes = 0
        // to suppress all logical optimizer passes.  Physical optimizations
        // (e.g. repartitioning, sort enforcement) are unaffected — DataFusion
        // does not expose a single physical-optimization kill-switch.
        if !config.enable_optimization() {
            df_config.options_mut().optimizer.max_passes = 0;
        } else {
            // Apply per-rule settings from the translation table.
            for (rule, _raw_value) in config.optimizer_rules() {
                match rule.as_str() {
                    "skip_failed_rules" => {
                        if let Some(val) = config.optimizer_rule(rule) {
                            df_config.options_mut().optimizer.skip_failed_rules = val;
                        }
                    }
                    "enable_round_robin_repartition" => {
                        if let Some(val) = config.optimizer_rule(rule) {
                            df_config
                                .options_mut()
                                .optimizer
                                .enable_round_robin_repartition = val;
                        }
                    }
                    "prefer_hash_join" => {
                        if let Some(val) = config.optimizer_rule(rule) {
                            df_config.options_mut().optimizer.prefer_hash_join = val;
                        }
                    }
                    "join_reordering" => {
                        // Maps to top_down_join_key_reordering — the closest DataFusion 53
                        // analog for join-order optimisation control.
                        if let Some(val) = config.optimizer_rule(rule) {
                            df_config
                                .options_mut()
                                .optimizer
                                .top_down_join_key_reordering = val;
                        }
                    }
                    // These rules are unconditionally enabled inside DataFusion 53;
                    // there is no per-rule toggle exposed through SessionConfig.
                    // Leaving them unmapped here does NOT disable them.
                    "filter_pushdown" | "predicate_pushdown" | "projection_pushdown" => {}
                    // Unknown rule names are silently ignored to stay forward-compatible.
                    _ => {}
                }
            }
        }

        df_config
    }

    /// Builds the DataFusion `RuntimeEnv` for this configuration.
    ///
    /// When a memory limit is configured it is applied via
    /// `RuntimeEnvBuilder::with_memory_limit` (the DataFusion 40+ API that
    /// replaced the removed `SessionConfig::with_mem_limit`). That build is
    /// fallible and any error is propagated, so the caller decides how to
    /// handle a runtime that could not honor the requested limit rather than
    /// silently receiving an unlimited one. With no memory limit configured,
    /// the default (unlimited) runtime is returned.
    fn build_runtime_env(
        config: &DistributedConfig,
    ) -> Result<Arc<datafusion::execution::runtime_env::RuntimeEnv>> {
        use datafusion::execution::runtime_env::RuntimeEnvBuilder;

        if let Some(limit) = config.memory_limit() {
            RuntimeEnvBuilder::new()
                .with_memory_limit(limit, 1.0)
                .build_arc()
                .map_err(|e| {
                    Error::InvalidOperation(format!(
                        "Failed to build DataFusion RuntimeEnv with memory limit {} bytes: {}",
                        limit, e
                    ))
                })
        } else {
            Ok(Arc::new(
                datafusion::execution::runtime_env::RuntimeEnv::default(),
            ))
        }
    }

    /// Assembles a [`DataFusionContext`] from its prepared parts. Shared by
    /// [`Self::new`] and [`Self::try_new`].
    fn assemble(
        config: &DistributedConfig,
        df_config: datafusion::execution::context::SessionConfig,
        runtime_env: Arc<datafusion::execution::runtime_env::RuntimeEnv>,
    ) -> Self {
        // Create DataFusion context with runtime
        let context = datafusion::execution::context::SessionContext::new_with_config_rt(
            df_config,
            runtime_env,
        );

        Self {
            context,
            runtime: Arc::new(std::sync::OnceLock::new()),
            config: config.clone(),
            registered_tables: HashMap::new(),
            metrics: ExecutionMetrics::new(),
        }
    }

    /// Returns the shared Tokio runtime, building it on first use.
    ///
    /// The runtime is a multi-thread runtime with all drivers enabled so that
    /// DataFusion's repartition/sort/join operators (which spawn tasks) run
    /// correctly. Construction is fallible; the built runtime is memoised in
    /// the shared `OnceLock` so subsequent calls (and clones) reuse it.
    fn runtime(&self) -> Result<Arc<tokio::runtime::Runtime>> {
        if let Some(rt) = self.runtime.get() {
            return Ok(rt.clone());
        }
        let built = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|e| {
                    Error::InvalidOperation(format!("Failed to build Tokio runtime: {}", e))
                })?,
        );
        match self.runtime.set(built.clone()) {
            Ok(()) => Ok(built),
            // Lost an initialisation race: use the runtime that won.
            Err(_) => Ok(self.runtime.get().cloned().unwrap_or(built)),
        }
    }

    /// Drives a DataFusion future to completion on the owned runtime.
    ///
    /// Returns an error instead of panicking when called from within an
    /// existing Tokio runtime (e.g. from an async handler), because
    /// `Runtime::block_on` panics with "Cannot start a runtime from within a
    /// runtime" in that situation.
    fn block_on<F: std::future::Future>(&self, future: F) -> Result<F::Output> {
        if tokio::runtime::Handle::try_current().is_ok() {
            return Err(Error::InvalidOperation(
                "The distributed DataFusion API is synchronous and cannot be called from within a \
                 Tokio runtime; call it from a synchronous context or off the async executor."
                    .to_string(),
            ));
        }
        let runtime = self.runtime()?;
        Ok(runtime.block_on(future))
    }
}

#[cfg(feature = "distributed")]
impl ExecutionContext for DataFusionContext {
    fn execute_plan(&mut self, plan: ExecutionPlan) -> Result<ExecutionResult> {
        // Validate the plan against the registered input schemas, unless the
        // caller opted out. This runs BEFORE SQL generation so that a bad
        // column reference yields a precise error rather than an opaque
        // DataFusion parse failure.
        if !self.config.skip_validation() {
            self.validate_plan(&plan)?;
        }

        // Convert the execution plan to SQL
        let sql = self.plan_to_sql(&plan)?;

        // Execute the SQL
        self.sql(&sql)
    }

    fn register_in_memory_table(&mut self, name: &str, partitions: PartitionSet) -> Result<()> {
        self.register_partitions(name, &partitions)?;
        // Store in our registry
        self.registered_tables.insert(name.to_string(), partitions);
        Ok(())
    }

    fn register_csv(&mut self, name: &str, path: &str) -> Result<()> {
        use datafusion::datasource::file_format::csv::CsvFormat;
        use datafusion::datasource::listing::ListingOptions;
        use std::sync::Arc;

        // Create CSV format options
        let file_format = Arc::new(CsvFormat::default().with_has_header(true));

        // Create listing options
        let listing_options = ListingOptions::new(file_format).with_file_extension(".csv");

        // Create table path
        let table_path = datafusion::datasource::listing::ListingTableUrl::parse(path)
            .map_err(|e| Error::InvalidValue(format!("Invalid CSV path: {}", e)))?;

        // Register CSV file with DataFusion
        self.block_on(async {
            self.context
                .register_listing_table(name, table_path, listing_options, None, None)
                .await
        })?
        .map_err(|e| Error::InvalidValue(format!("Failed to register CSV table: {}", e)))?;

        Ok(())
    }

    fn register_parquet(&mut self, name: &str, path: &str) -> Result<()> {
        use datafusion::datasource::file_format::parquet::ParquetFormat;
        use datafusion::datasource::listing::ListingOptions;
        use std::sync::Arc;

        // Create Parquet format options
        let file_format = Arc::new(ParquetFormat::default());

        // Create listing options
        let listing_options = ListingOptions::new(file_format).with_file_extension(".parquet");

        // Create table path
        let table_path = datafusion::datasource::listing::ListingTableUrl::parse(path)
            .map_err(|e| Error::InvalidValue(format!("Invalid Parquet path: {}", e)))?;

        // Register Parquet file with DataFusion
        self.block_on(async {
            self.context
                .register_listing_table(name, table_path, listing_options, None, None)
                .await
        })?
        .map_err(|e| Error::InvalidValue(format!("Failed to register Parquet table: {}", e)))?;

        Ok(())
    }

    fn sql(&mut self, query: &str) -> Result<ExecutionResult> {
        use crate::distributed::core::partition::{Partition, PartitionSet};

        // Execute SQL query using DataFusion on the owned Tokio runtime,
        // timing the whole execute+collect so the metrics are measured, not
        // fabricated.
        let start = std::time::Instant::now();
        let sql_result = self
            .block_on(async {
                let df = self.context.sql(query).await?;
                df.collect().await
            })?
            .map_err(|e| Error::InvalidValue(format!("SQL execution failed: {}", e)))?;
        let elapsed_ms = start.elapsed().as_millis() as u64;

        // Derive real execution metrics from the collected result.
        let rows: usize = sql_result.iter().map(|b| b.num_rows()).sum();
        let bytes: usize = sql_result.iter().map(|b| b.get_array_memory_size()).sum();
        let partitions_count = sql_result.len();

        let metrics = ExecutionMetrics::new()
            .with_execution_time(elapsed_ms)
            .with_rows_processed(rows)
            .with_partitions_processed(partitions_count)
            .with_bytes_processed(bytes)
            .with_bytes_output(bytes)
            .with_output_rows(rows);
        self.metrics = metrics.clone();

        // Convert result to our format
        let mut partitions = Vec::new();
        for (i, batch) in sql_result.iter().enumerate() {
            partitions.push(Arc::new(Partition::new(i, batch.clone())));
        }

        let schema = if sql_result.is_empty() {
            use arrow::datatypes::{Field, Schema};
            std::sync::Arc::new(Schema::new(vec![] as Vec<Field>))
        } else {
            sql_result[0].schema()
        };

        let partition_set = PartitionSet::new(partitions, schema.clone());

        Ok(ExecutionResult::new(partition_set, schema, metrics))
    }

    fn table_schema(&self, name: &str) -> Result<arrow::datatypes::SchemaRef> {
        // Try to get table schema from DataFusion context
        if let Some(table) = self
            .block_on(async { self.context.table(name).await.ok() })
            .ok()
            .flatten()
        {
            let schema = table.schema();
            Ok(Arc::new(schema.as_arrow().clone()))
        } else {
            // If table not found, check our registered tables
            if let Some(partition_set) = self.registered_tables.get(name) {
                partition_set.schema().cloned().ok_or_else(|| {
                    Error::InvalidValue(format!("Schema not found for table '{}'", name))
                })
            } else {
                Err(Error::InvalidValue(format!("Table '{}' not found", name)))
            }
        }
    }

    fn explain_plan(&self, plan: &ExecutionPlan, with_statistics: bool) -> Result<String> {
        // Convert execution plan to SQL and explain it
        let sql = self.plan_to_sql(plan)?;

        // DataFusion's `DFParser` accepts `EXPLAIN [ANALYZE] [VERBOSE] <stmt>`,
        // NOT the parenthesised PostgreSQL form `EXPLAIN (ANALYZE true, ...)`.
        let explain_sql = if with_statistics {
            format!("EXPLAIN ANALYZE VERBOSE {}", sql)
        } else {
            format!("EXPLAIN {}", sql)
        };

        // Execute the explain query
        let result = self
            .block_on(async {
                let df = self.context.sql(&explain_sql).await?;
                df.collect().await
            })?
            .map_err(|e| Error::InvalidValue(format!("Plan explanation failed: {}", e)))?;

        // Convert result to string
        let mut explanation = String::new();
        for batch in result {
            if let Some(column) = batch
                .column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
            {
                for i in 0..(column as &dyn arrow::array::Array).len() {
                    let line = column.value(i);
                    explanation.push_str(line);
                    explanation.push('\n');
                }
            }
        }

        Ok(explanation)
    }

    fn write_parquet(&mut self, result: &ExecutionResult, path: &str) -> Result<()> {
        use parquet::arrow::arrow_writer::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        use std::fs::File;

        // Create writer properties with compression
        let props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .build();

        // Create output file
        let file = File::create(path)
            .map_err(|e| Error::InvalidValue(format!("Failed to create Parquet file: {}", e)))?;

        // Create Arrow writer
        let mut writer = ArrowWriter::try_new(file, result.schema().clone(), Some(props))
            .map_err(|e| Error::InvalidValue(format!("Failed to create Parquet writer: {}", e)))?;

        // Write all partitions
        for partition in result.partitions().partitions() {
            if let Some(batch) = partition.data() {
                writer.write(batch).map_err(|e| {
                    Error::InvalidValue(format!("Failed to write Parquet batch: {}", e))
                })?;
            }
        }

        // Close writer
        writer
            .close()
            .map_err(|e| Error::InvalidValue(format!("Failed to close Parquet writer: {}", e)))?;

        Ok(())
    }

    fn write_csv(&mut self, result: &ExecutionResult, path: &str) -> Result<()> {
        use arrow::csv::Writer;
        use std::fs::File;

        // Create output file
        let file = File::create(path)
            .map_err(|e| Error::InvalidValue(format!("Failed to create CSV file: {}", e)))?;

        // Create CSV writer with headers
        let mut writer = Writer::new(file);

        // Write all partitions
        for partition in result.partitions().partitions() {
            if let Some(batch) = partition.data() {
                writer.write(batch).map_err(|e| {
                    Error::InvalidValue(format!("Failed to write CSV batch: {}", e))
                })?;
            }
        }

        Ok(())
    }

    fn metrics(&self) -> Result<ExecutionMetrics> {
        Ok(self.metrics.clone())
    }

    fn clone(&self) -> Box<dyn ExecutionContext> {
        // Create new context with same configuration
        let mut new_context = DataFusionContext::new(&self.config);

        // Share the same Tokio runtime rather than spinning up a new pool.
        new_context.runtime = self.runtime.clone();

        // Copy metrics
        new_context.metrics = self.metrics.clone();

        // Re-register every in-memory table into the fresh SessionContext so
        // the clone is actually queryable — a name-map-only copy leaves the
        // DataFusion catalog empty and every query fails "table not found".
        // Only record tables that registered successfully, keeping the local
        // registry consistent with the DataFusion catalog.
        for (name, partitions) in &self.registered_tables {
            if new_context.register_partitions(name, partitions).is_ok() {
                new_context
                    .registered_tables
                    .insert(name.clone(), partitions.clone());
            }
        }

        Box::new(new_context)
    }
}

impl DataFusionContext {
    /// Registers a partition set's record batches as an in-memory table in the
    /// DataFusion `SessionContext` (without touching the local registry).
    fn register_partitions(&self, name: &str, partitions: &PartitionSet) -> Result<()> {
        use datafusion::datasource::MemTable;

        let mut batches = Vec::new();
        let mut schema = None;
        for partition in partitions.partitions() {
            if let Some(data) = partition.data() {
                if schema.is_none() {
                    schema = Some(data.schema());
                }
                batches.push(data.clone());
            }
        }
        if batches.is_empty() {
            return Err(Error::InvalidValue("No data in partition set".to_string()));
        }
        let schema = schema
            .ok_or_else(|| Error::InvalidValue("No schema found in partitions".to_string()))?;
        let mem_table = MemTable::try_new(schema, vec![batches])
            .map_err(|e| Error::InvalidValue(format!("Failed to create memory table: {}", e)))?;
        self.context
            .register_table(name, Arc::new(mem_table))
            .map_err(|e| Error::InvalidValue(format!("Failed to register table: {}", e)))?;
        Ok(())
    }

    /// Validates an execution plan against the schemas of its input tables.
    ///
    /// Schemas are resolved from the DataFusion catalog (covering in-memory,
    /// CSV and Parquet registrations). When no input schema can be resolved
    /// (e.g. a lazily-registered file whose schema is not yet known) validation
    /// is skipped rather than turned into a hard failure. Empty-operation plans
    /// (a bare `SELECT * FROM t`) are always valid.
    fn validate_plan(&self, plan: &ExecutionPlan) -> Result<()> {
        use crate::distributed::schema_validator::SchemaValidator;

        if plan.operations().is_empty() {
            return Ok(());
        }

        // Gather every table the plan references: its input plus any Join right
        // side, so that Join validation has both schemas available.
        let mut table_names = vec![plan.input().to_string()];
        for op in plan.operations() {
            if let Operation::Join { right, .. } = op {
                table_names.push(right.clone());
            }
        }

        let mut validator = SchemaValidator::new();
        let mut resolved_any = false;
        for tname in &table_names {
            if let Ok(arrow_schema) = self.table_schema(tname) {
                if validator
                    .register_arrow_schema(tname.clone(), arrow_schema)
                    .is_ok()
                {
                    resolved_any = true;
                }
            }
        }

        // Nothing resolvable => cannot validate; defer to DataFusion's own
        // planning-time checks rather than rejecting a possibly-valid plan.
        if !resolved_any {
            return Ok(());
        }

        validator.validate_plan(plan)
    }

    /// Converts an `ExecutionPlan` to a single SQL string.
    ///
    /// Every operation wraps the running query as an aliased subquery, which
    /// keeps the generated SQL unambiguous (no fragile `str::replace`) and lets
    /// structured identifiers be quoted safely. Columns, aliases, table names
    /// and join keys are double-quote quoted; aggregate function names are
    /// checked against an allow-list. `Filter` conditions and `Window`
    /// expressions are caller-supplied raw SQL fragments and pass through
    /// verbatim.
    fn plan_to_sql(&self, plan: &ExecutionPlan) -> Result<String> {
        let mut sql = format!("SELECT * FROM {}", quote_ident(plan.input()));

        for operation in plan.operations() {
            match operation {
                Operation::Filter(condition) => {
                    sql = format!(
                        "SELECT * FROM ({}) AS __pandrs_src WHERE {}",
                        sql, condition
                    );
                }
                Operation::Select(columns) => {
                    let column_list = columns
                        .iter()
                        .map(|c| quote_ident(c))
                        .collect::<Vec<_>>()
                        .join(", ");
                    sql = format!("SELECT {} FROM ({}) AS __pandrs_src", column_list, sql);
                }
                Operation::Aggregate(group_by, aggregates) => {
                    sql = aggregate_sql(&sql, group_by, aggregates)?;
                }
                // GroupBy{} is the struct form of Aggregate() — same SQL.
                Operation::GroupBy { keys, aggregates } => {
                    sql = aggregate_sql(&sql, keys, aggregates)?;
                }
                Operation::OrderBy(sort_exprs) => {
                    let sort_list: Vec<String> = sort_exprs
                        .iter()
                        .map(|expr| {
                            format!(
                                "{} {}",
                                quote_ident(&expr.column),
                                if expr.ascending { "ASC" } else { "DESC" }
                            )
                        })
                        .collect();
                    sql = format!(
                        "SELECT * FROM ({}) AS __pandrs_src ORDER BY {}",
                        sql,
                        sort_list.join(", ")
                    );
                }
                Operation::Limit(n) => {
                    sql = format!("SELECT * FROM ({}) AS __pandrs_src LIMIT {}", sql, n);
                }
                Operation::Join {
                    join_type,
                    right,
                    left_keys,
                    right_keys,
                } => {
                    sql = join_sql(&sql, *join_type, right, left_keys, right_keys)?;
                }
                Operation::Distinct => {
                    sql = format!("SELECT DISTINCT * FROM ({}) AS __pandrs_src", sql);
                }
                Operation::Window(exprs) => {
                    if !exprs.is_empty() {
                        // Caller-supplied raw window SQL fragments.
                        let window_list = exprs.join(", ");
                        sql = format!("SELECT *, {} FROM ({}) AS __pandrs_src", window_list, sql);
                    }
                }
                Operation::Project(projections) => {
                    let proj_list: Vec<String> = projections
                        .iter()
                        .map(|(alias, expr)| format!("{} AS {}", expr, quote_ident(alias)))
                        .collect();
                    if !proj_list.is_empty() {
                        sql = format!(
                            "SELECT *, {} FROM ({}) AS __pandrs_src",
                            proj_list.join(", "),
                            sql
                        );
                    }
                }
                Operation::Union(other) => {
                    sql = format!("({}) UNION ALL (SELECT * FROM {})", sql, quote_ident(other));
                }
                Operation::Intersect(other) => {
                    sql = format!("({}) INTERSECT (SELECT * FROM {})", sql, quote_ident(other));
                }
                Operation::Except(other) => {
                    sql = format!("({}) EXCEPT (SELECT * FROM {})", sql, quote_ident(other));
                }
                Operation::Custom { name, params } => {
                    sql = custom_op_sql(&sql, name, params)?;
                }
            }
        }

        Ok(sql)
    }
}

/// Quotes a SQL identifier with double quotes, escaping embedded double quotes
/// by doubling them. This prevents identifier injection through the structured
/// (non-raw-SQL) operation API.
#[cfg(feature = "distributed")]
fn quote_ident(ident: &str) -> String {
    format!("\"{}\"", ident.replace('"', "\"\""))
}

/// Maps a caller-provided aggregate function name to its canonical SQL form,
/// rejecting anything not on the allow-list. This prevents function-name
/// injection (`function = "count(*)); DROP …"`) through the aggregate API.
#[cfg(feature = "distributed")]
fn sanitize_agg_function(func: &str) -> Result<&'static str> {
    match func.trim().to_lowercase().as_str() {
        "sum" => Ok("SUM"),
        "avg" | "mean" => Ok("AVG"),
        "count" => Ok("COUNT"),
        "min" => Ok("MIN"),
        "max" => Ok("MAX"),
        "stddev" | "std" => Ok("STDDEV"),
        // DataFusion 53 registers sample variance as `var` (aliases `var_samp`,
        // `var_sample`); there is NO `variance` function, so emitting
        // `VARIANCE(...)` fails planning with "Invalid function 'variance'".
        "variance" | "var" => Ok("VAR"),
        "median" => Ok("MEDIAN"),
        other => Err(Error::InvalidOperation(format!(
            "Unsupported aggregate function '{}'; allowed: sum, avg, count, min, max, stddev, \
             variance, median",
            other
        ))),
    }
}

/// Builds aggregate SQL, wrapping `inner` as an aliased subquery.
#[cfg(feature = "distributed")]
fn aggregate_sql(inner: &str, keys: &[String], aggregates: &[AggregateExpr]) -> Result<String> {
    let mut agg_exprs = Vec::with_capacity(aggregates.len());
    for agg in aggregates {
        let func = sanitize_agg_function(&agg.function)?;
        // `COUNT(*)` is special: `*` is not a quotable identifier.
        let col_sql = if agg.column.trim() == "*" {
            "*".to_string()
        } else {
            quote_ident(&agg.column)
        };
        // Use the caller-provided alias when present, else derive a stable
        // {func}_{column} name; the output identifier is always quoted.
        let alias = if agg.alias.trim().is_empty() {
            format!("{}_{}", agg.function.trim().to_lowercase(), agg.column)
        } else {
            agg.alias.clone()
        };
        agg_exprs.push(format!("{}({}) AS {}", func, col_sql, quote_ident(&alias)));
    }

    if keys.is_empty() {
        Ok(format!(
            "SELECT {} FROM ({}) AS __pandrs_src",
            agg_exprs.join(", "),
            inner
        ))
    } else {
        let group_columns = keys
            .iter()
            .map(|k| quote_ident(k))
            .collect::<Vec<_>>()
            .join(", ");
        Ok(format!(
            "SELECT {}, {} FROM ({}) AS __pandrs_src GROUP BY {}",
            group_columns,
            agg_exprs.join(", "),
            inner,
            group_columns
        ))
    }
}

/// Builds an equi-join (or cross join) SQL, wrapping `inner` as the left side.
///
/// The prior implementation emitted `ON true`, which silently produced a
/// Cartesian product; this emits the real equi-join predicate from
/// `left_keys`/`right_keys`.
#[cfg(feature = "distributed")]
fn join_sql(
    inner: &str,
    join_type: JoinType,
    right: &str,
    left_keys: &[String],
    right_keys: &[String],
) -> Result<String> {
    let right_q = quote_ident(right);

    if join_type == JoinType::Cross || left_keys.is_empty() {
        return Ok(format!(
            "SELECT * FROM ({}) AS __left CROSS JOIN {} AS __right",
            inner, right_q
        ));
    }

    if left_keys.len() != right_keys.len() {
        return Err(Error::InvalidOperation(format!(
            "Join key count mismatch: {} left key(s) vs {} right key(s)",
            left_keys.len(),
            right_keys.len()
        )));
    }

    let jt = match join_type {
        JoinType::Inner => "INNER JOIN",
        JoinType::Left => "LEFT JOIN",
        JoinType::Right => "RIGHT JOIN",
        JoinType::Full => "FULL OUTER JOIN",
        JoinType::Cross => "CROSS JOIN", // unreachable: handled above
    };

    let on = left_keys
        .iter()
        .zip(right_keys.iter())
        .map(|(l, r)| format!("__left.{} = __right.{}", quote_ident(l), quote_ident(r)))
        .collect::<Vec<_>>()
        .join(" AND ");

    Ok(format!(
        "SELECT * FROM ({}) AS __left {} {} AS __right ON {}",
        inner, jt, right_q, on
    ))
}

/// Translates a `Custom` operation to SQL. `select_expr`/`with_column` come from
/// the typed `ProjectionExt` API; their translation was previously only present
/// in the orphaned `distributed/datafusion` tree, so the whole `ProjectionExt`
/// surface failed with "cannot be translated". `create_udf` is DDL and does not
/// fit the single-SELECT model, so it is an honest `NotImplemented`.
#[cfg(feature = "distributed")]
fn custom_op_sql(
    inner: &str,
    name: &str,
    params: &std::collections::HashMap<String, String>,
) -> Result<String> {
    use crate::distributed::expr::ColumnProjection;

    match name {
        "select_expr" => {
            let projections_json = params.get("projections").ok_or_else(|| {
                Error::InvalidOperation(
                    "select_expr operation requires a 'projections' parameter".to_string(),
                )
            })?;
            let projections: Vec<ColumnProjection> = serde_json::from_str(projections_json)
                .map_err(|e| {
                    Error::InvalidOperation(format!("Failed to parse projections: {}", e))
                })?;
            if projections.is_empty() {
                return Err(Error::InvalidOperation(
                    "select_expr operation requires at least one projection".to_string(),
                ));
            }
            let cols = projections
                .iter()
                .map(|p| p.to_sql())
                .collect::<Vec<_>>()
                .join(", ");
            Ok(format!("SELECT {} FROM ({}) AS __pandrs_src", cols, inner))
        }
        "with_column" => {
            let projection_json = params.get("projection").ok_or_else(|| {
                Error::InvalidOperation(
                    "with_column operation requires a 'projection' parameter".to_string(),
                )
            })?;
            let projection: ColumnProjection =
                serde_json::from_str(projection_json).map_err(|e| {
                    Error::InvalidOperation(format!("Failed to parse projection: {}", e))
                })?;
            Ok(format!(
                "SELECT *, {} FROM ({}) AS __pandrs_src",
                projection.to_sql(),
                inner
            ))
        }
        "create_udf" => Err(Error::NotImplemented(
            "create_udf is DDL (CREATE FUNCTION) and cannot be expressed as a single query; \
             register scalar UDFs on the DataFusion SessionContext directly"
                .to_string(),
        )),
        other => Err(Error::NotImplemented(format!(
            "Custom operation '{}' cannot be translated to SQL",
            other
        ))),
    }
}

#[cfg(all(test, feature = "distributed"))]
mod tests {
    use super::*;
    use crate::distributed::core::config::DistributedConfig;

    /// Helper: build a fresh DataFusionContext and extract a read-only view of
    /// the DataFusion `SessionConfig.options().optimizer` it used.
    fn build_context_optimizer(cfg: DistributedConfig) -> datafusion::config::OptimizerOptions {
        let ctx = DataFusionContext::new(&cfg);
        ctx.context.copied_config().options().optimizer.clone()
    }

    #[test]
    fn test_optimizer_prefer_hash_join_false() {
        let cfg = DistributedConfig::new().with_optimizer_rule("prefer_hash_join", false);
        let opts = build_context_optimizer(cfg);
        assert!(
            !opts.prefer_hash_join,
            "prefer_hash_join should be false when set via DistributedConfig"
        );
    }

    #[test]
    fn test_optimizer_skip_failed_rules_true() {
        let cfg = DistributedConfig::new().with_optimizer_rule("skip_failed_rules", true);
        let opts = build_context_optimizer(cfg);
        assert!(
            opts.skip_failed_rules,
            "skip_failed_rules should be true when set via DistributedConfig"
        );
    }

    #[test]
    fn test_enable_optimization_false_sets_max_passes_zero() {
        let cfg = DistributedConfig::new().with_optimization(false);
        let opts = build_context_optimizer(cfg);
        assert_eq!(
            opts.max_passes, 0,
            "max_passes should be 0 when enable_optimization is false"
        );
    }

    #[test]
    fn test_optimizer_enable_round_robin_repartition_false() {
        let cfg =
            DistributedConfig::new().with_optimizer_rule("enable_round_robin_repartition", false);
        let opts = build_context_optimizer(cfg);
        assert!(
            !opts.enable_round_robin_repartition,
            "enable_round_robin_repartition should be false when disabled via DistributedConfig"
        );
    }

    #[test]
    fn test_optimizer_join_reordering_false() {
        let cfg = DistributedConfig::new().with_optimizer_rule("join_reordering", false);
        let opts = build_context_optimizer(cfg);
        assert!(
            !opts.top_down_join_key_reordering,
            "top_down_join_key_reordering should be false when join_reordering is disabled"
        );
    }

    #[test]
    fn test_optimizer_defaults_with_optimization_enabled() {
        // Default config has enable_optimization=true; max_passes should remain at DataFusion's
        // built-in default (3) — we must NOT zero it out.
        let cfg = DistributedConfig::new();
        let opts = build_context_optimizer(cfg);
        assert!(
            opts.max_passes > 0,
            "max_passes must remain > 0 when enable_optimization is true"
        );
    }
}
