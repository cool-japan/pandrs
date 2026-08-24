//! Wave-4 regression tests for the distributed engine cleanup (A5 + A6).
//!
//! A5 — `ProjectionExt::validate_projections` used to be a no-op that returned
//!      `Ok(())` for every input (fabrication class: it validated nothing).
//!      It now performs real validation, so these tests lock in that a
//!      genuinely-invalid projection set is rejected while a valid one passes.
//!
//! A6 — `DataFusionContext` used to silently discard a configured memory limit
//!      when the limited `RuntimeEnv` build failed, falling back to an
//!      unlimited runtime. A fallible `try_new` now honors the limit strictly
//!      and `create_context` (hence `to_distributed`) is routed through it.
//!      These tests assert the positive path builds cleanly.

#![cfg(feature = "distributed")]
#![allow(clippy::result_large_err)]

use pandrs::dataframe::DataFrame;
use pandrs::distributed::engines::datafusion::DataFusionContext;
use pandrs::distributed::expr::{
    ColumnMeta, ColumnProjection, Expr, ExprDataType, ExprSchema, ProjectionExt,
};
use pandrs::distributed::{DistributedConfig, ToDistributed};
use pandrs::error::Result;
use pandrs::series::Series;

fn config() -> DistributedConfig {
    DistributedConfig::new()
        .with_executor("datafusion")
        .with_concurrency(2)
}

/// A two-column local DataFrame converted to a distributed DataFrame, used as
/// the receiver for the `ProjectionExt::validate_projections` calls. The
/// receiver's own columns are irrelevant to validation (which is performed
/// against the explicitly-passed schema); it only needs to be a valid instance.
fn sample_ddf() -> Result<pandrs::distributed::DistributedDataFrame> {
    let mut df = DataFrame::new();
    df.add_column(
        "a".to_string(),
        Series::new(vec![1i64, 2, 3], Some("a".to_string()))?,
    )?;
    df.add_column(
        "b".to_string(),
        Series::new(vec![1.5f64, 2.5, 3.5], Some("b".to_string()))?,
    )?;
    df.to_distributed(config())
}

/// Schema with an integer column `a` and a float column `b`.
fn sample_schema() -> ExprSchema {
    let mut schema = ExprSchema::new();
    schema
        .add_column(ColumnMeta::new("a", ExprDataType::Integer, false, None))
        .add_column(ColumnMeta::new("b", ExprDataType::Float, false, None));
    schema
}

// ---------------------------------------------------------------------------
// A5: validate_projections
// ---------------------------------------------------------------------------

/// A well-formed projection set (existing columns, distinct output names,
/// non-empty aliases) validates successfully.
#[test]
fn validate_projections_accepts_valid_set() -> Result<()> {
    let ddf = sample_ddf()?;
    let schema = sample_schema();

    let projections = vec![
        ColumnProjection::column("a"),
        ColumnProjection::with_alias(Expr::col("b"), "b_out"),
    ];

    ddf.validate_projections(&projections, &schema)?;
    Ok(())
}

/// A projection referencing a column that is absent from the schema is rejected
/// — the core of the fix: the old no-op accepted this silently.
#[test]
fn validate_projections_rejects_unknown_column() -> Result<()> {
    let ddf = sample_ddf()?;
    let schema = sample_schema();

    let projections = vec![
        ColumnProjection::column("a"),
        ColumnProjection::column("does_not_exist"),
    ];

    let result = ddf.validate_projections(&projections, &schema);
    assert!(
        result.is_err(),
        "a projection over a column absent from the schema must be rejected, got {:?}",
        result
    );
    let msg = format!("{}", result.expect_err("checked is_err above"));
    assert!(
        msg.contains("does_not_exist"),
        "error should name the offending column, got: {}",
        msg
    );
    Ok(())
}

/// Two projections that resolve to the same output name collide and are
/// rejected (the underlying `ExprValidator::validate_projections` would have
/// silently overwritten one in its result map).
#[test]
fn validate_projections_rejects_duplicate_output_name() -> Result<()> {
    let ddf = sample_ddf()?;
    let schema = sample_schema();

    // Both resolve to output name "a": the first is the bare column `a`, the
    // second aliases `b` to `a`. Both expressions are individually valid.
    let projections = vec![
        ColumnProjection::column("a"),
        ColumnProjection::with_alias(Expr::col("b"), "a"),
    ];

    let result = ddf.validate_projections(&projections, &schema);
    assert!(
        result.is_err(),
        "duplicate output names must be rejected, got {:?}",
        result
    );
    let msg = format!("{}", result.expect_err("checked is_err above"));
    assert!(
        msg.contains("Duplicate") && msg.contains("'a'"),
        "error should describe the duplicate output name, got: {}",
        msg
    );
    Ok(())
}

/// A supplied alias that is empty or pure whitespace is malformed and rejected.
#[test]
fn validate_projections_rejects_blank_alias() -> Result<()> {
    let ddf = sample_ddf()?;
    let schema = sample_schema();

    let projections = vec![ColumnProjection::with_alias(Expr::col("a"), "   ")];

    let result = ddf.validate_projections(&projections, &schema);
    assert!(
        result.is_err(),
        "a blank/whitespace alias must be rejected, got {:?}",
        result
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// A6: memory-limit-honoring context construction
// ---------------------------------------------------------------------------

/// The fallible `DataFusionContext::try_new` succeeds when a valid memory limit
/// is configured — the limit is applied to the `RuntimeEnv`, not silently
/// dropped.
#[test]
fn try_new_honors_memory_limit() {
    let cfg = config().with_memory_limit(64 * 1024 * 1024); // 64 MiB
    let ctx = DataFusionContext::try_new(&cfg);
    assert!(
        ctx.is_ok(),
        "try_new must build a context that honors a valid memory limit, got {:?}",
        ctx.err()
    );
}

/// `try_new` also succeeds with no memory limit configured (the default
/// unlimited runtime path).
#[test]
fn try_new_succeeds_without_memory_limit() {
    let ctx = DataFusionContext::try_new(&config());
    assert!(
        ctx.is_ok(),
        "try_new must build a context with no memory limit, got {:?}",
        ctx.err()
    );
}

/// A `to_distributed` conversion with a memory limit set exercises the routed
/// `create_context` -> `try_new` path end-to-end and must build successfully.
#[test]
fn to_distributed_with_memory_limit_builds() -> Result<()> {
    let mut df = DataFrame::new();
    df.add_column(
        "x".to_string(),
        Series::new(vec![10i64, 20, 30], Some("x".to_string()))?,
    )?;

    let cfg = config().with_memory_limit(64 * 1024 * 1024);
    let _ddf = df.to_distributed(cfg)?;
    Ok(())
}
