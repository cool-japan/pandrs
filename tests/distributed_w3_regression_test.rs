//! Wave-3 regression tests for the distributed (DataFusion) engine.
//!
//! These lock in the fixes for:
//!  * typed `collect()` round-trip (no `format!("{:?}")` / hashed-string data
//!    destruction; integer columns come back as real integers, not `"1.0"`);
//!  * equi-joins emitting the real `ON left = right` predicate rather than
//!    `ON true` (which silently produced a Cartesian product);
//!  * chained `filter().collect()` actually returning filtered rows (the lazy
//!    plan is executed against the registered base table, not an unregistered
//!    derived id);
//!  * SQL identifier / function-name injection being neutralised (identifiers
//!    are quoted; aggregate function names are allow-listed and rejected).

#![cfg(feature = "distributed")]
#![allow(clippy::result_large_err)]

use pandrs::dataframe::DataFrame;
use pandrs::distributed::expr::Expr;
use pandrs::distributed::{
    DistributedConfig, DistributedContext, ExecutionPlan, JoinType, Operation, ToDistributed,
};
use pandrs::error::Result;
use pandrs::series::Series;

fn config() -> DistributedConfig {
    DistributedConfig::new()
        .with_executor("datafusion")
        .with_concurrency(2)
}

/// A typed DataFrame round-trips through the distributed engine with real
/// values and real types — integers stay integers, floats stay floats, bools
/// stay bools, strings stay the exact string.
#[test]
fn typed_collect_round_trip() -> Result<()> {
    let mut df = DataFrame::new();
    df.add_column(
        "id".to_string(),
        Series::new(vec![1i64, 2, 3], Some("id".to_string()))?,
    )?;
    df.add_column(
        "price".to_string(),
        Series::new(vec![1.5f64, 2.5, 3.5], Some("price".to_string()))?,
    )?;
    df.add_column(
        "flag".to_string(),
        Series::new(vec![true, false, true], Some("flag".to_string()))?,
    )?;
    df.add_column(
        "name".to_string(),
        Series::new(
            vec!["ab".to_string(), "cd".to_string(), "ef".to_string()],
            Some("name".to_string()),
        )?,
    )?;

    let mut ddf = df.to_distributed(config())?;
    let out = ddf.collect()?;

    assert_eq!(out.row_count(), 3, "row count must be preserved");

    // Integers come back as a typed Series<i64> — NOT "1.0" strings and NOT
    // routed through f64. (Sort defensively: a bare SELECT * preserves order
    // in practice, but the assertion is about the multiset of values, not the
    // scan order.)
    let mut ids = out.get_column::<i64>("id")?.values().to_vec();
    ids.sort();
    assert_eq!(ids, vec![1i64, 2, 3]);

    let mut prices = out.get_column::<f64>("price")?.values().to_vec();
    prices.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    assert_eq!(prices, vec![1.5f64, 2.5, 3.5]);

    let mut flags = out.get_column::<bool>("flag")?.values().to_vec();
    flags.sort();
    assert_eq!(flags, vec![false, true, true]);

    // Strings are the exact value — not a DefaultHasher digest and not
    // `format!("{:?}")` (which would add quotes).
    let mut names = out.get_column_string_values("name")?;
    names.sort();
    assert_eq!(names, vec!["ab", "cd", "ef"]);

    Ok(())
}

/// An unparseable string column survives the round-trip verbatim (the old code
/// replaced it with a `DefaultHasher` digest cast to f64, destroying the data).
#[test]
fn unparseable_strings_survive_round_trip() -> Result<()> {
    let mut df = DataFrame::new();
    df.add_column(
        "sku".to_string(),
        Series::new(
            vec![
                "A-007".to_string(),
                "widget".to_string(),
                "zzz!".to_string(),
            ],
            Some("sku".to_string()),
        )?,
    )?;

    let mut ddf = df.to_distributed(config())?;
    let out = ddf.collect()?;

    let mut skus = out.get_column_string_values("sku")?;
    skus.sort();
    assert_eq!(skus, vec!["A-007", "widget", "zzz!"]);
    Ok(())
}

/// An inner equi-join returns only the matching rows (2), not the Cartesian
/// product (3 x 3 = 9). This is the `ON __left.k = __right.k` fix.
#[test]
fn equi_join_is_not_cartesian() -> Result<()> {
    let mut left = DataFrame::new();
    left.add_column(
        "id".to_string(),
        Series::new(vec![1i64, 2, 3], Some("id".to_string()))?,
    )?;
    left.add_column(
        "lval".to_string(),
        Series::new(vec![10i64, 20, 30], Some("lval".to_string()))?,
    )?;

    let mut right = DataFrame::new();
    right.add_column(
        "id".to_string(),
        Series::new(vec![2i64, 3, 4], Some("id".to_string()))?,
    )?;
    right.add_column(
        "rval".to_string(),
        Series::new(vec![200i64, 300, 400], Some("rval".to_string()))?,
    )?;

    let mut context = DistributedContext::new(config())?;
    context.register_dataframe("left_tbl", &left)?;
    context.register_dataframe("right_tbl", &right)?;

    let mut plan = ExecutionPlan::new("left_tbl");
    plan.add_operation(Operation::Join {
        right: "right_tbl".to_string(),
        join_type: JoinType::Inner,
        left_keys: vec!["id".to_string()],
        right_keys: vec!["id".to_string()],
    });

    let exec = context.execution_context();
    let mut guard = exec
        .lock()
        .map_err(|_| pandrs::error::Error::InvalidOperation("lock poisoned".to_string()))?;
    let result = guard.execute_plan(plan)?;

    // ids {1,2,3} INNER JOIN {2,3,4} on id => {2,3} => 2 rows. A Cartesian
    // product would be 9.
    assert_eq!(
        result.row_count(),
        2,
        "equi-join must return only matching rows, not the Cartesian product"
    );
    Ok(())
}

/// A lazily chained `filter().collect()` returns the filtered rows. Previously
/// the derived table id was never registered ("table not found") and the plan
/// input pointed at the wrong table.
#[test]
fn chained_filter_collect_returns_filtered_rows() -> Result<()> {
    let mut df = DataFrame::new();
    df.add_column(
        "id".to_string(),
        Series::new(vec![1i64, 2, 3, 4, 5], Some("id".to_string()))?,
    )?;
    df.add_column(
        "v".to_string(),
        Series::new(vec![10i64, 20, 30, 40, 50], Some("v".to_string()))?,
    )?;

    let mut ddf = df.to_distributed(config())?;
    let mut filtered = ddf.filter("id > 2")?;
    let out = filtered.collect()?;

    assert_eq!(out.row_count(), 3, "filter id > 2 keeps 3 of 5 rows");
    let ids = out.get_column::<i64>("id")?;
    let mut got = ids.values().to_vec();
    got.sort();
    assert_eq!(got, vec![3i64, 4, 5]);
    Ok(())
}

/// Branching a pipeline from the same parent does not corrupt the parent: two
/// independent filters yield independent results (the old code pushed pending
/// operations onto `self`, so the second branch inherited the first's filter).
#[test]
fn branching_pipeline_does_not_corrupt_parent() -> Result<()> {
    let mut df = DataFrame::new();
    df.add_column(
        "id".to_string(),
        Series::new(vec![1i64, 2, 3, 4, 5], Some("id".to_string()))?,
    )?;

    let mut ddf = df.to_distributed(config())?;

    let mut a = ddf.filter("id > 2")?; // 3 rows: 3,4,5
    let mut b = ddf.filter("id > 4")?; // 1 row: 5

    assert_eq!(a.collect()?.row_count(), 3);
    assert_eq!(b.collect()?.row_count(), 1);
    Ok(())
}

/// The typed expression API quotes and escapes column identifiers, so a column
/// name containing SQL metacharacters is rendered as a quoted identifier rather
/// than executable SQL.
#[test]
fn identifier_injection_is_quoted() {
    let rendered = format!("{}", Expr::col("x\"; DROP TABLE t; --"));
    // Double quotes doubled, whole thing wrapped in double quotes: inert.
    assert_eq!(rendered, "\"x\"\"; DROP TABLE t; --\"");
}

/// An out-of-allow-list aggregate function name is rejected rather than spliced
/// into SQL (function-name injection defence).
#[test]
fn malicious_aggregate_function_is_rejected() -> Result<()> {
    let mut df = DataFrame::new();
    df.add_column(
        "v".to_string(),
        Series::new(vec![1i64, 2, 3], Some("v".to_string()))?,
    )?;

    let mut ddf = df.to_distributed(config())?;
    // `aggregate` is lazy; the injection attempt surfaces at collect().
    let mut agg = ddf.aggregate(&[], &[("v", "sum); DROP TABLE t; --", "out")])?;
    let result = agg.collect();

    assert!(
        result.is_err(),
        "an unknown/malicious aggregate function must be rejected, not executed"
    );
    Ok(())
}

/// Every aggregate function on the generator's allow-list is actually accepted
/// and executed by DataFusion 53 (guards against an allow-list entry that would
/// move a rejection from validation-time to an opaque SQL error at run-time).
#[test]
fn allow_listed_aggregate_functions_execute() -> Result<()> {
    let mut df = DataFrame::new();
    df.add_column(
        "v".to_string(),
        Series::new(vec![1i64, 2, 3, 4], Some("v".to_string()))?,
    )?;

    let mut ddf = df.to_distributed(config())?;
    let mut agg = ddf.aggregate(
        &[],
        &[
            ("v", "sum", "v_sum"),
            ("v", "avg", "v_avg"),
            ("v", "mean", "v_mean"), // aliases to AVG
            ("v", "count", "v_count"),
            ("v", "min", "v_min"),
            ("v", "max", "v_max"),
            ("v", "stddev", "v_std"),
            ("v", "variance", "v_var"),
            ("v", "median", "v_median"),
        ],
    )?;
    let out = agg.collect()?;
    assert_eq!(out.row_count(), 1);
    Ok(())
}

/// Documents the empty-string / null behaviour of the `Series<String>` column
/// model: a string column round-trips its values, but because the base
/// DataFrame's `Series<String>` cannot represent a null distinctly from the
/// empty string, `""` and null are the same byte. This test records the
/// user-visible behaviour so the module documentation stays honest.
#[test]
fn empty_string_roundtrip_behaviour() -> Result<()> {
    let mut df = DataFrame::new();
    df.add_column(
        "s".to_string(),
        Series::new(
            vec!["".to_string(), "x".to_string(), "y".to_string()],
            Some("s".to_string()),
        )?,
    )?;

    let mut ddf = df.to_distributed(config())?;
    let out = ddf.collect()?;

    let mut vals = out.get_column_string_values("s")?;
    vals.sort();
    // The empty string comes back as an empty string (it is materialised as a
    // null in Arrow and back to "" — indistinguishable from a genuine null,
    // which the Series<String> model also stores as "").
    assert_eq!(vals, vec!["", "x", "y"]);
    Ok(())
}

/// A legitimate aggregate over an integer column is NOT rejected by validation
/// (guards against the validator vocabulary drifting away from the generator).
#[test]
fn integer_sum_aggregate_is_accepted() -> Result<()> {
    let mut df = DataFrame::new();
    df.add_column(
        "grp".to_string(),
        Series::new(
            vec!["a".to_string(), "a".to_string(), "b".to_string()],
            Some("grp".to_string()),
        )?,
    )?;
    df.add_column(
        "v".to_string(),
        Series::new(vec![1i64, 2, 3], Some("v".to_string()))?,
    )?;

    let mut ddf = df.to_distributed(config())?;
    let mut agg = ddf.aggregate(&["grp"], &[("v", "sum", "v_sum")])?;
    let out = agg.collect()?;

    assert_eq!(out.row_count(), 2, "two groups: a and b");
    Ok(())
}
