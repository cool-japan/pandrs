//! New `LazyFrame` that wraps a `LogicalPlan` and integrates query optimization.
//!
//! This is the primary public API for the lazy evaluation engine.  Users build
//! up a logical query plan by chaining method calls on `LazyFrame`, and then
//! call `collect()` to run the optimizer and execute the plan against the
//! underlying `OptimizedDataFrame`.

use std::collections::HashMap;
use std::sync::Arc;

use crate::column::{
    BooleanColumn, Column, ColumnTrait, ColumnType, Float64Column, Int64Column, StringColumn,
};
use crate::core::error::{Error, Result};
use crate::optimized::dataframe::OptimizedDataFrame;
use crate::optimized::operations::JoinType;

use super::optimizer::Optimizer;
use super::plan::{AggExpr, BinaryOp, Expr, LiteralValue, LogicalPlan, UnaryOp};

// ─────────────────────────────────────────────────────────────────────────────
// GroupByBuilder — intermediate type returned by `.groupby()`
// ─────────────────────────────────────────────────────────────────────────────

/// Intermediate builder produced by `LazyFrame::groupby`.
/// Call `.agg(exprs)` to complete the aggregation.
pub struct GroupByBuilder {
    frame: LazyFrame,
    keys: Vec<Expr>,
}

impl GroupByBuilder {
    /// Specify the aggregation expressions and return a new `LazyFrame`.
    pub fn agg(self, aggs: Vec<Expr>) -> LazyFrame {
        let plan = LogicalPlan::Aggregate {
            keys: self.keys,
            aggs,
            input: Box::new(self.frame.plan),
        };
        LazyFrame { plan }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LazyFrame
// ─────────────────────────────────────────────────────────────────────────────

/// A lazily-evaluated DataFrame that builds a `LogicalPlan` and defers
/// execution until `collect()` is called.
///
/// # Examples
///
/// ```rust,no_run
/// use pandrs::compute::lazy::{LazyFrame, Expr};
/// use pandrs::optimized::dataframe::OptimizedDataFrame;
///
/// let df = OptimizedDataFrame::new();
/// let result = LazyFrame::scan(df)
///     .filter(Expr::col("age").gt(Expr::lit_int(30)))
///     .select(vec![Expr::col("name"), Expr::col("age")])
///     .collect();
/// ```
#[derive(Clone)]
pub struct LazyFrame {
    plan: LogicalPlan,
}

impl LazyFrame {
    /// Create a `LazyFrame` from an in-memory `OptimizedDataFrame`.
    pub fn scan(df: OptimizedDataFrame) -> Self {
        LazyFrame {
            plan: LogicalPlan::Scan {
                source: Arc::new(df),
                projection: None,
            },
        }
    }

    /// Create a `LazyFrame` directly from a `LogicalPlan` (advanced use).
    pub fn from_plan(plan: LogicalPlan) -> Self {
        LazyFrame { plan }
    }

    /// Consume self and return the underlying `LogicalPlan`.
    pub fn into_plan(self) -> LogicalPlan {
        self.plan
    }

    // ── Transformation methods ────────────────────────────────────────────────

    /// Filter rows where `predicate` evaluates to `true`.
    pub fn filter(self, predicate: Expr) -> Self {
        LazyFrame {
            plan: LogicalPlan::Filter {
                predicate,
                input: Box::new(self.plan),
            },
        }
    }

    /// Select (and optionally rename/transform) columns.
    pub fn select(self, exprs: Vec<Expr>) -> Self {
        LazyFrame {
            plan: LogicalPlan::Project {
                exprs,
                input: Box::new(self.plan),
            },
        }
    }

    /// Add or replace a single column computed by `expr`, naming it `name`.
    pub fn with_column(self, name: impl Into<String>, expr: Expr) -> Self {
        let name = name.into();
        let aliased = expr.alias(name);
        LazyFrame {
            plan: LogicalPlan::Project {
                exprs: vec![Expr::Wildcard, aliased],
                input: Box::new(self.plan),
            },
        }
    }

    /// Sort by one or more expressions.
    ///
    /// `ascending` must have the same length as `by`; if it is shorter, the
    /// remaining columns are sorted ascending.
    pub fn sort(self, by: Vec<Expr>, ascending: Vec<bool>) -> Self {
        let by_len = by.len();
        let mut ascending = ascending;
        ascending.resize(by_len, true);

        LazyFrame {
            plan: LogicalPlan::Sort {
                by,
                ascending,
                input: Box::new(self.plan),
            },
        }
    }

    /// Limit output to at most `n` rows.
    pub fn limit(self, n: usize) -> Self {
        LazyFrame {
            plan: LogicalPlan::Limit {
                n,
                input: Box::new(self.plan),
            },
        }
    }

    /// Join with `other` on matching key expressions.
    pub fn join(
        self,
        other: LazyFrame,
        left_on: Expr,
        right_on: Expr,
        join_type: JoinType,
    ) -> Self {
        LazyFrame {
            plan: LogicalPlan::Join {
                left: Box::new(self.plan),
                right: Box::new(other.plan),
                left_on,
                right_on,
                join_type,
            },
        }
    }

    /// Start building a group-by aggregation.
    ///
    /// Finish by calling `.agg(exprs)` on the returned `GroupByBuilder`.
    pub fn groupby(self, keys: Vec<Expr>) -> GroupByBuilder {
        GroupByBuilder { frame: self, keys }
    }

    // ── Execution ─────────────────────────────────────────────────────────────

    /// Return the unoptimized logical plan as a human-readable string.
    pub fn explain(&self) -> String {
        format!("Logical Plan (unoptimized):\n{}", self.plan.display())
    }

    /// Return the optimized logical plan as a human-readable string.
    pub fn explain_optimized(&self) -> Result<String> {
        let optimizer = Optimizer::default_rules();
        let optimized = optimizer.optimize(self.plan.clone())?;
        Ok(format!(
            "Logical Plan (optimized via [{}]):\n{}",
            optimizer.rule_names().join(", "),
            optimized.display()
        ))
    }

    /// Run the optimizer pipeline and execute the plan, returning a concrete
    /// `OptimizedDataFrame`.
    pub fn collect(self) -> Result<OptimizedDataFrame> {
        let optimizer = Optimizer::default_rules();
        let optimized_plan = optimizer.optimize(self.plan)?;
        Executor::execute(optimized_plan)
    }

    /// Execute without any optimization (useful for testing / debugging).
    pub fn collect_unoptimized(self) -> Result<OptimizedDataFrame> {
        Executor::execute(self.plan)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Plan executor
// ─────────────────────────────────────────────────────────────────────────────

/// Walks a `LogicalPlan` tree and produces a concrete `OptimizedDataFrame`.
struct Executor;

impl Executor {
    fn execute(plan: LogicalPlan) -> Result<OptimizedDataFrame> {
        match plan {
            LogicalPlan::Scan { source, projection } => {
                let df = (*source).clone();
                match projection {
                    None => Ok(df),
                    Some(cols) => {
                        let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
                        df.select(&col_refs)
                    }
                }
            }

            LogicalPlan::Filter { predicate, input } => {
                let df = Self::execute(*input)?;
                Self::apply_filter(df, &predicate)
            }

            LogicalPlan::Project { exprs, input } => {
                let df = Self::execute(*input)?;
                Self::apply_project(df, &exprs)
            }

            LogicalPlan::Aggregate { keys, aggs, input } => {
                let df = Self::execute(*input)?;
                Self::apply_aggregate(df, &keys, &aggs)
            }

            LogicalPlan::Sort {
                by,
                ascending,
                input,
            } => {
                let df = Self::execute(*input)?;
                Self::apply_sort(df, &by, &ascending)
            }

            LogicalPlan::Limit { n, input } => {
                let df = Self::execute(*input)?;
                Self::apply_limit(df, n)
            }

            LogicalPlan::Join {
                left,
                right,
                left_on,
                right_on,
                join_type,
            } => {
                let left_df = Self::execute(*left)?;
                let right_df = Self::execute(*right)?;
                Self::apply_join(left_df, right_df, &left_on, &right_on, join_type)
            }

            LogicalPlan::Union { left, right } => {
                let left_df = Self::execute(*left)?;
                let right_df = Self::execute(*right)?;
                Self::union_all(left_df, right_df)
            }
        }
    }

    // ── Filter ────────────────────────────────────────────────────────────────

    fn apply_filter(df: OptimizedDataFrame, predicate: &Expr) -> Result<OptimizedDataFrame> {
        let row_count = df.row_count();
        let mut mask = vec![true; row_count];

        for i in 0..row_count {
            let val = Self::eval_expr_at_row(&df, predicate, i)?;
            mask[i] = match val {
                ScalarValue::Boolean(b) => b,
                ScalarValue::Null => false,
                other => {
                    return Err(Error::InvalidOperation(format!(
                        "Filter predicate must evaluate to Boolean, got: {:?}",
                        other
                    )))
                }
            };
        }

        // Build new DataFrame with only the rows where mask is true
        let selected_indices: Vec<usize> = mask
            .into_iter()
            .enumerate()
            .filter(|(_, v)| *v)
            .map(|(i, _)| i)
            .collect();

        Self::select_rows(&df, &selected_indices)
    }

    // ── Project ───────────────────────────────────────────────────────────────

    fn apply_project(df: OptimizedDataFrame, exprs: &[Expr]) -> Result<OptimizedDataFrame> {
        let mut result = OptimizedDataFrame::new();
        let row_count = df.row_count();

        // Expand Wildcard before processing
        let expanded = Self::expand_exprs(&df, exprs);

        for expr in &expanded {
            let (col_name, column) = Self::eval_expr_as_column(&df, expr, row_count)?;
            result.add_column(col_name, column)?;
        }

        Ok(result)
    }

    /// Expand `Wildcard` expressions into individual column references.
    fn expand_exprs(df: &OptimizedDataFrame, exprs: &[Expr]) -> Vec<Expr> {
        let mut expanded = Vec::new();
        for expr in exprs {
            match expr {
                Expr::Wildcard => {
                    for col in df.column_names() {
                        expanded.push(Expr::Column(col.clone()));
                    }
                }
                other => expanded.push(other.clone()),
            }
        }
        expanded
    }

    /// Evaluate an expression over all rows, returning (output_name, Column).
    fn eval_expr_as_column(
        df: &OptimizedDataFrame,
        expr: &Expr,
        row_count: usize,
    ) -> Result<(String, Column)> {
        match expr {
            Expr::Column(name) => {
                let view = df.column(name)?;
                Ok((name.clone(), view.column().clone()))
            }
            Expr::Alias { expr: inner, name } => {
                let (_, col) = Self::eval_expr_as_column(df, inner, row_count)?;
                Ok((name.clone(), col))
            }
            other => {
                // Evaluate row-by-row and determine column type from first row
                let mut values: Vec<ScalarValue> = Vec::with_capacity(row_count);
                for i in 0..row_count {
                    values.push(Self::eval_expr_at_row(df, other, i)?);
                }

                let name = other
                    .output_name()
                    .unwrap_or_else(|| format!("expr_{}", other));

                let col = Self::scalar_values_to_column(values)?;
                Ok((name, col))
            }
        }
    }

    fn scalar_values_to_column(values: Vec<ScalarValue>) -> Result<Column> {
        if values.is_empty() {
            return Ok(Column::Boolean(BooleanColumn::new(vec![])));
        }

        // Determine type from first non-null value
        let first_type = values.iter().find_map(|v| match v {
            ScalarValue::Int64(_) => Some("i64"),
            ScalarValue::Float64(_) => Some("f64"),
            ScalarValue::Utf8(_) => Some("str"),
            ScalarValue::Boolean(_) => Some("bool"),
            ScalarValue::Null => None,
        });

        match first_type {
            Some("i64") => {
                let data: Vec<i64> = values
                    .into_iter()
                    .map(|v| match v {
                        ScalarValue::Int64(x) => x,
                        ScalarValue::Float64(x) => x as i64,
                        _ => 0,
                    })
                    .collect();
                Ok(Column::Int64(Int64Column::new(data)))
            }
            Some("f64") => {
                let data: Vec<f64> = values
                    .into_iter()
                    .map(|v| match v {
                        ScalarValue::Float64(x) => x,
                        ScalarValue::Int64(x) => x as f64,
                        _ => 0.0,
                    })
                    .collect();
                Ok(Column::Float64(Float64Column::new(data)))
            }
            Some("str") => {
                let data: Vec<String> = values
                    .into_iter()
                    .map(|v| match v {
                        ScalarValue::Utf8(s) => s,
                        other => format!("{:?}", other),
                    })
                    .collect();
                Ok(Column::String(StringColumn::new(data)))
            }
            Some("bool") | None => {
                let data: Vec<bool> = values
                    .into_iter()
                    .map(|v| matches!(v, ScalarValue::Boolean(true)))
                    .collect();
                Ok(Column::Boolean(BooleanColumn::new(data)))
            }
            _ => Err(Error::InvalidOperation(
                "Cannot determine column type".to_string(),
            )),
        }
    }

    // ── Aggregate ─────────────────────────────────────────────────────────────

    fn apply_aggregate(
        df: OptimizedDataFrame,
        keys: &[Expr],
        aggs: &[Expr],
    ) -> Result<OptimizedDataFrame> {
        if keys.is_empty() {
            // Global aggregation (no grouping)
            return Self::global_aggregate(&df, aggs);
        }

        // Build groups: map from group_key_string -> row indices
        let mut groups: HashMap<Vec<String>, Vec<usize>> = HashMap::new();

        for row_idx in 0..df.row_count() {
            let mut group_key = Vec::with_capacity(keys.len());
            for key_expr in keys {
                let val = Self::eval_expr_at_row(&df, key_expr, row_idx)?;
                group_key.push(val.to_sort_key());
            }
            groups.entry(group_key).or_default().push(row_idx);
        }

        let mut result = OptimizedDataFrame::new();

        // Collect group key column data and agg results
        let num_groups = groups.len();
        let key_names: Vec<String> = keys
            .iter()
            .filter_map(|e| match e {
                Expr::Column(n) => Some(n.clone()),
                Expr::Alias { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect();

        let mut key_data: Vec<Vec<String>> = vec![Vec::with_capacity(num_groups); keys.len()];
        let mut agg_data: Vec<(String, Vec<f64>)> = aggs
            .iter()
            .map(|e| {
                let name = e.output_name().unwrap_or_else(|| format!("{}", e));
                (name, Vec::with_capacity(num_groups))
            })
            .collect();

        for (group_key, row_indices) in &groups {
            // Group key column values
            for (i, key_val) in group_key.iter().enumerate() {
                if let Some(kd) = key_data.get_mut(i) {
                    kd.push(key_val.clone());
                }
            }

            // Aggregate expressions
            for (agg_idx, agg_expr) in aggs.iter().enumerate() {
                let value = Self::eval_agg_expr(&df, agg_expr, row_indices)?;
                if let Some((_, data)) = agg_data.get_mut(agg_idx) {
                    data.push(value);
                }
            }
        }

        // Add key columns as String columns
        for (i, name) in key_names.iter().enumerate() {
            let col = StringColumn::new(key_data[i].clone());
            result.add_column(name.clone(), Column::String(col))?;
        }

        // Add agg result columns as Float64 columns
        for (name, data) in agg_data {
            let col = Float64Column::new(data);
            result.add_column(name, Column::Float64(col))?;
        }

        Ok(result)
    }

    fn global_aggregate(df: &OptimizedDataFrame, aggs: &[Expr]) -> Result<OptimizedDataFrame> {
        let all_indices: Vec<usize> = (0..df.row_count()).collect();
        let mut result = OptimizedDataFrame::new();

        for agg_expr in aggs {
            let name = agg_expr
                .output_name()
                .unwrap_or_else(|| format!("{}", agg_expr));
            let value = Self::eval_agg_expr(df, agg_expr, &all_indices)?;
            let col = Float64Column::new(vec![value]);
            result.add_column(name, Column::Float64(col))?;
        }

        Ok(result)
    }

    /// Evaluate an aggregate expression over a set of row indices.
    fn eval_agg_expr(df: &OptimizedDataFrame, expr: &Expr, row_indices: &[usize]) -> Result<f64> {
        match expr {
            Expr::Agg(agg_box) => Self::eval_agg(df, agg_box, row_indices),
            Expr::Alias { expr: inner, .. } => Self::eval_agg_expr(df, inner, row_indices),
            _ => Err(Error::InvalidOperation(format!(
                "Expected an aggregation expression, got: {}",
                expr
            ))),
        }
    }

    fn eval_agg(df: &OptimizedDataFrame, agg: &AggExpr, row_indices: &[usize]) -> Result<f64> {
        match agg {
            AggExpr::Count(inner) => {
                // Count non-null values
                let mut count = 0usize;
                for &idx in row_indices {
                    let val = Self::eval_expr_at_row(df, inner, idx)?;
                    if !matches!(val, ScalarValue::Null) {
                        count += 1;
                    }
                }
                Ok(count as f64)
            }
            AggExpr::Sum(inner) => {
                let mut sum = 0.0f64;
                for &idx in row_indices {
                    let val = Self::eval_expr_at_row(df, inner, idx)?;
                    sum += val.to_f64().unwrap_or(0.0);
                }
                Ok(sum)
            }
            AggExpr::Mean(inner) => {
                let mut sum = 0.0f64;
                let mut count = 0usize;
                for &idx in row_indices {
                    let val = Self::eval_expr_at_row(df, inner, idx)?;
                    if let Some(f) = val.to_f64() {
                        sum += f;
                        count += 1;
                    }
                }
                if count == 0 {
                    Ok(f64::NAN)
                } else {
                    Ok(sum / count as f64)
                }
            }
            AggExpr::Min(inner) => {
                let mut min = f64::INFINITY;
                for &idx in row_indices {
                    let val = Self::eval_expr_at_row(df, inner, idx)?;
                    if let Some(f) = val.to_f64() {
                        if f < min {
                            min = f;
                        }
                    }
                }
                Ok(if min == f64::INFINITY { f64::NAN } else { min })
            }
            AggExpr::Max(inner) => {
                let mut max = f64::NEG_INFINITY;
                for &idx in row_indices {
                    let val = Self::eval_expr_at_row(df, inner, idx)?;
                    if let Some(f) = val.to_f64() {
                        if f > max {
                            max = f;
                        }
                    }
                }
                Ok(if max == f64::NEG_INFINITY {
                    f64::NAN
                } else {
                    max
                })
            }
            AggExpr::StdDev(inner) => {
                let mut values = Vec::with_capacity(row_indices.len());
                for &idx in row_indices {
                    let val = Self::eval_expr_at_row(df, inner, idx)?;
                    if let Some(f) = val.to_f64() {
                        values.push(f);
                    }
                }
                if values.is_empty() {
                    return Ok(f64::NAN);
                }
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance =
                    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
                Ok(variance.sqrt())
            }
        }
    }

    // ── Sort ──────────────────────────────────────────────────────────────────

    fn apply_sort(
        df: OptimizedDataFrame,
        by: &[Expr],
        ascending: &[bool],
    ) -> Result<OptimizedDataFrame> {
        let row_count = df.row_count();
        let mut sort_keys: Vec<Vec<SortKey>> = Vec::with_capacity(row_count);

        for i in 0..row_count {
            let mut row_keys = Vec::with_capacity(by.len());
            for sort_expr in by {
                let val = Self::eval_expr_at_row(&df, sort_expr, i)?;
                row_keys.push(SortKey(val.to_sort_key()));
            }
            sort_keys.push(row_keys);
        }

        let mut indices: Vec<usize> = (0..row_count).collect();
        indices.sort_by(|&a, &b| {
            for (col_idx, asc) in ascending.iter().enumerate() {
                let ka = sort_keys[a]
                    .get(col_idx)
                    .map(|s| s.0.as_str())
                    .unwrap_or("");
                let kb = sort_keys[b]
                    .get(col_idx)
                    .map(|s| s.0.as_str())
                    .unwrap_or("");
                let cmp = ka.cmp(kb);
                if cmp != std::cmp::Ordering::Equal {
                    return if *asc { cmp } else { cmp.reverse() };
                }
            }
            std::cmp::Ordering::Equal
        });

        Self::select_rows(&df, &indices)
    }

    // ── Limit ─────────────────────────────────────────────────────────────────

    fn apply_limit(df: OptimizedDataFrame, n: usize) -> Result<OptimizedDataFrame> {
        let actual = n.min(df.row_count());
        let indices: Vec<usize> = (0..actual).collect();
        Self::select_rows(&df, &indices)
    }

    // ── Join ──────────────────────────────────────────────────────────────────

    fn apply_join(
        left: OptimizedDataFrame,
        right: OptimizedDataFrame,
        left_on: &Expr,
        right_on: &Expr,
        join_type: JoinType,
    ) -> Result<OptimizedDataFrame> {
        // Extract the column names from simple Column expressions
        let left_col = match left_on {
            Expr::Column(name) => name.as_str(),
            _ => {
                return Err(Error::InvalidOperation(
                    "Join key must be a column expression".to_string(),
                ))
            }
        };
        let right_col = match right_on {
            Expr::Column(name) => name.as_str(),
            _ => {
                return Err(Error::InvalidOperation(
                    "Join key must be a column expression".to_string(),
                ))
            }
        };

        match join_type {
            JoinType::Inner => left.inner_join(&right, left_col, right_col),
            JoinType::Left => left.left_join(&right, left_col, right_col),
            JoinType::Right => right.left_join(&left, right_col, left_col),
            JoinType::Outer => left.outer_join(&right, left_col, right_col),
        }
    }

    // ── Union ─────────────────────────────────────────────────────────────────

    fn union_all(
        left: OptimizedDataFrame,
        right: OptimizedDataFrame,
    ) -> Result<OptimizedDataFrame> {
        let mut result = OptimizedDataFrame::new();

        for col_name in left.column_names() {
            let left_view = left.column(col_name)?;
            let right_view = right.column(col_name)?;

            let col = Self::concat_columns(left_view.column(), right_view.column())?;
            result.add_column(col_name.clone(), col)?;
        }

        Ok(result)
    }

    fn concat_columns(left: &Column, right: &Column) -> Result<Column> {
        match (left, right) {
            (Column::Int64(l), Column::Int64(r)) => {
                let mut data: Vec<i64> = (0..l.len())
                    .map(|i| l.get(i).ok().flatten().unwrap_or(0))
                    .collect();
                for i in 0..r.len() {
                    data.push(r.get(i).ok().flatten().unwrap_or(0));
                }
                Ok(Column::Int64(Int64Column::new(data)))
            }
            (Column::Float64(l), Column::Float64(r)) => {
                let mut data: Vec<f64> = (0..l.len())
                    .map(|i| l.get(i).ok().flatten().unwrap_or(0.0))
                    .collect();
                for i in 0..r.len() {
                    data.push(r.get(i).ok().flatten().unwrap_or(0.0));
                }
                Ok(Column::Float64(Float64Column::new(data)))
            }
            (Column::String(l), Column::String(r)) => {
                let mut data: Vec<String> = (0..l.len())
                    .map(|i| {
                        l.get(i)
                            .ok()
                            .flatten()
                            .map(|s| s.to_string())
                            .unwrap_or_default()
                    })
                    .collect();
                for i in 0..r.len() {
                    data.push(
                        r.get(i)
                            .ok()
                            .flatten()
                            .map(|s| s.to_string())
                            .unwrap_or_default(),
                    );
                }
                Ok(Column::String(StringColumn::new(data)))
            }
            (Column::Boolean(l), Column::Boolean(r)) => {
                let mut data: Vec<bool> = (0..l.len())
                    .map(|i| l.get(i).ok().flatten().unwrap_or(false))
                    .collect();
                for i in 0..r.len() {
                    data.push(r.get(i).ok().flatten().unwrap_or(false));
                }
                Ok(Column::Boolean(BooleanColumn::new(data)))
            }
            _ => Err(Error::TypeMismatch(
                "Cannot union columns of different types".to_string(),
            )),
        }
    }

    // ── Row selection helper ──────────────────────────────────────────────────

    fn select_rows(df: &OptimizedDataFrame, indices: &[usize]) -> Result<OptimizedDataFrame> {
        let mut result = OptimizedDataFrame::new();

        for col_name in df.column_names() {
            let view = df.column(col_name)?;
            let col = Self::reindex_column(view.column(), indices)?;
            result.add_column(col_name.clone(), col)?;
        }

        Ok(result)
    }

    fn reindex_column(col: &Column, indices: &[usize]) -> Result<Column> {
        match col {
            Column::Int64(c) => {
                let data: Vec<i64> = indices
                    .iter()
                    .map(|&i| c.get(i).ok().flatten().unwrap_or(0))
                    .collect();
                Ok(Column::Int64(Int64Column::new(data)))
            }
            Column::Float64(c) => {
                let data: Vec<f64> = indices
                    .iter()
                    .map(|&i| c.get(i).ok().flatten().unwrap_or(0.0))
                    .collect();
                Ok(Column::Float64(Float64Column::new(data)))
            }
            Column::String(c) => {
                let data: Vec<String> = indices
                    .iter()
                    .map(|&i| {
                        c.get(i)
                            .ok()
                            .flatten()
                            .map(|s| s.to_string())
                            .unwrap_or_default()
                    })
                    .collect();
                Ok(Column::String(StringColumn::new(data)))
            }
            Column::Boolean(c) => {
                let data: Vec<bool> = indices
                    .iter()
                    .map(|&i| c.get(i).ok().flatten().unwrap_or(false))
                    .collect();
                Ok(Column::Boolean(BooleanColumn::new(data)))
            }
        }
    }

    // ── Expression evaluator ──────────────────────────────────────────────────

    /// Evaluate an `Expr` at a single row index, returning a `ScalarValue`.
    fn eval_expr_at_row(
        df: &OptimizedDataFrame,
        expr: &Expr,
        row_idx: usize,
    ) -> Result<ScalarValue> {
        match expr {
            Expr::Column(name) => Self::read_scalar(df, name, row_idx),

            Expr::Literal(lit) => Ok(match lit {
                LiteralValue::Int64(v) => ScalarValue::Int64(*v),
                LiteralValue::Float64(v) => ScalarValue::Float64(*v),
                LiteralValue::Utf8(v) => ScalarValue::Utf8(v.clone()),
                LiteralValue::Boolean(v) => ScalarValue::Boolean(*v),
                LiteralValue::Null => ScalarValue::Null,
            }),

            Expr::BinaryOp { left, op, right } => {
                let lv = Self::eval_expr_at_row(df, left, row_idx)?;
                let rv = Self::eval_expr_at_row(df, right, row_idx)?;
                Self::eval_binary_op(lv, op, rv)
            }

            Expr::UnaryOp { op, expr: inner } => {
                let val = Self::eval_expr_at_row(df, inner, row_idx)?;
                Self::eval_unary_op(op, val)
            }

            Expr::IsNull(inner) => {
                let val = Self::eval_expr_at_row(df, inner, row_idx)?;
                Ok(ScalarValue::Boolean(matches!(val, ScalarValue::Null)))
            }

            Expr::IsNotNull(inner) => {
                let val = Self::eval_expr_at_row(df, inner, row_idx)?;
                Ok(ScalarValue::Boolean(!matches!(val, ScalarValue::Null)))
            }

            Expr::If {
                condition,
                then_expr,
                else_expr,
            } => {
                let cond = Self::eval_expr_at_row(df, condition, row_idx)?;
                match cond {
                    ScalarValue::Boolean(true) => Self::eval_expr_at_row(df, then_expr, row_idx),
                    ScalarValue::Boolean(false) | ScalarValue::Null => {
                        Self::eval_expr_at_row(df, else_expr, row_idx)
                    }
                    other => Err(Error::InvalidOperation(format!(
                        "IF condition must be Boolean, got {:?}",
                        other
                    ))),
                }
            }

            Expr::Alias { expr: inner, .. } => Self::eval_expr_at_row(df, inner, row_idx),

            Expr::Cast {
                expr: inner,
                data_type,
            } => {
                let val = Self::eval_expr_at_row(df, inner, row_idx)?;
                Self::cast_scalar(val, data_type)
            }

            Expr::Agg(_) => Err(Error::InvalidOperation(
                "Aggregate expressions cannot be evaluated row-by-row outside groupby context"
                    .to_string(),
            )),

            Expr::Wildcard => Err(Error::InvalidOperation(
                "Wildcard cannot be evaluated as a scalar".to_string(),
            )),
        }
    }

    fn read_scalar(df: &OptimizedDataFrame, col_name: &str, row_idx: usize) -> Result<ScalarValue> {
        let view = df.column(col_name)?;
        match view.column() {
            Column::Int64(c) => {
                let val = c
                    .get(row_idx)
                    .map_err(|e| Error::OperationFailed(e.to_string()))?;
                Ok(val.map(ScalarValue::Int64).unwrap_or(ScalarValue::Null))
            }
            Column::Float64(c) => {
                let val = c
                    .get(row_idx)
                    .map_err(|e| Error::OperationFailed(e.to_string()))?;
                Ok(val.map(ScalarValue::Float64).unwrap_or(ScalarValue::Null))
            }
            Column::String(c) => {
                let val = c
                    .get(row_idx)
                    .map_err(|e| Error::OperationFailed(e.to_string()))?;
                Ok(val
                    .map(|s| ScalarValue::Utf8(s.to_string()))
                    .unwrap_or(ScalarValue::Null))
            }
            Column::Boolean(c) => {
                let val = c
                    .get(row_idx)
                    .map_err(|e| Error::OperationFailed(e.to_string()))?;
                Ok(val.map(ScalarValue::Boolean).unwrap_or(ScalarValue::Null))
            }
        }
    }

    fn eval_binary_op(left: ScalarValue, op: &BinaryOp, right: ScalarValue) -> Result<ScalarValue> {
        use ScalarValue::*;
        match (left, right) {
            (Null, _) | (_, Null) => Ok(Null),
            (Int64(l), Int64(r)) => match op {
                BinaryOp::Add => Ok(Int64(l + r)),
                BinaryOp::Sub => Ok(Int64(l - r)),
                BinaryOp::Mul => Ok(Int64(l * r)),
                BinaryOp::Div => {
                    if r == 0 {
                        Err(Error::InvalidOperation("Division by zero".to_string()))
                    } else {
                        Ok(Int64(l / r))
                    }
                }
                BinaryOp::Eq => Ok(Boolean(l == r)),
                BinaryOp::NotEq => Ok(Boolean(l != r)),
                BinaryOp::Lt => Ok(Boolean(l < r)),
                BinaryOp::LtEq => Ok(Boolean(l <= r)),
                BinaryOp::Gt => Ok(Boolean(l > r)),
                BinaryOp::GtEq => Ok(Boolean(l >= r)),
                BinaryOp::And | BinaryOp::Or => Err(Error::TypeMismatch(
                    "AND/OR requires Boolean operands".to_string(),
                )),
            },
            (Float64(l), Float64(r)) => match op {
                BinaryOp::Add => Ok(Float64(l + r)),
                BinaryOp::Sub => Ok(Float64(l - r)),
                BinaryOp::Mul => Ok(Float64(l * r)),
                BinaryOp::Div => Ok(Float64(l / r)),
                BinaryOp::Eq => Ok(Boolean(l == r)),
                BinaryOp::NotEq => Ok(Boolean(l != r)),
                BinaryOp::Lt => Ok(Boolean(l < r)),
                BinaryOp::LtEq => Ok(Boolean(l <= r)),
                BinaryOp::Gt => Ok(Boolean(l > r)),
                BinaryOp::GtEq => Ok(Boolean(l >= r)),
                BinaryOp::And | BinaryOp::Or => Err(Error::TypeMismatch(
                    "AND/OR requires Boolean operands".to_string(),
                )),
            },
            (Int64(l), Float64(r)) => Self::eval_binary_op(Float64(l as f64), op, Float64(r)),
            (Float64(l), Int64(r)) => Self::eval_binary_op(Float64(l), op, Float64(r as f64)),
            (Boolean(l), Boolean(r)) => match op {
                BinaryOp::And => Ok(Boolean(l && r)),
                BinaryOp::Or => Ok(Boolean(l || r)),
                BinaryOp::Eq => Ok(Boolean(l == r)),
                BinaryOp::NotEq => Ok(Boolean(l != r)),
                _ => Err(Error::TypeMismatch(format!(
                    "Operator {:?} not supported for Boolean operands",
                    op
                ))),
            },
            (Utf8(l), Utf8(r)) => match op {
                BinaryOp::Eq => Ok(Boolean(l == r)),
                BinaryOp::NotEq => Ok(Boolean(l != r)),
                BinaryOp::Lt => Ok(Boolean(l < r)),
                BinaryOp::LtEq => Ok(Boolean(l <= r)),
                BinaryOp::Gt => Ok(Boolean(l > r)),
                BinaryOp::GtEq => Ok(Boolean(l >= r)),
                BinaryOp::Add => Ok(Utf8(format!("{}{}", l, r))),
                _ => Err(Error::TypeMismatch(format!(
                    "Operator {:?} not supported for String operands",
                    op
                ))),
            },
            (l, r) => Err(Error::TypeMismatch(format!(
                "Type mismatch in binary operation: {:?} {:?} {:?}",
                l, op, r
            ))),
        }
    }

    fn eval_unary_op(op: &UnaryOp, val: ScalarValue) -> Result<ScalarValue> {
        match (op, val) {
            (UnaryOp::Not, ScalarValue::Boolean(b)) => Ok(ScalarValue::Boolean(!b)),
            (UnaryOp::Neg, ScalarValue::Int64(v)) => Ok(ScalarValue::Int64(-v)),
            (UnaryOp::Neg, ScalarValue::Float64(v)) => Ok(ScalarValue::Float64(-v)),
            (_, ScalarValue::Null) => Ok(ScalarValue::Null),
            (op, val) => Err(Error::TypeMismatch(format!(
                "Unary operator {:?} not supported for {:?}",
                op, val
            ))),
        }
    }

    fn cast_scalar(val: ScalarValue, target: &super::plan::CastType) -> Result<ScalarValue> {
        use super::plan::CastType;
        match (val, target) {
            (ScalarValue::Null, _) => Ok(ScalarValue::Null),
            (ScalarValue::Int64(v), CastType::Float64) => Ok(ScalarValue::Float64(v as f64)),
            (ScalarValue::Int64(v), CastType::Utf8) => Ok(ScalarValue::Utf8(v.to_string())),
            (ScalarValue::Int64(v), CastType::Boolean) => Ok(ScalarValue::Boolean(v != 0)),
            (ScalarValue::Int64(v), CastType::Int64) => Ok(ScalarValue::Int64(v)),
            (ScalarValue::Float64(v), CastType::Int64) => Ok(ScalarValue::Int64(v as i64)),
            (ScalarValue::Float64(v), CastType::Utf8) => Ok(ScalarValue::Utf8(v.to_string())),
            (ScalarValue::Float64(v), CastType::Boolean) => Ok(ScalarValue::Boolean(v != 0.0)),
            (ScalarValue::Float64(v), CastType::Float64) => Ok(ScalarValue::Float64(v)),
            (ScalarValue::Boolean(b), CastType::Int64) => Ok(ScalarValue::Int64(b as i64)),
            (ScalarValue::Boolean(b), CastType::Float64) => {
                Ok(ScalarValue::Float64(if b { 1.0 } else { 0.0 }))
            }
            (ScalarValue::Boolean(b), CastType::Utf8) => Ok(ScalarValue::Utf8(b.to_string())),
            (ScalarValue::Boolean(b), CastType::Boolean) => Ok(ScalarValue::Boolean(b)),
            (ScalarValue::Utf8(s), CastType::Int64) => s
                .parse::<i64>()
                .map(ScalarValue::Int64)
                .map_err(|e| Error::cast(format!("Cannot cast '{}' to Int64: {}", s, e))),
            (ScalarValue::Utf8(s), CastType::Float64) => s
                .parse::<f64>()
                .map(ScalarValue::Float64)
                .map_err(|e| Error::cast(format!("Cannot cast '{}' to Float64: {}", s, e))),
            (ScalarValue::Utf8(s), CastType::Boolean) => {
                let b = matches!(s.to_lowercase().as_str(), "true" | "1" | "yes");
                Ok(ScalarValue::Boolean(b))
            }
            (ScalarValue::Utf8(s), CastType::Utf8) => Ok(ScalarValue::Utf8(s)),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal scalar value type
// ─────────────────────────────────────────────────────────────────────────────

/// Scalar value produced by expression evaluation.
#[derive(Debug, Clone, PartialEq)]
enum ScalarValue {
    Int64(i64),
    Float64(f64),
    Utf8(String),
    Boolean(bool),
    Null,
}

impl ScalarValue {
    /// Convert to f64 if numeric; returns None for non-numeric types.
    fn to_f64(&self) -> Option<f64> {
        match self {
            ScalarValue::Int64(v) => Some(*v as f64),
            ScalarValue::Float64(v) => Some(*v),
            ScalarValue::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    /// Convert to a string that can be used as a sort key.
    fn to_sort_key(&self) -> String {
        match self {
            // Pad integers to fixed width for lexicographic sort correctness
            ScalarValue::Int64(v) => format!("{:020}", v),
            // Pad floats — prefix with sign indicator
            ScalarValue::Float64(v) => format!("{:030.10}", v),
            ScalarValue::Utf8(s) => s.clone(),
            ScalarValue::Boolean(b) => b.to_string(),
            ScalarValue::Null => String::new(),
        }
    }
}

/// Simple sort key wrapper (exists to implement Ord cleanly).
struct SortKey(String);

// ─────────────────────────────────────────────────────────────────────────────
// Error helper
// ─────────────────────────────────────────────────────────────────────────────

impl Error {
    fn cast(msg: String) -> Self {
        Error::Cast(msg)
    }
}
