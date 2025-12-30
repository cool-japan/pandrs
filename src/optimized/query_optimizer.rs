//! Advanced Query Optimizer for LazyFrame
//!
//! This module provides sophisticated query optimization techniques including:
//! - Predicate pushdown: Move filters as early as possible
//! - Projection pushdown: Only load columns that are needed
//! - Operation fusion: Combine compatible operations
//! - Cost-based optimization: Reorder operations based on selectivity
//! - Common subexpression elimination

use crate::error::{Error, Result};
use std::collections::{HashMap, HashSet};

/// Optimization level for query execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OptimizationLevel {
    /// No optimization - execute operations in order
    None,
    /// Basic optimization - predicate pushdown only
    Basic,
    /// Standard optimization - pushdown + fusion
    Standard,
    /// Aggressive optimization - all techniques
    Aggressive,
}

impl Default for OptimizationLevel {
    fn default() -> Self {
        OptimizationLevel::Standard
    }
}

/// Statistics about a column for cost estimation
#[derive(Debug, Clone)]
pub struct ColumnStats {
    /// Column name
    pub name: String,
    /// Number of distinct values (cardinality)
    pub distinct_count: usize,
    /// Minimum value (if numeric)
    pub min_value: Option<f64>,
    /// Maximum value (if numeric)
    pub max_value: Option<f64>,
    /// Null count
    pub null_count: usize,
    /// Total row count
    pub row_count: usize,
    /// Estimated average value length (for strings)
    pub avg_length: Option<f64>,
}

impl ColumnStats {
    /// Create new column statistics
    pub fn new(name: String, row_count: usize) -> Self {
        ColumnStats {
            name,
            distinct_count: row_count,
            min_value: None,
            max_value: None,
            null_count: 0,
            row_count,
            avg_length: None,
        }
    }

    /// Estimate selectivity of a filter condition
    pub fn estimate_selectivity(&self, op: &FilterOp) -> f64 {
        match op {
            FilterOp::Equals(_) => {
                // Assume uniform distribution
                if self.distinct_count > 0 {
                    1.0 / self.distinct_count as f64
                } else {
                    0.1
                }
            }
            FilterOp::NotEquals(_) => {
                if self.distinct_count > 0 {
                    1.0 - (1.0 / self.distinct_count as f64)
                } else {
                    0.9
                }
            }
            FilterOp::LessThan(val) | FilterOp::LessOrEqual(val) => {
                if let (Some(min), Some(max)) = (self.min_value, self.max_value) {
                    if max > min {
                        ((val - min) / (max - min)).clamp(0.0, 1.0)
                    } else {
                        0.5
                    }
                } else {
                    0.33
                }
            }
            FilterOp::GreaterThan(val) | FilterOp::GreaterOrEqual(val) => {
                if let (Some(min), Some(max)) = (self.min_value, self.max_value) {
                    if max > min {
                        ((max - val) / (max - min)).clamp(0.0, 1.0)
                    } else {
                        0.5
                    }
                } else {
                    0.33
                }
            }
            FilterOp::Between(low, high) => {
                if let (Some(min), Some(max)) = (self.min_value, self.max_value) {
                    if max > min {
                        ((high - low) / (max - min)).clamp(0.0, 1.0)
                    } else {
                        0.5
                    }
                } else {
                    0.25
                }
            }
            FilterOp::IsNull => {
                if self.row_count > 0 {
                    self.null_count as f64 / self.row_count as f64
                } else {
                    0.01
                }
            }
            FilterOp::IsNotNull => {
                if self.row_count > 0 {
                    1.0 - (self.null_count as f64 / self.row_count as f64)
                } else {
                    0.99
                }
            }
            FilterOp::In(values) => {
                let n_values = values.len() as f64;
                if self.distinct_count > 0 {
                    (n_values / self.distinct_count as f64).min(1.0)
                } else {
                    (n_values * 0.1).min(1.0)
                }
            }
            FilterOp::Like(_) => 0.1,   // Conservative estimate for LIKE
            FilterOp::Custom(_) => 0.5, // Unknown selectivity
        }
    }
}

/// Filter operation types for optimization
#[derive(Debug, Clone)]
pub enum FilterOp {
    Equals(f64),
    NotEquals(f64),
    LessThan(f64),
    LessOrEqual(f64),
    GreaterThan(f64),
    GreaterOrEqual(f64),
    Between(f64, f64),
    IsNull,
    IsNotNull,
    In(Vec<f64>),
    Like(String),
    Custom(String),
}

/// Represents an optimizable operation in the query plan
#[derive(Debug, Clone)]
pub enum OptimizableOp {
    /// Select specific columns
    Select(Vec<String>),
    /// Filter with condition (column, operation, estimated selectivity)
    Filter {
        column: String,
        op: FilterOp,
        selectivity: f64,
    },
    /// Aggregation (group_by columns, aggregate column, function)
    Aggregate {
        group_by: Vec<String>,
        aggregates: Vec<(String, AggregateFunc)>,
    },
    /// Sort by columns
    Sort {
        columns: Vec<String>,
        ascending: Vec<bool>,
    },
    /// Join operation
    Join {
        right_columns: Vec<String>,
        left_key: String,
        right_key: String,
        join_type: JoinType,
    },
    /// Map/Transform operation
    Map {
        input_columns: Vec<String>,
        output_column: String,
    },
    /// Limit number of rows
    Limit(usize),
    /// Offset (skip rows)
    Offset(usize),
}

/// Aggregate function types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateFunc {
    Sum,
    Mean,
    Min,
    Max,
    Count,
    Std,
    Var,
    Median,
    First,
    Last,
}

/// Join types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Outer,
}

/// Query plan representation for optimization
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Operations in the plan
    pub operations: Vec<OptimizableOp>,
    /// Column statistics for cost estimation
    pub column_stats: HashMap<String, ColumnStats>,
    /// Total row count estimate
    pub estimated_rows: usize,
    /// Columns needed by the final result
    pub required_columns: HashSet<String>,
}

impl QueryPlan {
    /// Create a new query plan
    pub fn new(estimated_rows: usize) -> Self {
        QueryPlan {
            operations: Vec::new(),
            column_stats: HashMap::new(),
            estimated_rows,
            required_columns: HashSet::new(),
        }
    }

    /// Add an operation to the plan
    pub fn add_operation(&mut self, op: OptimizableOp) {
        self.operations.push(op);
    }

    /// Set column statistics
    pub fn set_column_stats(&mut self, stats: ColumnStats) {
        self.column_stats.insert(stats.name.clone(), stats);
    }

    /// Get the columns required by an operation
    fn get_required_columns(&self, op: &OptimizableOp) -> HashSet<String> {
        let mut cols = HashSet::new();
        match op {
            OptimizableOp::Select(columns) => {
                cols.extend(columns.iter().cloned());
            }
            OptimizableOp::Filter { column, .. } => {
                cols.insert(column.clone());
            }
            OptimizableOp::Aggregate {
                group_by,
                aggregates,
            } => {
                cols.extend(group_by.iter().cloned());
                for (col, _) in aggregates {
                    cols.insert(col.clone());
                }
            }
            OptimizableOp::Sort { columns, .. } => {
                cols.extend(columns.iter().cloned());
            }
            OptimizableOp::Join {
                left_key,
                right_key,
                right_columns,
                ..
            } => {
                cols.insert(left_key.clone());
                cols.insert(right_key.clone());
                cols.extend(right_columns.iter().cloned());
            }
            OptimizableOp::Map { input_columns, .. } => {
                cols.extend(input_columns.iter().cloned());
            }
            OptimizableOp::Limit(_) | OptimizableOp::Offset(_) => {}
        }
        cols
    }
}

/// Query optimizer implementation
#[derive(Debug)]
pub struct QueryOptimizer {
    /// Optimization level
    level: OptimizationLevel,
    /// Statistics about execution
    stats: OptimizerStats,
}

/// Statistics about optimizer performance
#[derive(Debug, Default, Clone)]
pub struct OptimizerStats {
    /// Number of optimizations applied
    pub optimizations_applied: usize,
    /// Number of operations before optimization
    pub operations_before: usize,
    /// Number of operations after optimization
    pub operations_after: usize,
    /// Estimated cost reduction
    pub estimated_cost_reduction: f64,
    /// Predicates pushed down
    pub predicates_pushed: usize,
    /// Projections pushed down
    pub projections_pushed: usize,
    /// Operations fused
    pub operations_fused: usize,
}

impl QueryOptimizer {
    /// Create a new query optimizer
    pub fn new(level: OptimizationLevel) -> Self {
        QueryOptimizer {
            level,
            stats: OptimizerStats::default(),
        }
    }

    /// Optimize a query plan
    pub fn optimize(&mut self, mut plan: QueryPlan) -> Result<QueryPlan> {
        self.stats = OptimizerStats::default();
        self.stats.operations_before = plan.operations.len();

        if self.level == OptimizationLevel::None {
            self.stats.operations_after = plan.operations.len();
            return Ok(plan);
        }

        // Phase 1: Compute required columns (backwards pass)
        self.compute_required_columns(&mut plan);

        // Phase 2: Predicate pushdown
        if self.level >= OptimizationLevel::Basic {
            plan = self.predicate_pushdown(plan)?;
        }

        // Phase 3: Projection pushdown
        if self.level >= OptimizationLevel::Standard {
            plan = self.projection_pushdown(plan)?;
        }

        // Phase 4: Operation fusion
        if self.level >= OptimizationLevel::Standard {
            plan = self.fuse_operations(plan)?;
        }

        // Phase 5: Cost-based reordering
        if self.level >= OptimizationLevel::Aggressive {
            plan = self.cost_based_reorder(plan)?;
        }

        self.stats.operations_after = plan.operations.len();
        Ok(plan)
    }

    /// Get optimizer statistics
    pub fn stats(&self) -> &OptimizerStats {
        &self.stats
    }

    /// Compute required columns for projection pushdown
    fn compute_required_columns(&mut self, plan: &mut QueryPlan) {
        // Start with columns required by the final operations
        let mut required = HashSet::new();

        // Walk backwards through operations
        for op in plan.operations.iter().rev() {
            let op_required = plan.get_required_columns(op);
            required.extend(op_required);
        }

        plan.required_columns = required;
    }

    /// Push predicates (filters) as close to the source as possible
    fn predicate_pushdown(&mut self, mut plan: QueryPlan) -> Result<QueryPlan> {
        let mut filters = Vec::new();
        let mut other_ops = Vec::new();

        // Separate filters from other operations
        for op in plan.operations {
            match op {
                OptimizableOp::Filter { .. } => filters.push(op),
                _ => other_ops.push(op),
            }
        }

        // Sort filters by selectivity (most selective first)
        filters.sort_by(|a, b| {
            let sel_a = if let OptimizableOp::Filter { selectivity, .. } = a {
                *selectivity
            } else {
                1.0
            };
            let sel_b = if let OptimizableOp::Filter { selectivity, .. } = b {
                *selectivity
            } else {
                1.0
            };
            sel_a
                .partial_cmp(&sel_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        self.stats.predicates_pushed = filters.len();

        // Reconstruct plan with filters first
        let mut new_ops = Vec::new();

        // Add filters that can be pushed to the front
        for filter in filters {
            if self.can_push_filter_before(&filter, &other_ops) {
                new_ops.push(filter);
                self.stats.optimizations_applied += 1;
            } else {
                // Find the right position for this filter
                let insert_pos = self.find_filter_position(&filter, &other_ops);
                other_ops.insert(insert_pos, filter);
            }
        }

        // Add remaining operations
        new_ops.extend(other_ops);

        plan.operations = new_ops;
        Ok(plan)
    }

    /// Check if a filter can be pushed before all given operations
    fn can_push_filter_before(&self, filter: &OptimizableOp, ops: &[OptimizableOp]) -> bool {
        let filter_col = if let OptimizableOp::Filter { column, .. } = filter {
            column
        } else {
            return false;
        };

        for op in ops {
            match op {
                // Can't push filter before aggregation that uses the filter column
                OptimizableOp::Aggregate { aggregates, .. } => {
                    for (col, _) in aggregates {
                        if col == filter_col {
                            return false;
                        }
                    }
                }
                // Can't push filter before a map that creates the filter column
                OptimizableOp::Map { output_column, .. } => {
                    if output_column == filter_col {
                        return false;
                    }
                }
                // Can push before most other operations
                _ => {}
            }
        }
        true
    }

    /// Find the correct position for a filter that can't be pushed all the way
    fn find_filter_position(&self, filter: &OptimizableOp, ops: &[OptimizableOp]) -> usize {
        let filter_col = if let OptimizableOp::Filter { column, .. } = filter {
            column
        } else {
            return ops.len();
        };

        for (i, op) in ops.iter().enumerate() {
            match op {
                OptimizableOp::Map { output_column, .. } => {
                    if output_column == filter_col {
                        return i + 1;
                    }
                }
                OptimizableOp::Aggregate { aggregates, .. } => {
                    for (col, _) in aggregates {
                        if col == filter_col {
                            return i + 1;
                        }
                    }
                }
                _ => {}
            }
        }
        0
    }

    /// Push column projections to load only needed columns
    fn projection_pushdown(&mut self, mut plan: QueryPlan) -> Result<QueryPlan> {
        let mut final_required = plan.required_columns.clone();

        // Walk through operations and compute required columns at each stage
        let mut required_at_stage: Vec<HashSet<String>> = Vec::new();

        for op in plan.operations.iter().rev() {
            let stage_required = final_required.clone();

            match op {
                OptimizableOp::Select(cols) => {
                    // Only need columns that are both selected and required later
                    final_required = cols
                        .iter()
                        .filter(|c| stage_required.contains(*c))
                        .cloned()
                        .collect();
                }
                OptimizableOp::Filter { column, .. } => {
                    final_required.insert(column.clone());
                }
                OptimizableOp::Aggregate {
                    group_by,
                    aggregates,
                } => {
                    final_required.extend(group_by.iter().cloned());
                    for (col, _) in aggregates {
                        final_required.insert(col.clone());
                    }
                }
                OptimizableOp::Map {
                    input_columns,
                    output_column,
                } => {
                    if stage_required.contains(output_column) {
                        final_required.extend(input_columns.iter().cloned());
                    }
                }
                OptimizableOp::Sort { columns, .. } => {
                    final_required.extend(columns.iter().cloned());
                }
                OptimizableOp::Join {
                    left_key,
                    right_key,
                    right_columns,
                    ..
                } => {
                    final_required.insert(left_key.clone());
                    final_required.insert(right_key.clone());
                    final_required.extend(right_columns.iter().cloned());
                }
                _ => {}
            }

            required_at_stage.push(stage_required);
        }

        // Add early projection if it reduces columns significantly
        if !final_required.is_empty() {
            // Check if this projection is useful (reduces columns)
            let has_wide_source = plan.column_stats.len() > final_required.len();

            if has_wide_source {
                let early_select = OptimizableOp::Select(final_required.into_iter().collect());
                plan.operations.insert(0, early_select);
                self.stats.projections_pushed += 1;
                self.stats.optimizations_applied += 1;
            }
        }

        Ok(plan)
    }

    /// Fuse compatible operations
    fn fuse_operations(&mut self, mut plan: QueryPlan) -> Result<QueryPlan> {
        let mut i = 0;
        while i < plan.operations.len().saturating_sub(1) {
            let can_fuse = match (&plan.operations[i], &plan.operations[i + 1]) {
                // Fuse consecutive filters on same column
                (
                    OptimizableOp::Filter { column: col1, .. },
                    OptimizableOp::Filter { column: col2, .. },
                ) => col1 == col2,

                // Fuse consecutive sorts (only keep last one if columns match)
                (
                    OptimizableOp::Sort { columns: cols1, .. },
                    OptimizableOp::Sort { columns: cols2, .. },
                ) => cols1 == cols2,

                // Fuse consecutive selects
                (OptimizableOp::Select(_), OptimizableOp::Select(_)) => true,

                // Fuse limit after offset
                (OptimizableOp::Offset(_), OptimizableOp::Limit(_)) => true,

                _ => false,
            };

            if can_fuse {
                // Keep only the second operation for most cases
                plan.operations.remove(i);
                self.stats.operations_fused += 1;
                self.stats.optimizations_applied += 1;
            } else {
                i += 1;
            }
        }

        Ok(plan)
    }

    /// Reorder operations based on cost estimation
    fn cost_based_reorder(&mut self, mut plan: QueryPlan) -> Result<QueryPlan> {
        // Calculate estimated cost for each filter
        let mut filter_costs: Vec<(usize, f64)> = Vec::new();

        for (i, op) in plan.operations.iter().enumerate() {
            if let OptimizableOp::Filter {
                column,
                selectivity,
                ..
            } = op
            {
                // Cost = selectivity * estimated execution cost
                // Lower selectivity filters should come first
                let execution_cost = plan
                    .column_stats
                    .get(column)
                    .map(|s| s.row_count as f64)
                    .unwrap_or(1000.0);

                let cost = selectivity * execution_cost;
                filter_costs.push((i, cost));
            }
        }

        // Sort filters by cost (lowest first)
        filter_costs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Reorder operations if beneficial
        if !filter_costs.is_empty() {
            let original_order: Vec<usize> = filter_costs.iter().map(|(i, _)| *i).collect();
            let optimal_order: Vec<usize> = {
                let mut sorted = filter_costs.clone();
                sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                sorted.iter().map(|(i, _)| *i).collect()
            };

            if original_order != optimal_order {
                // Reorder filters
                let mut new_operations = plan.operations.clone();
                for (new_pos, &old_pos) in optimal_order.iter().enumerate() {
                    if let Some((orig_idx, _)) = filter_costs.get(new_pos) {
                        new_operations[*orig_idx] = plan.operations[old_pos].clone();
                    }
                }
                plan.operations = new_operations;
                self.stats.optimizations_applied += 1;
            }
        }

        // Estimate cost reduction
        self.stats.estimated_cost_reduction = filter_costs
            .iter()
            .enumerate()
            .map(|(i, (_, cost))| {
                // Each earlier filter reduces the rows for subsequent operations
                let reduction_factor = 0.5f64.powi(i as i32);
                cost * (1.0 - reduction_factor)
            })
            .sum();

        Ok(plan)
    }
}

/// Builder for creating optimized query plans
#[derive(Debug)]
pub struct QueryPlanBuilder {
    plan: QueryPlan,
}

impl QueryPlanBuilder {
    /// Create a new query plan builder
    pub fn new(estimated_rows: usize) -> Self {
        QueryPlanBuilder {
            plan: QueryPlan::new(estimated_rows),
        }
    }

    /// Add a select operation
    pub fn select(mut self, columns: Vec<String>) -> Self {
        self.plan.add_operation(OptimizableOp::Select(columns));
        self
    }

    /// Add a filter operation
    pub fn filter(mut self, column: String, op: FilterOp) -> Self {
        // Estimate selectivity if we have stats
        let selectivity = self
            .plan
            .column_stats
            .get(&column)
            .map(|s| s.estimate_selectivity(&op))
            .unwrap_or(0.5);

        self.plan.add_operation(OptimizableOp::Filter {
            column,
            op,
            selectivity,
        });
        self
    }

    /// Add an aggregation operation
    pub fn aggregate(
        mut self,
        group_by: Vec<String>,
        aggregates: Vec<(String, AggregateFunc)>,
    ) -> Self {
        self.plan.add_operation(OptimizableOp::Aggregate {
            group_by,
            aggregates,
        });
        self
    }

    /// Add a sort operation
    pub fn sort(mut self, columns: Vec<String>, ascending: Vec<bool>) -> Self {
        self.plan
            .add_operation(OptimizableOp::Sort { columns, ascending });
        self
    }

    /// Add a limit operation
    pub fn limit(mut self, n: usize) -> Self {
        self.plan.add_operation(OptimizableOp::Limit(n));
        self
    }

    /// Add column statistics
    pub fn with_stats(mut self, stats: ColumnStats) -> Self {
        self.plan.set_column_stats(stats);
        self
    }

    /// Build the query plan
    pub fn build(self) -> QueryPlan {
        self.plan
    }
}

/// Trait for explaining query plans
pub trait Explainable {
    /// Generate a human-readable explanation of the plan
    fn explain(&self) -> String;

    /// Generate a detailed explanation with costs
    fn explain_analyze(&self) -> String;
}

impl Explainable for QueryPlan {
    fn explain(&self) -> String {
        let mut output = String::new();
        output.push_str("Query Plan:\n");
        output.push_str(&format!("  Estimated rows: {}\n", self.estimated_rows));
        output.push_str("  Operations:\n");

        for (i, op) in self.operations.iter().enumerate() {
            let op_str = match op {
                OptimizableOp::Select(cols) => format!("SELECT [{}]", cols.join(", ")),
                OptimizableOp::Filter {
                    column,
                    selectivity,
                    ..
                } => {
                    format!(
                        "FILTER {} (selectivity: {:.2}%)",
                        column,
                        selectivity * 100.0
                    )
                }
                OptimizableOp::Aggregate {
                    group_by,
                    aggregates,
                } => {
                    let agg_str: Vec<String> = aggregates
                        .iter()
                        .map(|(col, func)| format!("{:?}({})", func, col))
                        .collect();
                    if group_by.is_empty() {
                        format!("AGGREGATE {}", agg_str.join(", "))
                    } else {
                        format!(
                            "AGGREGATE BY [{}] => {}",
                            group_by.join(", "),
                            agg_str.join(", ")
                        )
                    }
                }
                OptimizableOp::Sort { columns, ascending } => {
                    let sort_str: Vec<String> = columns
                        .iter()
                        .zip(ascending)
                        .map(|(c, asc)| format!("{} {}", c, if *asc { "ASC" } else { "DESC" }))
                        .collect();
                    format!("SORT BY {}", sort_str.join(", "))
                }
                OptimizableOp::Join {
                    join_type,
                    left_key,
                    right_key,
                    ..
                } => {
                    format!("{:?} JOIN ON {} = {}", join_type, left_key, right_key)
                }
                OptimizableOp::Map {
                    input_columns,
                    output_column,
                } => {
                    format!("MAP [{}] -> {}", input_columns.join(", "), output_column)
                }
                OptimizableOp::Limit(n) => format!("LIMIT {}", n),
                OptimizableOp::Offset(n) => format!("OFFSET {}", n),
            };
            output.push_str(&format!("    {}. {}\n", i + 1, op_str));
        }

        output
    }

    fn explain_analyze(&self) -> String {
        let mut output = self.explain();
        output.push_str("\nColumn Statistics:\n");

        for (name, stats) in &self.column_stats {
            output.push_str(&format!("  {}:\n", name));
            output.push_str(&format!("    - Distinct: {}\n", stats.distinct_count));
            output.push_str(&format!("    - Nulls: {}\n", stats.null_count));
            if let Some(min) = stats.min_value {
                output.push_str(&format!("    - Min: {}\n", min));
            }
            if let Some(max) = stats.max_value {
                output.push_str(&format!("    - Max: {}\n", max));
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_plan_builder() {
        let plan = QueryPlanBuilder::new(1000)
            .select(vec!["a".to_string(), "b".to_string()])
            .filter("a".to_string(), FilterOp::GreaterThan(5.0))
            .aggregate(
                vec!["a".to_string()],
                vec![("b".to_string(), AggregateFunc::Sum)],
            )
            .build();

        assert_eq!(plan.operations.len(), 3);
        assert_eq!(plan.estimated_rows, 1000);
    }

    #[test]
    fn test_predicate_pushdown() {
        let mut stats = ColumnStats::new("price".to_string(), 10000);
        stats.min_value = Some(0.0);
        stats.max_value = Some(1000.0);

        let plan = QueryPlanBuilder::new(10000)
            .with_stats(stats)
            .select(vec!["name".to_string(), "price".to_string()])
            .filter("price".to_string(), FilterOp::GreaterThan(500.0))
            .build();

        let mut optimizer = QueryOptimizer::new(OptimizationLevel::Aggressive);
        let optimized = optimizer.optimize(plan).unwrap();

        // Filter should be moved before select
        assert!(matches!(
            optimized.operations[0],
            OptimizableOp::Filter { .. }
        ));
    }

    #[test]
    fn test_selectivity_estimation() {
        let mut stats = ColumnStats::new("category".to_string(), 1000);
        stats.distinct_count = 10;
        stats.min_value = Some(0.0);
        stats.max_value = Some(100.0);

        // Equals should have low selectivity for high-cardinality column
        let eq_sel = stats.estimate_selectivity(&FilterOp::Equals(5.0));
        assert!(eq_sel < 0.2);

        // Range should estimate based on value distribution
        let range_sel = stats.estimate_selectivity(&FilterOp::GreaterThan(50.0));
        assert!((range_sel - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_operation_fusion() {
        let plan = QueryPlanBuilder::new(1000)
            .select(vec!["a".to_string(), "b".to_string()])
            .select(vec!["a".to_string()]) // Should fuse with previous
            .build();

        let mut optimizer = QueryOptimizer::new(OptimizationLevel::Standard);
        let optimized = optimizer.optimize(plan).unwrap();

        // Two selects should be fused into one
        assert!(optimized.operations.len() < 2);
    }

    #[test]
    fn test_explain_plan() {
        let plan = QueryPlanBuilder::new(10000)
            .filter("price".to_string(), FilterOp::GreaterThan(100.0))
            .aggregate(
                vec!["category".to_string()],
                vec![("price".to_string(), AggregateFunc::Sum)],
            )
            .sort(vec!["price".to_string()], vec![false])
            .limit(10)
            .build();

        let explanation = plan.explain();
        assert!(explanation.contains("FILTER"));
        assert!(explanation.contains("AGGREGATE"));
        assert!(explanation.contains("SORT"));
        assert!(explanation.contains("LIMIT"));
    }

    #[test]
    fn test_optimizer_stats() {
        let plan = QueryPlanBuilder::new(1000)
            .filter("a".to_string(), FilterOp::Equals(1.0))
            .filter("b".to_string(), FilterOp::GreaterThan(10.0))
            .select(vec!["a".to_string(), "b".to_string()])
            .build();

        let mut optimizer = QueryOptimizer::new(OptimizationLevel::Aggressive);
        let _ = optimizer.optimize(plan).unwrap();

        let stats = optimizer.stats();
        assert!(stats.predicates_pushed > 0);
    }

    #[test]
    fn test_cost_based_reorder() {
        let mut stats_a = ColumnStats::new("a".to_string(), 10000);
        stats_a.distinct_count = 2; // Low cardinality = high selectivity for equals

        let mut stats_b = ColumnStats::new("b".to_string(), 10000);
        stats_b.distinct_count = 1000; // High cardinality = low selectivity for equals

        let plan = QueryPlanBuilder::new(10000)
            .with_stats(stats_a)
            .with_stats(stats_b)
            .filter("a".to_string(), FilterOp::Equals(1.0)) // Less selective
            .filter("b".to_string(), FilterOp::Equals(500.0)) // More selective
            .build();

        let mut optimizer = QueryOptimizer::new(OptimizationLevel::Aggressive);
        let optimized = optimizer.optimize(plan).unwrap();

        // More selective filter (b) should come first
        if let OptimizableOp::Filter { column, .. } = &optimized.operations[0] {
            assert_eq!(column, "b");
        }
    }
}
