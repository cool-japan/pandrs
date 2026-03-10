//! Builder and utility functions for hierarchical aggregation specifications
//!
//! Extracted from hierarchical_groupby.rs to comply with <2000 lines policy.

use std::sync::Arc;

use crate::dataframe::groupby::{AggFunc, CustomAggFn};
use crate::dataframe::hierarchical_groupby::HierarchicalAgg;

/// Builder for creating hierarchical aggregation specifications
pub struct HierarchicalAggBuilder {
    column: String,
    level_functions: Vec<(usize, AggFunc, String)>,
    custom_fn: Option<CustomAggFn>,
    propagate_up: bool,
    include_intermediate: bool,
}

impl HierarchicalAggBuilder {
    /// Create a new hierarchical aggregation builder
    pub fn new(column: String) -> Self {
        Self {
            column,
            level_functions: Vec::new(),
            custom_fn: None,
            propagate_up: false,
            include_intermediate: true,
        }
    }

    /// Add aggregation for a specific level
    pub fn at_level(mut self, level: usize, func: AggFunc, alias: String) -> Self {
        self.level_functions.push((level, func, alias));
        self
    }

    /// Add aggregation for all levels
    pub fn at_all_levels(mut self, func: AggFunc, base_alias: String) -> Self {
        // This would be updated when we know the number of levels
        // For now, just add to level 0
        self.level_functions.push((0, func, base_alias));
        self
    }

    /// Enable propagation of aggregations up the hierarchy
    pub fn with_propagation(mut self) -> Self {
        self.propagate_up = true;
        self
    }

    /// Set custom aggregation function
    pub fn with_custom<F>(mut self, func: F) -> Self
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        self.custom_fn = Some(Arc::new(func));
        self
    }

    /// Build the hierarchical aggregation
    pub fn build(self) -> HierarchicalAgg {
        HierarchicalAgg {
            column: self.column,
            level_functions: self.level_functions,
            custom_fn: self.custom_fn,
            propagate_up: self.propagate_up,
            include_intermediate: self.include_intermediate,
        }
    }
}

/// Utility functions for hierarchical groupby operations
pub mod utils {
    use super::*;

    /// Create a simple hierarchical aggregation
    pub fn simple_hierarchical_agg(column: &str, func: AggFunc, level: usize) -> HierarchicalAgg {
        HierarchicalAggBuilder::new(column.to_string())
            .at_level(level, func, format!("{}_{}", func.as_str(), column))
            .build()
    }

    /// Create aggregations for all levels
    pub fn all_levels_agg(column: &str, func: AggFunc, max_levels: usize) -> Vec<HierarchicalAgg> {
        (0..max_levels)
            .map(|level| simple_hierarchical_agg(column, func, level))
            .collect()
    }

    /// Create comprehensive aggregations (sum, mean, count) for a column
    pub fn comprehensive_agg(column: &str, level: usize) -> Vec<HierarchicalAgg> {
        vec![
            simple_hierarchical_agg(column, AggFunc::Sum, level),
            simple_hierarchical_agg(column, AggFunc::Mean, level),
            simple_hierarchical_agg(column, AggFunc::Count, level),
        ]
    }
}
