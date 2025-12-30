//! Enhanced GroupBy functionality for DataFrames with pandas-like named aggregations
//!
//! This module provides comprehensive groupby operations with support for:
//! - Named aggregations (similar to pandas .agg({'col': {'alias': 'func'}}) syntax)
//! - Multiple aggregation functions per column
//! - Custom aggregation functions
//! - Multi-level column names for results

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::base::Series;

/// Enumeration representing aggregation operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunc {
    /// Sum aggregation
    Sum,
    /// Mean (average) aggregation
    Mean,
    /// Minimum value aggregation
    Min,
    /// Maximum value aggregation
    Max,
    /// Count of non-null values
    Count,
    /// Standard deviation
    Std,
    /// Variance
    Var,
    /// Median value
    Median,
    /// First value
    First,
    /// Last value
    Last,
    /// Count of unique values
    Nunique,
    /// Custom aggregation function
    Custom,
}

impl AggFunc {
    /// Get the string representation of the aggregation function
    pub fn as_str(&self) -> &'static str {
        match self {
            AggFunc::Sum => "sum",
            AggFunc::Mean => "mean",
            AggFunc::Min => "min",
            AggFunc::Max => "max",
            AggFunc::Count => "count",
            AggFunc::Std => "std",
            AggFunc::Var => "var",
            AggFunc::Median => "median",
            AggFunc::First => "first",
            AggFunc::Last => "last",
            AggFunc::Nunique => "nunique",
            AggFunc::Custom => "custom",
        }
    }
}

/// Type for custom aggregation functions
pub type CustomAggFn = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;

/// Specification for a named aggregation
#[derive(Clone)]
pub struct NamedAgg {
    /// Column to aggregate
    pub column: String,
    /// Aggregation function to apply
    pub func: AggFunc,
    /// Alias for the result column
    pub alias: String,
    /// Optional custom function (required when func is Custom)
    pub custom_fn: Option<CustomAggFn>,
}

impl std::fmt::Debug for NamedAgg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NamedAgg")
            .field("column", &self.column)
            .field("func", &self.func)
            .field("alias", &self.alias)
            .field(
                "custom_fn",
                &self.custom_fn.as_ref().map(|_| "<custom_function>"),
            )
            .finish()
    }
}

impl NamedAgg {
    /// Create a new named aggregation
    pub fn new(column: String, func: AggFunc, alias: String) -> Self {
        Self {
            column,
            func,
            alias,
            custom_fn: None,
        }
    }

    /// Create a named aggregation with a custom function
    pub fn custom<F>(column: String, alias: String, func: F) -> Self
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        Self {
            column,
            func: AggFunc::Custom,
            alias,
            custom_fn: Some(Arc::new(func)),
        }
    }
}

/// Builder for creating multiple named aggregations for a single column
pub struct ColumnAggBuilder {
    column: String,
    aggregations: Vec<(AggFunc, String, Option<CustomAggFn>)>,
}

impl std::fmt::Debug for ColumnAggBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColumnAggBuilder")
            .field("column", &self.column)
            .field(
                "aggregations",
                &self
                    .aggregations
                    .iter()
                    .map(|(func, alias, custom_fn)| {
                        (func, alias, custom_fn.as_ref().map(|_| "<custom_function>"))
                    })
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl ColumnAggBuilder {
    /// Create a new column aggregation builder
    pub fn new(column: String) -> Self {
        Self {
            column,
            aggregations: Vec::new(),
        }
    }

    /// Add a standard aggregation function with an alias
    pub fn agg(mut self, func: AggFunc, alias: String) -> Self {
        self.aggregations.push((func, alias, None));
        self
    }

    /// Add a custom aggregation function with an alias
    pub fn custom<F>(mut self, alias: String, func: F) -> Self
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        self.aggregations
            .push((AggFunc::Custom, alias, Some(Arc::new(func))));
        self
    }

    /// Build the named aggregations
    pub fn build(self) -> Vec<NamedAgg> {
        self.aggregations
            .into_iter()
            .map(|(func, alias, custom_fn)| NamedAgg {
                column: self.column.clone(),
                func,
                alias,
                custom_fn,
            })
            .collect()
    }
}

/// DataFrame GroupBy with enhanced functionality
#[derive(Debug)]
pub struct DataFrameGroupBy {
    /// Original DataFrame
    df: DataFrame,
    /// Grouping column(s)
    group_by_columns: Vec<String>,
    /// Grouped indices for each group key
    groups: HashMap<Vec<String>, Vec<usize>>,
}

impl DataFrameGroupBy {
    /// Create a new DataFrame GroupBy
    pub fn new(df: DataFrame, group_by_columns: Vec<String>) -> Result<Self> {
        // Verify that all grouping columns exist
        for col in &group_by_columns {
            if !df.contains_column(col) {
                return Err(Error::ColumnNotFound(col.clone()));
            }
        }

        // Create groups based on the grouping columns
        let mut groups: HashMap<Vec<String>, Vec<usize>> = HashMap::new();

        for row_idx in 0..df.row_count() {
            let mut key = Vec::with_capacity(group_by_columns.len());

            for col_name in &group_by_columns {
                let col_values = df.get_column_string_values(col_name)?;
                if row_idx < col_values.len() {
                    key.push(col_values[row_idx].clone());
                } else {
                    key.push("NULL".to_string());
                }
            }

            groups.entry(key).or_default().push(row_idx);
        }

        Ok(Self {
            df,
            group_by_columns,
            groups,
        })
    }

    /// Get the number of groups
    pub fn ngroups(&self) -> usize {
        self.groups.len()
    }

    /// Get the size of each group
    pub fn size(&self) -> Result<DataFrame> {
        let mut result = DataFrame::new();

        let mut group_keys = Vec::new();
        let mut sizes = Vec::new();

        for (key, indices) in &self.groups {
            let key_str = key.join("_");
            group_keys.push(key_str);
            sizes.push(indices.len().to_string());
        }

        let group_series = Series::new(group_keys, Some("group".to_string()))?;
        let size_series = Series::new(sizes, Some("size".to_string()))?;

        result.add_column("group".to_string(), group_series)?;
        result.add_column("size".to_string(), size_series)?;

        Ok(result)
    }

    /// Apply named aggregations (pandas-like .agg() functionality)
    pub fn agg(&self, named_aggs: Vec<NamedAgg>) -> Result<DataFrame> {
        if named_aggs.is_empty() {
            return Err(Error::InvalidValue(
                "At least one aggregation must be specified".to_string(),
            ));
        }

        // Verify all columns exist
        for agg in &named_aggs {
            if !self.df.contains_column(&agg.column) {
                return Err(Error::ColumnNotFound(agg.column.clone()));
            }
        }

        let mut result = DataFrame::new();

        // Create columns for group keys
        for (i, group_col) in self.group_by_columns.iter().enumerate() {
            let mut group_values = Vec::new();
            for key in self.groups.keys() {
                group_values.push(key[i].clone());
            }
            let group_series = Series::new(group_values, Some(group_col.clone()))?;
            result.add_column(group_col.clone(), group_series)?;
        }

        // Apply each named aggregation
        for agg in &named_aggs {
            let mut agg_values = Vec::new();

            for indices in self.groups.values() {
                let agg_result =
                    self.calculate_aggregation(&agg.column, agg.func, indices, &agg.custom_fn)?;
                agg_values.push(agg_result.to_string());
            }

            let agg_series = Series::new(agg_values, Some(agg.alias.clone()))?;
            result.add_column(agg.alias.clone(), agg_series)?;
        }

        Ok(result)
    }

    /// Apply multiple aggregations using a builder pattern
    pub fn agg_multi(&self, builders: Vec<ColumnAggBuilder>) -> Result<DataFrame> {
        let mut named_aggs = Vec::new();

        for builder in builders {
            named_aggs.extend(builder.build());
        }

        self.agg(named_aggs)
    }

    /// Apply aggregations using a HashMap specification (similar to pandas)
    /// Example: {"price": [("mean", "avg_price"), ("std", "price_std")]}
    pub fn agg_dict(&self, agg_spec: HashMap<String, Vec<(AggFunc, String)>>) -> Result<DataFrame> {
        let mut named_aggs = Vec::new();

        for (column, specs) in agg_spec {
            for (func, alias) in specs {
                named_aggs.push(NamedAgg::new(column.clone(), func, alias));
            }
        }

        self.agg(named_aggs)
    }

    /// Convenience method for simple aggregations
    pub fn sum(&self, column: &str) -> Result<DataFrame> {
        let agg = NamedAgg::new(column.to_string(), AggFunc::Sum, format!("{}_sum", column));
        self.agg(vec![agg])
    }

    /// Convenience method for mean aggregation
    pub fn mean(&self, column: &str) -> Result<DataFrame> {
        let agg = NamedAgg::new(
            column.to_string(),
            AggFunc::Mean,
            format!("{}_mean", column),
        );
        self.agg(vec![agg])
    }

    /// Convenience method for count aggregation
    pub fn count(&self, column: &str) -> Result<DataFrame> {
        let agg = NamedAgg::new(
            column.to_string(),
            AggFunc::Count,
            format!("{}_count", column),
        );
        self.agg(vec![agg])
    }

    /// Convenience method for min aggregation
    pub fn min(&self, column: &str) -> Result<DataFrame> {
        let agg = NamedAgg::new(column.to_string(), AggFunc::Min, format!("{}_min", column));
        self.agg(vec![agg])
    }

    /// Convenience method for max aggregation
    pub fn max(&self, column: &str) -> Result<DataFrame> {
        let agg = NamedAgg::new(column.to_string(), AggFunc::Max, format!("{}_max", column));
        self.agg(vec![agg])
    }

    /// Convenience method for std aggregation
    pub fn std(&self, column: &str) -> Result<DataFrame> {
        let agg = NamedAgg::new(column.to_string(), AggFunc::Std, format!("{}_std", column));
        self.agg(vec![agg])
    }

    /// Convenience method for var aggregation
    pub fn var(&self, column: &str) -> Result<DataFrame> {
        let agg = NamedAgg::new(column.to_string(), AggFunc::Var, format!("{}_var", column));
        self.agg(vec![agg])
    }

    /// Convenience method for median aggregation
    pub fn median(&self, column: &str) -> Result<DataFrame> {
        let agg = NamedAgg::new(
            column.to_string(),
            AggFunc::Median,
            format!("{}_median", column),
        );
        self.agg(vec![agg])
    }

    /// Convenience method for nunique aggregation
    pub fn nunique(&self, column: &str) -> Result<DataFrame> {
        let agg = NamedAgg::new(
            column.to_string(),
            AggFunc::Nunique,
            format!("{}_nunique", column),
        );
        self.agg(vec![agg])
    }

    /// Apply a custom aggregation function
    pub fn apply<F>(&self, column: &str, alias: &str, func: F) -> Result<DataFrame>
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        let agg = NamedAgg::custom(column.to_string(), alias.to_string(), func);
        self.agg(vec![agg])
    }

    /// Filter groups based on a condition
    pub fn filter<F>(&self, condition: F) -> Result<DataFrame>
    where
        F: Fn(&DataFrame) -> bool,
    {
        let mut filtered_indices = Vec::new();

        for indices in self.groups.values() {
            // Create a subset DataFrame for this group
            let group_df = self.create_group_dataframe(indices)?;

            // Apply the condition
            if condition(&group_df) {
                filtered_indices.extend(indices);
            }
        }

        // Create result DataFrame with filtered rows
        self.create_subset_dataframe(&filtered_indices)
    }

    /// Transform groups using a function
    pub fn transform<F>(&self, func: F) -> Result<DataFrame>
    where
        F: Fn(&DataFrame) -> Result<DataFrame>,
    {
        let mut transformed_parts = Vec::new();

        for indices in self.groups.values() {
            let group_df = self.create_group_dataframe(indices)?;
            let transformed = func(&group_df)?;
            transformed_parts.push(transformed);
        }

        // Concatenate all transformed parts
        self.concatenate_dataframes(transformed_parts)
    }

    /// Calculate aggregation for a column and group
    fn calculate_aggregation(
        &self,
        column: &str,
        func: AggFunc,
        indices: &[usize],
        custom_fn: &Option<CustomAggFn>,
    ) -> Result<f64> {
        let column_values = self.df.get_column_string_values(column)?;

        // Extract numeric values for this group
        let group_values: Vec<f64> = indices
            .iter()
            .filter_map(|&idx| {
                if idx < column_values.len() {
                    column_values[idx].parse::<f64>().ok()
                } else {
                    None
                }
            })
            .collect();

        if group_values.is_empty() {
            return Ok(0.0);
        }

        match func {
            AggFunc::Sum => Ok(group_values.iter().sum()),
            AggFunc::Mean => Ok(group_values.iter().sum::<f64>() / group_values.len() as f64),
            AggFunc::Min => Ok(group_values.iter().fold(f64::INFINITY, |a, &b| a.min(b))),
            AggFunc::Max => Ok(group_values
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b))),
            AggFunc::Count => Ok(group_values.len() as f64),
            AggFunc::Std => {
                if group_values.len() <= 1 {
                    Ok(0.0)
                } else {
                    let mean = group_values.iter().sum::<f64>() / group_values.len() as f64;
                    let variance = group_values
                        .iter()
                        .map(|&x| (x - mean).powi(2))
                        .sum::<f64>()
                        / (group_values.len() - 1) as f64;
                    Ok(variance.sqrt())
                }
            }
            AggFunc::Var => {
                if group_values.len() <= 1 {
                    Ok(0.0)
                } else {
                    let mean = group_values.iter().sum::<f64>() / group_values.len() as f64;
                    Ok(group_values
                        .iter()
                        .map(|&x| (x - mean).powi(2))
                        .sum::<f64>()
                        / (group_values.len() - 1) as f64)
                }
            }
            AggFunc::Median => {
                let mut sorted = group_values;
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mid = sorted.len() / 2;
                if sorted.len() % 2 == 0 {
                    Ok((sorted[mid - 1] + sorted[mid]) / 2.0)
                } else {
                    Ok(sorted[mid])
                }
            }
            AggFunc::First => Ok(group_values[0]),
            AggFunc::Last => Ok(*group_values.last().unwrap()),
            AggFunc::Nunique => {
                let mut unique_values = group_values;
                unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                unique_values.dedup();
                Ok(unique_values.len() as f64)
            }
            AggFunc::Custom => {
                if let Some(custom_fn) = custom_fn {
                    Ok(custom_fn(&group_values))
                } else {
                    Err(Error::InvalidValue(
                        "Custom function not provided".to_string(),
                    ))
                }
            }
        }
    }

    /// Create a DataFrame for a specific group
    fn create_group_dataframe(&self, indices: &[usize]) -> Result<DataFrame> {
        let mut group_df = DataFrame::new();

        for col_name in self.df.column_names() {
            let column_values = self.df.get_column_string_values(&col_name)?;
            let group_values: Vec<String> = indices
                .iter()
                .filter_map(|&idx| {
                    if idx < column_values.len() {
                        Some(column_values[idx].clone())
                    } else {
                        None
                    }
                })
                .collect();

            let group_series = Series::new(group_values, Some(col_name.clone()))?;
            group_df.add_column(col_name, group_series)?;
        }

        Ok(group_df)
    }

    /// Create a subset DataFrame with specific row indices
    fn create_subset_dataframe(&self, indices: &[usize]) -> Result<DataFrame> {
        let mut subset_df = DataFrame::new();

        for col_name in self.df.column_names() {
            let column_values = self.df.get_column_string_values(&col_name)?;
            let subset_values: Vec<String> = indices
                .iter()
                .filter_map(|&idx| {
                    if idx < column_values.len() {
                        Some(column_values[idx].clone())
                    } else {
                        None
                    }
                })
                .collect();

            let subset_series = Series::new(subset_values, Some(col_name.clone()))?;
            subset_df.add_column(col_name, subset_series)?;
        }

        Ok(subset_df)
    }

    /// Concatenate multiple DataFrames
    fn concatenate_dataframes(&self, dataframes: Vec<DataFrame>) -> Result<DataFrame> {
        if dataframes.is_empty() {
            return Ok(DataFrame::new());
        }

        let mut result = DataFrame::new();
        let first_df = &dataframes[0];

        for col_name in first_df.column_names() {
            let mut all_values = Vec::new();

            for df in &dataframes {
                let column_values = df.get_column_string_values(&col_name)?;
                all_values.extend(column_values);
            }

            let concat_series = Series::new(all_values, Some(col_name.clone()))?;
            result.add_column(col_name, concat_series)?;
        }

        Ok(result)
    }
}

/// Extension trait to add groupby functionality to DataFrame
pub trait GroupByExt {
    /// Group DataFrame by one or more columns
    fn groupby<S: AsRef<str>>(&self, columns: &[S]) -> Result<DataFrameGroupBy>;

    /// Group DataFrame by a single column (convenience method)
    fn groupby_single(&self, column: &str) -> Result<DataFrameGroupBy>;
}

impl GroupByExt for DataFrame {
    fn groupby<S: AsRef<str>>(&self, columns: &[S]) -> Result<DataFrameGroupBy> {
        let group_columns: Vec<String> = columns.iter().map(|s| s.as_ref().to_string()).collect();
        DataFrameGroupBy::new(self.clone(), group_columns)
    }

    fn groupby_single(&self, column: &str) -> Result<DataFrameGroupBy> {
        DataFrameGroupBy::new(self.clone(), vec![column.to_string()])
    }
}

/// Helper macros for creating aggregation specifications

/// Create a named aggregation
#[macro_export]
macro_rules! named_agg {
    ($column:expr, $func:expr, $alias:expr) => {
        NamedAgg::new($column.to_string(), $func, $alias.to_string())
    };
}

/// Create multiple named aggregations for a column
#[macro_export]
macro_rules! column_aggs {
    ($column:expr, $(($func:expr, $alias:expr)),+) => {
        {
            let mut builder = ColumnAggBuilder::new($column.to_string());
            $(
                builder = builder.agg($func, $alias.to_string());
            )+
            builder
        }
    };
}

/// Create aggregation specification (similar to pandas)
#[macro_export]
macro_rules! agg_spec {
    ($($column:expr => [$(($func:expr, $alias:expr)),+]),+) => {
        {
            let mut spec = std::collections::HashMap::new();
            $(
                spec.insert($column.to_string(), vec![$(($func, $alias.to_string())),+]);
            )+
            spec
        }
    };
}
