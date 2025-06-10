//! Statistical functions module for OptimizedDataFrame
//!
//! This module provides statistical functionality for data analysis.
//! It supports ANOVA, t-tests, chi-square tests, Mann-Whitney U tests, and more.

use crate::column::{Column, ColumnTrait};
use crate::error::Result;
use crate::optimized::split_dataframe::OptimizedDataFrame;
use crate::stats::{
    self, AnovaResult, ChiSquareResult, DescriptiveStats, LinearRegressionResult,
    MannWhitneyResult, TTestResult,
};
use std::collections::HashMap;

/// Statistical result type for OptimizedDataFrame
#[derive(Debug, Clone)]
pub enum StatResult {
    /// Descriptive statistics results
    Descriptive(DescriptiveStats),
    /// t-test results
    TTest(TTestResult),
    /// Analysis of variance (ANOVA) results
    Anova(AnovaResult),
    /// Mann-Whitney U test results
    MannWhitneyU(MannWhitneyResult),
    /// Chi-square test results
    ChiSquare(ChiSquareResult),
    /// Linear regression results
    LinearRegression(LinearRegressionResult),
}

/// Output format for descriptive statistics results
#[derive(Debug, Clone)]
pub struct StatDescribe {
    /// Map of statistics
    pub stats: HashMap<String, f64>,
    /// List of statistics (ordered)
    pub stats_list: Vec<(String, f64)>,
}

/// Statistical functionality extension for OptimizedDataFrame
impl OptimizedDataFrame {
    /// Calculate basic statistics for a specific column
    ///
    /// # Arguments
    /// * `column_name` - Name of the column to calculate statistics for
    ///
    /// # Returns
    /// A structure containing descriptive statistics
    pub fn describe(&self, column_name: &str) -> Result<StatDescribe> {
        let col = self.column(column_name)?;

        if let Some(float_col) = col.as_float64() {
            // For floating-point columns
            let values: Vec<f64> = (0..self.row_count())
                .filter_map(|i| float_col.get(i).ok().flatten())
                .collect();

            // Use stats module
            let stats = stats::describe(&values)?;

            // Store results in HashMap
            let mut result = HashMap::new();
            result.insert("count".to_string(), stats.count as f64);
            result.insert("mean".to_string(), stats.mean);
            result.insert("std".to_string(), stats.std);
            result.insert("min".to_string(), stats.min);
            result.insert("25%".to_string(), stats.q1);
            result.insert("50%".to_string(), stats.median);
            result.insert("75%".to_string(), stats.q3);
            result.insert("max".to_string(), stats.max);

            // Also provide ordered list
            let stats_list = vec![
                ("count".to_string(), stats.count as f64),
                ("mean".to_string(), stats.mean),
                ("std".to_string(), stats.std),
                ("min".to_string(), stats.min),
                ("25%".to_string(), stats.q1),
                ("50%".to_string(), stats.median),
                ("75%".to_string(), stats.q3),
                ("max".to_string(), stats.max),
            ];

            let mut result = HashMap::new();
            result.insert("count".to_string(), stats.count as f64);
            result.insert("mean".to_string(), stats.mean);
            result.insert("std".to_string(), stats.std);
            result.insert("min".to_string(), stats.min);
            result.insert("25%".to_string(), stats.q1);
            result.insert("50%".to_string(), stats.median);
            result.insert("75%".to_string(), stats.q3);
            result.insert("max".to_string(), stats.max);

            Ok(StatDescribe {
                stats: result,
                stats_list,
            })
        } else if let Some(int_col) = col.as_int64() {
            // For integer columns, convert to floating-point for calculation
            let values: Vec<f64> = (0..self.row_count())
                .filter_map(|i| int_col.get(i).ok().flatten().map(|v| v as f64))
                .collect();

            // Use stats module
            let stats = stats::describe(&values)?;

            // Store results in HashMap
            let mut result = HashMap::new();
            result.insert("count".to_string(), stats.count as f64);
            result.insert("mean".to_string(), stats.mean);
            result.insert("std".to_string(), stats.std);
            result.insert("min".to_string(), stats.min);
            result.insert("25%".to_string(), stats.q1);
            result.insert("50%".to_string(), stats.median);
            result.insert("75%".to_string(), stats.q3);
            result.insert("max".to_string(), stats.max);

            // Also provide ordered list
            let stats_list = vec![
                ("count".to_string(), stats.count as f64),
                ("mean".to_string(), stats.mean),
                ("std".to_string(), stats.std),
                ("min".to_string(), stats.min),
                ("25%".to_string(), stats.q1),
                ("50%".to_string(), stats.median),
                ("75%".to_string(), stats.q3),
                ("max".to_string(), stats.max),
            ];

            let mut result = HashMap::new();
            result.insert("count".to_string(), stats.count as f64);
            result.insert("mean".to_string(), stats.mean);
            result.insert("std".to_string(), stats.std);
            result.insert("min".to_string(), stats.min);
            result.insert("25%".to_string(), stats.q1);
            result.insert("50%".to_string(), stats.median);
            result.insert("75%".to_string(), stats.q3);
            result.insert("max".to_string(), stats.max);

            Ok(StatDescribe {
                stats: result,
                stats_list,
            })
        } else {
            Err(crate::error::Error::Type(format!(
                "Column '{}' is not a numeric type",
                column_name
            )))
        }
    }

    /// Calculate descriptive statistics for multiple columns at once
    ///
    /// # Returns
    /// A mapping from column names to statistical results
    pub fn describe_all(&self) -> Result<HashMap<String, StatDescribe>> {
        let mut results = HashMap::new();

        for col_name in self.column_names() {
            // Target only numeric columns
            let col = self.column(col_name)?;
            if col.as_float64().is_some() || col.as_int64().is_some() {
                if let Ok(desc) = self.describe(col_name) {
                    results.insert(col_name.to_string(), desc);
                }
            }
        }

        Ok(results)
    }

    /// Perform t-test on two columns
    ///
    /// # Arguments
    /// * `col1` - First column name
    /// * `col2` - Second column name
    /// * `alpha` - Significance level (default: 0.05)
    /// * `equal_var` - Whether to assume equal variance (default: true)
    ///
    /// # Returns
    /// Results of the t-test
    pub fn ttest(
        &self,
        col1: &str,
        col2: &str,
        alpha: Option<f64>,
        equal_var: Option<bool>,
    ) -> Result<TTestResult> {
        let alpha = alpha.unwrap_or(0.05);
        let equal_var = equal_var.unwrap_or(true);

        // Get column data
        let column1 = self.column(col1)?;
        let column2 = self.column(col2)?;

        // Convert to floating-point vectors
        let values1: Vec<f64> = match column1 {
            col if col.as_float64().is_some() => {
                let float_col = col.as_float64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| float_col.get(i).ok().flatten())
                    .collect()
            }
            col if col.as_int64().is_some() => {
                let int_col = col.as_int64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| int_col.get(i).ok().flatten().map(|v| v as f64))
                    .collect()
            }
            _ => {
                return Err(crate::error::Error::Type(format!(
                    "Column '{}' is not a numeric type",
                    col1
                )))
            }
        };

        let values2: Vec<f64> = match column2 {
            col if col.as_float64().is_some() => {
                let float_col = col.as_float64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| float_col.get(i).ok().flatten())
                    .collect()
            }
            col if col.as_int64().is_some() => {
                let int_col = col.as_int64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| int_col.get(i).ok().flatten().map(|v| v as f64))
                    .collect()
            }
            _ => {
                return Err(crate::error::Error::Type(format!(
                    "Column '{}' is not a numeric type",
                    col2
                )))
            }
        };

        // Perform t-test
        stats::ttest(&values1, &values2, alpha, equal_var)
    }

    /// Perform one-way analysis of variance (ANOVA)
    ///
    /// # Arguments
    /// * `value_col` - Column name containing the measured values
    /// * `group_col` - Column name for grouping
    /// * `alpha` - Significance level (default: 0.05)
    ///
    /// # Returns
    /// Results of the ANOVA
    pub fn anova(
        &self,
        value_col: &str,
        group_col: &str,
        alpha: Option<f64>,
    ) -> Result<AnovaResult> {
        let alpha = alpha.unwrap_or(0.05);

        // Get the value column
        let value_column = self.column(value_col)?;

        // Get the group column
        let group_column = self.column(group_col)?;
        let group_col_string = group_column.as_string().ok_or_else(|| {
            crate::error::Error::Type(format!("Column '{}' must be a string type", group_col))
        })?;

        // Convert values to floating-point
        let values: Vec<(f64, String)> = match value_column {
            col if col.as_float64().is_some() => {
                let float_col = col.as_float64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| {
                        let val = float_col.get(i).ok().flatten()?;
                        let group = group_col_string.get(i).ok().flatten()?;
                        Some((val, group.to_string()))
                    })
                    .collect()
            }
            col if col.as_int64().is_some() => {
                let int_col = col.as_int64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| {
                        let val = int_col.get(i).ok().flatten()? as f64;
                        let group = group_col_string.get(i).ok().flatten()?;
                        Some((val, group.to_string()))
                    })
                    .collect()
            }
            _ => {
                return Err(crate::error::Error::Type(format!(
                    "Column '{}' is not a numeric type",
                    value_col
                )))
            }
        };

        // Organize data by group
        let mut groups: HashMap<String, Vec<f64>> = HashMap::new();
        for (val, group) in values {
            groups.entry(group).or_insert_with(Vec::new).push(val);
        }

        // Ensure there are at least 2 groups
        if groups.len() < 2 {
            return Err(crate::error::Error::InsufficientData(
                "ANOVA requires at least 2 groups".to_string(),
            ));
        }

        // Convert to &str group map
        let str_groups: HashMap<&str, Vec<f64>> = groups
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();

        // Perform ANOVA
        stats::anova(&str_groups, alpha)
    }

    /// Perform Mann-Whitney U test (non-parametric test)
    ///
    /// # Arguments
    /// * `col1` - First column name
    /// * `col2` - Second column name
    /// * `alpha` - Significance level (default: 0.05)
    ///
    /// # Returns
    /// Results of the Mann-Whitney U test
    pub fn mann_whitney_u(
        &self,
        col1: &str,
        col2: &str,
        alpha: Option<f64>,
    ) -> Result<MannWhitneyResult> {
        let alpha = alpha.unwrap_or(0.05);

        // Get column data
        let column1 = self.column(col1)?;
        let column2 = self.column(col2)?;

        // Convert to floating-point vectors
        let values1: Vec<f64> = match column1 {
            col if col.as_float64().is_some() => {
                let float_col = col.as_float64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| float_col.get(i).ok().flatten())
                    .collect()
            }
            col if col.as_int64().is_some() => {
                let int_col = col.as_int64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| int_col.get(i).ok().flatten().map(|v| v as f64))
                    .collect()
            }
            _ => {
                return Err(crate::error::Error::Type(format!(
                    "Column '{}' is not a numeric type",
                    col1
                )))
            }
        };

        let values2: Vec<f64> = match column2 {
            col if col.as_float64().is_some() => {
                let float_col = col.as_float64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| float_col.get(i).ok().flatten())
                    .collect()
            }
            col if col.as_int64().is_some() => {
                let int_col = col.as_int64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| int_col.get(i).ok().flatten().map(|v| v as f64))
                    .collect()
            }
            _ => {
                return Err(crate::error::Error::Type(format!(
                    "Column '{}' is not a numeric type",
                    col2
                )))
            }
        };

        // Perform Mann-Whitney U test
        stats::mann_whitney_u(&values1, &values2, alpha)
    }

    /// Perform chi-square test
    ///
    /// # Arguments
    /// * `row_col` - Column name determining rows
    /// * `col_col` - Column name determining columns
    /// * `count_col` - Column name containing counts/frequencies
    /// * `alpha` - Significance level (default: 0.05)
    ///
    /// # Returns
    /// Results of the chi-square test
    pub fn chi_square_test(
        &self,
        row_col: &str,
        col_col: &str,
        count_col: &str,
        alpha: Option<f64>,
    ) -> Result<ChiSquareResult> {
        let alpha = alpha.unwrap_or(0.05);

        // Get column data
        let row_column = self.column(row_col)?;
        let col_column = self.column(col_col)?;
        let count_column = self.column(count_col)?;

        // Get string columns
        let row_strings = row_column.as_string().ok_or_else(|| {
            crate::error::Error::Type(format!("Column '{}' must be a string type", row_col))
        })?;

        let col_strings = col_column.as_string().ok_or_else(|| {
            crate::error::Error::Type(format!("Column '{}' must be a string type", col_col))
        })?;

        // Get count values
        let count_values: Vec<f64> = match count_column {
            col if col.as_float64().is_some() => {
                let float_col = col.as_float64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| float_col.get(i).ok().flatten())
                    .collect()
            }
            col if col.as_int64().is_some() => {
                let int_col = col.as_int64().unwrap();
                (0..self.row_count())
                    .filter_map(|i| int_col.get(i).ok().flatten().map(|v| v as f64))
                    .collect()
            }
            _ => {
                return Err(crate::error::Error::Type(format!(
                    "Column '{}' is not a numeric type",
                    count_col
                )))
            }
        };

        // Generate contingency table
        // Extract unique row and column values
        let mut unique_rows = vec![];
        let mut unique_cols = vec![];

        for i in 0..self.row_count() {
            if let Ok(Some(row_val)) = row_strings.get(i) {
                if !unique_rows.contains(&row_val) {
                    unique_rows.push(row_val);
                }
            }

            if let Ok(Some(col_val)) = col_strings.get(i) {
                if !unique_cols.contains(&col_val) {
                    unique_cols.push(col_val);
                }
            }
        }

        // Build observed data matrix
        let mut observed = vec![vec![0.0; unique_cols.len()]; unique_rows.len()];

        for i in 0..self.row_count() {
            if let (Ok(Some(row_val)), Ok(Some(col_val)), count) =
                (row_strings.get(i), col_strings.get(i), count_values.get(i))
            {
                if let (Some(row_idx), Some(col_idx)) = (
                    unique_rows.iter().position(|r| r == &row_val),
                    unique_cols.iter().position(|c| c == &col_val),
                ) {
                    // Add count value if available, otherwise add 1.0
                    if let Some(cnt) = count {
                        observed[row_idx][col_idx] += *cnt;
                    } else {
                        observed[row_idx][col_idx] += 1.0;
                    }
                }
            }
        }

        // Perform chi-square test
        stats::chi_square_test(&observed, alpha)
    }

    /// Perform linear regression analysis
    ///
    /// # Arguments
    /// * `y_col` - Name of the target (dependent) variable column
    /// * `x_cols` - List of explanatory (independent) variable column names
    ///
    /// # Returns
    /// Results of the linear regression analysis
    pub fn linear_regression(
        &self,
        y_col: &str,
        x_cols: &[&str],
    ) -> Result<LinearRegressionResult> {
        // Convert to DataFrame format
        let mut df = crate::dataframe::DataFrame::new();

        // Add the target variable
        let y_column = self.column(y_col)?;
        if let Some(float_col) = y_column.as_float64() {
            let values: Vec<f64> = (0..self.row_count())
                .filter_map(|i| float_col.get(i).ok().flatten())
                .collect();

            let series = crate::series::Series::new(values, Some(y_col.to_string()))?;
            df.add_column(y_col.to_string(), series)?;
        } else if let Some(int_col) = y_column.as_int64() {
            // Convert integer column to floating-point
            let values: Vec<f64> = (0..self.row_count())
                .filter_map(|i| int_col.get(i).ok().flatten().map(|v| v as f64))
                .collect();

            let series = crate::series::Series::new(values, Some(y_col.to_string()))?;
            df.add_column(y_col.to_string(), series)?;
        } else {
            return Err(crate::error::Error::Type(format!(
                "Column '{}' must be a numeric type",
                y_col
            )));
        }

        // Add explanatory variable columns
        for &x_col in x_cols {
            let x_column = self.column(x_col)?;
            if let Some(float_col) = x_column.as_float64() {
                let values: Vec<f64> = (0..self.row_count())
                    .filter_map(|i| float_col.get(i).ok().flatten())
                    .collect();

                let series = crate::series::Series::new(values, Some(x_col.to_string()))?;
                df.add_column(x_col.to_string(), series)?;
            } else if let Some(int_col) = x_column.as_int64() {
                // Convert integer column to floating-point
                let values: Vec<f64> = (0..self.row_count())
                    .filter_map(|i| int_col.get(i).ok().flatten().map(|v| v as f64))
                    .collect();

                let series = crate::series::Series::new(values, Some(x_col.to_string()))?;
                df.add_column(x_col.to_string(), series)?;
            } else {
                return Err(crate::error::Error::Type(format!(
                    "Column '{}' must be a numeric type",
                    x_col
                )));
            }
        }

        // Build linear regression model
        stats::linear_regression(&df, y_col, x_cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::{Column, Float64Column, StringColumn};
    use crate::optimized::split_dataframe::OptimizedDataFrame;

    #[test]
    fn test_describe() {
        let mut df = OptimizedDataFrame::new();

        // Create test data
        let values = Float64Column::with_name(vec![1.0, 2.0, 3.0, 4.0, 5.0], "values");
        df.add_column("values", Column::Float64(values)).unwrap();

        // Test describe function
        let desc = df.describe("values").unwrap();

        // Verify results
        assert_eq!(desc.stats.get("count").unwrap().clone() as usize, 5);
        assert!((desc.stats.get("mean").unwrap() - 3.0).abs() < 1e-10);
        assert!((desc.stats.get("min").unwrap() - 1.0).abs() < 1e-10);
        assert!((desc.stats.get("max").unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_ttest() {
        let mut df = OptimizedDataFrame::new();

        // Create test data
        let values1 = Float64Column::with_name(vec![1.0, 2.0, 3.0, 4.0, 5.0], "sample1");
        let values2 = Float64Column::with_name(vec![2.0, 3.0, 4.0, 5.0, 6.0], "sample2");

        df.add_column("sample1", Column::Float64(values1)).unwrap();
        df.add_column("sample2", Column::Float64(values2)).unwrap();

        // Run t-test
        let result = df
            .ttest("sample1", "sample2", Some(0.05), Some(true))
            .unwrap();

        // Verify results
        assert!(result.statistic < 0.0); // Because sample2 has larger values
        assert_eq!(result.df, 8); // Degrees of freedom is total sample size - 2
    }

    #[test]
    fn test_anova() {
        let mut df = OptimizedDataFrame::new();

        // Create test data
        let values = Float64Column::with_name(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            ],
            "values",
        );

        let groups = StringColumn::with_name(
            vec![
                "A".to_string(),
                "A".to_string(),
                "A".to_string(),
                "A".to_string(),
                "A".to_string(),
                "B".to_string(),
                "B".to_string(),
                "B".to_string(),
                "B".to_string(),
                "B".to_string(),
                "C".to_string(),
                "C".to_string(),
                "C".to_string(),
                "C".to_string(),
                "C".to_string(),
            ],
            "group",
        );

        df.add_column("values", Column::Float64(values)).unwrap();
        df.add_column("group", Column::String(groups)).unwrap();

        // Perform ANOVA
        let result = df.anova("values", "group", Some(0.05)).unwrap();

        // Verify results
        assert!(result.f_statistic > 0.0);
        assert_eq!(result.df_between, 2); // Number of groups - 1
        assert_eq!(result.df_within, 12); // Total sample size - number of groups
        assert_eq!(result.df_total, 14); // Total sample size - 1
    }
}
