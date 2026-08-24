//! Statistical models for [`OptimizedDataFrame`]
//!
//! The columnar `OptimizedDataFrame` stores each column as a typed, nullable
//! vector. This module reads those columns directly and feeds them to the
//! statistical routines in [`crate::stats`], so the frame never has to be
//! materialized into a row-oriented representation first.

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::series::Series;
use crate::stats::{self, LinearRegressionResult};

use super::core::OptimizedDataFrame;

impl OptimizedDataFrame {
    /// Read a numeric column as one `Option<f64>` per row.
    ///
    /// Integer columns are widened to `f64`. String and boolean columns are
    /// rejected rather than coerced: silently parsing them would turn a typo in
    /// a column name into a regression on zeros.
    fn numeric_column_values(&self, column_name: &str) -> Result<Vec<Option<f64>>> {
        let view = self.column(column_name)?;

        if let Some(float_col) = view.as_float64() {
            (0..self.row_count()).map(|i| float_col.get(i)).collect()
        } else if let Some(int_col) = view.as_int64() {
            (0..self.row_count())
                .map(|i| int_col.get(i).map(|value| value.map(|v| v as f64)))
                .collect()
        } else {
            Err(Error::Type(format!(
                "Column '{}' must be of type Int64 or Float64 to be used in a regression",
                column_name
            )))
        }
    }

    /// Fit an ordinary least squares linear regression.
    ///
    /// The model is `y = b0 + b1*x1 + ... + bp*xp`, estimated by the normal
    /// equations. Rows are selected by complete-case analysis (listwise
    /// deletion): a row contributes only when the target and *every*
    /// explanatory column hold a value, which keeps the observation vectors
    /// aligned. Dropping missing values per column instead would shift
    /// observations against each other and silently fit a different data set.
    ///
    /// # Arguments
    /// * `y_column` - Name of the target (dependent) variable column
    /// * `x_columns` - Names of the explanatory (independent) variable columns
    ///
    /// # Returns
    /// A [`LinearRegressionResult`] carrying the intercept, one coefficient per
    /// explanatory column (in the order given), R², adjusted R², per-coefficient
    /// p-values, fitted values, and residuals.
    ///
    /// # Errors
    /// Returns an error when a column is missing or non-numeric, when
    /// `x_columns` is empty or contains duplicates, or when too few complete
    /// rows remain to identify the model.
    ///
    /// # Example
    /// ```rust
    /// use pandrs::column::{Column, Float64Column};
    /// use pandrs::OptimizedDataFrame;
    ///
    /// let mut df = OptimizedDataFrame::new();
    /// df.add_column(
    ///     "x".to_string(),
    ///     Column::Float64(Float64Column::new(vec![1.0, 2.0, 3.0, 4.0])),
    /// )?;
    /// df.add_column(
    ///     "y".to_string(),
    ///     Column::Float64(Float64Column::new(vec![3.0, 5.0, 7.0, 9.0])),
    /// )?;
    ///
    /// let model = df.linear_regression("y", &["x"])?;
    /// assert!((model.intercept - 1.0).abs() < 1e-9);
    /// assert!((model.coefficients[0] - 2.0).abs() < 1e-9);
    /// # Ok::<(), pandrs::error::Error>(())
    /// ```
    pub fn linear_regression(
        &self,
        y_column: &str,
        x_columns: &[&str],
    ) -> Result<LinearRegressionResult> {
        if x_columns.is_empty() {
            return Err(Error::InvalidOperation(
                "Linear regression requires at least one explanatory column".into(),
            ));
        }

        // A repeated predictor makes X'X singular, and using the target as its
        // own predictor is always a mistake. Both are cheaper to reject here
        // than to diagnose from a failed matrix inversion.
        for (position, &name) in x_columns.iter().enumerate() {
            if name == y_column {
                return Err(Error::InvalidOperation(format!(
                    "Column '{}' cannot be both the target and an explanatory variable",
                    name
                )));
            }
            if x_columns[position + 1..].contains(&name) {
                return Err(Error::InvalidOperation(format!(
                    "Explanatory column '{}' is listed more than once",
                    name
                )));
            }
        }

        let y_raw = self.numeric_column_values(y_column)?;
        let x_raw = x_columns
            .iter()
            .map(|&name| self.numeric_column_values(name))
            .collect::<Result<Vec<_>>>()?;

        let mut y_values: Vec<f64> = Vec::with_capacity(self.row_count());
        let mut x_values: Vec<Vec<f64>> = vec![Vec::with_capacity(self.row_count()); x_raw.len()];
        let mut complete_row: Vec<f64> = Vec::with_capacity(x_raw.len());

        for row in 0..self.row_count() {
            let target = match y_raw[row] {
                Some(value) => value,
                None => continue,
            };

            complete_row.clear();
            for column in &x_raw {
                match column[row] {
                    Some(value) => complete_row.push(value),
                    None => break,
                }
            }
            if complete_row.len() != x_raw.len() {
                continue;
            }

            y_values.push(target);
            for (destination, &value) in x_values.iter_mut().zip(complete_row.iter()) {
                destination.push(value);
            }
        }

        // The model estimates one intercept plus one slope per predictor, so at
        // least that many observations are needed before X'X can be inverted.
        let parameter_count = x_columns.len() + 1;
        if y_values.len() < parameter_count {
            return Err(Error::InsufficientData(format!(
                "Linear regression on '{}' needs at least {} complete rows for {} explanatory \
                 column(s), but only {} row(s) have no missing values",
                y_column,
                parameter_count,
                x_columns.len(),
                y_values.len()
            )));
        }

        let mut frame = DataFrame::new();
        frame.add_column(
            y_column.to_string(),
            Series::new(y_values, Some(y_column.to_string()))?,
        )?;
        for (&name, values) in x_columns.iter().zip(x_values) {
            frame.add_column(
                name.to_string(),
                Series::new(values, Some(name.to_string()))?,
            )?;
        }

        stats::linear_regression(&frame, y_column, x_columns)
    }
}

#[cfg(test)]
mod tests {
    use crate::column::{Column, Float64Column, Int64Column, StringColumn};
    use crate::optimized::OptimizedDataFrame;

    /// Build a frame with the given columns, panicking only on setup failure.
    fn frame(columns: Vec<(&str, Column)>) -> OptimizedDataFrame {
        let mut df = OptimizedDataFrame::new();
        for (name, column) in columns {
            df.add_column(name.to_string(), column)
                .expect("test column should be addable");
        }
        df
    }

    #[test]
    fn recovers_exact_simple_regression() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| 2.0 * v + 1.0).collect();
        let df = frame(vec![
            ("x", Column::Float64(Float64Column::new(x))),
            ("y", Column::Float64(Float64Column::new(y))),
        ]);

        let model = df.linear_regression("y", &["x"]).expect("regression");

        assert!((model.intercept - 1.0).abs() < 1e-9, "{}", model.intercept);
        assert_eq!(model.coefficients.len(), 1);
        assert!((model.coefficients[0] - 2.0).abs() < 1e-9);
        assert!((model.r_squared - 1.0).abs() < 1e-9);
        assert_eq!(model.fitted_values.len(), 10);
        assert!(model.residuals.iter().all(|r| r.abs() < 1e-9));
    }

    #[test]
    fn recovers_exact_multiple_regression_from_mixed_types() {
        // y = 3 + 2*x1 - 1*x2, with x2 stored as an integer column.
        let x1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x2: Vec<i64> = vec![0, 1, 3, 2, 5, 4];
        let y: Vec<f64> = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| 3.0 + 2.0 * a - b as f64)
            .collect();
        let df = frame(vec![
            ("x1", Column::Float64(Float64Column::new(x1))),
            ("x2", Column::Int64(Int64Column::new(x2))),
            ("y", Column::Float64(Float64Column::new(y))),
        ]);

        let model = df
            .linear_regression("y", &["x1", "x2"])
            .expect("regression");

        assert!((model.intercept - 3.0).abs() < 1e-8, "{}", model.intercept);
        assert!((model.coefficients[0] - 2.0).abs() < 1e-8);
        assert!((model.coefficients[1] + 1.0).abs() < 1e-8);
        assert_eq!(model.p_values.len(), 3);
    }

    #[test]
    fn rejects_non_numeric_and_malformed_requests() {
        let df = frame(vec![
            (
                "x",
                Column::Float64(Float64Column::new(vec![1.0, 2.0, 3.0])),
            ),
            (
                "y",
                Column::Float64(Float64Column::new(vec![2.0, 4.0, 6.0])),
            ),
            (
                "label",
                Column::String(StringColumn::new(vec![
                    "a".to_string(),
                    "b".to_string(),
                    "c".to_string(),
                ])),
            ),
        ]);

        assert!(df.linear_regression("y", &["label"]).is_err());
        assert!(df.linear_regression("label", &["x"]).is_err());
        assert!(df.linear_regression("y", &[]).is_err());
        assert!(df.linear_regression("y", &["x", "x"]).is_err());
        assert!(df.linear_regression("y", &["y"]).is_err());
        assert!(df.linear_regression("y", &["missing"]).is_err());
    }

    #[test]
    fn reports_insufficient_complete_rows() {
        let df = frame(vec![
            ("x", Column::Float64(Float64Column::new(vec![1.0, 2.0]))),
            ("y", Column::Float64(Float64Column::new(vec![2.0, 4.0]))),
            ("z", Column::Float64(Float64Column::new(vec![1.0, 3.0]))),
        ]);

        // Two rows cannot identify an intercept plus two slopes.
        let error = df
            .linear_regression("y", &["x", "z"])
            .expect_err("under-determined system must be rejected");
        assert!(
            error.to_string().contains("complete rows"),
            "unexpected error: {}",
            error
        );
    }
}
