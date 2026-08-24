//! Data preprocessing for machine learning
//!
//! This module provides tools for preprocessing data before feeding it to
//! machine learning algorithms, including scaling, normalization, encoding
//! categorical variables, generating polynomial features, discretising
//! continuous features and handling missing values.
//!
//! # Missing values
//!
//! Missing numeric data is represented by `NaN` (the representation the rest of
//! pandrs' numeric ML paths use). Every transformer in this module treats `NaN` as
//! *absent*: statistics are computed from the observed values only, and — except in
//! [`Imputer`], whose entire job is to fill them — missing entries stay missing in the
//! output rather than being silently replaced by `0.0`.

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::series::Series;
use std::collections::{BTreeSet, HashMap};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Whether `name` names a column whose *concrete storage type* is numeric.
///
/// The check is an exact downcast rather than "can these strings be parsed as
/// numbers", so digit-coded categorical labels stored as `Series<String>` (`"1"`,
/// `"2"`, …) are not mistaken for continuous features.
fn is_numeric_column(df: &DataFrame, name: &str) -> bool {
    df.get_column::<f64>(name).is_ok()
        || df.get_column::<f32>(name).is_ok()
        || df.get_column::<i64>(name).is_ok()
        || df.get_column::<i32>(name).is_ok()
        || df.get_column::<bool>(name).is_ok()
}

/// Whether `name` names a column whose concrete storage type is `String`.
fn is_string_column(df: &DataFrame, name: &str) -> bool {
    df.get_column::<String>(name).is_ok()
}

/// Resolve the numeric target columns of a transformer.
///
/// * `Some(cols)` — every listed column must exist and be numeric (otherwise `Err`).
/// * `None` — every column whose concrete type is numeric. Non-numeric columns are
///   *filtered out* here rather than making the whole call fail: `columns = None`
///   used to hard-error on the first non-`f64` column in the frame, which made the
///   documented "scale all numeric columns" default unusable on any mixed frame.
fn resolve_numeric_targets(df: &DataFrame, columns: &Option<Vec<String>>) -> Result<Vec<String>> {
    match columns {
        Some(cols) => {
            for name in cols {
                if !df.has_column(name) {
                    return Err(Error::ColumnNotFound(name.clone()));
                }
                if !is_numeric_column(df, name) {
                    return Err(Error::InvalidValue(format!(
                        "Column '{}' is not numeric and cannot be used here; \
                         remove it from the column list or encode it first",
                        name
                    )));
                }
            }
            Ok(cols.clone())
        }
        None => Ok(df
            .column_names()
            .iter()
            .filter(|name| is_numeric_column(df, name.as_str()))
            .cloned()
            .collect()),
    }
}

/// Copy a column into `result` preserving its concrete element type.
fn clone_column_into(df: &DataFrame, result: &mut DataFrame, name: &str) -> Result<()> {
    macro_rules! try_clone {
        ($($ty:ty),+ $(,)?) => {
            $(
                if let Ok(series) = df.get_column::<$ty>(name) {
                    result.add_column(name.to_string(), series.clone())?;
                    return Ok(());
                }
            )+
        };
    }

    try_clone!(f64, f32, i64, i32, i16, i8, u64, u32, u16, u8, usize, isize, bool, String);

    Err(Error::InvalidValue(format!(
        "Column '{}' has an element type that cannot be copied through this \
         transformer (supported: f64, f32, i64, i32, i16, i8, u64, u32, u16, u8, \
         usize, isize, bool, String)",
        name
    )))
}

/// Numeric values of a column as `f64`, with `NaN` preserved as "missing".
fn numeric_values(df: &DataFrame, name: &str) -> Result<Vec<f64>> {
    df.get_column_numeric_values(name)
}

/// The observed (non-missing) values of a numeric column.
fn observed(values: &[f64]) -> Vec<f64> {
    values.iter().copied().filter(|v| !v.is_nan()).collect()
}

/// Ascending sort of a slice of finite/observed values.
fn sorted_ascending(values: &[f64]) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted
}

/// Linear-interpolated quantile of an ascending-sorted, non-empty slice.
///
/// Matches NumPy's default (`method="linear"`), which is also what
/// `sklearn.preprocessing.KBinsDiscretizer(strategy="quantile")` uses.
fn quantile_sorted(sorted: &[f64], q: f64) -> Result<f64> {
    if sorted.is_empty() {
        return Err(Error::InvalidValue(
            "Cannot compute a quantile of an empty (all-missing) column".into(),
        ));
    }
    if sorted.len() == 1 {
        return Ok(sorted[0]);
    }
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;
    if lower == upper {
        Ok(sorted[lower])
    } else {
        let weight = pos - lower as f64;
        Ok(sorted[lower] + weight * (sorted[upper] - sorted[lower]))
    }
}

/// Median of an ascending-sorted, non-empty slice.
fn median_sorted(sorted: &[f64]) -> Result<f64> {
    if sorted.is_empty() {
        return Err(Error::InvalidValue(
            "Cannot compute a median of an empty (all-missing) column".into(),
        ));
    }
    let n = sorted.len();
    if n % 2 == 1 {
        Ok(sorted[n / 2])
    } else {
        Ok((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0)
    }
}

/// Most frequent value; ties are broken by the smaller value (as in sklearn).
fn most_frequent(values: &[f64]) -> Result<f64> {
    if values.is_empty() {
        return Err(Error::InvalidValue(
            "Cannot compute a most-frequent value of an empty (all-missing) column".into(),
        ));
    }
    let mut counts: HashMap<u64, usize> = HashMap::new();
    for &v in values {
        *counts.entry(v.to_bits()).or_insert(0) += 1;
    }
    let mut best = values[0];
    let mut best_count = 0usize;
    for (&bits, &count) in &counts {
        let value = f64::from_bits(bits);
        if count > best_count || (count == best_count && value < best) {
            best_count = count;
            best = value;
        }
    }
    Ok(best)
}

// ---------------------------------------------------------------------------
// StandardScaler
// ---------------------------------------------------------------------------

/// Standard scaler for normalizing features to zero mean and unit variance
#[derive(Debug, Clone)]
pub struct StandardScaler {
    /// Mean values for each feature
    pub means: Option<HashMap<String, f64>>,
    /// Standard deviation values for each feature
    pub stds: Option<HashMap<String, f64>>,
    /// Columns to scale (if None, scale all numeric columns)
    pub columns: Option<Vec<String>>,
}

impl StandardScaler {
    /// Create a new StandardScaler
    pub fn new() -> Self {
        StandardScaler {
            means: None,
            stds: None,
            columns: None,
        }
    }

    /// Specify columns to scale
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }

    /// Fit the scaler to the data
    ///
    /// Statistics are computed from the observed (non-`NaN`) values of each target
    /// column. A column with no observed values at all is an error: there is no mean
    /// or standard deviation to learn from it.
    pub fn fit(&mut self, df: &DataFrame) -> Result<()> {
        let columns = resolve_numeric_targets(df, &self.columns)?;

        let mut means = HashMap::new();
        let mut stds = HashMap::new();

        for col_name in columns {
            let values = numeric_values(df, &col_name)?;
            if values.is_empty() {
                continue;
            }

            let present = observed(&values);
            if present.is_empty() {
                return Err(Error::InvalidValue(format!(
                    "StandardScaler: column '{}' has no non-missing values; \
                     no mean/standard deviation can be computed from it",
                    col_name
                )));
            }

            let n = present.len() as f64;
            let mean = present.iter().sum::<f64>() / n;
            let variance = present.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;

            means.insert(col_name.clone(), mean);
            stds.insert(col_name, variance.sqrt());
        }

        self.means = Some(means);
        self.stds = Some(stds);

        Ok(())
    }

    /// Transform data using the fitted scaler
    ///
    /// Columns the scaler was not fitted on (including non-numeric ones) are copied
    /// through unchanged, and missing values stay missing.
    pub fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        let means = self
            .means
            .as_ref()
            .ok_or_else(|| Error::InvalidValue("StandardScaler not fitted".into()))?;
        let stds = self
            .stds
            .as_ref()
            .ok_or_else(|| Error::InvalidValue("StandardScaler not fitted".into()))?;

        let mut result = DataFrame::new();

        for col_name in df.column_names() {
            match (means.get(col_name.as_str()), stds.get(col_name.as_str())) {
                (Some(&mean), Some(&std_dev)) => {
                    let values = numeric_values(df, col_name)?;
                    let scaled: Vec<f64> = values
                        .iter()
                        .map(|&x| {
                            if x.is_nan() {
                                // Missing in, missing out.
                                f64::NAN
                            } else if std_dev > 1e-10 {
                                (x - mean) / std_dev
                            } else {
                                // Zero-variance column: centring is all that is
                                // defined (sklearn keeps scale_ = 1.0 here).
                                x - mean
                            }
                        })
                        .collect();
                    result.add_column(
                        col_name.clone(),
                        Series::new(scaled, Some(col_name.clone()))?,
                    )?;
                }
                _ => clone_column_into(df, &mut result, col_name)?,
            }
        }

        Ok(result)
    }

    /// Fit the scaler to the data and transform it in one step
    pub fn fit_transform(&mut self, df: &DataFrame) -> Result<DataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MinMaxScaler
// ---------------------------------------------------------------------------

/// Min-Max scaler for scaling features to a specific range
#[derive(Debug, Clone)]
pub struct MinMaxScaler {
    /// Minimum values for each feature
    pub min_values: Option<HashMap<String, f64>>,
    /// Maximum values for each feature
    pub max_values: Option<HashMap<String, f64>>,
    /// Columns to scale (if None, scale all numeric columns)
    pub columns: Option<Vec<String>>,
    /// Feature range (min, max)
    pub feature_range: (f64, f64),
}

impl MinMaxScaler {
    /// Create a new MinMaxScaler with default range [0, 1]
    pub fn new() -> Self {
        MinMaxScaler {
            min_values: None,
            max_values: None,
            columns: None,
            feature_range: (0.0, 1.0),
        }
    }

    /// Create a new MinMaxScaler with custom range
    pub fn new_with_range(min: f64, max: f64) -> Self {
        MinMaxScaler {
            min_values: None,
            max_values: None,
            columns: None,
            feature_range: (min, max),
        }
    }

    /// Set the feature range (builder pattern)
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.feature_range = (min, max);
        self
    }

    /// Specify columns to scale
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }

    /// Fit the scaler to the data
    pub fn fit(&mut self, df: &DataFrame) -> Result<()> {
        let columns = resolve_numeric_targets(df, &self.columns)?;

        let mut min_values = HashMap::new();
        let mut max_values = HashMap::new();

        for col_name in columns {
            let values = numeric_values(df, &col_name)?;
            if values.is_empty() {
                continue;
            }

            let present = observed(&values);
            if present.is_empty() {
                return Err(Error::InvalidValue(format!(
                    "MinMaxScaler: column '{}' has no non-missing values; \
                     no minimum/maximum can be computed from it",
                    col_name
                )));
            }

            let min_val = present.iter().copied().fold(f64::INFINITY, f64::min);
            let max_val = present.iter().copied().fold(f64::NEG_INFINITY, f64::max);

            min_values.insert(col_name.clone(), min_val);
            max_values.insert(col_name, max_val);
        }

        self.min_values = Some(min_values);
        self.max_values = Some(max_values);

        Ok(())
    }

    /// Transform data using the fitted scaler
    pub fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        let min_values = self
            .min_values
            .as_ref()
            .ok_or_else(|| Error::InvalidValue("MinMaxScaler not fitted".into()))?;
        let max_values = self
            .max_values
            .as_ref()
            .ok_or_else(|| Error::InvalidValue("MinMaxScaler not fitted".into()))?;
        let (feature_min, feature_max) = self.feature_range;

        let mut result = DataFrame::new();

        for col_name in df.column_names() {
            match (
                min_values.get(col_name.as_str()),
                max_values.get(col_name.as_str()),
            ) {
                (Some(&min_val), Some(&max_val)) => {
                    let values = numeric_values(df, col_name)?;
                    let span = max_val - min_val;
                    let scaled: Vec<f64> = values
                        .iter()
                        .map(|&x| {
                            if x.is_nan() {
                                f64::NAN
                            } else if span.abs() > 1e-10 {
                                (x - min_val) / span * (feature_max - feature_min) + feature_min
                            } else {
                                // Degenerate (constant) column: sklearn maps it to
                                // feature_range[0].
                                feature_min
                            }
                        })
                        .collect();
                    result.add_column(
                        col_name.clone(),
                        Series::new(scaled, Some(col_name.clone()))?,
                    )?;
                }
                _ => clone_column_into(df, &mut result, col_name)?,
            }
        }

        Ok(result)
    }

    /// Fit the scaler to the data and transform it in one step
    pub fn fit_transform(&mut self, df: &DataFrame) -> Result<DataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// OneHotEncoder
// ---------------------------------------------------------------------------

/// What [`OneHotEncoder::transform`] does with a category it never saw during `fit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandleUnknown {
    /// Return an error (scikit-learn's default)
    Error,
    /// Emit an all-zero indicator row for that value
    Ignore,
}

/// One-hot encoder for categorical variables
///
/// `fit` learns the distinct categories of each target column (sorted, so the emitted
/// column order is stable), and `transform` replaces each encoded column with one 0/1
/// indicator column per category.
#[derive(Debug, Clone)]
pub struct OneHotEncoder {
    /// Categories for each feature, learned by `fit`
    pub categories: Option<HashMap<String, Vec<String>>>,
    /// Columns to encode (if None, encode all string-typed columns)
    pub columns: Option<Vec<String>>,
    /// Whether to drop the first category (dummy encoding, avoids collinearity)
    pub drop_first: bool,
    /// Prefix for new column names (defaults to the source column name)
    pub prefix: Option<String>,
    /// Behaviour on unseen categories at transform time
    pub handle_unknown: HandleUnknown,
}

impl OneHotEncoder {
    /// Create a new OneHotEncoder
    pub fn new() -> Self {
        OneHotEncoder {
            categories: None,
            columns: None,
            drop_first: false,
            prefix: None,
            handle_unknown: HandleUnknown::Error,
        }
    }

    /// Specify columns to encode
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }

    /// Set whether to drop the first category
    pub fn drop_first(mut self, drop_first: bool) -> Self {
        self.drop_first = drop_first;
        self
    }

    /// Set prefix for new column names
    pub fn with_prefix(mut self, prefix: String) -> Self {
        self.prefix = Some(prefix);
        self
    }

    /// Set the behaviour for categories not seen during `fit`
    pub fn with_handle_unknown(mut self, handle_unknown: HandleUnknown) -> Self {
        self.handle_unknown = handle_unknown;
        self
    }

    /// Resolve which columns to encode: the explicit list, or every string column.
    fn resolve_targets(&self, df: &DataFrame) -> Result<Vec<String>> {
        match &self.columns {
            Some(cols) => {
                for name in cols {
                    if !df.has_column(name) {
                        return Err(Error::ColumnNotFound(name.clone()));
                    }
                }
                Ok(cols.clone())
            }
            None => Ok(df
                .column_names()
                .iter()
                .filter(|name| is_string_column(df, name.as_str()))
                .cloned()
                .collect()),
        }
    }

    /// Learn the categories of each target column.
    pub fn fit(&mut self, df: &DataFrame) -> Result<()> {
        let targets = self.resolve_targets(df)?;

        let mut categories = HashMap::new();
        for col_name in targets {
            let values = df.get_column_string_values(&col_name)?;
            let unique: Vec<String> = values
                .into_iter()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            if unique.is_empty() {
                return Err(Error::InvalidValue(format!(
                    "OneHotEncoder: column '{}' has no values to learn categories from",
                    col_name
                )));
            }
            categories.insert(col_name, unique);
        }

        self.categories = Some(categories);
        Ok(())
    }

    /// Replace every fitted column with its 0/1 indicator columns.
    ///
    /// Indicator columns are named `"{prefix or column}_{category}"`. Columns the
    /// encoder was not fitted on are copied through unchanged.
    pub fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        let categories = self
            .categories
            .as_ref()
            .ok_or_else(|| Error::InvalidValue("OneHotEncoder not fitted".into()))?;

        let mut result = DataFrame::new();

        for col_name in df.column_names() {
            let Some(cats) = categories.get(col_name.as_str()) else {
                clone_column_into(df, &mut result, col_name)?;
                continue;
            };

            let values = df.get_column_string_values(col_name)?;

            if self.handle_unknown == HandleUnknown::Error {
                let known: BTreeSet<&String> = cats.iter().collect();
                for (row, value) in values.iter().enumerate() {
                    if !known.contains(value) {
                        return Err(Error::InvalidValue(format!(
                            "OneHotEncoder: column '{}' contains category '{}' at row {} \
                             which was not seen during fit; known categories: {:?}. \
                             Use with_handle_unknown(HandleUnknown::Ignore) to encode it \
                             as an all-zero row instead",
                            col_name, value, row, cats
                        )));
                    }
                }
            }

            let base = self.prefix.clone().unwrap_or_else(|| col_name.clone());
            let skip = usize::from(self.drop_first);
            for category in cats.iter().skip(skip) {
                let indicator: Vec<f64> = values
                    .iter()
                    .map(|v| if v == category { 1.0 } else { 0.0 })
                    .collect();
                let new_name = format!("{}_{}", base, category);
                result.add_column(
                    new_name.clone(),
                    Series::new(indicator, Some(new_name.clone()))?,
                )?;
            }
        }

        Ok(result)
    }

    /// Fit to the data and transform it in one step
    pub fn fit_transform(&mut self, df: &DataFrame) -> Result<DataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

impl Default for OneHotEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PolynomialFeatures
// ---------------------------------------------------------------------------

/// Generate polynomial and interaction features
///
/// Produces the complete multiset basis up to `degree`: every product
/// `x_{i1}·x_{i2}·…·x_{ik}` with `1 <= k <= degree` and `i1 <= i2 <= … <= ik`
/// (`i1 < i2 < … < ik` when `interaction_only` is set), i.e. exactly the terms
/// `sklearn.preprocessing.PolynomialFeatures` emits, in the same order.
#[derive(Debug, Clone)]
pub struct PolynomialFeatures {
    /// Degree of the polynomial
    pub degree: usize,
    /// Whether to include bias term (constant feature)
    pub include_bias: bool,
    /// Whether to include interaction features only (no powers of a single feature)
    pub interaction_only: bool,
    /// Columns to use (if None, use all numeric columns)
    pub columns: Option<Vec<String>>,
    /// Input feature columns recorded by `fit`
    pub input_features: Option<Vec<String>>,
}

impl PolynomialFeatures {
    /// Create a new PolynomialFeatures instance
    pub fn new(degree: usize) -> Self {
        PolynomialFeatures {
            degree,
            include_bias: true,
            interaction_only: false,
            columns: None,
            input_features: None,
        }
    }

    /// Set whether to include bias term
    pub fn include_bias(mut self, include_bias: bool) -> Self {
        self.include_bias = include_bias;
        self
    }

    /// Set whether to include interaction features only
    pub fn interaction_only(mut self, interaction_only: bool) -> Self {
        self.interaction_only = interaction_only;
        self
    }

    /// Specify columns to use
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }

    /// Record the input feature columns the expansion will be built from.
    pub fn fit(&mut self, df: &DataFrame) -> Result<()> {
        if self.degree == 0 && !self.include_bias {
            return Err(Error::InvalidInput(
                "PolynomialFeatures with degree = 0 and include_bias = false would \
                 generate no features at all"
                    .into(),
            ));
        }

        let targets = resolve_numeric_targets(df, &self.columns)?;
        if targets.is_empty() {
            return Err(Error::InvalidValue(
                "PolynomialFeatures: no numeric feature columns to expand".into(),
            ));
        }

        self.input_features = Some(targets);
        Ok(())
    }

    /// Exponent tuples of the expansion, in scikit-learn's order.
    ///
    /// Degree 1 terms come first (in column order), then degree 2, and so on; within
    /// a degree, index multisets are visited in lexicographic order.
    fn exponent_tuples(&self, n_features: usize) -> Vec<Vec<usize>> {
        let mut tuples = Vec::new();

        for degree in 1..=self.degree {
            let mut current: Vec<usize> = Vec::with_capacity(degree);
            combinations(
                n_features,
                degree,
                0,
                self.interaction_only,
                &mut current,
                &mut |indices: &[usize]| {
                    let mut exponents = vec![0usize; n_features];
                    for &idx in indices {
                        exponents[idx] += 1;
                    }
                    tuples.push(exponents);
                },
            );
        }

        tuples
    }

    /// Human-readable name of one expansion term, e.g. `"a"`, `"a^2"`, `"a b^2"`.
    fn term_name(features: &[String], exponents: &[usize]) -> String {
        let mut parts = Vec::new();
        for (idx, &exp) in exponents.iter().enumerate() {
            match exp {
                0 => {}
                1 => parts.push(features[idx].clone()),
                _ => parts.push(format!("{}^{}", features[idx], exp)),
            }
        }
        parts.join(" ")
    }

    /// Generate the polynomial expansion.
    ///
    /// Columns that are not part of the expansion are copied through unchanged, so the
    /// target column of a supervised problem survives the transform. The degree-1
    /// terms reproduce the input columns under their original names. Rows with a
    /// missing value in any input column produce missing values for every term that
    /// involves that column.
    pub fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        let features = self
            .input_features
            .as_ref()
            .ok_or_else(|| Error::InvalidValue("PolynomialFeatures not fitted".into()))?;

        let mut columns_data = Vec::with_capacity(features.len());
        for name in features {
            if !df.has_column(name) {
                return Err(Error::ColumnNotFound(name.clone()));
            }
            columns_data.push(numeric_values(df, name)?);
        }

        let n_rows = columns_data.first().map(|c| c.len()).unwrap_or(0);
        for (idx, column) in columns_data.iter().enumerate() {
            if column.len() != n_rows {
                return Err(Error::DimensionMismatch(format!(
                    "Feature column '{}' has {} rows but '{}' has {}",
                    features[idx],
                    column.len(),
                    features[0],
                    n_rows
                )));
            }
        }

        let mut result = DataFrame::new();

        // Pass through everything that is not an expansion input.
        for col_name in df.column_names() {
            if !features.contains(col_name) {
                clone_column_into(df, &mut result, col_name)?;
            }
        }

        if self.include_bias {
            let bias = vec![1.0_f64; n_rows];
            result.add_column("bias".to_string(), Series::new(bias, Some("bias".into()))?)?;
        }

        for exponents in self.exponent_tuples(features.len()) {
            let name = Self::term_name(features, &exponents);
            let mut values = Vec::with_capacity(n_rows);
            for row in 0..n_rows {
                let mut product = 1.0_f64;
                for (idx, &exp) in exponents.iter().enumerate() {
                    if exp == 0 {
                        continue;
                    }
                    let value = columns_data[idx][row];
                    if value.is_nan() {
                        product = f64::NAN;
                        break;
                    }
                    product *= value.powi(exp as i32);
                }
                values.push(product);
            }
            result.add_column(name.clone(), Series::new(values, Some(name))?)?;
        }

        Ok(result)
    }

    /// Fit to the data and transform it in one step
    pub fn fit_transform(&mut self, df: &DataFrame) -> Result<DataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

/// Visit every index multiset of size `size` drawn from `0..n` in lexicographic order.
///
/// With `strict` set, indices must be strictly increasing (combinations without
/// replacement — the `interaction_only` basis); otherwise they are non-decreasing
/// (combinations *with* replacement — the full polynomial basis).
fn combinations(
    n: usize,
    size: usize,
    start: usize,
    strict: bool,
    current: &mut Vec<usize>,
    visit: &mut dyn FnMut(&[usize]),
) {
    if current.len() == size {
        visit(current);
        return;
    }
    for idx in start..n {
        current.push(idx);
        let next_start = if strict { idx + 1 } else { idx };
        combinations(n, size, next_start, strict, current, visit);
        current.pop();
    }
}

// ---------------------------------------------------------------------------
// Binner
// ---------------------------------------------------------------------------

/// Bin continuous features into discrete bins
///
/// Bin `i` covers `[edges[i], edges[i+1])`, with the final bin closed on the right, so
/// every observed value maps into `0 ..= n_bins-1` (values outside the fitted range are
/// clipped into the first/last bin) — the same convention as
/// `sklearn.preprocessing.KBinsDiscretizer(encode="ordinal")`.
#[derive(Debug, Clone)]
pub struct Binner {
    /// Number of bins
    pub n_bins: usize,
    /// Bin edges for each feature (length `n_bins + 1` before duplicate removal)
    pub bin_edges: Option<HashMap<String, Vec<f64>>>,
    /// Strategy for binning ('uniform', 'quantile', or 'kmeans')
    pub strategy: String,
    /// Columns to bin (if None, bin all numeric columns)
    pub columns: Option<Vec<String>>,
}

impl Binner {
    /// Create a new Binner with uniform strategy
    pub fn new(n_bins: usize) -> Self {
        Binner {
            n_bins,
            bin_edges: None,
            strategy: "uniform".to_string(),
            columns: None,
        }
    }

    /// Set binning strategy (`"uniform"`, `"quantile"` or `"kmeans"`)
    pub fn with_strategy(mut self, strategy: &str) -> Self {
        self.strategy = strategy.to_string();
        self
    }

    /// Specify columns to bin
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }

    /// Learn the bin edges of every target column.
    pub fn fit(&mut self, df: &DataFrame) -> Result<()> {
        if self.n_bins < 2 {
            return Err(Error::InvalidInput(format!(
                "Binner requires at least 2 bins, got {}",
                self.n_bins
            )));
        }

        let targets = resolve_numeric_targets(df, &self.columns)?;
        let mut bin_edges = HashMap::new();

        for col_name in targets {
            let values = numeric_values(df, &col_name)?;
            let present = observed(&values);
            if present.is_empty() {
                return Err(Error::InvalidValue(format!(
                    "Binner: column '{}' has no non-missing values to derive bin edges from",
                    col_name
                )));
            }

            let sorted = sorted_ascending(&present);
            let edges = match self.strategy.to_lowercase().as_str() {
                "uniform" => Self::uniform_edges(&sorted, self.n_bins),
                "quantile" => Self::quantile_edges(&sorted, self.n_bins),
                "kmeans" => Self::kmeans_edges(&sorted, self.n_bins),
                other => Err(Error::InvalidValue(format!(
                    "Unknown binning strategy '{}'; expected 'uniform', 'quantile' or 'kmeans'",
                    other
                ))),
            }?;

            let edges = dedupe_edges(&edges);
            if edges.len() < 2 {
                return Err(Error::InvalidValue(format!(
                    "Binner: column '{}' is constant (or has fewer distinct values than \
                     bins); it cannot be split into {} bins",
                    col_name, self.n_bins
                )));
            }

            bin_edges.insert(col_name, edges);
        }

        self.bin_edges = Some(bin_edges);
        Ok(())
    }

    /// Equal-width edges spanning `[min, max]`.
    fn uniform_edges(sorted: &[f64], n_bins: usize) -> Result<Vec<f64>> {
        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let width = (max - min) / n_bins as f64;
        let mut edges: Vec<f64> = (0..=n_bins).map(|i| min + width * i as f64).collect();
        // Guard against floating-point drift on the closing edge.
        if let Some(last) = edges.last_mut() {
            *last = max;
        }
        Ok(edges)
    }

    /// Equal-frequency edges: the `i/n_bins` quantiles, `i = 0..=n_bins`.
    fn quantile_edges(sorted: &[f64], n_bins: usize) -> Result<Vec<f64>> {
        let mut edges = Vec::with_capacity(n_bins + 1);
        for i in 0..=n_bins {
            edges.push(quantile_sorted(sorted, i as f64 / n_bins as f64)?);
        }
        Ok(edges)
    }

    /// 1-D k-means edges: cluster the column, then cut at the midpoints between
    /// neighbouring centroids (the strategy `KBinsDiscretizer(strategy="kmeans")`
    /// uses). Centroids are initialised on an equally spaced grid, so the result is
    /// deterministic and needs no RNG.
    fn kmeans_edges(sorted: &[f64], n_bins: usize) -> Result<Vec<f64>> {
        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        if (max - min).abs() <= f64::EPSILON {
            return Ok(vec![min, max]);
        }

        let step = (max - min) / (n_bins - 1) as f64;
        let mut centers: Vec<f64> = (0..n_bins).map(|i| min + step * i as f64).collect();

        // Lloyd's algorithm. It decreases the objective monotonically, so a bounded
        // number of sweeps is safe; convergence usually happens in a handful.
        for _ in 0..300 {
            let mut sums = vec![0.0_f64; n_bins];
            let mut counts = vec![0usize; n_bins];

            for &value in sorted {
                let mut best = 0usize;
                let mut best_dist = (value - centers[0]).abs();
                for (idx, &center) in centers.iter().enumerate().skip(1) {
                    let dist = (value - center).abs();
                    if dist < best_dist {
                        best = idx;
                        best_dist = dist;
                    }
                }
                sums[best] += value;
                counts[best] += 1;
            }

            let mut shift = 0.0_f64;
            for idx in 0..n_bins {
                if counts[idx] > 0 {
                    let updated = sums[idx] / counts[idx] as f64;
                    shift = shift.max((updated - centers[idx]).abs());
                    centers[idx] = updated;
                }
                // An empty cluster keeps its previous centre: moving it would need an
                // arbitrary reseeding rule, and the duplicate edge it produces is
                // removed below.
            }

            if shift <= 1e-12 * (max - min).abs().max(1.0) {
                break;
            }
        }

        centers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut edges = Vec::with_capacity(n_bins + 1);
        edges.push(min);
        for pair in centers.windows(2) {
            edges.push((pair[0] + pair[1]) / 2.0);
        }
        edges.push(max);
        Ok(edges)
    }

    /// Replace every fitted column with its bin index.
    ///
    /// Missing values stay missing (`NaN` in, `NaN` out); columns the binner was not
    /// fitted on are copied through unchanged.
    pub fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        let bin_edges = self
            .bin_edges
            .as_ref()
            .ok_or_else(|| Error::InvalidValue("Binner not fitted".into()))?;

        let mut result = DataFrame::new();

        for col_name in df.column_names() {
            let Some(edges) = bin_edges.get(col_name.as_str()) else {
                clone_column_into(df, &mut result, col_name)?;
                continue;
            };

            let values = numeric_values(df, col_name)?;
            let n_bins = edges.len() - 1;
            let inner = &edges[1..edges.len() - 1];

            let binned: Vec<f64> = values
                .iter()
                .map(|&v| {
                    if v.is_nan() {
                        f64::NAN
                    } else {
                        // Number of inner edges at or below `v` -> bin index, clipped
                        // into the fitted range.
                        let idx = inner.partition_point(|&edge| edge <= v);
                        idx.min(n_bins - 1) as f64
                    }
                })
                .collect();

            result.add_column(
                col_name.clone(),
                Series::new(binned, Some(col_name.clone()))?,
            )?;
        }

        Ok(result)
    }

    /// Fit to the data and transform it in one step
    pub fn fit_transform(&mut self, df: &DataFrame) -> Result<DataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

/// Drop consecutive duplicate edges (degenerate bins) while keeping ascending order.
fn dedupe_edges(edges: &[f64]) -> Vec<f64> {
    let mut result: Vec<f64> = Vec::with_capacity(edges.len());
    for &edge in edges {
        match result.last() {
            Some(&last) if (edge - last).abs() <= f64::EPSILON * last.abs().max(1.0) => {}
            _ => result.push(edge),
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Imputer
// ---------------------------------------------------------------------------

/// Strategy for imputing missing values
#[derive(Debug, Clone, PartialEq)]
pub enum ImputeStrategy {
    /// Impute with mean
    Mean,
    /// Impute with median
    Median,
    /// Impute with most frequent value
    MostFrequent,
    /// Impute with constant value
    Constant(f64),
}

/// Imputer for handling missing values
///
/// `fit` derives one fill value per target column from that column's *observed*
/// values; `transform` substitutes it for every `NaN`. A column with no observed
/// values at all is an error for the data-derived strategies: there is nothing to
/// compute a mean/median/mode from, and inventing `0.0` would be indistinguishable
/// downstream from a genuine zero reading.
#[derive(Debug, Clone)]
pub struct Imputer {
    /// Strategy for imputing missing values
    pub strategy: ImputeStrategy,
    /// Fill values for each feature, learned by `fit`
    pub fill_values: Option<HashMap<String, f64>>,
    /// Columns to impute (if None, impute all numeric columns)
    pub columns: Option<Vec<String>>,
}

impl Imputer {
    /// Create a new Imputer with mean strategy
    pub fn new() -> Self {
        Imputer {
            strategy: ImputeStrategy::Mean,
            fill_values: None,
            columns: None,
        }
    }

    /// Set imputation strategy
    pub fn with_strategy(mut self, strategy: ImputeStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Specify columns to impute
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }

    /// Learn one fill value per target column.
    pub fn fit(&mut self, df: &DataFrame) -> Result<()> {
        let targets = resolve_numeric_targets(df, &self.columns)?;
        let mut fill_values = HashMap::new();

        for col_name in targets {
            let values = numeric_values(df, &col_name)?;
            let present = observed(&values);

            if present.is_empty() && !matches!(self.strategy, ImputeStrategy::Constant(_)) {
                return Err(Error::InvalidValue(format!(
                    "Imputer: column '{}' has no non-missing values; a {:?} fill value \
                     cannot be computed from it",
                    col_name, self.strategy
                )));
            }

            let fill = match self.strategy {
                ImputeStrategy::Mean => present.iter().sum::<f64>() / present.len() as f64,
                ImputeStrategy::Median => median_sorted(&sorted_ascending(&present))?,
                ImputeStrategy::MostFrequent => most_frequent(&present)?,
                ImputeStrategy::Constant(value) => value,
            };

            fill_values.insert(col_name, fill);
        }

        self.fill_values = Some(fill_values);
        Ok(())
    }

    /// Fill missing values of the fitted columns.
    ///
    /// Columns the imputer was not fitted on are copied through unchanged — including
    /// their missing values, which are *not* silently replaced.
    pub fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        let fill_values = self
            .fill_values
            .as_ref()
            .ok_or_else(|| Error::InvalidValue("Imputer not fitted".into()))?;

        let mut result = DataFrame::new();

        for col_name in df.column_names() {
            let Some(&fill) = fill_values.get(col_name.as_str()) else {
                clone_column_into(df, &mut result, col_name)?;
                continue;
            };

            let values = numeric_values(df, col_name)?;
            let imputed: Vec<f64> = values
                .iter()
                .map(|&v| if v.is_nan() { fill } else { v })
                .collect();

            result.add_column(
                col_name.clone(),
                Series::new(imputed, Some(col_name.clone()))?,
            )?;
        }

        Ok(result)
    }

    /// Fit to the data and transform it in one step
    pub fn fit_transform(&mut self, df: &DataFrame) -> Result<DataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

impl Default for Imputer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FeatureSelector
// ---------------------------------------------------------------------------

/// Feature selector for selecting a subset of features
#[derive(Debug, Clone)]
pub struct FeatureSelector {
    /// Columns to select
    pub columns: Vec<String>,
}

impl FeatureSelector {
    /// Create a new FeatureSelector
    pub fn new(columns: Vec<String>) -> Self {
        FeatureSelector { columns }
    }

    /// Transform data by selecting features, preserving each column's element type.
    pub fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        let mut result = DataFrame::new();

        for col_name in &self.columns {
            if !df.has_column(col_name) {
                return Err(Error::ColumnNotFound(col_name.to_string()));
            }
            clone_column_into(df, &mut result, col_name)?;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn df_from_f64(columns: Vec<(&str, Vec<f64>)>) -> DataFrame {
        let mut df = DataFrame::new();
        for (name, values) in columns {
            df.add_column(
                name.to_string(),
                Series::new(values, Some(name.to_string())).expect("series"),
            )
            .expect("add column");
        }
        df
    }

    #[test]
    fn test_polynomial_term_count_and_names() {
        let df = df_from_f64(vec![("a", vec![1.0, 2.0]), ("b", vec![3.0, 4.0])]);
        let mut poly = PolynomialFeatures::new(2).include_bias(false);
        let out = poly.fit_transform(&df).expect("fit_transform");

        // a, b, a^2, a b, b^2
        let names = out.column_names().to_vec();
        assert_eq!(names.len(), 5, "unexpected columns: {:?}", names);
        for expected in ["a", "b", "a^2", "a b", "b^2"] {
            assert!(
                names.iter().any(|n| n == expected),
                "missing term '{}' in {:?}",
                expected,
                names
            );
        }
    }

    #[test]
    fn test_binner_uniform_edges() {
        let df = df_from_f64(vec![("x", vec![0.0, 1.0, 2.0, 3.0, 4.0])]);
        let mut binner = Binner::new(2);
        binner.fit(&df).expect("fit");
        let edges = binner
            .bin_edges
            .as_ref()
            .expect("edges")
            .get("x")
            .expect("column edges")
            .clone();
        assert_eq!(edges, vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn test_imputer_mean_skips_missing() {
        let df = df_from_f64(vec![("x", vec![1.0, f64::NAN, 3.0])]);
        let mut imputer = Imputer::new();
        let out = imputer.fit_transform(&df).expect("fit_transform");
        let values = out
            .get_column::<f64>("x")
            .expect("column")
            .values()
            .to_vec();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }
}
