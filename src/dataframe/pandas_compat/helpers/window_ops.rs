//! Window operation helper functions for rolling and expanding calculations

use crate::core::error::Result;
use crate::dataframe::DataFrame;

/// Compute rolling sum with configurable window
pub fn rolling_sum(
    df: &DataFrame,
    column: &str,
    window: usize,
    min_periods: Option<usize>,
) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
    let min_periods = min_periods.unwrap_or(window);

    let mut result = Vec::with_capacity(values.len());

    for i in 0..values.len() {
        let start = i.saturating_sub(window.saturating_sub(1));
        let window_values: Vec<f64> = values[start..=i]
            .iter()
            .filter(|v| !v.is_nan())
            .copied()
            .collect();

        if window_values.len() >= min_periods {
            result.push(window_values.iter().sum());
        } else {
            result.push(f64::NAN);
        }
    }

    Ok(result)
}

/// Compute rolling mean
pub fn rolling_mean(
    df: &DataFrame,
    column: &str,
    window: usize,
    min_periods: Option<usize>,
) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
    let min_periods = min_periods.unwrap_or(window);

    let mut result = Vec::with_capacity(values.len());

    for i in 0..values.len() {
        let start = i.saturating_sub(window.saturating_sub(1));
        let window_values: Vec<f64> = values[start..=i]
            .iter()
            .filter(|v| !v.is_nan())
            .copied()
            .collect();

        if window_values.len() >= min_periods {
            let sum: f64 = window_values.iter().sum();
            result.push(sum / window_values.len() as f64);
        } else {
            result.push(f64::NAN);
        }
    }

    Ok(result)
}

/// Compute rolling standard deviation
pub fn rolling_std(
    df: &DataFrame,
    column: &str,
    window: usize,
    min_periods: Option<usize>,
) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
    let min_periods = min_periods.unwrap_or(window);

    let mut result = Vec::with_capacity(values.len());

    for i in 0..values.len() {
        let start = i.saturating_sub(window.saturating_sub(1));
        let window_values: Vec<f64> = values[start..=i]
            .iter()
            .filter(|v| !v.is_nan())
            .copied()
            .collect();

        if window_values.len() >= min_periods && window_values.len() > 1 {
            let n = window_values.len() as f64;
            let mean = window_values.iter().sum::<f64>() / n;
            let variance = window_values
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>()
                / (n - 1.0);
            result.push(variance.sqrt());
        } else {
            result.push(f64::NAN);
        }
    }

    Ok(result)
}

/// Compute rolling minimum
pub fn rolling_min(
    df: &DataFrame,
    column: &str,
    window: usize,
    min_periods: Option<usize>,
) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
    let min_periods = min_periods.unwrap_or(window);

    let mut result = Vec::with_capacity(values.len());

    for i in 0..values.len() {
        let start = i.saturating_sub(window.saturating_sub(1));
        let window_values: Vec<f64> = values[start..=i]
            .iter()
            .filter(|v| !v.is_nan())
            .copied()
            .collect();

        if window_values.len() >= min_periods {
            result.push(window_values.iter().cloned().fold(f64::INFINITY, f64::min));
        } else {
            result.push(f64::NAN);
        }
    }

    Ok(result)
}

/// Compute rolling maximum
pub fn rolling_max(
    df: &DataFrame,
    column: &str,
    window: usize,
    min_periods: Option<usize>,
) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
    let min_periods = min_periods.unwrap_or(window);

    let mut result = Vec::with_capacity(values.len());

    for i in 0..values.len() {
        let start = i.saturating_sub(window.saturating_sub(1));
        let window_values: Vec<f64> = values[start..=i]
            .iter()
            .filter(|v| !v.is_nan())
            .copied()
            .collect();

        if window_values.len() >= min_periods {
            result.push(
                window_values
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max),
            );
        } else {
            result.push(f64::NAN);
        }
    }

    Ok(result)
}

/// Compute rolling variance
pub fn rolling_var(
    df: &DataFrame,
    column: &str,
    window: usize,
    min_periods: Option<usize>,
) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
    let min_periods = min_periods.unwrap_or(window);

    let mut result = Vec::with_capacity(values.len());

    for i in 0..values.len() {
        let start = i.saturating_sub(window.saturating_sub(1));
        let window_values: Vec<f64> = values[start..=i]
            .iter()
            .filter(|v| !v.is_nan())
            .copied()
            .collect();

        if window_values.len() >= min_periods && window_values.len() > 1 {
            let mean: f64 = window_values.iter().sum::<f64>() / window_values.len() as f64;
            let variance: f64 = window_values
                .iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>()
                / (window_values.len() - 1) as f64;
            result.push(variance);
        } else {
            result.push(f64::NAN);
        }
    }

    Ok(result)
}

/// Compute rolling median
pub fn rolling_median(
    df: &DataFrame,
    column: &str,
    window: usize,
    min_periods: Option<usize>,
) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
    let min_periods = min_periods.unwrap_or(window);

    let mut result = Vec::with_capacity(values.len());

    for i in 0..values.len() {
        let start = i.saturating_sub(window.saturating_sub(1));
        let mut window_values: Vec<f64> = values[start..=i]
            .iter()
            .filter(|v| !v.is_nan())
            .copied()
            .collect();

        if window_values.len() >= min_periods {
            window_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = window_values.len() / 2;
            let median = if window_values.len() % 2 == 0 {
                (window_values[mid - 1] + window_values[mid]) / 2.0
            } else {
                window_values[mid]
            };
            result.push(median);
        } else {
            result.push(f64::NAN);
        }
    }

    Ok(result)
}

/// Compute rolling count of non-NaN values
pub fn rolling_count(df: &DataFrame, column: &str, window: usize) -> Result<Vec<usize>> {
    let values = df.get_column_numeric_values(column)?;

    let mut result = Vec::with_capacity(values.len());

    for i in 0..values.len() {
        let start = i.saturating_sub(window.saturating_sub(1));
        let count = values[start..=i].iter().filter(|v| !v.is_nan()).count();
        result.push(count);
    }

    Ok(result)
}

/// Apply custom function to rolling window
pub fn rolling_apply<F>(
    df: &DataFrame,
    column: &str,
    window: usize,
    func: F,
    min_periods: Option<usize>,
) -> Result<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    let values = df.get_column_numeric_values(column)?;
    let min_periods = min_periods.unwrap_or(window);

    let mut result = Vec::with_capacity(values.len());

    for i in 0..values.len() {
        let start = i.saturating_sub(window.saturating_sub(1));
        let window_values: Vec<f64> = values[start..=i]
            .iter()
            .filter(|v| !v.is_nan())
            .copied()
            .collect();

        if window_values.len() >= min_periods {
            result.push(func(&window_values));
        } else {
            result.push(f64::NAN);
        }
    }

    Ok(result)
}

/// Compute expanding sum (cumulative from start)
pub fn expanding_sum(df: &DataFrame, column: &str, min_periods: usize) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
    let mut result = Vec::with_capacity(values.len());
    let mut sum = 0.0;
    let mut count = 0usize;

    for v in &values {
        if !v.is_nan() {
            sum += v;
            count += 1;
        }
        if count >= min_periods {
            result.push(sum);
        } else {
            result.push(f64::NAN);
        }
    }

    Ok(result)
}

/// Compute expanding mean
pub fn expanding_mean(df: &DataFrame, column: &str, min_periods: usize) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
    let mut result = Vec::with_capacity(values.len());
    let mut sum = 0.0;
    let mut count = 0usize;

    for v in &values {
        if !v.is_nan() {
            sum += v;
            count += 1;
        }
        if count >= min_periods {
            result.push(sum / count as f64);
        } else {
            result.push(f64::NAN);
        }
    }

    Ok(result)
}

/// Compute expanding standard deviation
pub fn expanding_std(df: &DataFrame, column: &str, min_periods: usize) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
    let mut result = Vec::with_capacity(values.len());
    let mut seen: Vec<f64> = Vec::new();

    for v in &values {
        if !v.is_nan() {
            seen.push(*v);
        }
        if seen.len() >= min_periods && seen.len() > 1 {
            let n = seen.len() as f64;
            let mean = seen.iter().sum::<f64>() / n;
            let variance = seen.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
            result.push(variance.sqrt());
        } else {
            result.push(f64::NAN);
        }
    }

    Ok(result)
}

/// Compute expanding minimum
pub fn expanding_min(df: &DataFrame, column: &str, min_periods: usize) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
    let mut result = Vec::with_capacity(values.len());
    let mut min_val = f64::INFINITY;
    let mut count = 0usize;

    for v in &values {
        if !v.is_nan() {
            min_val = min_val.min(*v);
            count += 1;
        }
        if count >= min_periods {
            result.push(min_val);
        } else {
            result.push(f64::NAN);
        }
    }

    Ok(result)
}

/// Compute expanding maximum
pub fn expanding_max(df: &DataFrame, column: &str, min_periods: usize) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
    let mut result = Vec::with_capacity(values.len());
    let mut max_val = f64::NEG_INFINITY;
    let mut count = 0usize;

    for v in &values {
        if !v.is_nan() {
            max_val = max_val.max(*v);
            count += 1;
        }
        if count >= min_periods {
            result.push(max_val);
        } else {
            result.push(f64::NAN);
        }
    }

    Ok(result)
}

/// Compute expanding variance
pub fn expanding_var(df: &DataFrame, column: &str, min_periods: usize) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
    let mut result = Vec::with_capacity(values.len());
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut count = 0usize;

    for v in &values {
        if !v.is_nan() {
            sum += v;
            sum_sq += v * v;
            count += 1;
        }
        if count >= min_periods && count > 1 {
            let mean = sum / count as f64;
            let variance = (sum_sq - count as f64 * mean * mean) / (count - 1) as f64;
            result.push(variance);
        } else {
            result.push(f64::NAN);
        }
    }

    Ok(result)
}

/// Apply custom function to expanding window
pub fn expanding_apply<F>(
    df: &DataFrame,
    column: &str,
    func: F,
    min_periods: usize,
) -> Result<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    let values = df.get_column_numeric_values(column)?;
    let mut result = Vec::with_capacity(values.len());
    let mut window_values: Vec<f64> = Vec::new();

    for v in &values {
        if !v.is_nan() {
            window_values.push(*v);
        }
        if window_values.len() >= min_periods {
            result.push(func(&window_values));
        } else {
            result.push(f64::NAN);
        }
    }

    Ok(result)
}
