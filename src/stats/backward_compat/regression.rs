//! Regression analysis module
//!
//! This module provides functionality for regression analysis, including
//! linear regression, polynomial regression, and regression diagnostics.

use crate::error::{Result, Error};
use crate::dataframe::DataFrame;
use crate::stats::LinearRegressionResult;

/// Internal implementation of linear regression
pub(crate) fn linear_regression_impl(
    df: &DataFrame,
    y_column: &str,
    x_columns: &[&str],
) -> Result<LinearRegressionResult> {
    // Check if y_column exists
    if !df.has_column(y_column) {
        return Err(Error::InvalidColumn(format!("Column '{}' does not exist", y_column)));
    }
    
    // Check if all x_columns exist
    for &col_name in x_columns {
        if !df.has_column(col_name) {
            return Err(Error::InvalidColumn(format!("Column '{}' does not exist", col_name)));
        }
    }
    
    // Get y data
    let y_series = df.get_column(y_column)?;
    let y_data = y_series.as_f64()?;
    
    // Check for empty data
    if y_data.is_empty() {
        return Err(Error::EmptyData("Regression analysis requires data".into()));
    }
    
    let n_samples = y_data.len();
    let n_features = x_columns.len();
    
    // Check if we have enough samples
    if n_samples <= n_features {
        return Err(Error::InsufficientData(
            format!("Regression analysis requires more samples than features (samples: {}, features: {})", n_samples, n_features)
        ));
    }
    
    // Create design matrix X (with intercept column)
    let mut x_matrix = vec![vec![1.0; n_samples]]; // First column is all 1's for intercept
    for &col_name in x_columns {
        let x_series = df.get_column(col_name)?;
        let x_data = x_series.as_f64()?;
        
        // Make sure all X columns have the same length as y
        if x_data.len() != n_samples {
            return Err(Error::DimensionMismatch(
                format!("Column '{}' has length {} but expected {}", col_name, x_data.len(), n_samples)
            ));
        }
        
        x_matrix.push(x_data);
    }
    
    // Transpose X for easier calculations
    let mut x_transpose = vec![vec![0.0; n_features + 1]; n_samples];
    for i in 0..n_samples {
        for j in 0..(n_features + 1) {
            x_transpose[i][j] = x_matrix[j][i];
        }
    }
    
    // Calculate X^T * X
    let mut xtx = vec![vec![0.0; n_features + 1]; n_features + 1];
    for i in 0..(n_features + 1) {
        for j in 0..(n_features + 1) {
            for k in 0..n_samples {
                xtx[i][j] += x_matrix[i][k] * x_matrix[j][k];
            }
        }
    }
    
    // Calculate X^T * y
    let mut xty = vec![0.0; n_features + 1];
    for i in 0..(n_features + 1) {
        for k in 0..n_samples {
            xty[i] += x_matrix[i][k] * y_data[k];
        }
    }
    
    // Solve for coefficients (inv(X^T * X) * X^T * y)
    // For simplicity, we'll use a basic matrix inversion algorithm
    // Real implementation should use a numerical library with more stable methods
    let xtx_inv = invert_matrix(&xtx)?;
    
    // Calculate coefficients
    let mut coeffs = vec![0.0; n_features + 1];
    for i in 0..(n_features + 1) {
        for j in 0..(n_features + 1) {
            coeffs[i] += xtx_inv[i][j] * xty[j];
        }
    }
    
    // Calculate fitted values and residuals
    let mut fitted_values = vec![0.0; n_samples];
    let mut residuals = vec![0.0; n_samples];
    
    for i in 0..n_samples {
        fitted_values[i] = coeffs[0]; // Intercept
        for j in 0..n_features {
            fitted_values[i] += coeffs[j + 1] * x_transpose[i][j + 1];
        }
        residuals[i] = y_data[i] - fitted_values[i];
    }
    
    // Calculate R-squared
    let y_mean = y_data.iter().sum::<f64>() / n_samples as f64;
    let total_sum_squares: f64 = y_data.iter().map(|&y| (y - y_mean).powi(2)).sum();
    let residual_sum_squares: f64 = residuals.iter().map(|&r| r.powi(2)).sum();
    let r_squared = 1.0 - (residual_sum_squares / total_sum_squares);
    
    // Calculate adjusted R-squared
    let adj_r_squared = 1.0 - ((1.0 - r_squared) * (n_samples as f64 - 1.0) / (n_samples as f64 - n_features as f64 - 1.0));
    
    // Calculate standard error
    let residual_mean_square = residual_sum_squares / (n_samples - n_features - 1) as f64;
    
    // Standard errors of the coefficients
    let mut std_errors = vec![0.0; n_features + 1];
    for i in 0..(n_features + 1) {
        std_errors[i] = (xtx_inv[i][i] * residual_mean_square).sqrt();
    }
    
    // Calculate p-values for coefficients
    let mut p_values = vec![0.0; n_features + 1];
    for i in 0..(n_features + 1) {
        let t_stat = coeffs[i] / std_errors[i];
        // Use t-distribution for p-value
        let df = n_samples - n_features - 1;
        p_values[i] = 2.0 * (1.0 - t_distribution_cdf(t_stat.abs(), df));
    }
    
    // Extract coefficients for features (excluding intercept)
    let feature_coeffs = coeffs.iter().skip(1).cloned().collect();
    
    Ok(LinearRegressionResult {
        intercept: coeffs[0],
        coefficients: feature_coeffs,
        r_squared,
        adj_r_squared,
        p_values,
        fitted_values,
        residuals,
    })
}

/// Simple t-distribution CDF approximation (private use for p-values)
fn t_distribution_cdf(t: f64, df: usize) -> f64 {
    // Use normal distribution approximation (for large degrees of freedom)
    if df > 30 {
        return normal_cdf(t);
    }
    
    // Approximation calculation for t-distribution
    let df_f64 = df as f64;
    let x = df_f64 / (df_f64 + t * t);
    let a = 0.5 * df_f64;
    
    // Simplified approximation
    let beta_approx = if t > 0.0 {
        1.0 - 0.5 * x.powf(a)
    } else {
        0.5 * x.powf(a)
    };
    
    beta_approx
}

/// Simple normal CDF approximation (private use for p-values)
fn normal_cdf(z: f64) -> f64 {
    // Approximation calculation for standard normal distribution CDF
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let x = z.abs() / (2.0_f64).sqrt();
    
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    
    0.5 * (1.0 + sign * y)
}

/// Simple matrix inversion function for regression calculations
///
/// Uses Gauss-Jordan elimination with partial pivoting
/// Note: This is a simplified implementation. A real-world implementation
/// should use a numerical library for better stability and performance.
fn invert_matrix(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = matrix.len();
    
    // Check if the matrix is square
    for row in matrix {
        if row.len() != n {
            return Err(Error::InvalidInput("Matrix must be square for inversion".into()));
        }
    }
    
    // Create augmented matrix [A|I]
    let mut augmented = vec![vec![0.0; 2 * n]; n];
    
    for i in 0..n {
        for j in 0..n {
            augmented[i][j] = matrix[i][j];
        }
        augmented[i][i + n] = 1.0; // Identity matrix on the right
    }
    
    // Gauss-Jordan elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        let mut max_val = augmented[i][i].abs();
        
        for k in (i + 1)..n {
            if augmented[k][i].abs() > max_val {
                max_row = k;
                max_val = augmented[k][i].abs();
            }
        }
        
        // Check for singularity
        if max_val < 1e-10 {
            return Err(Error::ComputationError("Matrix is singular or nearly singular".into()));
        }
        
        // Swap rows if needed
        if max_row != i {
            for j in 0..(2 * n) {
                let temp = augmented[i][j];
                augmented[i][j] = augmented[max_row][j];
                augmented[max_row][j] = temp;
            }
        }
        
        // Scale the pivot row
        let pivot = augmented[i][i];
        for j in 0..(2 * n) {
            augmented[i][j] /= pivot;
        }
        
        // Eliminate other rows
        for k in 0..n {
            if k != i {
                let factor = augmented[k][i];
                for j in 0..(2 * n) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }
    
    // Extract the inverse from the right side of the augmented matrix
    let mut inverse = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inverse[i][j] = augmented[i][j + n];
        }
    }
    
    Ok(inverse)
}

/// Perform polynomial regression
///
/// # Description
/// Fits a polynomial regression model of specified degree.
///
/// # Arguments
/// * `x` - Independent variable values
/// * `y` - Dependent variable values
/// * `degree` - Polynomial degree to fit
///
/// # Example
/// ```
/// use pandrs::stats::regression;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![1.0, 4.0, 9.0, 16.0, 25.0];
/// let result = regression::polynomial_regression(&x, &y, 2).unwrap();
/// println!("Coefficients: {:?}", result.coefficients);
/// println!("R-squared: {}", result.r_squared);
/// ```
pub fn polynomial_regression(
    x: &[f64],
    y: &[f64],
    degree: usize
) -> Result<LinearRegressionResult> {
    if x.is_empty() || y.is_empty() {
        return Err(Error::EmptyData("Regression analysis requires data".into()));
    }
    
    if x.len() != y.len() {
        return Err(Error::DimensionMismatch(
            format!("X and Y must have the same length: x={}, y={}", x.len(), y.len())
        ));
    }
    
    let n = x.len();
    
    if n <= degree {
        return Err(Error::InsufficientData(
            format!("Polynomial regression of degree {} requires at least {} data points", 
                   degree, degree + 1)
        ));
    }
    
    // Create design matrix X (with intercept and polynomial terms)
    let mut x_matrix = Vec::with_capacity(degree + 1);
    
    // Intercept column (all 1's)
    x_matrix.push(vec![1.0; n]);
    
    // Add polynomial terms
    for d in 1..=degree {
        let col: Vec<f64> = x.iter().map(|&xi| xi.powi(d as i32)).collect();
        x_matrix.push(col);
    }
    
    // Transpose X for easier calculations
    let mut x_transpose = vec![vec![0.0; degree + 1]; n];
    for i in 0..n {
        for j in 0..(degree + 1) {
            x_transpose[i][j] = x_matrix[j][i];
        }
    }
    
    // Calculate X^T * X
    let mut xtx = vec![vec![0.0; degree + 1]; degree + 1];
    for i in 0..(degree + 1) {
        for j in 0..(degree + 1) {
            for k in 0..n {
                xtx[i][j] += x_matrix[i][k] * x_matrix[j][k];
            }
        }
    }
    
    // Calculate X^T * y
    let mut xty = vec![0.0; degree + 1];
    for i in 0..(degree + 1) {
        for k in 0..n {
            xty[i] += x_matrix[i][k] * y[k];
        }
    }
    
    // Solve for coefficients
    let xtx_inv = invert_matrix(&xtx)?;
    
    // Calculate coefficients
    let mut coeffs = vec![0.0; degree + 1];
    for i in 0..(degree + 1) {
        for j in 0..(degree + 1) {
            coeffs[i] += xtx_inv[i][j] * xty[j];
        }
    }
    
    // Calculate fitted values and residuals
    let mut fitted_values = vec![0.0; n];
    let mut residuals = vec![0.0; n];
    
    for i in 0..n {
        fitted_values[i] = coeffs[0]; // Intercept
        for d in 1..=degree {
            fitted_values[i] += coeffs[d] * x[i].powi(d as i32);
        }
        residuals[i] = y[i] - fitted_values[i];
    }
    
    // Calculate R-squared
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let total_sum_squares: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let residual_sum_squares: f64 = residuals.iter().map(|&r| r.powi(2)).sum();
    let r_squared = 1.0 - (residual_sum_squares / total_sum_squares);
    
    // Calculate adjusted R-squared
    let adj_r_squared = 1.0 - ((1.0 - r_squared) * (n as f64 - 1.0) / (n as f64 - degree as f64 - 1.0));
    
    // Calculate standard error
    let residual_mean_square = residual_sum_squares / (n - degree - 1) as f64;
    
    // Standard errors of the coefficients
    let mut std_errors = vec![0.0; degree + 1];
    for i in 0..(degree + 1) {
        std_errors[i] = (xtx_inv[i][i] * residual_mean_square).sqrt();
    }
    
    // Calculate p-values for coefficients
    let mut p_values = vec![0.0; degree + 1];
    for i in 0..(degree + 1) {
        let t_stat = coeffs[i] / std_errors[i];
        // Use t-distribution for p-value
        let df = n - degree - 1;
        p_values[i] = 2.0 * (1.0 - t_distribution_cdf(t_stat.abs(), df));
    }
    
    // Extract coefficients for polynomial terms (excluding intercept)
    let feature_coeffs = coeffs.iter().skip(1).cloned().collect();
    
    Ok(LinearRegressionResult {
        intercept: coeffs[0],
        coefficients: feature_coeffs,
        r_squared,
        adj_r_squared,
        p_values,
        fitted_values,
        residuals,
    })
}

/// Perform simple linear regression
///
/// # Description
/// Fits a simple linear regression model (y = a + bx).
///
/// # Arguments
/// * `x` - Independent variable values
/// * `y` - Dependent variable values
///
/// # Example
/// ```
/// use pandrs::stats::regression;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
///
/// let result = regression::simple_linear_regression(&x, &y).unwrap();
/// println!("Intercept: {}", result.intercept);
/// println!("Slope: {}", result.coefficients[0]);
/// println!("R-squared: {}", result.r_squared);
/// ```
pub fn simple_linear_regression(x: &[f64], y: &[f64]) -> Result<LinearRegressionResult> {
    // Simple linear regression is just polynomial regression of degree 1
    polynomial_regression(x, y, 1)
}

/// Calculate residual diagnostics for regression model
///
/// # Description
/// Computes diagnostics for regression residuals to check model assumptions.
///
/// # Arguments
/// * `residuals` - The residuals from a regression model
///
/// # Returns
/// * A map of diagnostic names to values
///
/// # Example
/// ```
/// use pandrs::stats::regression;
///
/// let residuals = vec![0.5, -0.3, 0.2, -0.4, 0.1];
/// let diagnostics = regression::residual_diagnostics(&residuals).unwrap();
/// println!("Shapiro p-value (normality): {}", diagnostics["shapiro_p_value"]);
/// println!("Durbin-Watson (autocorrelation): {}", diagnostics["durbin_watson"]);
/// ```
pub fn residual_diagnostics(residuals: &[f64]) -> Result<std::collections::HashMap<String, f64>> {
    if residuals.is_empty() {
        return Err(Error::EmptyData("Residual diagnostics requires data".into()));
    }
    
    let n = residuals.len();
    
    if n < 3 {
        return Err(Error::InsufficientData("Residual diagnostics requires at least 3 data points".into()));
    }
    
    let mut diagnostics = std::collections::HashMap::new();
    
    // Basic statistics
    let mean = residuals.iter().sum::<f64>() / n as f64;
    
    let variance = residuals.iter()
        .map(|&r| (r - mean).powi(2))
        .sum::<f64>() / (n as f64);
    
    diagnostics.insert("mean".to_string(), mean);
    diagnostics.insert("variance".to_string(), variance);
    
    // Durbin-Watson statistic (tests for autocorrelation)
    let mut dw_num = 0.0;
    let mut dw_denom = 0.0;
    
    for i in 1..n {
        dw_num += (residuals[i] - residuals[i - 1]).powi(2);
    }
    
    for &r in residuals {
        dw_denom += r.powi(2);
    }
    
    let durbin_watson = dw_num / dw_denom;
    diagnostics.insert("durbin_watson".to_string(), durbin_watson);
    
    // Jarque-Bera test (tests for normality)
    let skewness = residuals.iter()
        .map(|&r| (r - mean).powi(3))
        .sum::<f64>() / (n as f64 * variance.powf(1.5));
    
    let kurtosis = residuals.iter()
        .map(|&r| (r - mean).powi(4))
        .sum::<f64>() / (n as f64 * variance.powi(2)) - 3.0;
    
    let jarque_bera = n as f64 * (skewness.powi(2) / 6.0 + kurtosis.powi(2) / 24.0);
    let jb_p_value = 1.0 - chi2_cdf(jarque_bera, 2);
    
    diagnostics.insert("skewness".to_string(), skewness);
    diagnostics.insert("kurtosis".to_string(), kurtosis);
    diagnostics.insert("jarque_bera".to_string(), jarque_bera);
    diagnostics.insert("jb_p_value".to_string(), jb_p_value);
    
    // Additional approximation: Shapiro-Wilk test (simplified)
    let mut sorted = residuals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    // Simplified Shapiro-Wilk approximation (not accurate for small samples)
    let shapiro_w = approximate_shapiro_wilk(&sorted);
    
    // Convert W to p-value using approximation
    let shapiro_p = approximate_shapiro_p(shapiro_w, n);
    
    diagnostics.insert("shapiro_w".to_string(), shapiro_w);
    diagnostics.insert("shapiro_p_value".to_string(), shapiro_p);
    
    Ok(diagnostics)
}

/// Chi-square CDF approximation (for residual diagnostics)
fn chi2_cdf(chi2: f64, df: usize) -> f64 {
    // Simple approximation for chi-square distribution
    let k = df as f64 / 2.0;
    let x = chi2 / 2.0;
    
    // Simplified approximation
    if chi2 <= 0.0 {
        return 0.0;
    }
    
    let p = 1.0 - (-x).exp() * (1.0 + x + x.powi(2) / 2.0);
    p.min(1.0).max(0.0)
}

/// Approximate Shapiro-Wilk test (simplified for residual diagnostics)
fn approximate_shapiro_wilk(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    
    // Mean and variance
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let variance = sorted.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / (n as f64);
    
    if variance.abs() < 1e-10 {
        return 0.0; // Avoid division by zero
    }
    
    // Calculate normal order statistics medians
    let mut a = Vec::with_capacity(n);
    for i in 0..n {
        let p = (i as f64 + 0.375) / (n as f64 + 0.25);
        // Inverse normal CDF approximation
        let z = if p < 0.5 {
            -(-2.0 * p.ln()).sqrt()
        } else {
            (-2.0 * (1.0 - p).ln()).sqrt()
        };
        a.push(z);
    }
    
    // Normalize a
    let a_sq_sum: f64 = a.iter().map(|&x| x.powi(2)).sum();
    a.iter_mut().for_each(|x| *x /= a_sq_sum.sqrt());
    
    // Calculate W statistic
    let numerator: f64 = a.iter().zip(sorted.iter())
        .map(|(&ai, &xi)| ai * xi)
        .sum::<f64>().powi(2);
    
    let denominator: f64 = sorted.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>();
    
    numerator / denominator
}

/// Approximate Shapiro-Wilk p-value (simplified for residual diagnostics)
fn approximate_shapiro_p(w: f64, n: usize) -> f64 {
    // Very simplified approximation
    if w >= 1.0 {
        return 1.0;
    }
    
    if w <= 0.0 {
        return 0.0;
    }
    
    // Convert W to approximate p-value
    let ln_w = (1.0 - w).ln();
    let n_f64 = n as f64;
    
    let mu = -1.5861 - 0.31082 * n_f64.ln() - 0.083751 * n_f64.powi(2).ln();
    let sigma = 0.6897 + 0.0113 * n_f64.ln();
    
    let z = (ln_w - mu) / sigma;
    
    // Convert z to p-value
    1.0 - normal_cdf(z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::DataFrame;
    use crate::series::Series;
    
    #[test]
    fn test_invert_matrix() {
        // 2x2 matrix
        let matrix = vec![
            vec![4.0, 7.0],
            vec![2.0, 6.0]
        ];
        
        let inverse = invert_matrix(&matrix).unwrap();
        
        // Expected inverse: [0.6, -0.7; -0.2, 0.4]
        assert!((inverse[0][0] - 0.6).abs() < 1e-10);
        assert!((inverse[0][1] + 0.7).abs() < 1e-10);
        assert!((inverse[1][0] + 0.2).abs() < 1e-10);
        assert!((inverse[1][1] - 0.4).abs() < 1e-10);
        
        // Test with identity matrix
        let identity = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0]
        ];
        
        let identity_inv = invert_matrix(&identity).unwrap();
        assert!((identity_inv[0][0] - 1.0).abs() < 1e-10);
        assert!((identity_inv[0][1] - 0.0).abs() < 1e-10);
        assert!((identity_inv[1][0] - 0.0).abs() < 1e-10);
        assert!((identity_inv[1][1] - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_simple_linear_regression() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let result = simple_linear_regression(&x, &y).unwrap();
        
        // Perfect linear relationship y = 2x
        assert!((result.intercept - 0.0).abs() < 1e-10);
        assert!((result.coefficients[0] - 2.0).abs() < 1e-10);
        assert!((result.r_squared - 1.0).abs() < 1e-10);
        
        // Test with some noise
        let y_noisy = vec![2.1, 3.9, 6.2, 7.8, 10.1];
        let result_noisy = simple_linear_regression(&x, &y_noisy).unwrap();
        
        // Should still be close to y = 2x
        assert!(result_noisy.intercept.abs() < 0.5);
        assert!((result_noisy.coefficients[0] - 2.0).abs() < 0.2);
        assert!(result_noisy.r_squared > 0.99);
    }
    
    #[test]
    fn test_polynomial_regression() {
        // Quadratic data: y = x^2
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 4.0, 9.0, 16.0, 25.0];
        
        // Linear regression won't fit well
        let linear_result = polynomial_regression(&x, &y, 1).unwrap();
        assert!(linear_result.r_squared < 0.9);
        
        // Quadratic regression should fit perfectly
        let quad_result = polynomial_regression(&x, &y, 2).unwrap();
        
        // Should be close to y = 0 + 0*x + 1*x^2
        assert!(quad_result.intercept.abs() < 1e-8);
        assert!(quad_result.coefficients[0].abs() < 1e-8); // x coefficient
        assert!((quad_result.coefficients[1] - 1.0).abs() < 1e-8); // x^2 coefficient
        assert!((quad_result.r_squared - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_residual_diagnostics() {
        // Well-behaved residuals (approximately normal)
        let normal_residuals = vec![0.1, -0.2, 0.3, -0.15, 0.25, -0.1, 0.05, -0.3];
        let normal_diag = residual_diagnostics(&normal_residuals).unwrap();
        
        // Mean should be close to zero
        assert!(normal_diag["mean"].abs() < 0.1);
        
        // Durbin-Watson should be around 2 for uncorrelated residuals
        assert!((normal_diag["durbin_watson"] - 2.0).abs() < 1.0);
        
        // Shapiro-Wilk p-value should be high for normal residuals
        assert!(normal_diag["shapiro_p_value"] > 0.05);
        
        // Skewed residuals
        let skewed_residuals = vec![0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.2, 1.5, 2.0];
        let skewed_diag = residual_diagnostics(&skewed_residuals).unwrap();
        
        // Should have positive skewness
        assert!(skewed_diag["skewness"] > 0.5);
    }
    
    #[test]
    fn test_linear_regression_implementation() {
        // Create a simple DataFrame for testing
        let mut df = DataFrame::new();
        
        // x1 and x2 as predictors
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![2.0, 3.0, 5.0, 4.0, 8.0];
        
        // y = 1 + 2*x1 + 0.5*x2 (with a bit of noise)
        let y = vec![5.1, 8.9, 12.7, 14.8, 19.2];
        
        // Add columns to DataFrame
        df.add_column("x1".to_string(), Series::new(x1, Some("x1".to_string())).unwrap()).unwrap();
        df.add_column("x2".to_string(), Series::new(x2, Some("x2".to_string())).unwrap()).unwrap();
        df.add_column("y".to_string(), Series::new(y, Some("y".to_string())).unwrap()).unwrap();
        
        // Test linear_regression_impl
        let result = linear_regression_impl(&df, "y", &["x1", "x2"]).unwrap();
        
        // Coefficient for x1 should be close to 2
        assert!((result.coefficients[0] - 2.0).abs() < 0.5);
        
        // Coefficient for x2 should be close to 0.5
        assert!((result.coefficients[1] - 0.5).abs() < 0.5);
        
        // Intercept should be close to 1
        assert!((result.intercept - 1.0).abs() < 0.5);
        
        // R-squared should be high
        assert!(result.r_squared > 0.95);
    }
}