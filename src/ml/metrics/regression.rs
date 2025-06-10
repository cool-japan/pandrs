//! Regression model evaluation metrics

use crate::error::{Error, Result};
use crate::optimized::OptimizedDataFrame;
use crate::series::Series;
use std::cmp::Ordering;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_squared_error() {
        let y_true = vec![3.0, 5.0, 2.5, 7.0, 10.0];
        let y_pred = vec![2.8, 4.8, 2.7, 7.2, 9.8];

        let mse = mean_squared_error(&y_true, &y_pred).unwrap();

        // Actual calculation: (0.2)² + (0.2)² + (0.2)² + (0.2)² + (0.2)² = 0.2
        assert!((mse - 0.2 / 5.0).abs() < 1e-6); // 0.04 is the correct expected value, compared to six decimal places
    }

    #[test]
    fn test_r2_score() {
        let y_true = vec![3.0, 5.0, 2.5, 7.0, 10.0];
        let y_pred = vec![2.8, 4.8, 2.7, 7.2, 9.8];

        let r2 = r2_score(&y_true, &y_pred).unwrap();
        assert!(r2 > 0.99); // R² score should be greater than 0.99
    }

    #[test]
    fn test_empty_input() {
        let empty: Vec<f64> = vec![];

        let mse_result = mean_squared_error(&empty, &empty);
        assert!(mse_result.is_err());

        let r2_result = r2_score(&empty, &empty);
        assert!(r2_result.is_err());
    }

    #[test]
    fn test_different_length() {
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![1.0, 2.0];

        let mse_result = mean_squared_error(&y_true, &y_pred);
        assert!(mse_result.is_err());

        let r2_result = r2_score(&y_true, &y_pred);
        assert!(r2_result.is_err());
    }
}

/// Calculate Mean Squared Error (MSE)
///
/// # Arguments
/// * `y_true` - True values
/// * `y_pred` - Predicted values
///
/// # Returns
/// * `Result<f64>` - Mean Squared Error
pub fn mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    if y_true.len() != y_pred.len() {
        return Err(Error::DimensionMismatch(format!(
            "Length mismatch between true and predicted values: {} vs {}",
            y_true.len(),
            y_pred.len()
        )));
    }

    if y_true.is_empty() {
        return Err(Error::InvalidOperation(
            "Cannot calculate with empty data".to_string(),
        ));
    }

    let sum_squared_error = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&true_val, &pred_val)| {
            let error = true_val - pred_val;
            error * error
        })
        .sum::<f64>();

    Ok(sum_squared_error / y_true.len() as f64)
}

/// Calculate Mean Absolute Error (MAE)
///
/// # Arguments
/// * `y_true` - True values
/// * `y_pred` - Predicted values
///
/// # Returns
/// * `Result<f64>` - Mean Absolute Error
pub fn mean_absolute_error(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    if y_true.len() != y_pred.len() {
        return Err(Error::DimensionMismatch(format!(
            "Length mismatch between true and predicted values: {} vs {}",
            y_true.len(),
            y_pred.len()
        )));
    }

    if y_true.is_empty() {
        return Err(Error::InvalidOperation(
            "Cannot calculate with empty data".to_string(),
        ));
    }

    let sum_absolute_error = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&true_val, &pred_val)| (true_val - pred_val).abs())
        .sum::<f64>();

    Ok(sum_absolute_error / y_true.len() as f64)
}

/// Calculate Root Mean Squared Error (RMSE)
///
/// # Arguments
/// * `y_true` - True values
/// * `y_pred` - Predicted values
///
/// # Returns
/// * `Result<f64>` - Root Mean Squared Error
pub fn root_mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    let mse = mean_squared_error(y_true, y_pred)?;
    Ok(mse.sqrt())
}

/// Calculate R² score (coefficient of determination)
///
/// # Arguments
/// * `y_true` - True values
/// * `y_pred` - Predicted values
///
/// # Returns
/// * `Result<f64>` - R² score (1 is best, can be negative if the model is worse than a constant model)
pub fn r2_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    if y_true.len() != y_pred.len() {
        return Err(Error::DimensionMismatch(format!(
            "Length mismatch between true and predicted values: {} vs {}",
            y_true.len(),
            y_pred.len()
        )));
    }

    if y_true.is_empty() {
        return Err(Error::InvalidOperation(
            "Cannot calculate with empty data".to_string(),
        ));
    }

    // Calculate the mean of true values
    let y_mean = y_true.iter().sum::<f64>() / y_true.len() as f64;

    // Calculate total sum of squares (variance of true values)
    let ss_tot = y_true
        .iter()
        .map(|&true_val| {
            let diff = true_val - y_mean;
            diff * diff
        })
        .sum::<f64>();

    // Calculate residual sum of squares
    let ss_res = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&true_val, &pred_val)| {
            let error = true_val - pred_val;
            error * error
        })
        .sum::<f64>();

    // Handle edge cases
    if ss_tot == 0.0 {
        if ss_res == 0.0 {
            Ok(1.0) // Perfect prediction
        } else {
            Ok(0.0) // Constant prediction with error
        }
    } else {
        Ok(1.0 - (ss_res / ss_tot))
    }
}

/// Calculate Explained Variance Score
///
/// # Arguments
/// * `y_true` - True values
/// * `y_pred` - Predicted values
///
/// # Returns
/// * `Result<f64>` - Explained Variance Score (1 is best)
pub fn explained_variance_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    if y_true.len() != y_pred.len() {
        return Err(Error::DimensionMismatch(format!(
            "Length mismatch between true and predicted values: {} vs {}",
            y_true.len(),
            y_pred.len()
        )));
    }

    if y_true.is_empty() {
        return Err(Error::InvalidOperation(
            "Cannot calculate with empty data".to_string(),
        ));
    }

    // Calculate the mean of true and predicted values
    let y_true_mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
    let y_pred_mean = y_pred.iter().sum::<f64>() / y_pred.len() as f64;

    // Calculate variance of true values
    let var_y_true = y_true
        .iter()
        .map(|&val| {
            let diff = val - y_true_mean;
            diff * diff
        })
        .sum::<f64>()
        / y_true.len() as f64;

    // Calculate variance of residuals
    let var_residual = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| {
            let residual = (t - p) - (y_true_mean - y_pred_mean);
            residual * residual
        })
        .sum::<f64>()
        / y_true.len() as f64;

    // Handle edge cases
    if var_y_true == 0.0 {
        if var_residual == 0.0 {
            Ok(1.0)
        } else {
            Ok(0.0)
        }
    } else {
        Ok(1.0 - (var_residual / var_y_true))
    }
}
