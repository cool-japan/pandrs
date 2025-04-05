//! 回帰モデル評価のためのメトリクス

use crate::error::{Error, Result};
use crate::optimized::OptimizedDataFrame;
use crate::series::Series;
use std::cmp::Ordering;

/// 平均二乗誤差（Mean Squared Error）を計算
///
/// # Arguments
/// * `y_true` - 真の値
/// * `y_pred` - 予測値
///
/// # Returns
/// * `Result<f64>` - 平均二乗誤差
pub fn mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    if y_true.len() != y_pred.len() {
        return Err(Error::DimensionMismatch(format!(
            "真の値と予測値の長さが一致しません: {} vs {}",
            y_true.len(),
            y_pred.len()
        )));
    }
    
    if y_true.is_empty() {
        return Err(Error::InvalidOperation("空のデータで計算することはできません".to_string()));
    }
    
    let sum_squared_error = y_true.iter()
        .zip(y_pred.iter())
        .map(|(&true_val, &pred_val)| {
            let error = true_val - pred_val;
            error * error
        })
        .sum::<f64>();
    
    Ok(sum_squared_error / y_true.len() as f64)
}

/// 平均絶対誤差（Mean Absolute Error）を計算
///
/// # Arguments
/// * `y_true` - 真の値
/// * `y_pred` - 予測値
///
/// # Returns
/// * `Result<f64>` - 平均絶対誤差
pub fn mean_absolute_error(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    if y_true.len() != y_pred.len() {
        return Err(Error::DimensionMismatch(format!(
            "真の値と予測値の長さが一致しません: {} vs {}",
            y_true.len(),
            y_pred.len()
        )));
    }
    
    if y_true.is_empty() {
        return Err(Error::InvalidOperation("空のデータで計算することはできません".to_string()));
    }
    
    let sum_absolute_error = y_true.iter()
        .zip(y_pred.iter())
        .map(|(&true_val, &pred_val)| {
            (true_val - pred_val).abs()
        })
        .sum::<f64>();
    
    Ok(sum_absolute_error / y_true.len() as f64)
}

/// 平均二乗誤差の平方根（Root Mean Squared Error）を計算
///
/// # Arguments
/// * `y_true` - 真の値
/// * `y_pred` - 予測値
///
/// # Returns
/// * `Result<f64>` - 平均二乗誤差の平方根
pub fn root_mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    let mse = mean_squared_error(y_true, y_pred)?;
    Ok(mse.sqrt())
}

/// 決定係数（R^2 score）を計算
///
/// # Arguments
/// * `y_true` - 真の値
/// * `y_pred` - 予測値
///
/// # Returns
/// * `Result<f64>` - 決定係数（1が最高、悪化すると負の値になり得る）
pub fn r2_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    if y_true.len() != y_pred.len() {
        return Err(Error::DimensionMismatch(format!(
            "真の値と予測値の長さが一致しません: {} vs {}",
            y_true.len(),
            y_pred.len()
        )));
    }
    
    if y_true.is_empty() {
        return Err(Error::InvalidOperation("空のデータで計算することはできません".to_string()));
    }
    
    // 真の値の平均を計算
    let y_mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
    
    // 全変動（total sum of squares）を計算
    let ss_tot = y_true.iter()
        .map(|&true_val| {
            let diff = true_val - y_mean;
            diff * diff
        })
        .sum::<f64>();
    
    // 残差平方和（residual sum of squares）を計算
    let ss_res = y_true.iter()
        .zip(y_pred.iter())
        .map(|(&true_val, &pred_val)| {
            let error = true_val - pred_val;
            error * error
        })
        .sum::<f64>();
    
    // ss_totが0の場合（全てのy_trueが同じ値）
    if ss_tot == 0.0 {
        if ss_res == 0.0 {
            // 完全な予測（y_true = y_predのとき）
            Ok(1.0)
        } else {
            // 定数予測で誤差がある場合
            Ok(0.0)
        }
    } else {
        Ok(1.0 - (ss_res / ss_tot))
    }
}

/// 説明分散（Explained variance score）を計算
///
/// # Arguments
/// * `y_true` - 真の値
/// * `y_pred` - 予測値
///
/// # Returns
/// * `Result<f64>` - 説明分散スコア（1が最高）
pub fn explained_variance_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    if y_true.len() != y_pred.len() {
        return Err(Error::DimensionMismatch(format!(
            "真の値と予測値の長さが一致しません: {} vs {}",
            y_true.len(),
            y_pred.len()
        )));
    }
    
    if y_true.is_empty() {
        return Err(Error::InvalidOperation("空のデータで計算することはできません".to_string()));
    }
    
    // 真の値の平均と予測値の平均
    let y_true_mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
    let y_pred_mean = y_pred.iter().sum::<f64>() / y_pred.len() as f64;
    
    // 真の値の分散を計算
    let var_y_true = y_true.iter()
        .map(|&val| {
            let diff = val - y_true_mean;
            diff * diff
        })
        .sum::<f64>() / y_true.len() as f64;
    
    // 残差の分散を計算
    let var_residual = y_true.iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| {
            let residual = (t - p) - (y_true_mean - y_pred_mean);
            residual * residual
        })
        .sum::<f64>() / y_true.len() as f64;
    
    // 分散が0の場合
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