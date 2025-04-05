//! 分類モデル評価のためのメトリクス

use crate::error::{Error, Result};
use std::cmp::Ordering;

/// 精度（Accuracy）を計算
///
/// # Arguments
/// * `y_true` - 真のラベル
/// * `y_pred` - 予測ラベル
///
/// # Returns
/// * `Result<f64>` - 精度（0〜1）
pub fn accuracy_score<T: PartialEq>(y_true: &[T], y_pred: &[T]) -> Result<f64> {
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
    
    let correct_count = y_true.iter()
        .zip(y_pred.iter())
        .filter(|(t, p)| t == p)
        .count();
    
    Ok(correct_count as f64 / y_true.len() as f64)
}

/// 適合率（Precision）を計算（2クラス分類）
///
/// # Arguments
/// * `y_true` - 真のラベル（trueまたはfalse）
/// * `y_pred` - 予測ラベル（trueまたはfalse）
///
/// # Returns
/// * `Result<f64>` - 適合率（0〜1）
pub fn precision_score(y_true: &[bool], y_pred: &[bool]) -> Result<f64> {
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
    
    // True Positiveの数
    let tp = y_true.iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| t && p)
        .count();
    
    // False Positiveの数
    let fp = y_true.iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| !t && p)
        .count();
    
    if tp + fp == 0 {
        return Ok(0.0); // 正と予測したサンプルがない場合
    }
    
    Ok(tp as f64 / (tp + fp) as f64)
}

/// 再現率（Recall）を計算（2クラス分類）
///
/// # Arguments
/// * `y_true` - 真のラベル（trueまたはfalse）
/// * `y_pred` - 予測ラベル（trueまたはfalse）
///
/// # Returns
/// * `Result<f64>` - 再現率（0〜1）
pub fn recall_score(y_true: &[bool], y_pred: &[bool]) -> Result<f64> {
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
    
    // True Positiveの数
    let tp = y_true.iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| t && p)
        .count();
    
    // False Negativeの数
    let fn_ = y_true.iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| t && !p)
        .count();
    
    if tp + fn_ == 0 {
        return Ok(0.0); // 実際に正のサンプルがない場合
    }
    
    Ok(tp as f64 / (tp + fn_) as f64)
}

/// F1スコアを計算（2クラス分類）
///
/// # Arguments
/// * `y_true` - 真のラベル（trueまたはfalse）
/// * `y_pred` - 予測ラベル（trueまたはfalse）
///
/// # Returns
/// * `Result<f64>` - F1スコア（0〜1）
pub fn f1_score(y_true: &[bool], y_pred: &[bool]) -> Result<f64> {
    let precision = precision_score(y_true, y_pred)?;
    let recall = recall_score(y_true, y_pred)?;
    
    if precision + recall == 0.0 {
        return Ok(0.0); // 分母がゼロになる場合
    }
    
    Ok(2.0 * precision * recall / (precision + recall))
}