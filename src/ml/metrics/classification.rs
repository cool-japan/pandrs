//! 分類モデル評価のためのメトリクス

use crate::error::{Error, Result};
use std::cmp::Ordering;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy_score() {
        let true_labels = vec![true, false, true, true, false, false];
        let pred_labels = vec![true, false, false, true, true, false];
        
        let accuracy = accuracy_score(&true_labels, &pred_labels).unwrap();
        assert!((accuracy - 0.6666666).abs() < 1e-6); // 4/6 = 0.6666...
    }

    #[test]
    fn test_precision_score() {
        let true_labels = vec![true, false, true, true, false, false];
        let pred_labels = vec![true, false, false, true, true, false];
        
        let precision = precision_score(&true_labels, &pred_labels).unwrap();
        assert!((precision - 0.6666666).abs() < 1e-6); // TP=2, FP=1, 2/(2+1) = 0.6666...
    }

    #[test]
    fn test_recall_score() {
        let true_labels = vec![true, false, true, true, false, false];
        let pred_labels = vec![true, false, false, true, true, false];
        
        let recall = recall_score(&true_labels, &pred_labels).unwrap();
        assert!((recall - 0.6666666).abs() < 1e-6); // TP=2, FN=1, 2/(2+1) = 0.6666...
    }

    #[test]
    fn test_f1_score() {
        let true_labels = vec![true, false, true, true, false, false];
        let pred_labels = vec![true, false, false, true, true, false];
        
        let f1 = f1_score(&true_labels, &pred_labels).unwrap();
        assert!((f1 - 0.6666666).abs() < 1e-6); // precision=recall=0.6666..., F1 = 2*p*r/(p+r) = 0.6666...
    }

    #[test]
    fn test_empty_input() {
        let empty: Vec<bool> = vec![];
        
        let accuracy_result = accuracy_score(&empty, &empty);
        assert!(accuracy_result.is_err());
        
        let precision_result = precision_score(&empty, &empty);
        assert!(precision_result.is_err());
    }

    #[test]
    fn test_different_length() {
        let true_labels = vec![true, false, true];
        let pred_labels = vec![true, false];
        
        let accuracy_result = accuracy_score(&true_labels, &pred_labels);
        assert!(accuracy_result.is_err());
        
        let precision_result = precision_score(&true_labels, &pred_labels);
        assert!(precision_result.is_err());
    }
}

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