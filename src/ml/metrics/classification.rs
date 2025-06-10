//! Classification model evaluation metrics

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

/// Calculate accuracy
///
/// # Arguments
/// * `y_true` - True labels
/// * `y_pred` - Predicted labels
///
/// # Returns
/// * `Result<f64>` - Accuracy (0 to 1)
pub fn accuracy_score<T: PartialEq>(y_true: &[T], y_pred: &[T]) -> Result<f64> {
    if y_true.len() != y_pred.len() {
        return Err(Error::DimensionMismatch(format!(
            "Length mismatch between true and predicted labels: {} vs {}",
            y_true.len(),
            y_pred.len()
        )));
    }

    if y_true.is_empty() {
        return Err(Error::InvalidOperation(
            "Cannot calculate with empty data".to_string(),
        ));
    }

    let correct_count = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(t, p)| t == p)
        .count();

    Ok(correct_count as f64 / y_true.len() as f64)
}

/// Calculate precision (binary classification)
///
/// # Arguments
/// * `y_true` - True labels (true or false)
/// * `y_pred` - Predicted labels (true or false)
///
/// # Returns
/// * `Result<f64>` - Precision (0 to 1)
pub fn precision_score(y_true: &[bool], y_pred: &[bool]) -> Result<f64> {
    if y_true.len() != y_pred.len() {
        return Err(Error::DimensionMismatch(format!(
            "Length mismatch between true and predicted labels: {} vs {}",
            y_true.len(),
            y_pred.len()
        )));
    }

    if y_true.is_empty() {
        return Err(Error::InvalidOperation(
            "Cannot calculate with empty data".to_string(),
        ));
    }

    // Count true positives
    let tp = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| t && p)
        .count();

    // Count false positives
    let fp = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| !t && p)
        .count();

    if tp + fp == 0 {
        return Ok(0.0); // No positive predictions
    }

    Ok(tp as f64 / (tp + fp) as f64)
}

/// Calculate recall (binary classification)
///
/// # Arguments
/// * `y_true` - True labels (true or false)
/// * `y_pred` - Predicted labels (true or false)
///
/// # Returns
/// * `Result<f64>` - Recall (0 to 1)
pub fn recall_score(y_true: &[bool], y_pred: &[bool]) -> Result<f64> {
    if y_true.len() != y_pred.len() {
        return Err(Error::DimensionMismatch(format!(
            "Length mismatch between true and predicted labels: {} vs {}",
            y_true.len(),
            y_pred.len()
        )));
    }

    if y_true.is_empty() {
        return Err(Error::InvalidOperation(
            "Cannot calculate with empty data".to_string(),
        ));
    }

    // Count true positives
    let tp = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| t && p)
        .count();

    // Count false negatives
    let fn_ = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| t && !p)
        .count();

    if tp + fn_ == 0 {
        return Ok(0.0); // No actual positive samples
    }

    Ok(tp as f64 / (tp + fn_) as f64)
}

/// Calculate F1 score (binary classification)
///
/// # Arguments
/// * `y_true` - True labels (true or false)
/// * `y_pred` - Predicted labels (true or false)
///
/// # Returns
/// * `Result<f64>` - F1 score (0 to 1)
pub fn f1_score(y_true: &[bool], y_pred: &[bool]) -> Result<f64> {
    let precision = precision_score(y_true, y_pred)?;
    let recall = recall_score(y_true, y_pred)?;

    if precision + recall == 0.0 {
        return Ok(0.0); // Avoid division by zero
    }

    Ok(2.0 * precision * recall / (precision + recall))
}
