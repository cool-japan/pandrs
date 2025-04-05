//! 評価指標モジュール
//!
//! 機械学習モデルの評価のための指標を提供します。

use crate::error::Result;
use crate::series::Series;
use crate::na::DataValue;

/// 回帰モデルの評価指標
pub mod regression {
    use super::*;
    
    /// 平均絶対誤差（Mean Absolute Error; MAE）を計算します
    pub fn mean_absolute_error(y_true: &Series, y_pred: &Series) -> Result<f64> {
        let n = y_true.len() as f64;
        let mut sum = 0.0;
        
        for (true_val, pred_val) in y_true.iter().zip(y_pred.iter()) {
            match (true_val, pred_val) {
                (DataValue::Float64(t), DataValue::Float64(p)) => sum += (t - p).abs(),
                (DataValue::Int64(t), DataValue::Int64(p)) => sum += (t - p).abs() as f64,
                (DataValue::Float64(t), DataValue::Int64(p)) => sum += (t - *p as f64).abs(),
                (DataValue::Int64(t), DataValue::Float64(p)) => sum += (*t as f64 - p).abs(),
                _ => return Err(crate::error::Error::InvalidOperation(
                    "Only numeric values can be used for MAE calculation".to_string()
                )),
            }
        }
        
        Ok(sum / n)
    }
    
    /// 平均二乗誤差（Mean Squared Error; MSE）を計算します
    pub fn mean_squared_error(y_true: &Series, y_pred: &Series) -> Result<f64> {
        let n = y_true.len() as f64;
        let mut sum = 0.0;
        
        for (true_val, pred_val) in y_true.iter().zip(y_pred.iter()) {
            match (true_val, pred_val) {
                (DataValue::Float64(t), DataValue::Float64(p)) => sum += (t - p).powi(2),
                (DataValue::Int64(t), DataValue::Int64(p)) => sum += ((t - p) as f64).powi(2),
                (DataValue::Float64(t), DataValue::Int64(p)) => sum += (t - *p as f64).powi(2),
                (DataValue::Int64(t), DataValue::Float64(p)) => sum += (*t as f64 - p).powi(2),
                _ => return Err(crate::error::Error::InvalidOperation(
                    "Only numeric values can be used for MSE calculation".to_string()
                )),
            }
        }
        
        Ok(sum / n)
    }
    
    /// 平均二乗誤差の平方根（Root Mean Squared Error; RMSE）を計算します
    pub fn root_mean_squared_error(y_true: &Series, y_pred: &Series) -> Result<f64> {
        let mse = mean_squared_error(y_true, y_pred)?;
        Ok(mse.sqrt())
    }
    
    /// 決定係数（R2スコア）を計算します
    pub fn r2_score(y_true: &Series, y_pred: &Series) -> Result<f64> {
        let n = y_true.len() as f64;
        let mut ss_res = 0.0;  // 残差平方和
        let mut ss_tot = 0.0;  // 全平方和
        
        let y_mean = y_true.mean()?;
        
        for (true_val, pred_val) in y_true.iter().zip(y_pred.iter()) {
            match (true_val, pred_val) {
                (DataValue::Float64(t), DataValue::Float64(p)) => {
                    ss_res += (t - p).powi(2);
                    ss_tot += (t - y_mean).powi(2);
                },
                (DataValue::Int64(t), DataValue::Int64(p)) => {
                    ss_res += ((t - p) as f64).powi(2);
                    ss_tot += (*t as f64 - y_mean).powi(2);
                },
                (DataValue::Float64(t), DataValue::Int64(p)) => {
                    ss_res += (t - *p as f64).powi(2);
                    ss_tot += (t - y_mean).powi(2);
                },
                (DataValue::Int64(t), DataValue::Float64(p)) => {
                    ss_res += (*t as f64 - p).powi(2);
                    ss_tot += (*t as f64 - y_mean).powi(2);
                },
                _ => return Err(crate::error::Error::InvalidOperation(
                    "Only numeric values can be used for R2 calculation".to_string()
                )),
            }
        }
        
        if ss_tot == 0.0 {
            Ok(0.0)  // すべての真の値が同じ場合
        } else {
            Ok(1.0 - (ss_res / ss_tot))
        }
    }
}

/// 分類モデルの評価指標
pub mod classification {
    use super::*;
    use std::collections::HashMap;
    
    /// 精度（Accuracy）を計算します
    pub fn accuracy_score(y_true: &Series, y_pred: &Series) -> Result<f64> {
        let n = y_true.len() as f64;
        let mut correct = 0.0;
        
        for (true_val, pred_val) in y_true.iter().zip(y_pred.iter()) {
            if true_val == pred_val {
                correct += 1.0;
            }
        }
        
        Ok(correct / n)
    }
    
    /// 適合率（Precision）を計算します（バイナリ分類用）
    pub fn precision_score(y_true: &Series, y_pred: &Series, positive_label: &DataValue) -> Result<f64> {
        let mut true_positive = 0;
        let mut false_positive = 0;
        
        for (true_val, pred_val) in y_true.iter().zip(y_pred.iter()) {
            if pred_val == positive_label {
                if true_val == positive_label {
                    true_positive += 1;
                } else {
                    false_positive += 1;
                }
            }
        }
        
        if true_positive + false_positive == 0 {
            Ok(0.0)
        } else {
            Ok(true_positive as f64 / (true_positive + false_positive) as f64)
        }
    }
    
    /// 再現率（Recall）を計算します（バイナリ分類用）
    pub fn recall_score(y_true: &Series, y_pred: &Series, positive_label: &DataValue) -> Result<f64> {
        let mut true_positive = 0;
        let mut false_negative = 0;
        
        for (true_val, pred_val) in y_true.iter().zip(y_pred.iter()) {
            if true_val == positive_label {
                if pred_val == positive_label {
                    true_positive += 1;
                } else {
                    false_negative += 1;
                }
            }
        }
        
        if true_positive + false_negative == 0 {
            Ok(0.0)
        } else {
            Ok(true_positive as f64 / (true_positive + false_negative) as f64)
        }
    }
    
    /// F1スコアを計算します（バイナリ分類用）
    pub fn f1_score(y_true: &Series, y_pred: &Series, positive_label: &DataValue) -> Result<f64> {
        let precision = precision_score(y_true, y_pred, positive_label)?;
        let recall = recall_score(y_true, y_pred, positive_label)?;
        
        if precision + recall == 0.0 {
            Ok(0.0)
        } else {
            Ok(2.0 * precision * recall / (precision + recall))
        }
    }
    
    /// 混同行列（Confusion Matrix）を計算します
    pub fn confusion_matrix(y_true: &Series, y_pred: &Series) -> Result<HashMap<(DataValue, DataValue), usize>> {
        let mut matrix = HashMap::new();
        
        for (true_val, pred_val) in y_true.iter().zip(y_pred.iter()) {
            let true_val = true_val.clone();
            let pred_val = pred_val.clone();
            
            let count = matrix.entry((true_val, pred_val)).or_insert(0);
            *count += 1;
        }
        
        Ok(matrix)
    }
}