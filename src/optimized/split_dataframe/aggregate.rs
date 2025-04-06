use std::collections::HashMap;
use rayon::prelude::*;

use crate::column::{Column, ColumnTrait};
use crate::error::{Error, Result};
use crate::optimized::split_dataframe::core::OptimizedDataFrame;

impl OptimizedDataFrame {
    /// 列の合計を計算
    ///
    /// # Arguments
    /// * `column_name` - 合計する列名
    ///
    /// # Returns
    /// * `Result<f64>` - 合計値
    pub fn sum(&self, column_name: &str) -> Result<f64> {
        let column_idx = self.column_indices.get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;
        
        let column = &self.columns[*column_idx];
        
        match column {
            Column::Int64(col) => {
                let sum = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .sum::<i64>() as f64;
                Ok(sum)
            },
            Column::Float64(col) => {
                let sum = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .sum::<f64>();
                Ok(sum)
            },
            _ => Err(Error::TypeError(format!(
                "列 '{}' は数値型ではありません", column_name
            ))),
        }
    }
    
    /// 列の平均値を計算
    ///
    /// # Arguments
    /// * `column_name` - 平均する列名
    ///
    /// # Returns
    /// * `Result<f64>` - 平均値
    pub fn mean(&self, column_name: &str) -> Result<f64> {
        let column_idx = self.column_indices.get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;
        
        let column = &self.columns[*column_idx];
        
        match column {
            Column::Int64(col) => {
                let values: Vec<i64> = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .collect();
                
                if values.is_empty() {
                    return Err(Error::EmptyColumn(column_name.to_string()));
                }
                
                let sum: i64 = values.iter().sum();
                Ok(sum as f64 / values.len() as f64)
            },
            Column::Float64(col) => {
                let values: Vec<f64> = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .collect();
                
                if values.is_empty() {
                    return Err(Error::EmptyColumn(column_name.to_string()));
                }
                
                let sum: f64 = values.iter().sum();
                Ok(sum / values.len() as f64)
            },
            _ => Err(Error::TypeError(format!(
                "列 '{}' は数値型ではありません", column_name
            ))),
        }
    }
    
    /// 列の最大値を計算
    ///
    /// # Arguments
    /// * `column_name` - 最大値を求める列名
    ///
    /// # Returns
    /// * `Result<f64>` - 最大値
    pub fn max(&self, column_name: &str) -> Result<f64> {
        let column_idx = self.column_indices.get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;
        
        let column = &self.columns[*column_idx];
        
        match column {
            Column::Int64(col) => {
                let max = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .max();
                    
                match max {
                    Some(val) => Ok(val as f64),
                    None => Err(Error::EmptyColumn(column_name.to_string())),
                }
            },
            Column::Float64(col) => {
                let max = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .fold(f64::NEG_INFINITY, |a, b| a.max(b));
                    
                if max.is_infinite() {
                    Err(Error::EmptyColumn(column_name.to_string()))
                } else {
                    Ok(max)
                }
            },
            _ => Err(Error::TypeError(format!(
                "列 '{}' は数値型ではありません", column_name
            ))),
        }
    }
    
    /// 列の最小値を計算
    ///
    /// # Arguments
    /// * `column_name` - 最小値を求める列名
    ///
    /// # Returns
    /// * `Result<f64>` - 最小値
    pub fn min(&self, column_name: &str) -> Result<f64> {
        let column_idx = self.column_indices.get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;
        
        let column = &self.columns[*column_idx];
        
        match column {
            Column::Int64(col) => {
                let min = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .min();
                    
                match min {
                    Some(val) => Ok(val as f64),
                    None => Err(Error::EmptyColumn(column_name.to_string())),
                }
            },
            Column::Float64(col) => {
                let min = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .fold(f64::INFINITY, |a, b| a.min(b));
                    
                if min.is_infinite() {
                    Err(Error::EmptyColumn(column_name.to_string()))
                } else {
                    Ok(min)
                }
            },
            _ => Err(Error::TypeError(format!(
                "列 '{}' は数値型ではありません", column_name
            ))),
        }
    }
    
    /// 列の非NULLデータ数を数える
    ///
    /// # Arguments
    /// * `column_name` - 対象列名
    ///
    /// # Returns
    /// * `Result<usize>` - 非NULL値の個数
    pub fn count(&self, column_name: &str) -> Result<usize> {
        let column_idx = self.column_indices.get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;
        
        let column = &self.columns[*column_idx];
        
        let count = match column {
            Column::Int64(col) => (0..col.len())
                .filter(|&i| col.get(i).map_or(false, |opt| opt.is_some()))
                .count(),
            Column::Float64(col) => (0..col.len())
                .filter(|&i| col.get(i).map_or(false, |opt| opt.is_some()))
                .count(),
            Column::String(col) => (0..col.len())
                .filter(|&i| col.get(i).map_or(false, |opt| opt.is_some()))
                .count(),
            Column::Boolean(col) => (0..col.len())
                .filter(|&i| col.get(i).map_or(false, |opt| opt.is_some()))
                .count(),
        };
        
        Ok(count)
    }
    
    /// 複数列に対して集計操作を実行
    ///
    /// # Arguments
    /// * `column_names` - 対象列名配列
    /// * `operation` - 操作名称。"sum", "mean", "max", "min", "count"のいずれか
    ///
    /// # Returns
    /// * `Result<HashMap<String, f64>>` - 各列の計算結果を格納したハッシュマップ
    pub fn aggregate(&self, column_names: &[&str], operation: &str) -> Result<HashMap<String, f64>> {
        let mut results = HashMap::new();
        
        for &column_name in column_names {
            let result = match operation {
                "sum" => self.sum(column_name),
                "mean" => self.mean(column_name),
                "max" => self.max(column_name),
                "min" => self.min(column_name),
                "count" => self.count(column_name).map(|c| c as f64),
                _ => return Err(Error::OperationNotSupported(operation.to_string())),
            };
            
            // エラーの場合はその列をスキップ
            if let Ok(value) = result {
                results.insert(column_name.to_string(), value);
            }
        }
        
        if results.is_empty() {
            Err(Error::OperationFailed(format!("操作 '{}' がすべての列で失敗しました", operation)))
        } else {
            Ok(results)
        }
    }
    
    /// 全ての数値列に対して集計操作を実行
    ///
    /// # Arguments
    /// * `operation` - 操作名称。"sum", "mean", "max", "min", "count"のいずれか
    ///
    /// # Returns
    /// * `Result<HashMap<String, f64>>` - 各列の計算結果を格納したハッシュマップ
    pub fn aggregate_numeric(&self, operation: &str) -> Result<HashMap<String, f64>> {
        // 数値列名を収集
        let numeric_columns: Vec<&str> = self.column_names.iter()
            .filter(|&name| {
                let idx = self.column_indices.get(name).unwrap();
                matches!(self.columns[*idx], Column::Int64(_) | Column::Float64(_))
            })
            .map(|s| s.as_str())
            .collect();
        
        if numeric_columns.is_empty() {
            return Err(Error::OperationFailed("数値列が存在しません".to_string()));
        }
        
        self.aggregate(&numeric_columns, operation)
    }
}