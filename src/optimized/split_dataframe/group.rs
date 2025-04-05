//! OptimizedDataFrameのグループ化と集計機能

use std::collections::HashMap;

use super::core::OptimizedDataFrame;
use crate::column::{Column, ColumnType, Float64Column, Int64Column, StringColumn, BooleanColumn};
use crate::error::{Error, Result};

/// 集計操作を表す列挙型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateOp {
    /// 合計
    Sum,
    /// 平均
    Mean,
    /// 最小値
    Min,
    /// 最大値
    Max,
    /// 件数
    Count,
}

/// グループ化結果を表す構造体
pub struct GroupBy<'a> {
    /// 元のデータフレーム
    df: &'a OptimizedDataFrame,
    /// グループ化キー列
    group_by_columns: Vec<String>,
    /// グループごとの行インデックス
    groups: HashMap<Vec<String>, Vec<usize>>,
}

impl OptimizedDataFrame {
    /// データフレームをグループ化
    ///
    /// # Arguments
    /// * `columns` - グループ化するための列名
    ///
    /// # Returns
    /// * `Result<GroupBy>` - グループ化結果
    pub fn group_by<I, S>(&self, columns: I) -> Result<GroupBy<'_>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let group_by_columns: Vec<String> = columns
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();
        
        // 各列の存在確認
        for column in &group_by_columns {
            if !self.column_indices.contains_key(column) {
                return Err(Error::ColumnNotFound(column.clone()));
            }
        }
        
        // グループ化キーの作成
        let mut groups: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
        
        for row_idx in 0..self.row_count {
            let mut key = Vec::with_capacity(group_by_columns.len());
            
            for col_name in &group_by_columns {
                let col_idx = self.column_indices[col_name];
                let col = &self.columns[col_idx];
                
                let key_part = match col {
                    Column::Int64(int_col) => {
                        if let Ok(Some(val)) = int_col.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    },
                    Column::Float64(float_col) => {
                        if let Ok(Some(val)) = float_col.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    },
                    Column::String(str_col) => {
                        if let Ok(Some(val)) = str_col.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    },
                    Column::Boolean(bool_col) => {
                        if let Ok(Some(val)) = bool_col.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    },
                };
                
                key.push(key_part);
            }
            
            groups.entry(key).or_default().push(row_idx);
        }
        
        Ok(GroupBy {
            df: self,
            group_by_columns,
            groups,
        })
    }
}

impl<'a> GroupBy<'a> {
    /// グループごとに集計操作を実行
    ///
    /// # Arguments
    /// * `aggregations` - 集計操作のリスト（列名、操作、結果列名）
    ///
    /// # Returns
    /// * `Result<OptimizedDataFrame>` - 集計結果のDataFrame
    pub fn aggregate<I>(&self, aggregations: I) -> Result<OptimizedDataFrame>
    where
        I: IntoIterator<Item = (String, AggregateOp, String)>,
    {
        let aggregations: Vec<(String, AggregateOp, String)> = aggregations.into_iter().collect();
        
        // 各集計対象列の存在確認
        for (col_name, _, _) in &aggregations {
            if !self.df.column_indices.contains_key(col_name) {
                return Err(Error::ColumnNotFound(col_name.clone()));
            }
        }
        
        // 集計結果の格納先
        let mut result = OptimizedDataFrame::new();
        
        // グループ化キー列のデータ
        let mut group_key_data: HashMap<String, Vec<String>> = HashMap::new();
        for key in self.group_by_columns.iter() {
            group_key_data.insert(key.clone(), Vec::new());
        }
        
        // 集計結果列のデータ
        let mut agg_result_data: HashMap<String, Vec<f64>> = HashMap::new();
        for (_, _, alias) in &aggregations {
            agg_result_data.insert(alias.clone(), Vec::new());
        }
        
        // 各グループに対して集計を実行
        for (key, row_indices) in &self.groups {
            // グループ化キーの値を追加
            for (i, col_name) in self.group_by_columns.iter().enumerate() {
                group_key_data.get_mut(col_name).unwrap().push(key[i].clone());
            }
            
            // 集計操作を実行
            for (col_name, op, alias) in &aggregations {
                let col_idx = self.df.column_indices[col_name];
                let col = &self.df.columns[col_idx];
                
                let result_value = match (col, op) {
                    (Column::Int64(int_col), AggregateOp::Sum) => {
                        let mut sum = 0;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = int_col.get(idx) {
                                sum += val;
                            }
                        }
                        sum as f64
                    },
                    (Column::Int64(int_col), AggregateOp::Mean) => {
                        let mut sum = 0;
                        let mut count = 0;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = int_col.get(idx) {
                                sum += val;
                                count += 1;
                            }
                        }
                        if count > 0 {
                            sum as f64 / count as f64
                        } else {
                            0.0
                        }
                    },
                    (Column::Int64(int_col), AggregateOp::Min) => {
                        let mut min = i64::MAX;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = int_col.get(idx) {
                                min = min.min(val);
                            }
                        }
                        if min == i64::MAX {
                            0.0
                        } else {
                            min as f64
                        }
                    },
                    (Column::Int64(int_col), AggregateOp::Max) => {
                        let mut max = i64::MIN;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = int_col.get(idx) {
                                max = max.max(val);
                            }
                        }
                        if max == i64::MIN {
                            0.0
                        } else {
                            max as f64
                        }
                    },
                    (Column::Float64(float_col), AggregateOp::Sum) => {
                        let mut sum = 0.0;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = float_col.get(idx) {
                                sum += val;
                            }
                        }
                        sum
                    },
                    (Column::Float64(float_col), AggregateOp::Mean) => {
                        let mut sum = 0.0;
                        let mut count = 0;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = float_col.get(idx) {
                                sum += val;
                                count += 1;
                            }
                        }
                        if count > 0 {
                            sum / count as f64
                        } else {
                            0.0
                        }
                    },
                    (Column::Float64(float_col), AggregateOp::Min) => {
                        let mut min = f64::INFINITY;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = float_col.get(idx) {
                                min = min.min(val);
                            }
                        }
                        if min == f64::INFINITY {
                            0.0
                        } else {
                            min
                        }
                    },
                    (Column::Float64(float_col), AggregateOp::Max) => {
                        let mut max = f64::NEG_INFINITY;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = float_col.get(idx) {
                                max = max.max(val);
                            }
                        }
                        if max == f64::NEG_INFINITY {
                            0.0
                        } else {
                            max
                        }
                    },
                    (_, AggregateOp::Count) => {
                        row_indices.len() as f64
                    },
                    _ => {
                        return Err(Error::OperationFailed(format!(
                            "集計操作 {:?} は列タイプ {:?} に対応していません",
                            op, col.column_type()
                        )));
                    }
                };
                
                agg_result_data.get_mut(alias).unwrap().push(result_value);
            }
        }
        
        // グループ化キー列を追加
        for (col_name, values) in group_key_data {
            // 文字列列として追加
            let col = StringColumn::new(values);
            result.add_column(col_name, Column::String(col))?;
        }
        
        // 集計結果列を追加
        for (_, _, alias) in &aggregations {
            let values = agg_result_data.get(alias).unwrap();
            let col = Float64Column::new(values.clone());
            result.add_column(alias.clone(), Column::Float64(col))?;
        }
        
        Ok(result)
    }
    
    /// 集計のショートカットメソッド: 合計
    pub fn sum(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_sum", column);
        self.aggregate(vec![(column.to_string(), AggregateOp::Sum, agg_name)])
    }
    
    /// 集計のショートカットメソッド: 平均
    pub fn mean(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_mean", column);
        self.aggregate(vec![(column.to_string(), AggregateOp::Mean, agg_name)])
    }
    
    /// 集計のショートカットメソッド: 最小値
    pub fn min(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_min", column);
        self.aggregate(vec![(column.to_string(), AggregateOp::Min, agg_name)])
    }
    
    /// 集計のショートカットメソッド: 最大値
    pub fn max(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_max", column);
        self.aggregate(vec![(column.to_string(), AggregateOp::Max, agg_name)])
    }
    
    /// 集計のショートカットメソッド: 件数
    pub fn count(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_count", column);
        self.aggregate(vec![(column.to_string(), AggregateOp::Count, agg_name)])
    }
}
