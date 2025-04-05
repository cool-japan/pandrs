//! OptimizedDataFrameのデータ操作関連機能

use std::collections::HashMap;
use rayon::prelude::*;

use super::core::{OptimizedDataFrame, ColumnView};
use crate::column::{Column, ColumnType, Int64Column, Float64Column, StringColumn, BooleanColumn};
use crate::error::{Error, Result};
use crate::index::{DataFrameIndex, IndexTrait, Index};

impl OptimizedDataFrame {
    /// 列の選択（新しいDataFrameとして）
    pub fn select(&self, columns: &[&str]) -> Result<Self> {
        let mut result = Self::new();
        
        for &name in columns {
            let column_idx = self.column_indices.get(name)
                .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
            
            let column = self.columns[*column_idx].clone();
            result.add_column(name.to_string(), column)?;
        }
        
        // インデックスのコピー
        if let Some(ref idx) = self.index {
            result.index = Some(idx.clone());
        }
        
        Ok(result)
    }
    
    /// フィルタリング（新しいDataFrameとして）
    pub fn filter(&self, condition_column: &str) -> Result<Self> {
        // 条件列の取得
        let column_idx = self.column_indices.get(condition_column)
            .ok_or_else(|| Error::ColumnNotFound(condition_column.to_string()))?;
        
        let condition = &self.columns[*column_idx];
        
        // 条件列がブール型であることを確認
        if let Column::Boolean(bool_col) = condition {
            // trueの行のインデックスを収集
            let mut indices = Vec::new();
            for i in 0..bool_col.len() {
                if let Ok(Some(true)) = bool_col.get(i) {
                    indices.push(i);
                }
            }
            
            // 新しいDataFrameを作成
            let mut result = Self::new();
            
            // 各列をフィルタリング
            for (i, name) in self.column_names.iter().enumerate() {
                let column = &self.columns[i];
                
                let filtered_column = match column {
                    Column::Int64(col) => {
                        let mut filtered_data = Vec::with_capacity(indices.len());
                        for &idx in &indices {
                            if let Ok(Some(val)) = col.get(idx) {
                                filtered_data.push(val);
                            } else {
                                filtered_data.push(0); // デフォルト値
                            }
                        }
                        Column::Int64(Int64Column::new(filtered_data))
                    },
                    Column::Float64(col) => {
                        let mut filtered_data = Vec::with_capacity(indices.len());
                        for &idx in &indices {
                            if let Ok(Some(val)) = col.get(idx) {
                                filtered_data.push(val);
                            } else {
                                filtered_data.push(0.0); // デフォルト値
                            }
                        }
                        Column::Float64(Float64Column::new(filtered_data))
                    },
                    Column::String(col) => {
                        let mut filtered_data = Vec::with_capacity(indices.len());
                        for &idx in &indices {
                            if let Ok(Some(val)) = col.get(idx) {
                                filtered_data.push(val.to_string());
                            } else {
                                filtered_data.push(String::new()); // デフォルト値
                            }
                        }
                        Column::String(StringColumn::new(filtered_data))
                    },
                    Column::Boolean(col) => {
                        let mut filtered_data = Vec::with_capacity(indices.len());
                        for &idx in &indices {
                            if let Ok(Some(val)) = col.get(idx) {
                                filtered_data.push(val);
                            } else {
                                filtered_data.push(false); // デフォルト値
                            }
                        }
                        Column::Boolean(BooleanColumn::new(filtered_data))
                    },
                };
                
                result.add_column(name.clone(), filtered_column)?;
            }
            
            Ok(result)
        } else {
            Err(Error::ColumnTypeMismatch {
                name: condition_column.to_string(),
                expected: ColumnType::Boolean,
                found: condition.column_type(),
            })
        }
    }
    
    /// 指定された行インデックスでフィルタリング（内部ヘルパー）
    pub(crate) fn filter_by_indices(&self, indices: &[usize]) -> Result<Self> {
        let mut result = Self::new();
        
        // 各列をフィルタリング
        for (i, name) in self.column_names.iter().enumerate() {
            let column = &self.columns[i];
            
            let filtered_column = match column {
                Column::Int64(col) => {
                    let filtered_data: Vec<i64> = indices.iter()
                        .filter_map(|&idx| {
                            if idx < col.len() {
                                if let Ok(Some(val)) = col.get(idx) {
                                    Some(val)
                                } else {
                                    Some(0) // デフォルト値
                                }
                            } else {
                                None
                            }
                        })
                        .collect();
                    Column::Int64(Int64Column::new(filtered_data))
                },
                Column::Float64(col) => {
                    let filtered_data: Vec<f64> = indices.iter()
                        .filter_map(|&idx| {
                            if idx < col.len() {
                                if let Ok(Some(val)) = col.get(idx) {
                                    Some(val)
                                } else {
                                    Some(0.0) // デフォルト値
                                }
                            } else {
                                None
                            }
                        })
                        .collect();
                    Column::Float64(Float64Column::new(filtered_data))
                },
                Column::String(col) => {
                    let filtered_data: Vec<String> = indices.iter()
                        .filter_map(|&idx| {
                            if idx < col.len() {
                                if let Ok(Some(val)) = col.get(idx) {
                                    Some(val.to_string())
                                } else {
                                    Some(String::new()) // デフォルト値
                                }
                            } else {
                                None
                            }
                        })
                        .collect();
                    Column::String(StringColumn::new(filtered_data))
                },
                Column::Boolean(col) => {
                    let filtered_data: Vec<bool> = indices.iter()
                        .filter_map(|&idx| {
                            if idx < col.len() {
                                if let Ok(Some(val)) = col.get(idx) {
                                    Some(val)
                                } else {
                                    Some(false) // デフォルト値
                                }
                            } else {
                                None
                            }
                        })
                        .collect();
                    Column::Boolean(BooleanColumn::new(filtered_data))
                },
            };
            
            result.add_column(name.clone(), filtered_column)?;
        }
        
        // インデックスのコピー
        if let Some(ref idx) = self.index {
            result.index = Some(idx.clone());
        }
        
        Ok(result)
    }
    
    /// 先頭n行を取得
    pub fn head(&self, n: usize) -> Result<Self> {
        let n = std::cmp::min(n, self.row_count);
        let indices: Vec<usize> = (0..n).collect();
        self.filter_by_indices(&indices)
    }
    
    /// 末尾n行を取得
    pub fn tail(&self, n: usize) -> Result<Self> {
        let n = std::cmp::min(n, self.row_count);
        let start = self.row_count.saturating_sub(n);
        let indices: Vec<usize> = (start..self.row_count).collect();
        self.filter_by_indices(&indices)
    }
}
