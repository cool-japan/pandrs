//! OptimizedDataFrameのデータ操作関連機能

use std::collections::HashMap;
use rayon::prelude::*;

use super::core::{OptimizedDataFrame, ColumnView};
use crate::column::{Column, ColumnType, Int64Column, Float64Column, StringColumn, BooleanColumn, ColumnTrait};
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
    
    /// DataFrameを「長形式」に変換する（melt操作）
    /// 
    /// 複数の列を単一の「変数」列と「値」列に変換します。
    /// パフォーマンスを最優先した実装です。
    ///
    /// # Arguments
    /// * `id_vars` - 変換せずに保持する列名（識別子列）
    /// * `value_vars` - 変換する列名（値列）。指定しない場合はid_vars以外のすべての列
    /// * `var_name` - 変数名の列名（デフォルト: "variable"）
    /// * `value_name` - 値の列名（デフォルト: "value"）
    ///
    /// # Returns
    /// * `Result<Self>` - 長形式に変換されたDataFrame
    pub fn melt(
        &self,
        id_vars: &[&str],
        value_vars: Option<&[&str]>,
        var_name: Option<&str>,
        value_name: Option<&str>,
    ) -> Result<Self> {
        // 引数のデフォルト値を設定
        let var_name = var_name.unwrap_or("variable");
        let value_name = value_name.unwrap_or("value");
        
        // value_varsが指定されていない場合は、id_vars以外のすべての列を使用
        let value_vars = if let Some(vars) = value_vars {
            vars.to_vec()
        } else {
            self.column_names
                .iter()
                .filter(|name| !id_vars.contains(&name.as_str()))
                .map(|s| s.as_str())
                .collect()
        };
        
        // 存在しない列名をチェック
        for col in id_vars.iter().chain(value_vars.iter()) {
            if !self.column_indices.contains_key(*col) {
                return Err(Error::ColumnNotFound((*col).to_string()));
            }
        }
        
        // 結果のサイズを事前計算（パフォーマンス最適化）
        let result_rows = self.row_count * value_vars.len();
        
        // ID列のデータを抽出
        let mut id_columns = Vec::with_capacity(id_vars.len());
        for &id_col in id_vars {
            let idx = self.column_indices[id_col];
            id_columns.push((id_col, &self.columns[idx]));
        }
        
        // 値列のデータを抽出
        let mut value_columns = Vec::with_capacity(value_vars.len());
        for &val_col in &value_vars {
            let idx = self.column_indices[val_col];
            value_columns.push((val_col, &self.columns[idx]));
        }
        
        // 結果のDataFrameを生成
        let mut result = Self::new();
        
        // 変数名の列を作成
        let mut var_col_data = Vec::with_capacity(result_rows);
        for &value_col_name in &value_vars {
            for _ in 0..self.row_count {
                var_col_data.push(value_col_name.to_string());
            }
        }
        result.add_column(var_name.to_string(), Column::String(StringColumn::new(var_col_data)))?;
        
        // ID列をレプリケートして追加
        for &(id_col_name, col) in &id_columns {
            match col {
                Column::Int64(int_col) => {
                    // 整数型の列
                    let mut repeated_data = Vec::with_capacity(result_rows);
                    for _ in 0..value_vars.len() {
                        for i in 0..self.row_count {
                            if let Ok(Some(val)) = int_col.get(i) {
                                repeated_data.push(val);
                            } else {
                                // NULL値の場合はデフォルト値を使用
                                repeated_data.push(0);
                            }
                        }
                    }
                    result.add_column(id_col_name.to_string(), Column::Int64(Int64Column::new(repeated_data)))?;
                },
                Column::Float64(float_col) => {
                    // 浮動小数点型の列
                    let mut repeated_data = Vec::with_capacity(result_rows);
                    for _ in 0..value_vars.len() {
                        for i in 0..self.row_count {
                            if let Ok(Some(val)) = float_col.get(i) {
                                repeated_data.push(val);
                            } else {
                                // NULL値の場合はデフォルト値を使用
                                repeated_data.push(0.0);
                            }
                        }
                    }
                    result.add_column(id_col_name.to_string(), Column::Float64(Float64Column::new(repeated_data)))?;
                },
                Column::String(str_col) => {
                    // 文字列型の列
                    let mut repeated_data = Vec::with_capacity(result_rows);
                    for _ in 0..value_vars.len() {
                        for i in 0..self.row_count {
                            if let Ok(Some(val)) = str_col.get(i) {
                                repeated_data.push(val.to_string());
                            } else {
                                // NULL値の場合は空文字列を使用
                                repeated_data.push(String::new());
                            }
                        }
                    }
                    result.add_column(id_col_name.to_string(), Column::String(StringColumn::new(repeated_data)))?;
                },
                Column::Boolean(bool_col) => {
                    // ブール型の列
                    let mut repeated_data = Vec::with_capacity(result_rows);
                    for _ in 0..value_vars.len() {
                        for i in 0..self.row_count {
                            if let Ok(Some(val)) = bool_col.get(i) {
                                repeated_data.push(val);
                            } else {
                                // NULL値の場合はデフォルト値を使用
                                repeated_data.push(false);
                            }
                        }
                    }
                    result.add_column(id_col_name.to_string(), Column::Boolean(BooleanColumn::new(repeated_data)))?;
                },
            }
        }
        
        // 値列を作成（型に応じた最適な方法で）
        // 最適化のため、最終的な型を推測してからデータを追加する
        let mut all_values = Vec::with_capacity(result_rows);
        
        // まず全ての値を文字列として収集
        for (_, col) in value_columns {
            match col {
                Column::Int64(int_col) => {
                    for i in 0..self.row_count {
                        if let Ok(Some(val)) = int_col.get(i) {
                            all_values.push(val.to_string());
                        } else {
                            all_values.push(String::new());
                        }
                    }
                },
                Column::Float64(float_col) => {
                    for i in 0..self.row_count {
                        if let Ok(Some(val)) = float_col.get(i) {
                            all_values.push(val.to_string());
                        } else {
                            all_values.push(String::new());
                        }
                    }
                },
                Column::String(str_col) => {
                    for i in 0..self.row_count {
                        if let Ok(Some(val)) = str_col.get(i) {
                            all_values.push(val.to_string());
                        } else {
                            all_values.push(String::new());
                        }
                    }
                },
                Column::Boolean(bool_col) => {
                    for i in 0..self.row_count {
                        if let Ok(Some(val)) = bool_col.get(i) {
                            all_values.push(val.to_string());
                        } else {
                            all_values.push(String::new());
                        }
                    }
                },
            }
        }
        
        // 適切な型を推測してデータ追加（すべて数値なら数値型、等）
        let is_all_int = all_values.iter()
            .all(|s| s.parse::<i64>().is_ok() || s.is_empty());
        
        let is_all_float = !is_all_int && all_values.iter()
            .all(|s| s.parse::<f64>().is_ok() || s.is_empty());
        
        let is_all_bool = !is_all_int && !is_all_float && all_values.iter()
            .all(|s| {
                let lower = s.to_lowercase();
                lower.is_empty() || lower == "true" || lower == "false" || 
                lower == "1" || lower == "0" || lower == "yes" || lower == "no"
            });
        
        // 型に合わせて列を追加
        if is_all_int {
            let int_values: Vec<i64> = all_values.iter()
                .map(|s| s.parse::<i64>().unwrap_or(0))
                .collect();
            result.add_column(value_name.to_string(), Column::Int64(Int64Column::new(int_values)))?;
        } else if is_all_float {
            let float_values: Vec<f64> = all_values.iter()
                .map(|s| s.parse::<f64>().unwrap_or(0.0))
                .collect();
            result.add_column(value_name.to_string(), Column::Float64(Float64Column::new(float_values)))?;
        } else if is_all_bool {
            let bool_values: Vec<bool> = all_values.iter()
                .map(|s| {
                    let lower = s.to_lowercase();
                    lower == "true" || lower == "1" || lower == "yes"
                })
                .collect();
            result.add_column(value_name.to_string(), Column::Boolean(BooleanColumn::new(bool_values)))?;
        } else {
            // デフォルトは文字列型
            result.add_column(value_name.to_string(), Column::String(StringColumn::new(all_values)))?;
        }
        
        Ok(result)
    }
    
    /// データフレームを縦方向に連結する
    ///
    /// # Arguments
    /// * `other` - 連結するデータフレーム
    ///
    /// # Returns
    /// * `Result<Self>` - 連結されたデータフレーム
    pub fn append(&self, other: &Self) -> Result<Self> {
        if self.columns.is_empty() {
            return Ok(other.clone());
        }
        
        if other.columns.is_empty() {
            return Ok(self.clone());
        }
        
        // 結果のDataFrameを生成
        let mut result = Self::new();
        
        // 列名の集合を作成
        let mut all_columns = std::collections::HashSet::new();
        
        for name in &self.column_names {
            all_columns.insert(name.clone());
        }
        
        for name in &other.column_names {
            all_columns.insert(name.clone());
        }
        
        // 新しい列データを準備
        for col_name in all_columns {
            let self_has_column = self.column_indices.contains_key(&col_name);
            let other_has_column = other.column_indices.contains_key(&col_name);
            
            // 両方のDataFrameに列が存在する場合
            if self_has_column && other_has_column {
                let self_col_idx = self.column_indices[&col_name];
                let other_col_idx = other.column_indices[&col_name];
                
                let self_col = &self.columns[self_col_idx];
                let other_col = &other.columns[other_col_idx];
                
                // 同じ型の列を連結
                if self_col.column_type() == other_col.column_type() {
                    match (self_col, other_col) {
                        (Column::Int64(self_int), Column::Int64(other_int)) => {
                            let mut combined_data = Vec::with_capacity(self_int.len() + other_int.len());
                            
                            // 自分のデータを追加
                            for i in 0..self_int.len() {
                                if let Ok(Some(val)) = self_int.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(0); // デフォルト値
                                }
                            }
                            
                            // 相手のデータを追加
                            for i in 0..other_int.len() {
                                if let Ok(Some(val)) = other_int.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(0); // デフォルト値
                                }
                            }
                            
                            result.add_column(col_name, Column::Int64(Int64Column::new(combined_data)))?;
                        },
                        (Column::Float64(self_float), Column::Float64(other_float)) => {
                            let mut combined_data = Vec::with_capacity(self_float.len() + other_float.len());
                            
                            // 自分のデータを追加
                            for i in 0..self_float.len() {
                                if let Ok(Some(val)) = self_float.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(0.0); // デフォルト値
                                }
                            }
                            
                            // 相手のデータを追加
                            for i in 0..other_float.len() {
                                if let Ok(Some(val)) = other_float.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(0.0); // デフォルト値
                                }
                            }
                            
                            result.add_column(col_name, Column::Float64(Float64Column::new(combined_data)))?;
                        },
                        (Column::String(self_str), Column::String(other_str)) => {
                            let mut combined_data = Vec::with_capacity(self_str.len() + other_str.len());
                            
                            // 自分のデータを追加
                            for i in 0..self_str.len() {
                                if let Ok(Some(val)) = self_str.get(i) {
                                    combined_data.push(val.to_string());
                                } else {
                                    combined_data.push(String::new()); // デフォルト値
                                }
                            }
                            
                            // 相手のデータを追加
                            for i in 0..other_str.len() {
                                if let Ok(Some(val)) = other_str.get(i) {
                                    combined_data.push(val.to_string());
                                } else {
                                    combined_data.push(String::new()); // デフォルト値
                                }
                            }
                            
                            result.add_column(col_name, Column::String(StringColumn::new(combined_data)))?;
                        },
                        (Column::Boolean(self_bool), Column::Boolean(other_bool)) => {
                            let mut combined_data = Vec::with_capacity(self_bool.len() + other_bool.len());
                            
                            // 自分のデータを追加
                            for i in 0..self_bool.len() {
                                if let Ok(Some(val)) = self_bool.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(false); // デフォルト値
                                }
                            }
                            
                            // 相手のデータを追加
                            for i in 0..other_bool.len() {
                                if let Ok(Some(val)) = other_bool.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(false); // デフォルト値
                                }
                            }
                            
                            result.add_column(col_name, Column::Boolean(BooleanColumn::new(combined_data)))?;
                        },
                        _ => {
                            // 型が一致しない場合は文字列として連結
                            let mut combined_data = Vec::with_capacity(self.row_count + other.row_count);
                            
                            // 自分のデータを追加
                            for i in 0..self.row_count {
                                let value = match self_col {
                                    Column::Int64(col) => col.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_default(),
                                    Column::Float64(col) => col.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_default(),
                                    Column::String(col) => col.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_default(),
                                    Column::Boolean(col) => col.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_default(),
                                };
                                combined_data.push(value);
                            }
                            
                            // 相手のデータを追加
                            for i in 0..other.row_count {
                                let value = match other_col {
                                    Column::Int64(col) => col.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_default(),
                                    Column::Float64(col) => col.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_default(),
                                    Column::String(col) => col.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_default(),
                                    Column::Boolean(col) => col.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_default(),
                                };
                                combined_data.push(value);
                            }
                            
                            result.add_column(col_name, Column::String(StringColumn::new(combined_data)))?;
                        }
                    }
                }
                // 異なる型の列は文字列として連結
                else {
                    let mut combined_data = Vec::with_capacity(self.row_count + other.row_count);
                    
                    // 自分のデータを追加
                    for i in 0..self.row_count {
                        let value = match self_col {
                            Column::Int64(col) => col.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_default(),
                            Column::Float64(col) => col.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_default(),
                            Column::String(col) => col.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_default(),
                            Column::Boolean(col) => col.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_default(),
                        };
                        combined_data.push(value);
                    }
                    
                    // 相手のデータを追加
                    for i in 0..other.row_count {
                        let value = match other_col {
                            Column::Int64(col) => col.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_default(),
                            Column::Float64(col) => col.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_default(),
                            Column::String(col) => col.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_default(),
                            Column::Boolean(col) => col.get(i).ok().flatten().map(|v| v.to_string()).unwrap_or_default(),
                        };
                        combined_data.push(value);
                    }
                    
                    result.add_column(col_name, Column::String(StringColumn::new(combined_data)))?;
                }
            }
            // 片方にのみ列が存在する場合
            else {
                let total_rows = self.row_count + other.row_count;
                
                if self_has_column {
                    let col_idx = self.column_indices[&col_name];
                    let column = &self.columns[col_idx];
                    
                    match column {
                        Column::Int64(int_col) => {
                            let mut combined_data = Vec::with_capacity(total_rows);
                            
                            // 自分のデータを追加
                            for i in 0..int_col.len() {
                                if let Ok(Some(val)) = int_col.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(0); // デフォルト値
                                }
                            }
                            
                            // 存在しない側のデータはデフォルト値で埋める
                            combined_data.resize(total_rows, 0);
                            
                            result.add_column(col_name, Column::Int64(Int64Column::new(combined_data)))?;
                        },
                        Column::Float64(float_col) => {
                            let mut combined_data = Vec::with_capacity(total_rows);
                            
                            // 自分のデータを追加
                            for i in 0..float_col.len() {
                                if let Ok(Some(val)) = float_col.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(0.0); // デフォルト値
                                }
                            }
                            
                            // 存在しない側のデータはデフォルト値で埋める
                            combined_data.resize(total_rows, 0.0);
                            
                            result.add_column(col_name, Column::Float64(Float64Column::new(combined_data)))?;
                        },
                        Column::String(str_col) => {
                            let mut combined_data = Vec::with_capacity(total_rows);
                            
                            // 自分のデータを追加
                            for i in 0..str_col.len() {
                                if let Ok(Some(val)) = str_col.get(i) {
                                    combined_data.push(val.to_string());
                                } else {
                                    combined_data.push(String::new()); // デフォルト値
                                }
                            }
                            
                            // 存在しない側のデータはデフォルト値で埋める
                            combined_data.resize(total_rows, String::new());
                            
                            result.add_column(col_name, Column::String(StringColumn::new(combined_data)))?;
                        },
                        Column::Boolean(bool_col) => {
                            let mut combined_data = Vec::with_capacity(total_rows);
                            
                            // 自分のデータを追加
                            for i in 0..bool_col.len() {
                                if let Ok(Some(val)) = bool_col.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(false); // デフォルト値
                                }
                            }
                            
                            // 存在しない側のデータはデフォルト値で埋める
                            combined_data.resize(total_rows, false);
                            
                            result.add_column(col_name, Column::Boolean(BooleanColumn::new(combined_data)))?;
                        },
                    }
                } else if other_has_column {
                    let col_idx = other.column_indices[&col_name];
                    let column = &other.columns[col_idx];
                    
                    match column {
                        Column::Int64(int_col) => {
                            let mut combined_data = Vec::with_capacity(total_rows);
                            
                            // 自分側（存在しない）はデフォルト値で埋める
                            combined_data.resize(self.row_count, 0);
                            
                            // 相手のデータを追加
                            for i in 0..int_col.len() {
                                if let Ok(Some(val)) = int_col.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(0); // デフォルト値
                                }
                            }
                            
                            result.add_column(col_name, Column::Int64(Int64Column::new(combined_data)))?;
                        },
                        Column::Float64(float_col) => {
                            let mut combined_data = Vec::with_capacity(total_rows);
                            
                            // 自分側（存在しない）はデフォルト値で埋める
                            combined_data.resize(self.row_count, 0.0);
                            
                            // 相手のデータを追加
                            for i in 0..float_col.len() {
                                if let Ok(Some(val)) = float_col.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(0.0); // デフォルト値
                                }
                            }
                            
                            result.add_column(col_name, Column::Float64(Float64Column::new(combined_data)))?;
                        },
                        Column::String(str_col) => {
                            let mut combined_data = Vec::with_capacity(total_rows);
                            
                            // 自分側（存在しない）はデフォルト値で埋める
                            combined_data.resize(self.row_count, String::new());
                            
                            // 相手のデータを追加
                            for i in 0..str_col.len() {
                                if let Ok(Some(val)) = str_col.get(i) {
                                    combined_data.push(val.to_string());
                                } else {
                                    combined_data.push(String::new()); // デフォルト値
                                }
                            }
                            
                            result.add_column(col_name, Column::String(StringColumn::new(combined_data)))?;
                        },
                        Column::Boolean(bool_col) => {
                            let mut combined_data = Vec::with_capacity(total_rows);
                            
                            // 自分側（存在しない）はデフォルト値で埋める
                            combined_data.resize(self.row_count, false);
                            
                            // 相手のデータを追加
                            for i in 0..bool_col.len() {
                                if let Ok(Some(val)) = bool_col.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(false); // デフォルト値
                                }
                            }
                            
                            result.add_column(col_name, Column::Boolean(BooleanColumn::new(combined_data)))?;
                        },
                    }
                }
            }
        }
        
        Ok(result)
    }
}
