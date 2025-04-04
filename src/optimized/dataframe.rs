use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::sync::Arc;

use crate::column::{Column, ColumnTrait, ColumnType, Int64Column};
use crate::error::{Error, Result};
use crate::index::{DataFrameIndex, IndexTrait, Index};
use crate::optimized::operations::JoinType;

/// 最適化されたDataFrame実装
/// 列指向ストレージを使用し、高速なデータ処理を実現
#[derive(Clone)]
pub struct OptimizedDataFrame {
    // 列データ
    columns: Vec<Column>,
    // 列名→インデックスのマッピング
    column_indices: HashMap<String, usize>,
    // 列の順序
    column_names: Vec<String>,
    // 行数
    row_count: usize,
    // インデックス (オプション)
    index: Option<DataFrameIndex<String>>,
}

/// 列に対するビュー（参照）を表す構造体
#[derive(Clone)]
pub struct ColumnView {
    column: Column,
}

impl Debug for OptimizedDataFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 最大表示行数
        const MAX_ROWS: usize = 10;
        
        if self.columns.is_empty() {
            return write!(f, "OptimizedDataFrame (0 rows x 0 columns)");
        }
        
        writeln!(f, "OptimizedDataFrame ({} rows x {} columns):", self.row_count, self.columns.len())?;
        
        // 列ヘッダーの表示
        write!(f, "{:<5} |", "idx")?;
        for name in &self.column_names {
            write!(f, " {:<15} |", name)?;
        }
        writeln!(f)?;
        
        // 区切り線
        write!(f, "{:-<5}-+", "")?;
        for _ in &self.column_names {
            write!(f, "-{:-<15}-+", "")?;
        }
        writeln!(f)?;
        
        // 最大MAX_ROWS行まで表示
        let display_rows = std::cmp::min(self.row_count, MAX_ROWS);
        for i in 0..display_rows {
            if let Some(ref idx) = self.index {
                let idx_value = match idx {
                    DataFrameIndex::Simple(ref simple_idx) => {
                        if i < simple_idx.len() {
                            simple_idx.get_value(i).map(|s| s.to_string()).unwrap_or_else(|| i.to_string())
                        } else {
                            i.to_string()
                        }
                    },
                    DataFrameIndex::Multi(_) => i.to_string()
                };
                write!(f, "{:<5} |", idx_value)?;
            } else {
                write!(f, "{:<5} |", i)?;
            }
            
            for col_idx in 0..self.columns.len() {
                let col = &self.columns[col_idx];
                let value = match col {
                    Column::Int64(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            format!("{}", val)
                        } else {
                            "NULL".to_string()
                        }
                    },
                    Column::Float64(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            format!("{:.3}", val)
                        } else {
                            "NULL".to_string()
                        }
                    },
                    Column::String(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            format!("\"{}\"", val)
                        } else {
                            "NULL".to_string()
                        }
                    },
                    Column::Boolean(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            format!("{}", val)
                        } else {
                            "NULL".to_string()
                        }
                    },
                };
                write!(f, " {:<15} |", value)?;
            }
            writeln!(f)?;
        }
        
        // 省略表示
        if self.row_count > MAX_ROWS {
            writeln!(f, "... ({} more rows)", self.row_count - MAX_ROWS)?;
        }
        
        Ok(())
    }
}

impl OptimizedDataFrame {
    /// 新しい空のDataFrameを作成
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            column_indices: HashMap::new(),
            column_names: Vec::new(),
            row_count: 0,
            index: None,
        }
    }
    
    /// 列を追加
    pub fn add_column<C: Into<Column>>(&mut self, name: impl Into<String>, column: C) -> Result<()> {
        let name = name.into();
        let column = column.into();
        
        // 列名の重複チェック
        if self.column_indices.contains_key(&name) {
            return Err(Error::DuplicateColumnName(name));
        }
        
        // 行数の整合性チェック
        let column_len = column.len();
        if !self.columns.is_empty() && column_len != self.row_count {
            return Err(Error::InconsistentRowCount {
                expected: self.row_count,
                found: column_len,
            });
        }
        
        // 列の追加
        let column_idx = self.columns.len();
        self.columns.push(column);
        self.column_indices.insert(name.clone(), column_idx);
        self.column_names.push(name);
        
        // 最初の列の場合は行数を設定
        if self.row_count == 0 {
            self.row_count = column_len;
        }
        
        Ok(())
    }
    
    /// 列の参照を取得
    pub fn column(&self, name: &str) -> Result<ColumnView> {
        let column_idx = self.column_indices.get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
        
        let column = self.columns[*column_idx].clone();
        Ok(ColumnView { column })
    }
    
    /// 列の型を取得
    pub fn column_type(&self, name: &str) -> Result<ColumnType> {
        let column_idx = self.column_indices.get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
        
        Ok(self.columns[*column_idx].column_type())
    }
    
    /// 列名のリストを取得
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }
    
    /// 行数を取得
    pub fn row_count(&self) -> usize {
        self.row_count
    }
    
    /// 列数を取得
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }
    
    /// インデックスを設定
    pub fn set_index(&mut self, name: &str) -> Result<()> {
        // 列の存在チェック
        let column_idx = self.column_indices.get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
        
        let column = &self.columns[*column_idx];
        
        // 文字列列からインデックスを作成
        if let Column::String(string_col) = column {
            let mut index_values = Vec::new();
            let mut index_map = HashMap::new();
            
            for i in 0..string_col.len() {
                if let Ok(Some(value)) = string_col.get(i) {
                    let value_string = value.to_string();
                    index_values.push(value_string.clone());
                    index_map.insert(value_string, i);
                } else {
                    let value_string = i.to_string();
                    index_values.push(value_string.clone());
                    index_map.insert(value_string, i);
                }
            }
            
            let index = Index::with_name(index_values, Some(name.to_string()))?;
            self.index = Some(DataFrameIndex::from_simple(index));
            return Ok(());
        }
        
        // 整数列からインデックスを作成
        if let Column::Int64(int_col) = column {
            let mut index_values = Vec::new();
            let mut index_map = HashMap::new();
            
            for i in 0..int_col.len() {
                if let Ok(Some(value)) = int_col.get(i) {
                    let value_string = value.to_string();
                    index_values.push(value_string.clone());
                    index_map.insert(value_string, i);
                } else {
                    let value_string = i.to_string();
                    index_values.push(value_string.clone());
                    index_map.insert(value_string, i);
                }
            }
            
            let index = Index::with_name(index_values, Some(name.to_string()))?;
            self.index = Some(DataFrameIndex::from_simple(index));
            return Ok(());
        }
        
        Err(Error::Operation(format!(
            "列 '{}' はインデックスとして使用できる型ではありません", name
        )))
    }
    
    /// 整数インデックスを使用して行を取得（新しいDataFrameとして）
    pub fn get_row(&self, row_idx: usize) -> Result<Self> {
        if row_idx >= self.row_count {
            return Err(Error::IndexOutOfBounds {
                index: row_idx,
                size: self.row_count,
            });
        }
        
        let mut result = Self::new();
        
        for (i, name) in self.column_names.iter().enumerate() {
            let column = &self.columns[i];
            
            let new_column = match column {
                Column::Int64(col) => {
                    let value = col.get(row_idx)?.unwrap_or_default();
                    Column::Int64(Int64Column::new(vec![value]))
                },
                Column::Float64(col) => {
                    let value = col.get(row_idx)?.unwrap_or_default();
                    Column::Float64(crate::column::Float64Column::new(vec![value]))
                },
                Column::String(col) => {
                    let value = col.get(row_idx)?.unwrap_or_default().to_string();
                    Column::String(crate::column::StringColumn::new(vec![value]))
                },
                Column::Boolean(col) => {
                    let value = col.get(row_idx)?.unwrap_or_default();
                    Column::Boolean(crate::column::BooleanColumn::new(vec![value]))
                },
            };
            
            result.add_column(name.clone(), new_column)?;
        }
        
        Ok(result)
    }
    
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
                        Column::Float64(crate::column::Float64Column::new(filtered_data))
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
                        Column::String(crate::column::StringColumn::new(filtered_data))
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
                        Column::Boolean(crate::column::BooleanColumn::new(filtered_data))
                    },
                };
                
                result.add_column(name.clone(), filtered_column)?;
            }
            
            // 新しいインデックスの作成
            if let Some(ref idx) = self.index {
                let mut new_index_values = Vec::with_capacity(indices.len());
                let mut new_index_map = HashMap::new();
                
                for (new_idx, &old_idx) in indices.iter().enumerate() {
                    if old_idx < idx.len() {
                        // インデックス値を取得する代替方法
                        if let DataFrameIndex::Simple(ref simple_idx) = idx {
                            if old_idx < simple_idx.len() {
                                let value = simple_idx.get_value(old_idx).map(|s| s.to_string()).unwrap_or_else(|| old_idx.to_string());
                                new_index_values.push(value.clone());
                                new_index_map.insert(value, new_idx);
                            } else {
                                let value = old_idx.to_string();
                                new_index_values.push(value.clone());
                                new_index_map.insert(value, new_idx);
                            }
                        } else {
                            let value = old_idx.to_string();
                            new_index_values.push(value.clone());
                            new_index_map.insert(value, new_idx);
                        }
                    } else {
                        let value = old_idx.to_string();
                        new_index_values.push(value.clone());
                        new_index_map.insert(value, new_idx);
                    }
                }
                
                // 新しいインデックスを作成
                let name_opt = match idx {
                    DataFrameIndex::Simple(ref simple_idx) => simple_idx.name().map(|s| s.to_string()),
                    DataFrameIndex::Multi(ref multi_idx) => multi_idx.names().first().cloned().flatten(),
                };
                
                let new_idx = Index::with_name(new_index_values, name_opt)?;
                result.index = Some(DataFrameIndex::from_simple(new_idx));
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
    
    /// マッピング関数を適用（並列処理対応）
    pub fn par_apply<F>(&self, func: F) -> Result<Self>
    where
        F: Fn(&ColumnView) -> Result<Column> + Sync + Send,
    {
        use rayon::prelude::*;
        
        let column_views: Vec<_> = self.column_names.iter()
            .map(|name| self.column(name))
            .collect::<Result<_>>()?;
        
        // 並列処理
        let new_columns: Result<Vec<_>> = column_views.par_iter()
            .map(|view| func(view))
            .collect();
        
        // 結果を新しいDataFrameに格納
        let new_columns = new_columns?;
        let mut result = Self::new();
        
        for (name, column) in self.column_names.iter().zip(new_columns) {
            result.add_column(name.clone(), column)?;
        }
        
        // インデックスのコピー
        if let Some(ref idx) = self.index {
            result.index = Some(idx.clone());
        }
        
        Ok(result)
    }
    
    /// 行のフィルタリング実行（データサイズに応じて直列/並列処理を自動選択）
    pub fn par_filter(&self, condition_column: &str) -> Result<Self> {
        use rayon::prelude::*;
        
        // 最適並列化のための閾値（これより小さいデータサイズでは直列処理が有利）
        const PARALLEL_THRESHOLD: usize = 100_000;
        
        // 条件列の取得
        let column_idx = self.column_indices.get(condition_column)
            .ok_or_else(|| Error::ColumnNotFound(condition_column.to_string()))?;
        
        let condition = &self.columns[*column_idx];
        
        // 条件列がブール型であることを確認
        if let Column::Boolean(bool_col) = condition {
            let row_count = bool_col.len();
            
            // データサイズに基づいて直列/並列処理を選択
            let indices: Vec<usize> = if row_count < PARALLEL_THRESHOLD {
                // 直列処理（小規模データ）
                (0..row_count)
                    .filter_map(|i| {
                        if let Ok(Some(true)) = bool_col.get(i) {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                // 並列処理（大規模データ）
                // チャンクサイズを最適化して並列化オーバーヘッドを削減
                let chunk_size = (row_count / rayon::current_num_threads()).max(1000);
                
                // まずレンジを配列に変換してからチャンク処理
                (0..row_count).collect::<Vec<_>>()
                    .par_chunks(chunk_size)
                    .flat_map(|chunk| {
                        chunk.iter().filter_map(|&i| {
                            if let Ok(Some(true)) = bool_col.get(i) {
                                Some(i)
                            } else {
                                None
                            }
                        }).collect::<Vec<_>>()
                    })
                    .collect()
            };
            
            if indices.is_empty() {
                // 空のデータフレームを返す
                let mut result = Self::new();
                for name in &self.column_names {
                    let col_idx = self.column_indices[name];
                    let empty_col = match &self.columns[col_idx] {
                        Column::Int64(_) => Column::Int64(Int64Column::new(Vec::new())),
                        Column::Float64(_) => Column::Float64(crate::column::Float64Column::new(Vec::new())),
                        Column::String(_) => Column::String(crate::column::StringColumn::new(Vec::new())),
                        Column::Boolean(_) => Column::Boolean(crate::column::BooleanColumn::new(Vec::new())),
                    };
                    result.add_column(name.clone(), empty_col)?;
                }
                return Ok(result);
            }
            
            // 新しいDataFrameを作成
            let mut result = Self::new();
            
            // 結果の列を格納するベクトルを事前に確保
            let mut result_columns = Vec::with_capacity(self.column_names.len());
            
            // データサイズに基づいて列の処理方法を選択
            if indices.len() < PARALLEL_THRESHOLD || self.column_names.len() < 4 {
                // 直列処理（小規模データまたは列数が少ない場合）
                for name in &self.column_names {
                    let i = self.column_indices[name];
                    let column = &self.columns[i];
                    
                    let filtered_column = match column {
                        Column::Int64(col) => {
                            let filtered_data: Vec<i64> = indices.iter()
                                .map(|&idx| {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        val
                                    } else {
                                        0 // デフォルト値
                                    }
                                })
                                .collect();
                            Column::Int64(Int64Column::new(filtered_data))
                        },
                        Column::Float64(col) => {
                            let filtered_data: Vec<f64> = indices.iter()
                                .map(|&idx| {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        val
                                    } else {
                                        0.0 // デフォルト値
                                    }
                                })
                                .collect();
                            Column::Float64(crate::column::Float64Column::new(filtered_data))
                        },
                        Column::String(col) => {
                            let filtered_data: Vec<String> = indices.iter()
                                .map(|&idx| {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        val.to_string()
                                    } else {
                                        String::new() // デフォルト値
                                    }
                                })
                                .collect();
                            Column::String(crate::column::StringColumn::new(filtered_data))
                        },
                        Column::Boolean(col) => {
                            let filtered_data: Vec<bool> = indices.iter()
                                .map(|&idx| {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        val
                                    } else {
                                        false // デフォルト値
                                    }
                                })
                                .collect();
                            Column::Boolean(crate::column::BooleanColumn::new(filtered_data))
                        },
                    };
                    
                    result_columns.push((name.clone(), filtered_column));
                }
            } else {
                // 大規模データの並列処理
                // 各列に対して並列処理を行う（列レベルの粗粒度並列化）
                result_columns = self.column_names.par_iter()
                    .map(|name| {
                        let i = self.column_indices[name];
                        let column = &self.columns[i];
                        
                        let indices_len = indices.len();
                        let filtered_column = match column {
                            Column::Int64(col) => {
                                // 大きなインデックスリストは分割して処理
                                let chunk_size = (indices_len / 8).max(1000);
                                let mut filtered_data = Vec::with_capacity(indices_len);
                                
                                // chunks を使用してすべての要素を確実に処理
                                for chunk in indices.chunks(chunk_size) {
                                    let chunk_data: Vec<i64> = chunk.iter()
                                        .map(|&idx| {
                                            if let Ok(Some(val)) = col.get(idx) {
                                                val
                                            } else {
                                                0 // デフォルト値
                                            }
                                        })
                                        .collect();
                                    filtered_data.extend(chunk_data);
                                }
                                
                                Column::Int64(Int64Column::new(filtered_data))
                            },
                            Column::Float64(col) => {
                                // 大きなインデックスリストは分割して処理
                                let chunk_size = (indices_len / 8).max(1000);
                                let mut filtered_data = Vec::with_capacity(indices_len);
                                
                                // chunksを使用してすべての要素を確実に処理
                                for chunk in indices.chunks(chunk_size) {
                                    let chunk_data: Vec<f64> = chunk.iter()
                                        .map(|&idx| {
                                            if let Ok(Some(val)) = col.get(idx) {
                                                val
                                            } else {
                                                0.0 // デフォルト値
                                            }
                                        })
                                        .collect();
                                    filtered_data.extend(chunk_data);
                                }
                                
                                Column::Float64(crate::column::Float64Column::new(filtered_data))
                            },
                            Column::String(col) => {
                                // 文字列処理は特に重いので、より細かいチャンク処理
                                let chunk_size = (indices_len / 16).max(500);
                                let mut filtered_data = Vec::with_capacity(indices_len);
                                
                                // chunksを使用してすべての要素を確実に処理
                                for chunk in indices.chunks(chunk_size) {
                                    let chunk_data: Vec<String> = chunk.iter()
                                        .map(|&idx| {
                                            if let Ok(Some(val)) = col.get(idx) {
                                                val.to_string()
                                            } else {
                                                String::new() // デフォルト値
                                            }
                                        })
                                        .collect();
                                    filtered_data.extend(chunk_data);
                                }
                                
                                Column::String(crate::column::StringColumn::new(filtered_data))
                            },
                            Column::Boolean(col) => {
                                let filtered_data: Vec<bool> = indices.iter()
                                    .map(|&idx| {
                                        if let Ok(Some(val)) = col.get(idx) {
                                            val
                                        } else {
                                            false // デフォルト値
                                        }
                                    })
                                    .collect();
                                Column::Boolean(crate::column::BooleanColumn::new(filtered_data))
                            },
                        };
                        
                        (name.clone(), filtered_column)
                    })
                    .collect();
            }
            
            // 結果をデータフレームに追加
            for (name, column) in result_columns {
                result.add_column(name, column)?;
            }
            
            // インデックスのコピー
            if let Some(ref idx) = self.index {
                result.index = Some(idx.clone());
            }
            
            Ok(result)
        } else {
            Err(Error::OperationFailed(format!(
                "列 '{}' はブール型ではありません", condition_column
            )))
        }
    }
    
    /// グループ化操作を並列で実行（データサイズに応じて最適化）
    pub fn par_groupby(&self, group_by_columns: &[&str]) -> Result<HashMap<String, Self>> {
        use rayon::prelude::*;
        use std::collections::hash_map::Entry;
        use std::sync::{Arc, Mutex};
        
        // データサイズに基づく最適化閾値
        const PARALLEL_THRESHOLD: usize = 50_000;
        
        // グループ化キーのカラムインデックスを取得
        let mut group_col_indices = Vec::with_capacity(group_by_columns.len());
        for &col_name in group_by_columns {
            let col_idx = self.column_indices.get(col_name)
                .ok_or_else(|| Error::ColumnNotFound(col_name.to_string()))?;
            group_col_indices.push(*col_idx);
        }
        
        // グループキーを生成し、各行のインデックスをグループ化
        let groups: HashMap<String, Vec<usize>> = if self.row_count < PARALLEL_THRESHOLD {
            // 小規模データでは直列処理の方が効率的
            let mut groups = HashMap::new();
            
            for row_idx in 0..self.row_count {
                // この行のグループキーを生成
                let mut key_parts = Vec::with_capacity(group_col_indices.len());
                
                for &col_idx in &group_col_indices {
                    let column = &self.columns[col_idx];
                    let part = match column {
                        Column::Int64(col) => {
                            if let Ok(Some(val)) = col.get(row_idx) {
                                val.to_string()
                            } else {
                                "NA".to_string()
                            }
                        },
                        Column::Float64(col) => {
                            if let Ok(Some(val)) = col.get(row_idx) {
                                val.to_string()
                            } else {
                                "NA".to_string()
                            }
                        },
                        Column::String(col) => {
                            if let Ok(Some(val)) = col.get(row_idx) {
                                val.to_string()
                            } else {
                                "NA".to_string()
                            }
                        },
                        Column::Boolean(col) => {
                            if let Ok(Some(val)) = col.get(row_idx) {
                                val.to_string()
                            } else {
                                "NA".to_string()
                            }
                        },
                    };
                    key_parts.push(part);
                }
                
                let group_key = key_parts.join("_");
                
                match groups.entry(group_key) {
                    Entry::Vacant(e) => { e.insert(vec![row_idx]); },
                    Entry::Occupied(mut e) => { e.get_mut().push(row_idx); }
                }
            }
            
            groups
        } else {
            // 大規模データでは並列処理+ロックフリーアプローチ
            // 1. 並列でローカルなグループマップを作成
            // 2. それらをマージ
            let chunk_size = (self.row_count / rayon::current_num_threads()).max(1000);
            
            // ステップ1: 並列でローカルな中間グループマップを作成
            let local_maps: Vec<HashMap<String, Vec<usize>>> = 
                (0..self.row_count).collect::<Vec<_>>()
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut local_groups = HashMap::new();
                    
                    for &row_idx in chunk {
                        // この行のグループキーを生成
                        let mut key_parts = Vec::with_capacity(group_col_indices.len());
                        
                        for &col_idx in &group_col_indices {
                            let column = &self.columns[col_idx];
                            let part = match column {
                                Column::Int64(col) => {
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NA".to_string()
                                    }
                                },
                                Column::Float64(col) => {
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NA".to_string()
                                    }
                                },
                                Column::String(col) => {
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NA".to_string()
                                    }
                                },
                                Column::Boolean(col) => {
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NA".to_string()
                                    }
                                },
                            };
                            key_parts.push(part);
                        }
                        
                        let group_key = key_parts.join("_");
                        
                        match local_groups.entry(group_key) {
                            Entry::Vacant(e) => { e.insert(vec![row_idx]); },
                            Entry::Occupied(mut e) => { e.get_mut().push(row_idx); }
                        }
                    }
                    
                    local_groups
                })
                .collect();
            
            // ステップ2: 中間マップをマージ
            let mut merged_groups = HashMap::new();
            for local_map in local_maps {
                for (key, indices) in local_map {
                    match merged_groups.entry(key) {
                        Entry::Vacant(e) => { e.insert(indices); },
                        Entry::Occupied(mut e) => { e.get_mut().extend(indices); }
                    }
                }
            }
            
            merged_groups
        };
        
        // 各グループに対してDataFrameを効率的に作成
        let result = if groups.len() < 100 || self.row_count < PARALLEL_THRESHOLD {
            // グループ数が少ない場合や小規模データでは直列処理
            let mut result = HashMap::with_capacity(groups.len());
            for (key, indices) in groups {
                let group_df = self.filter_by_indices(&indices)?;
                result.insert(key, group_df);
            }
            result
        } else {
            // 大規模データでのグループ処理は並列化
            // 各グループを並列処理し、スレッドセーフに結果を集約
            let result_mutex = Arc::new(Mutex::new(HashMap::with_capacity(groups.len())));
            
            // チャンクサイズを調整して、オーバーヘッドを最小化
            let chunk_size = (groups.len() / rayon::current_num_threads()).max(10);
            
            // グループのリストを作成し、チャンクに分割して並列処理
            let group_items: Vec<(String, Vec<usize>)> = groups.into_iter().collect();
            
            group_items.par_chunks(chunk_size)
                .for_each(|chunk| {
                    // 各チャンクの処理結果を一時保存
                    let mut local_results = HashMap::new();
                    
                    for (key, indices) in chunk {
                        if let Ok(group_df) = self.filter_by_indices(indices) {
                            local_results.insert(key.clone(), group_df);
                        }
                    }
                    
                    // 結果をメインのHashMapにマージ
                    if let Ok(mut result_map) = result_mutex.lock() {
                        for (key, df) in local_results {
                            result_map.insert(key, df);
                        }
                    }
                });
            
            // 最終結果を取得
            match Arc::try_unwrap(result_mutex) {
                Ok(mutex) => mutex.into_inner().unwrap_or_default(),
                Err(_) => HashMap::new(), // アークの解除に失敗した場合
            }
        };
        
        Ok(result)
    }
    
    /// 指定された行インデックスでフィルタリング（内部ヘルパー）
    fn filter_by_indices(&self, indices: &[usize]) -> Result<Self> {
        use rayon::prelude::*;
        
        let mut result = Self::new();
        
        // 各列を並列でフィルタリング
        let column_results: Result<Vec<(String, Column)>> = self.column_names.par_iter()
            .map(|name| {
                let i = self.column_indices[name];
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
                        Column::Float64(crate::column::Float64Column::new(filtered_data))
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
                        Column::String(crate::column::StringColumn::new(filtered_data))
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
                        Column::Boolean(crate::column::BooleanColumn::new(filtered_data))
                    },
                };
                
                Ok((name.clone(), filtered_column))
            })
            .collect();
        
        // 結果をデータフレームに追加
        for (name, column) in column_results? {
            result.add_column(name, column)?;
        }
        
        // インデックスのコピー
        if let Some(ref idx) = self.index {
            result.index = Some(idx.clone());
        }
        
        Ok(result)
    }
    
    /// 内部結合
    pub fn inner_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        self.join_impl(other, left_on, right_on, JoinType::Inner)
    }
    
    /// 左結合
    pub fn left_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        self.join_impl(other, left_on, right_on, JoinType::Left)
    }
    
    /// 右結合
    pub fn right_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        self.join_impl(other, left_on, right_on, JoinType::Right)
    }
    
    /// 外部結合
    pub fn outer_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        self.join_impl(other, left_on, right_on, JoinType::Outer)
    }
    
    // 結合の実装（内部メソッド）
    fn join_impl(&self, other: &Self, left_on: &str, right_on: &str, join_type: JoinType) -> Result<Self> {
        // 結合キー列の取得
        let left_col_idx = self.column_indices.get(left_on)
            .ok_or_else(|| Error::ColumnNotFound(left_on.to_string()))?;
        
        let right_col_idx = other.column_indices.get(right_on)
            .ok_or_else(|| Error::ColumnNotFound(right_on.to_string()))?;
        
        let left_col = &self.columns[*left_col_idx];
        let right_col = &other.columns[*right_col_idx];
        
        // 両方の列が同じ型であることを確認
        if left_col.column_type() != right_col.column_type() {
            return Err(Error::ColumnTypeMismatch {
                name: format!("{} と {}", left_on, right_on),
                expected: left_col.column_type(),
                found: right_col.column_type(),
            });
        }
        
        // 結合キーのマッピングを構築
        let mut right_key_to_indices: HashMap<String, Vec<usize>> = HashMap::new();
        
        match right_col {
            Column::Int64(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        right_key_to_indices.entry(key).or_default().push(i);
                    }
                }
            },
            Column::Float64(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        right_key_to_indices.entry(key).or_default().push(i);
                    }
                }
            },
            Column::String(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        right_key_to_indices.entry(key).or_default().push(i);
                    }
                }
            },
            Column::Boolean(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        right_key_to_indices.entry(key).or_default().push(i);
                    }
                }
            },
        }
        
        // 結合行の作成
        let mut result = Self::new();
        let mut join_indices: Vec<(Option<usize>, Option<usize>)> = Vec::new();
        
        // 左側のDataFrameを処理
        match left_col {
            Column::Int64(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        if let Some(right_indices) = right_key_to_indices.get(&key) {
                            // マッチがある場合、各右側インデックスと結合
                            for &right_idx in right_indices {
                                join_indices.push((Some(i), Some(right_idx)));
                            }
                        } else if join_type == JoinType::Left || join_type == JoinType::Outer {
                            // 左結合または外部結合では、マッチがなくても左側の行を含める
                            join_indices.push((Some(i), None));
                        }
                    }
                }
            },
            Column::Float64(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        if let Some(right_indices) = right_key_to_indices.get(&key) {
                            for &right_idx in right_indices {
                                join_indices.push((Some(i), Some(right_idx)));
                            }
                        } else if join_type == JoinType::Left || join_type == JoinType::Outer {
                            join_indices.push((Some(i), None));
                        }
                    }
                }
            },
            Column::String(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        if let Some(right_indices) = right_key_to_indices.get(&key) {
                            for &right_idx in right_indices {
                                join_indices.push((Some(i), Some(right_idx)));
                            }
                        } else if join_type == JoinType::Left || join_type == JoinType::Outer {
                            join_indices.push((Some(i), None));
                        }
                    }
                }
            },
            Column::Boolean(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        if let Some(right_indices) = right_key_to_indices.get(&key) {
                            for &right_idx in right_indices {
                                join_indices.push((Some(i), Some(right_idx)));
                            }
                        } else if join_type == JoinType::Left || join_type == JoinType::Outer {
                            join_indices.push((Some(i), None));
                        }
                    }
                }
            },
        }
        
        // 右結合または外部結合の場合、右側で未マッチの行を追加
        if join_type == JoinType::Right || join_type == JoinType::Outer {
            let mut right_matched = vec![false; other.row_count];
            for (_, right_idx) in &join_indices {
                if let Some(idx) = right_idx {
                    right_matched[*idx] = true;
                }
            }
            
            for (i, matched) in right_matched.iter().enumerate() {
                if !matched {
                    join_indices.push((None, Some(i)));
                }
            }
        }
        
        // 結果が空の場合、早期リターン
        if join_indices.is_empty() {
            // 空のデータフレームを返す（列だけ設定）
            let mut result = Self::new();
            for name in &self.column_names {
                if name != left_on {
                    let col_idx = self.column_indices[name];
                    let col_type = self.columns[col_idx].column_type();
                    
                    // 型に応じた空の列を作成
                    let empty_col = match col_type {
                        ColumnType::Int64 => Column::Int64(Int64Column::new(Vec::new())),
                        ColumnType::Float64 => Column::Float64(crate::column::Float64Column::new(Vec::new())),
                        ColumnType::String => Column::String(crate::column::StringColumn::new(Vec::new())),
                        ColumnType::Boolean => Column::Boolean(crate::column::BooleanColumn::new(Vec::new())),
                    };
                    
                    result.add_column(name.clone(), empty_col)?;
                }
            }
            
            for name in &other.column_names {
                if name != right_on {
                    let suffix = "_right";
                    let new_name = if self.column_indices.contains_key(name) {
                        format!("{}{}", name, suffix)
                    } else {
                        name.clone()
                    };
                    
                    let col_idx = other.column_indices[name];
                    let col_type = other.columns[col_idx].column_type();
                    
                    // 型に応じた空の列を作成
                    let empty_col = match col_type {
                        ColumnType::Int64 => Column::Int64(Int64Column::new(Vec::new())),
                        ColumnType::Float64 => Column::Float64(crate::column::Float64Column::new(Vec::new())),
                        ColumnType::String => Column::String(crate::column::StringColumn::new(Vec::new())),
                        ColumnType::Boolean => Column::Boolean(crate::column::BooleanColumn::new(Vec::new())),
                    };
                    
                    result.add_column(new_name, empty_col)?;
                }
            }
            
            return Ok(result);
        }
        
        // 結果用の各列データを準備
        let row_count = join_indices.len();
        
        // 左側の列を追加
        for name in &self.column_names {
            if name != left_on { // 結合キー列は1つだけ追加する
                let col_idx = self.column_indices[name];
                let col = &self.columns[col_idx];
                
                let joined_col = match col {
                    Column::Int64(int_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (left_idx, _) in &join_indices {
                            if let Some(idx) = left_idx {
                                if let Ok(Some(val)) = int_col.get(*idx) {
                                    data.push(val);
                                } else {
                                    data.push(0); // デフォルト値
                                }
                            } else {
                                data.push(0); // デフォルト値（右側のみ）
                            }
                        }
                        Column::Int64(Int64Column::new(data))
                    },
                    Column::Float64(float_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (left_idx, _) in &join_indices {
                            if let Some(idx) = left_idx {
                                if let Ok(Some(val)) = float_col.get(*idx) {
                                    data.push(val);
                                } else {
                                    data.push(0.0); // デフォルト値
                                }
                            } else {
                                data.push(0.0); // デフォルト値（右側のみ）
                            }
                        }
                        Column::Float64(crate::column::Float64Column::new(data))
                    },
                    Column::String(str_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (left_idx, _) in &join_indices {
                            if let Some(idx) = left_idx {
                                if let Ok(Some(val)) = str_col.get(*idx) {
                                    data.push(val.to_string());
                                } else {
                                    data.push(String::new()); // デフォルト値
                                }
                            } else {
                                data.push(String::new()); // デフォルト値（右側のみ）
                            }
                        }
                        Column::String(crate::column::StringColumn::new(data))
                    },
                    Column::Boolean(bool_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (left_idx, _) in &join_indices {
                            if let Some(idx) = left_idx {
                                if let Ok(Some(val)) = bool_col.get(*idx) {
                                    data.push(val);
                                } else {
                                    data.push(false); // デフォルト値
                                }
                            } else {
                                data.push(false); // デフォルト値（右側のみ）
                            }
                        }
                        Column::Boolean(crate::column::BooleanColumn::new(data))
                    },
                };
                
                result.add_column(name.clone(), joined_col)?;
            }
        }
        
        // 結合キー列を追加（左側から）
        let left_key_col = &self.columns[*left_col_idx];
        let joined_key_col = match left_key_col {
            Column::Int64(int_col) => {
                let mut data = Vec::with_capacity(row_count);
                for (left_idx, right_idx) in &join_indices {
                    if let Some(idx) = left_idx {
                        if let Ok(Some(val)) = int_col.get(*idx) {
                            data.push(val);
                        } else {
                            data.push(0); // デフォルト値
                        }
                    } else if let Some(idx) = right_idx {
                        // 右側のキー値を使用
                        if let Column::Int64(right_int_col) = &other.columns[*right_col_idx] {
                            if let Ok(Some(val)) = right_int_col.get(*idx) {
                                data.push(val);
                            } else {
                                data.push(0); // デフォルト値
                            }
                        } else {
                            data.push(0); // デフォルト値
                        }
                    } else {
                        data.push(0); // デフォルト値
                    }
                }
                Column::Int64(Int64Column::new(data))
            },
            Column::Float64(float_col) => {
                let mut data = Vec::with_capacity(row_count);
                for (left_idx, right_idx) in &join_indices {
                    if let Some(idx) = left_idx {
                        if let Ok(Some(val)) = float_col.get(*idx) {
                            data.push(val);
                        } else {
                            data.push(0.0); // デフォルト値
                        }
                    } else if let Some(idx) = right_idx {
                        // 右側のキー値を使用
                        if let Column::Float64(right_float_col) = &other.columns[*right_col_idx] {
                            if let Ok(Some(val)) = right_float_col.get(*idx) {
                                data.push(val);
                            } else {
                                data.push(0.0); // デフォルト値
                            }
                        } else {
                            data.push(0.0); // デフォルト値
                        }
                    } else {
                        data.push(0.0); // デフォルト値
                    }
                }
                Column::Float64(crate::column::Float64Column::new(data))
            },
            Column::String(str_col) => {
                let mut data = Vec::with_capacity(row_count);
                for (left_idx, right_idx) in &join_indices {
                    if let Some(idx) = left_idx {
                        if let Ok(Some(val)) = str_col.get(*idx) {
                            data.push(val.to_string());
                        } else {
                            data.push(String::new()); // デフォルト値
                        }
                    } else if let Some(idx) = right_idx {
                        // 右側のキー値を使用
                        if let Column::String(right_str_col) = &other.columns[*right_col_idx] {
                            if let Ok(Some(val)) = right_str_col.get(*idx) {
                                data.push(val.to_string());
                            } else {
                                data.push(String::new()); // デフォルト値
                            }
                        } else {
                            data.push(String::new()); // デフォルト値
                        }
                    } else {
                        data.push(String::new()); // デフォルト値
                    }
                }
                Column::String(crate::column::StringColumn::new(data))
            },
            Column::Boolean(bool_col) => {
                let mut data = Vec::with_capacity(row_count);
                for (left_idx, right_idx) in &join_indices {
                    if let Some(idx) = left_idx {
                        if let Ok(Some(val)) = bool_col.get(*idx) {
                            data.push(val);
                        } else {
                            data.push(false); // デフォルト値
                        }
                    } else if let Some(idx) = right_idx {
                        // 右側のキー値を使用
                        if let Column::Boolean(right_bool_col) = &other.columns[*right_col_idx] {
                            if let Ok(Some(val)) = right_bool_col.get(*idx) {
                                data.push(val);
                            } else {
                                data.push(false); // デフォルト値
                            }
                        } else {
                            data.push(false); // デフォルト値
                        }
                    } else {
                        data.push(false); // デフォルト値
                    }
                }
                Column::Boolean(crate::column::BooleanColumn::new(data))
            },
        };
        
        result.add_column(left_on.to_string(), joined_key_col)?;
        
        // 右側の列を追加（結合キー以外）
        for name in &other.column_names {
            if name != right_on {
                let suffix = "_right";
                let new_name = if result.column_indices.contains_key(name) {
                    format!("{}{}", name, suffix)
                } else {
                    name.clone()
                };
                
                let col_idx = other.column_indices[name];
                let col = &other.columns[col_idx];
                
                let joined_col = match col {
                    Column::Int64(int_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (_, right_idx) in &join_indices {
                            if let Some(idx) = right_idx {
                                if let Ok(Some(val)) = int_col.get(*idx) {
                                    data.push(val);
                                } else {
                                    data.push(0); // デフォルト値
                                }
                            } else {
                                data.push(0); // デフォルト値（左側のみ）
                            }
                        }
                        Column::Int64(Int64Column::new(data))
                    },
                    Column::Float64(float_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (_, right_idx) in &join_indices {
                            if let Some(idx) = right_idx {
                                if let Ok(Some(val)) = float_col.get(*idx) {
                                    data.push(val);
                                } else {
                                    data.push(0.0); // デフォルト値
                                }
                            } else {
                                data.push(0.0); // デフォルト値（左側のみ）
                            }
                        }
                        Column::Float64(crate::column::Float64Column::new(data))
                    },
                    Column::String(str_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (_, right_idx) in &join_indices {
                            if let Some(idx) = right_idx {
                                if let Ok(Some(val)) = str_col.get(*idx) {
                                    data.push(val.to_string());
                                } else {
                                    data.push(String::new()); // デフォルト値
                                }
                            } else {
                                data.push(String::new()); // デフォルト値（左側のみ）
                            }
                        }
                        Column::String(crate::column::StringColumn::new(data))
                    },
                    Column::Boolean(bool_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (_, right_idx) in &join_indices {
                            if let Some(idx) = right_idx {
                                if let Ok(Some(val)) = bool_col.get(*idx) {
                                    data.push(val);
                                } else {
                                    data.push(false); // デフォルト値
                                }
                            } else {
                                data.push(false); // デフォルト値（左側のみ）
                            }
                        }
                        Column::Boolean(crate::column::BooleanColumn::new(data))
                    },
                };
                
                result.add_column(new_name, joined_col)?;
            }
        }
        
        Ok(result)
    }
}

/// ビューされた列に対する操作
impl ColumnView {
    /// 列の型を取得
    pub fn column_type(&self) -> ColumnType {
        self.column.column_type()
    }
    
    /// 列の長さを取得
    pub fn len(&self) -> usize {
        self.column.len()
    }
    
    /// 列が空かどうかを確認
    pub fn is_empty(&self) -> bool {
        self.column.is_empty()
    }
    
    /// 整数列としてアクセス
    pub fn as_int64(&self) -> Option<&crate::column::Int64Column> {
        if let Column::Int64(ref col) = self.column {
            Some(col)
        } else {
            None
        }
    }
    
    /// 浮動小数点列としてアクセス
    pub fn as_float64(&self) -> Option<&crate::column::Float64Column> {
        if let Column::Float64(ref col) = self.column {
            Some(col)
        } else {
            None
        }
    }
    
    /// 文字列列としてアクセス
    pub fn as_string(&self) -> Option<&crate::column::StringColumn> {
        if let Column::String(ref col) = self.column {
            Some(col)
        } else {
            None
        }
    }
    
    /// ブール列としてアクセス
    pub fn as_boolean(&self) -> Option<&crate::column::BooleanColumn> {
        if let Column::Boolean(ref col) = self.column {
            Some(col)
        } else {
            None
        }
    }
    
    /// 内部のColumnを取得
    pub fn into_column(self) -> Column {
        self.column
    }
}