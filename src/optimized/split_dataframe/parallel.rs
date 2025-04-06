//! OptimizedDataFrameの並列処理機能

use std::collections::HashMap;
use rayon::prelude::*;

use super::core::OptimizedDataFrame;
use crate::column::{Column, ColumnTrait, Int64Column, Float64Column, StringColumn, BooleanColumn};
use crate::error::{Error, Result};

impl OptimizedDataFrame {
    /// 並列処理を使用した行のフィルタリング
    ///
    /// 条件列（ブール型）の値がtrueの行のみを抽出し、大規模データセットでは並列処理を適用します。
    /// 
    /// # Arguments
    /// * `condition_column` - フィルタリング条件となるブール型の列名
    ///
    /// # Returns
    /// * `Result<Self>` - フィルタリングされた新しいDataFrame
    pub fn par_filter(&self, condition_column: &str) -> Result<Self> {
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
                        Column::Float64(_) => Column::Float64(Float64Column::new(Vec::new())),
                        Column::String(_) => Column::String(StringColumn::new(Vec::new())),
                        Column::Boolean(_) => Column::Boolean(BooleanColumn::new(Vec::new())),
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
                            Column::Float64(Float64Column::new(filtered_data))
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
                            Column::String(StringColumn::new(filtered_data))
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
                            Column::Boolean(BooleanColumn::new(filtered_data))
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
                                
                                Column::Float64(Float64Column::new(filtered_data))
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
                                
                                Column::String(StringColumn::new(filtered_data))
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
                                Column::Boolean(BooleanColumn::new(filtered_data))
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
}