use std::cmp::Ordering;
use std::collections::HashMap;
use rayon::prelude::*;

use crate::column::{Column, ColumnTrait, ColumnType};
use crate::error::{Error, Result};
use crate::optimized::split_dataframe::core::OptimizedDataFrame;

impl OptimizedDataFrame {
    /// DataFrameを列でソート
    ///
    /// # Arguments
    /// * `by` - ソートする列名
    /// * `ascending` - 昇順にするかどうか
    ///
    /// # Returns
    /// * `Result<Self>` - ソート済みの新しいDataFrame
    pub fn sort_by(&self, by: &str, ascending: bool) -> Result<Self> {
        // 列インデックスの取得
        let column_idx = self.column_indices.get(by)
            .ok_or_else(|| Error::ColumnNotFound(by.to_string()))?;
        
        let column = &self.columns[*column_idx];
        
        // 行インデックスの配列を作成（0から行数-1までの数値）
        let mut indices: Vec<usize> = (0..self.row_count()).collect();
        
        // 列の型によるソート
        match column.column_type() {
            ColumnType::Int64 => {
                let col = column.as_int64().unwrap();
                // 行インデックスと値のペアを作成
                let mut pairs: Vec<(usize, Option<i64>)> = indices.iter()
                    .map(|&idx| (idx, col.get(idx).ok().flatten()))
                    .collect();
                
                // ソート: NULLは最後に配置
                pairs.sort_by(|a, b| {
                    match (&a.1, &b.1) {
                        (None, None) => Ordering::Equal,
                        (None, _) => Ordering::Greater,
                        (_, None) => Ordering::Less,
                        (Some(val_a), Some(val_b)) => {
                            if ascending {
                                val_a.cmp(val_b)
                            } else {
                                val_b.cmp(val_a)
                            }
                        }
                    }
                });
                
                // ソート済み行インデックスを抽出
                indices = pairs.into_iter().map(|(idx, _)| idx).collect();
            },
            ColumnType::Float64 => {
                let col = column.as_float64().unwrap();
                // 行インデックスと値のペアを作成
                let mut pairs: Vec<(usize, Option<f64>)> = indices.iter()
                    .map(|&idx| (idx, col.get(idx).ok().flatten()))
                    .collect();
                
                // ソート: NULLは最後に配置
                pairs.sort_by(|a, b| {
                    match (&a.1, &b.1) {
                        (None, None) => Ordering::Equal,
                        (None, _) => Ordering::Greater,
                        (_, None) => Ordering::Less,
                        (Some(val_a), Some(val_b)) => {
                            if ascending {
                                val_a.partial_cmp(val_b).unwrap_or(Ordering::Equal)
                            } else {
                                val_b.partial_cmp(val_a).unwrap_or(Ordering::Equal)
                            }
                        }
                    }
                });
                
                // ソート済み行インデックスを抽出
                indices = pairs.into_iter().map(|(idx, _)| idx).collect();
            },
            ColumnType::String => {
                let col = column.as_string().unwrap();
                // 行インデックスと値のペアを作成
                let mut pairs: Vec<(usize, Option<String>)> = indices.iter()
                    .map(|&idx| (idx, col.get(idx).ok().flatten().map(|s| s.to_string())))
                    .collect();
                
                // ソート: NULLは最後に配置
                pairs.sort_by(|a, b| {
                    match (&a.1, &b.1) {
                        (None, None) => Ordering::Equal,
                        (None, _) => Ordering::Greater,
                        (_, None) => Ordering::Less,
                        (Some(val_a), Some(val_b)) => {
                            if ascending {
                                val_a.cmp(val_b)
                            } else {
                                val_b.cmp(val_a)
                            }
                        }
                    }
                });
                
                // ソート済み行インデックスを抽出
                indices = pairs.into_iter().map(|(idx, _)| idx).collect();
            },
            ColumnType::Boolean => {
                let col = column.as_boolean().unwrap();
                // 行インデックスと値のペアを作成
                let mut pairs: Vec<(usize, Option<bool>)> = indices.iter()
                    .map(|&idx| (idx, col.get(idx).ok().flatten()))
                    .collect();
                
                // ソート: NULLは最後に配置
                pairs.sort_by(|a, b| {
                    match (&a.1, &b.1) {
                        (None, None) => Ordering::Equal,
                        (None, _) => Ordering::Greater,
                        (_, None) => Ordering::Less,
                        (Some(val_a), Some(val_b)) => {
                            if ascending {
                                val_a.cmp(val_b)
                            } else {
                                val_b.cmp(val_a)
                            }
                        }
                    }
                });
                
                // ソート済み行インデックスを抽出
                indices = pairs.into_iter().map(|(idx, _)| idx).collect();
            },
        }
        
        // ソート済み行インデックスから新しいDataFrameを作成
        self.select_rows_by_indices_internal(&indices)
    }
    
    /// 複数列でDataFrameをソート
    ///
    /// # Arguments
    /// * `by` - ソートする列名の配列
    /// * `ascending` - 各列の昇順/降順の配列（Noneの場合は全て昇順）
    ///
    /// # Returns
    /// * `Result<Self>` - ソート済みの新しいDataFrame
    pub fn sort_by_columns(&self, by: &[&str], ascending: Option<&[bool]>) -> Result<Self> {
        if by.is_empty() {
            return Err(Error::EmptyColumnList);
        }
        
        // 列名の存在確認
        for &col_name in by {
            if !self.column_indices.contains_key(col_name) {
                return Err(Error::ColumnNotFound(col_name.to_string()));
            }
        }
        
        // 昇順配列設定
        let is_ascending: Vec<bool> = match ascending {
            Some(asc) => {
                if asc.len() != by.len() {
                    return Err(Error::InconsistentArrayLengths {
                        expected: by.len(),
                        found: asc.len(),
                    });
                }
                asc.to_vec()
            },
            None => vec![true; by.len()], // デフォルトは昇順
        };
        
        // 行インデックスの配列を作成（0から行数-1までの数値）
        let mut indices: Vec<usize> = (0..self.row_count()).collect();
        
        // 複数キーでソート処理
        indices.sort_by(|&a, &b| {
            // 各列で比較
            for (col_idx, (&col_name, &asc)) in by.iter().zip(is_ascending.iter()).enumerate() {
                let column_idx = self.column_indices[col_name];
                let column = &self.columns[column_idx];
                
                let cmp = match column.column_type() {
                    ColumnType::Int64 => {
                        let col = column.as_int64().unwrap();
                        let val_a = col.get(a).ok().flatten();
                        let val_b = col.get(b).ok().flatten();
                        
                        match (val_a, val_b) {
                            (None, None) => Ordering::Equal,
                            (None, _) => Ordering::Greater,
                            (_, None) => Ordering::Less,
                            (Some(v_a), Some(v_b)) => {
                                if asc { v_a.cmp(&v_b) } else { v_b.cmp(&v_a) }
                            }
                        }
                    },
                    ColumnType::Float64 => {
                        let col = column.as_float64().unwrap();
                        let val_a = col.get(a).ok().flatten();
                        let val_b = col.get(b).ok().flatten();
                        
                        match (val_a, val_b) {
                            (None, None) => Ordering::Equal,
                            (None, _) => Ordering::Greater,
                            (_, None) => Ordering::Less,
                            (Some(v_a), Some(v_b)) => {
                                if asc { 
                                    v_a.partial_cmp(&v_b).unwrap_or(Ordering::Equal) 
                                } else { 
                                    v_b.partial_cmp(&v_a).unwrap_or(Ordering::Equal) 
                                }
                            }
                        }
                    },
                    ColumnType::String => {
                        let col = column.as_string().unwrap();
                        let val_a = col.get(a).ok().flatten().map(|s| s.to_string());
                        let val_b = col.get(b).ok().flatten().map(|s| s.to_string());
                        
                        match (val_a, val_b) {
                            (None, None) => Ordering::Equal,
                            (None, _) => Ordering::Greater,
                            (_, None) => Ordering::Less,
                            (Some(v_a), Some(v_b)) => {
                                if asc { v_a.cmp(&v_b) } else { v_b.cmp(&v_a) }
                            }
                        }
                    },
                    ColumnType::Boolean => {
                        let col = column.as_boolean().unwrap();
                        let val_a = col.get(a).ok().flatten();
                        let val_b = col.get(b).ok().flatten();
                        
                        match (val_a, val_b) {
                            (None, None) => Ordering::Equal,
                            (None, _) => Ordering::Greater,
                            (_, None) => Ordering::Less,
                            (Some(v_a), Some(v_b)) => {
                                if asc { v_a.cmp(&v_b) } else { v_b.cmp(&v_a) }
                            }
                        }
                    },
                };
                
                // 等しくない場合は結果を返す
                if cmp != Ordering::Equal {
                    return cmp;
                }
                
                // 次の列へ進む
            }
            
            // 全ての列が等しい場合
            Ordering::Equal
        });
        
        // ソート済み行インデックスから新しいDataFrameを作成
        self.select_rows_by_indices_internal(&indices)
    }
    
    /// 行インデックスに基づいて行を選択（selectモジュールの実装を使用）
    fn select_rows_by_indices_internal(&self, indices: &[usize]) -> Result<Self> {
        // select.rsに実装された関数を使用
        use crate::optimized::split_dataframe::select;
        select::select_rows_by_indices_impl(self, indices)
    }
}