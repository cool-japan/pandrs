//! OptimizedDataFrameの行操作関連機能

use std::collections::HashMap;
use rayon::prelude::*;

use super::core::OptimizedDataFrame;
use super::data_ops; // データ操作モジュールを参照
use crate::column::{Column, Int64Column, Float64Column, StringColumn, BooleanColumn, ColumnTrait};
use crate::error::{Error, Result};
use crate::index::{DataFrameIndex, IndexTrait};

impl OptimizedDataFrame {
    /// 行のフィルタリング（新しいDataFrameとして）
    /// 
    /// 条件列（ブール型）の値がtrueの行のみを抽出します。
    /// 
    /// # Arguments
    /// * `condition_column` - フィルタリング条件となるブール型の列名
    ///
    /// # Returns
    /// * `Result<Self>` - フィルタリングされた新しいDataFrame
    ///
    /// # 注意
    /// この関数はデータ操作モジュールと同じシグネチャを持つため、実際の実装は `filter_rows` として提供されます。
    pub fn filter_rows(&self, condition_column: &str) -> Result<Self> {
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
            
            // インデックスの処理
            if let Some(ref idx) = self.index {
                if let DataFrameIndex::Simple(ref simple_idx) = idx {
                    let mut new_index_values = Vec::with_capacity(indices.len());
                    
                    for &old_idx in &indices {
                        if old_idx < simple_idx.len() {
                            let value = simple_idx.get_value(old_idx)
                                .map(|s| s.to_string())
                                .unwrap_or_else(|| old_idx.to_string());
                            new_index_values.push(value);
                        } else {
                            new_index_values.push(old_idx.to_string());
                        }
                    }
                    
                    let new_index = crate::index::Index::new(new_index_values)?;
                    result.set_index_from_simple_index(new_index)?;
                }
            }
            
            Ok(result)
        } else {
            Err(Error::ColumnTypeMismatch {
                name: condition_column.to_string(),
                expected: crate::column::ColumnType::Boolean,
                found: condition.column_type(),
            })
        }
    }
    
    /// 指定された行インデックスでフィルタリング
    ///
    /// # Arguments
    /// * `indices` - 抽出する行のインデックス配列
    ///
    /// # Returns
    /// * `Result<Self>` - フィルタリングされた新しいDataFrame
    ///
    /// # 注意
    /// この関数はデータ操作モジュールと同じシグネチャを持つため、実際の実装は `filter_rows_by_indices` として提供されます。
    pub fn filter_rows_by_indices(&self, indices: &[usize]) -> Result<Self> {
        // 並列処理を使用
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
                
                Ok((name.clone(), filtered_column))
            })
            .collect();
        
        // 結果を処理
        let columns = column_results?;
        for (name, column) in columns {
            result.add_column(name, column)?;
        }
        
        // インデックスの処理
        if let Some(ref idx) = self.index {
            if let DataFrameIndex::Simple(ref simple_idx) = idx {
                let valid_indices: Vec<usize> = indices.iter()
                    .filter(|&&i| i < self.row_count)
                    .cloned()
                    .collect();
                
                if !valid_indices.is_empty() {
                    let mut new_index_values = Vec::with_capacity(valid_indices.len());
                    
                    for &old_idx in &valid_indices {
                        if old_idx < simple_idx.len() {
                            let value = simple_idx.get_value(old_idx)
                                .map(|s| s.to_string())
                                .unwrap_or_else(|| old_idx.to_string());
                            new_index_values.push(value);
                        } else {
                            new_index_values.push(old_idx.to_string());
                        }
                    }
                    
                    let new_index = crate::index::Index::new(new_index_values)?;
                    result.set_index_from_simple_index(new_index)?;
                }
            }
        }
        
        Ok(result)
    }
    
    /// 先頭n行を取得
    ///
    /// # Arguments
    /// * `n` - 取得する行数
    ///
    /// # Returns
    /// * `Result<Self>` - 先頭n行の新しいDataFrame
    ///
    /// # 注意
    /// この関数はデータ操作モジュールと同じシグネチャを持つため、実際の実装は `head_rows` として提供されます。
    pub fn head_rows(&self, n: usize) -> Result<Self> {
        let n = std::cmp::min(n, self.row_count);
        let indices: Vec<usize> = (0..n).collect();
        self.filter_rows_by_indices(&indices)
    }
    
    /// 末尾n行を取得
    ///
    /// # Arguments
    /// * `n` - 取得する行数
    ///
    /// # Returns
    /// * `Result<Self>` - 末尾n行の新しいDataFrame
    /// 
    /// # 注意
    /// この関数はデータ操作モジュールと同じシグネチャを持つため、実際の実装は `tail_rows` として提供されます。
    pub fn tail_rows(&self, n: usize) -> Result<Self> {
        let n = std::cmp::min(n, self.row_count);
        let start = self.row_count.saturating_sub(n);
        let indices: Vec<usize> = (start..self.row_count).collect();
        self.filter_rows_by_indices(&indices)
    }
    
    /// サンプリングして行を取得
    ///
    /// # Arguments
    /// * `n` - サンプリングする行数
    /// * `replace` - 復元抽出するかどうか
    /// * `seed` - 乱数シードの値（再現性のため）
    ///
    /// # Returns
    /// * `Result<Self>` - サンプリングされた新しいDataFrame
    /// 
    /// # 注意
    /// この関数はデータ操作モジュールと同じシグネチャを持つため、実際の実装は `sample_rows` として提供されます。
    pub fn sample_rows(&self, n: usize, replace: bool, seed: Option<u64>) -> Result<Self> {
        use rand::{SeedableRng, Rng, seq::SliceRandom};
        use rand::rngs::StdRng;
        use rand::{rng, RngCore}; // thread_rngはrngに名前変更
        
        if self.row_count == 0 {
            return Ok(Self::new());
        }
        
        let row_indices: Vec<usize> = (0..self.row_count).collect();
        
        // 乱数生成器を初期化
        let mut rng = if let Some(seed_val) = seed {
            StdRng::seed_from_u64(seed_val)
        } else {
            // 依存関係の更新でAPIが変わったので、シードを生成する方法を使用
            let mut seed_bytes = [0u8; 32];
            rng().fill_bytes(&mut seed_bytes);
            StdRng::from_seed(seed_bytes)
        };
        
        // 行インデックスをサンプリング
        let sampled_indices = if replace {
            // 復元抽出
            let mut samples = Vec::with_capacity(n);
            for _ in 0..n {
                let idx = rng.random_range(0..self.row_count); // gen_rangeはrandom_rangeに名前変更
                samples.push(idx);
            }
            samples
        } else {
            // 非復元抽出
            let sample_size = std::cmp::min(n, self.row_count);
            let mut indices_copy = row_indices.clone();
            indices_copy.shuffle(&mut rng);
            indices_copy[0..sample_size].to_vec()
        };
        
        self.filter_rows_by_indices(&sampled_indices)
    }
}