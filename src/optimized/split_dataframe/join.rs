//! OptimizedDataFrameの結合（join）機能

use std::collections::HashMap;

use super::core::OptimizedDataFrame;
use crate::column::{Column, ColumnType, Int64Column, BooleanColumn};
use crate::error::{Error, Result};

/// 結合タイプを表す列挙型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// 内部結合（両方の表に存在する行のみ）
    Inner,
    /// 左結合（左側の表の全ての行と、それに一致する右側の表の行）
    Left,
    /// 右結合（右側の表の全ての行と、それに一致する左側の表の行）
    Right,
    /// 外部結合（両方の表の全ての行）
    Outer,
}

impl OptimizedDataFrame {
    /// 内部結合
    ///
    /// # Arguments
    /// * `other` - 結合する右側のDataFrame
    /// * `left_on` - 左側の結合キー列
    /// * `right_on` - 右側の結合キー列
    ///
    /// # Returns
    /// * `Result<Self>` - 結合結果のDataFrame
    pub fn inner_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        self.join_impl(other, left_on, right_on, JoinType::Inner)
    }

    /// 左結合
    ///
    /// # Arguments
    /// * `other` - 結合する右側のDataFrame
    /// * `left_on` - 左側の結合キー列
    /// * `right_on` - 右側の結合キー列
    ///
    /// # Returns
    /// * `Result<Self>` - 結合結果のDataFrame
    pub fn left_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        self.join_impl(other, left_on, right_on, JoinType::Left)
    }

    /// 右結合
    ///
    /// # Arguments
    /// * `other` - 結合する右側のDataFrame
    /// * `left_on` - 左側の結合キー列
    /// * `right_on` - 右側の結合キー列
    ///
    /// # Returns
    /// * `Result<Self>` - 結合結果のDataFrame
    pub fn right_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        self.join_impl(other, left_on, right_on, JoinType::Right)
    }

    /// 外部結合
    ///
    /// # Arguments
    /// * `other` - 結合する右側のDataFrame
    /// * `left_on` - 左側の結合キー列
    /// * `right_on` - 右側の結合キー列
    ///
    /// # Returns
    /// * `Result<Self>` - 結合結果のDataFrame
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