//! OptimizedDataFrameの関数適用機能

use std::collections::HashMap;
use rayon::prelude::*;

use super::core::{OptimizedDataFrame, ColumnView};
use crate::column::{Column, Int64Column, Float64Column, StringColumn, BooleanColumn, ColumnTrait};
use crate::error::{Error, Result};

impl OptimizedDataFrame {
    /// 列に関数を適用し、結果の新しいDataFrameを返す（パフォーマンス最適化版）
    ///
    /// # Arguments
    /// * `f` - 適用する関数（列のビューを取り、新しい列を返す）
    /// * `columns` - 処理対象の列名（Noneの場合はすべての列）
    /// # Returns
    /// * `Result<Self>` - 処理結果のDataFrame
    pub fn apply<F>(&self, f: F, columns: Option<&[&str]>) -> Result<Self>
    where
        F: Fn(&ColumnView) -> Result<Column> + Send + Sync,
    {
        let mut result = Self::new();
        
        // 処理対象の列を決定
        let target_columns = if let Some(cols) = columns {
            // 指定された列のみを対象とする
            cols.iter()
                .map(|&name| {
                    self.column_indices.get(name)
                        .ok_or_else(|| Error::ColumnNotFound(name.to_string()))
                        .map(|&idx| (name, idx))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            // すべての列を対象とする
            self.column_names.iter()
                .map(|name| {
                    let idx = self.column_indices[name];
                    (name.as_str(), idx)
                })
                .collect()
        };
        
        // 列ごとに関数を適用（パフォーマンス最適化のため並列処理を使用）
        let processed_columns: Result<Vec<(String, Column)>> = target_columns
            .into_par_iter()  // 並列イテレーション
            .map(|(name, idx)| {
                // 列のビューを作成
                let view = ColumnView {
                    column: self.columns[idx].clone(),
                };
                
                // 関数を適用して新しい列を生成
                let new_column = f(&view)?;
                
                // 元の列と同じ行数であることを確認
                if new_column.len() != self.row_count {
                    return Err(Error::LengthMismatch {
                        expected: self.row_count,
                        actual: new_column.len(),
                    });
                }
                
                Ok((name.to_string(), new_column))
            })
            .collect();
            
        // 処理結果の列をDataFrameに追加
        for (name, column) in processed_columns? {
            result.add_column(name, column)?;
        }
            
        // 処理対象外の列をそのままコピー
        if columns.is_some() {
            for (name, idx) in self.column_names.iter().map(|name| (name, self.column_indices[name])) {
                if !result.column_indices.contains_key(name) {
                    result.add_column(name.clone(), self.columns[idx].clone_column())?;
                }
            }
        }
        
        // インデックスを新しいDataFrameにコピー
        if let Some(ref idx) = self.index {
            result.index = Some(idx.clone());
        }
        
        Ok(result)
    }
    
    /// 要素ごとに関数を適用（applymap相当）
    ///
    /// # Arguments
    /// * `column_name` - 対象の列名
    /// * `f` - 適用する関数（列の型に応じた関数）
    /// # Returns
    /// * `Result<Self>` - 処理結果のDataFrame
    pub fn applymap<F, G, H, I>(&self, column_name: &str, f_str: F, f_int: G, f_float: H, f_bool: I) -> Result<Self>
    where
        F: Fn(&str) -> String + Send + Sync,
        G: Fn(&i64) -> i64 + Send + Sync,
        H: Fn(&f64) -> f64 + Send + Sync,
        I: Fn(&bool) -> bool + Send + Sync,
    {
        // 列の存在確認
        let col_idx = self.column_indices.get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;
        
        let column = &self.columns[*col_idx];
        
        // 型に応じた処理
        let new_column = match column {
            Column::Int64(int_col) => {
                let mut new_data = Vec::with_capacity(int_col.len());
                
                for i in 0..int_col.len() {
                    if let Ok(Some(val)) = int_col.get(i) {
                        new_data.push(f_int(&val));
                    } else {
                        // NULL値はそのまま
                        new_data.push(0);  // デフォルト値
                    }
                }
                
                Column::Int64(Int64Column::new(new_data))
            },
            Column::Float64(float_col) => {
                let mut new_data = Vec::with_capacity(float_col.len());
                
                for i in 0..float_col.len() {
                    if let Ok(Some(val)) = float_col.get(i) {
                        new_data.push(f_float(&val));
                    } else {
                        // NULL値はそのまま
                        new_data.push(0.0);  // デフォルト値
                    }
                }
                
                Column::Float64(Float64Column::new(new_data))
            },
            Column::String(str_col) => {
                let mut new_data = Vec::with_capacity(str_col.len());
                
                for i in 0..str_col.len() {
                    if let Ok(Some(val)) = str_col.get(i) {
                        new_data.push(f_str(val));
                    } else {
                        // NULL値はそのまま
                        new_data.push(String::new());  // デフォルト値
                    }
                }
                
                Column::String(StringColumn::new(new_data))
            },
            Column::Boolean(bool_col) => {
                let mut new_data = Vec::with_capacity(bool_col.len());
                
                for i in 0..bool_col.len() {
                    if let Ok(Some(val)) = bool_col.get(i) {
                        new_data.push(f_bool(&val));
                    } else {
                        // NULL値はそのまま
                        new_data.push(false);  // デフォルト値
                    }
                }
                
                Column::Boolean(BooleanColumn::new(new_data))
            },
        };
        
        // 結果のDataFrameを作成
        let mut result = self.clone();
        
        // 既存の列を置き換え
        result.columns[*col_idx] = new_column;
        
        Ok(result)
    }
    
    /// 並列処理を使用して列に関数を適用
    pub fn par_apply<F>(&self, func: F) -> Result<Self>
    where
        F: Fn(&ColumnView) -> Result<Column> + Sync + Send,
    {
        // 基本的にはapplyと同じですが、内部的には常に並列処理を使用します
        self.apply(func, None)
    }
}