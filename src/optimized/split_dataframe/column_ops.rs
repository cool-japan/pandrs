//! OptimizedDataFrameの列操作関連機能

use super::core::{OptimizedDataFrame, ColumnView};
use crate::column::{Column, Int64Column, Float64Column, StringColumn, BooleanColumn, ColumnType};
use crate::error::{Error, Result};

impl OptimizedDataFrame {
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
    
    /// 整数列を追加
    pub fn add_int_column(&mut self, name: impl Into<String>, data: Vec<i64>) -> Result<()> {
        self.add_column(name, Column::Int64(Int64Column::new(data)))
    }
    
    /// 浮動小数点列を追加
    pub fn add_float_column(&mut self, name: impl Into<String>, data: Vec<f64>) -> Result<()> {
        self.add_column(name, Column::Float64(Float64Column::new(data)))
    }
    
    /// 文字列列を追加
    pub fn add_string_column(&mut self, name: impl Into<String>, data: Vec<String>) -> Result<()> {
        self.add_column(name, Column::String(StringColumn::new(data)))
    }
    
    /// ブール列を追加
    pub fn add_boolean_column(&mut self, name: impl Into<String>, data: Vec<bool>) -> Result<()> {
        self.add_column(name, Column::Boolean(BooleanColumn::new(data)))
    }
    
    /// 列を削除
    pub fn remove_column(&mut self, name: &str) -> Result<Column> {
        let column_idx = self.column_indices.get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
        
        // 列とそのインデックスを削除
        let column_idx = *column_idx;
        let removed_column = self.columns.remove(column_idx);
        self.column_indices.remove(name);
        
        // 列名リストから削除
        let name_idx = self.column_names.iter().position(|n| n == name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
        self.column_names.remove(name_idx);
        
        // インデックスの再計算
        for (_, idx) in self.column_indices.iter_mut() {
            if *idx > column_idx {
                *idx -= 1;
            }
        }
        
        Ok(removed_column)
    }
    
    /// 列名を変更
    pub fn rename_column(&mut self, old_name: &str, new_name: impl Into<String>) -> Result<()> {
        let new_name = new_name.into();
        
        // 新しい名前が既に存在する場合はエラー
        if self.column_indices.contains_key(&new_name) && old_name != new_name {
            return Err(Error::DuplicateColumnName(new_name));
        }
        
        // 古い名前が存在するか確認
        let column_idx = *self.column_indices.get(old_name)
            .ok_or_else(|| Error::ColumnNotFound(old_name.to_string()))?;
        
        // インデックスと列名を更新
        self.column_indices.remove(old_name);
        self.column_indices.insert(new_name.clone(), column_idx);
        
        // 列名リストを更新
        let name_idx = self.column_names.iter().position(|n| n == old_name)
            .ok_or_else(|| Error::ColumnNotFound(old_name.to_string()))?;
        self.column_names[name_idx] = new_name;
        
        Ok(())
    }
    
    /// 列の参照を取得
    pub fn column(&self, name: &str) -> Result<ColumnView> {
        let column_idx = self.column_indices.get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
        
        let column = self.columns[*column_idx].clone();
        Ok(ColumnView { column })
    }
    
    /// 指定された行と列の値を取得
    pub fn get_value(&self, row_idx: usize, column_name: &str) -> Result<Option<String>> {
        if row_idx >= self.row_count {
            return Err(Error::IndexOutOfBounds {
                index: row_idx,
                size: self.row_count,
            });
        }
        
        let column_idx = self.column_indices.get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;
        
        let column = &self.columns[*column_idx];
        
        // 列の型に応じて値を取得
        let value = match column {
            Column::Int64(col) => {
                match col.get(row_idx)? {
                    Some(val) => Some(val.to_string()),
                    None => None,
                }
            },
            Column::Float64(col) => {
                match col.get(row_idx)? {
                    Some(val) => Some(val.to_string()),
                    None => None,
                }
            },
            Column::String(col) => {
                match col.get(row_idx)? {
                    Some(val) => Some(val.to_string()),
                    None => None,
                }
            },
            Column::Boolean(col) => {
                match col.get(row_idx)? {
                    Some(val) => Some(val.to_string()),
                    None => None,
                }
            },
        };
        
        Ok(value)
    }
    
    /// 列の型を取得
    pub fn column_type(&self, name: &str) -> Result<ColumnType> {
        let column_idx = self.column_indices.get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
        
        Ok(self.columns[*column_idx].column_type())
    }
}
