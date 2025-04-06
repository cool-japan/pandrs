//! ColumnViewの実装

use super::core::ColumnView;
use crate::column::{Column, Int64Column, Float64Column, StringColumn, BooleanColumn, ColumnType};
use crate::error::{Error, Result};
use std::any::Any;

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
    
    /// 内部のColumnへの参照を取得
    pub fn column(&self) -> &Column {
        &self.column
    }
    
    /// 内部のColumnを取得（消費的）
    pub fn into_column(self) -> Column {
        self.column
    }
    
    /// 特定のインデックスのfloat64値を取得
    pub fn get_f64(&self, index: usize) -> Result<Option<f64>> {
        match &self.column {
            Column::Float64(col) => col.get(index),
            _ => Err(Error::ColumnTypeMismatch {
                name: self.column.name().unwrap_or("").to_string(),
                expected: ColumnType::Float64,
                found: self.column.column_type(),
            }),
        }
    }
    
    /// 特定のインデックスのint64値を取得
    pub fn get_i64(&self, index: usize) -> Result<Option<i64>> {
        match &self.column {
            Column::Int64(col) => col.get(index),
            _ => Err(Error::ColumnTypeMismatch {
                name: self.column.name().unwrap_or("").to_string(),
                expected: ColumnType::Int64,
                found: self.column.column_type(),
            }),
        }
    }
    
    /// 特定のインデックスの文字列値を取得
    pub fn get_string(&self, index: usize) -> Result<Option<String>> {
        match &self.column {
            Column::String(col) => {
                match col.get(index)? {
                    Some(s) => Ok(Some(s.to_string())),
                    None => Ok(None)
                }
            },
            _ => Err(Error::ColumnTypeMismatch {
                name: self.column.name().unwrap_or("").to_string(),
                expected: ColumnType::String,
                found: self.column.column_type(),
            }),
        }
    }
    
    /// 特定のインデックスのブール値を取得
    pub fn get_bool(&self, index: usize) -> Result<Option<bool>> {
        match &self.column {
            Column::Boolean(col) => col.get(index),
            _ => Err(Error::ColumnTypeMismatch {
                name: self.column.name().unwrap_or("").to_string(),
                expected: ColumnType::Boolean,
                found: self.column.column_type(),
            }),
        }
    }
}
