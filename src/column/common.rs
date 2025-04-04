use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use crate::error::{Error, Result};

/// 列の型を識別するための列挙型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    Int64,
    Float64,
    String,
    Boolean,
}

/// 列の共通操作を定義するトレイト
pub trait ColumnTrait: Debug + Send + Sync {
    /// 列の長さを返す
    fn len(&self) -> usize;
    
    /// 列が空かどうかを返す
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// 列の型を返す
    fn column_type(&self) -> ColumnType;
    
    /// 列の名前を返す
    fn name(&self) -> Option<&str>;
    
    /// 列を複製する
    fn clone_column(&self) -> Column;
    
    /// 列をAny型として取得する
    fn as_any(&self) -> &dyn Any;
}

/// 型キャスト用の拡張トレイト（オブジェクト安全性問題を回避）
pub trait ColumnCast {
    /// 列をAs型としてキャストする
    fn as_boxed<T: 'static>(&self) -> Option<&T>;
}

impl<T: ColumnTrait> ColumnCast for T {
    fn as_boxed<U: 'static>(&self) -> Option<&U> {
        self.as_any().downcast_ref::<U>()
    }
}

/// 列を表す列挙型
#[derive(Debug, Clone)]
pub enum Column {
    Int64(crate::column::Int64Column),
    Float64(crate::column::Float64Column),
    String(crate::column::StringColumn),
    Boolean(crate::column::BooleanColumn),
}

/// NULL値を追跡するビットマスク
#[derive(Debug, Clone)]
pub struct BitMask {
    pub(crate) data: Arc<[u8]>,
    pub(crate) len: usize,
}

impl BitMask {
    /// 新しいビットマスクを作成する
    pub fn new(length: usize) -> Self {
        let bytes_needed = (length + 7) / 8;
        let data = vec![0u8; bytes_needed].into();
        
        Self {
            data,
            len: length,
        }
    }
    
    /// すべてのビットが0のビットマスクを作成する
    pub fn zeros(length: usize) -> Self {
        Self::new(length)
    }
    
    /// すべてのビットが1のビットマスクを作成する
    pub fn ones(length: usize) -> Self {
        let bytes_needed = (length + 7) / 8;
        let mut data = vec![0xFFu8; bytes_needed];
        
        // 不完全な最後のバイトを調整
        let remaining_bits = length % 8;
        if remaining_bits != 0 {
            let last_byte_mask = (1u8 << remaining_bits) - 1;
            if let Some(last) = data.last_mut() {
                *last = *last & last_byte_mask;
            }
        }
        
        Self {
            data: data.into(),
            len: length,
        }
    }
    
    /// ブール値のベクトルからビットマスクを作成する
    pub fn from_bools(bools: &[bool]) -> Self {
        let length = bools.len();
        let bytes_needed = (length + 7) / 8;
        let mut data = vec![0u8; bytes_needed];
        
        for (i, &is_set) in bools.iter().enumerate() {
            if is_set {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                data[byte_idx] |= 1 << bit_idx;
            }
        }
        
        Self {
            data: data.into(),
            len: length,
        }
    }
    
    /// ビットが設定されているかどうかを確認する
    pub fn get(&self, index: usize) -> Result<bool> {
        if index >= self.len {
            return Err(Error::IndexOutOfBounds {
                index,
                size: self.len,
            });
        }
        
        let byte_idx = index / 8;
        let bit_idx = index % 8;
        let byte = self.data[byte_idx];
        
        Ok((byte & (1 << bit_idx)) != 0)
    }
    
    /// ビットマスクの長さを返す
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// ビットマスクが空かどうかを返す
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// 列操作のユーティリティ関数
pub mod utils {
    use super::*;
    
    /// ブール値のベクトルからビットマスクを作成する
    pub fn create_bitmask(nulls: &[bool]) -> Arc<[u8]> {
        let length = nulls.len();
        let bytes_needed = (length + 7) / 8;
        let mut data = vec![0u8; bytes_needed];
        
        for (i, &is_null) in nulls.iter().enumerate() {
            if is_null {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                data[byte_idx] |= 1 << bit_idx;
            }
        }
        
        data.into()
    }
    
    /// ビットマスクからブール値のベクトルを作成する
    pub fn bitmask_to_bools(mask: &[u8], len: usize) -> Vec<bool> {
        let mut result = Vec::with_capacity(len);
        
        for i in 0..len {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            let is_set = (mask[byte_idx] & (1 << bit_idx)) != 0;
            result.push(is_set);
        }
        
        result
    }
}

// Column enumの実装
impl Column {
    /// 列の長さを返す
    pub fn len(&self) -> usize {
        match self {
            Column::Int64(col) => col.len(),
            Column::Float64(col) => col.len(),
            Column::String(col) => col.len(),
            Column::Boolean(col) => col.len(),
        }
    }
    
    /// 列が空かどうかを返す
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// 列の型を返す
    pub fn column_type(&self) -> ColumnType {
        match self {
            Column::Int64(_) => ColumnType::Int64,
            Column::Float64(_) => ColumnType::Float64,
            Column::String(_) => ColumnType::String,
            Column::Boolean(_) => ColumnType::Boolean,
        }
    }
    
    /// 列の名前を返す
    pub fn name(&self) -> Option<&str> {
        match self {
            Column::Int64(col) => col.name.as_deref(),
            Column::Float64(col) => col.name.as_deref(),
            Column::String(col) => col.name.as_deref(),
            Column::Boolean(col) => col.name.as_deref(),
        }
    }
    
    /// 列を複製する
    pub fn clone_column(&self) -> Self {
        self.clone()
    }
}

// 型変換のFrom実装
impl From<crate::column::Int64Column> for Column {
    fn from(col: crate::column::Int64Column) -> Self {
        Column::Int64(col)
    }
}

impl From<crate::column::Float64Column> for Column {
    fn from(col: crate::column::Float64Column) -> Self {
        Column::Float64(col)
    }
}

impl From<crate::column::StringColumn> for Column {
    fn from(col: crate::column::StringColumn) -> Self {
        Column::String(col)
    }
}

impl From<crate::column::BooleanColumn> for Column {
    fn from(col: crate::column::BooleanColumn) -> Self {
        Column::Boolean(col)
    }
}