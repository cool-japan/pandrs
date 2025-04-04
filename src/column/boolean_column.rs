use std::sync::Arc;
use std::any::Any;

use crate::column::common::{Column, ColumnTrait, ColumnType, BitMask};
use crate::error::{Error, Result};

/// ブール型の列を表す構造体（BitMaskで最適化）
#[derive(Debug, Clone)]
pub struct BooleanColumn {
    pub(crate) data: BitMask,
    pub(crate) null_mask: Option<Arc<[u8]>>,
    pub(crate) name: Option<String>,
    pub(crate) length: usize,
}

impl BooleanColumn {
    /// ブールベクトルから新しいBooleanColumnを作成する
    pub fn new(data: Vec<bool>) -> Self {
        let length = data.len();
        let bitmask = BitMask::from_bools(&data);
        
        Self {
            data: bitmask,
            null_mask: None,
            name: None,
            length,
        }
    }
    
    /// 名前付きのBooleanColumnを作成する
    pub fn with_name(data: Vec<bool>, name: impl Into<String>) -> Self {
        let length = data.len();
        let bitmask = BitMask::from_bools(&data);
        
        Self {
            data: bitmask,
            null_mask: None,
            name: Some(name.into()),
            length,
        }
    }
    
    /// NULL値を含むBooleanColumnを作成する
    pub fn with_nulls(data: Vec<bool>, nulls: Vec<bool>) -> Self {
        let null_mask = if nulls.iter().any(|&is_null| is_null) {
            Some(crate::column::common::utils::create_bitmask(&nulls))
        } else {
            None
        };
        
        let length = data.len();
        let bitmask = BitMask::from_bools(&data);
        
        Self {
            data: bitmask,
            null_mask,
            name: None,
            length,
        }
    }
    
    /// 名前を設定する
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }
    
    /// 名前を取得する
    pub fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    
    /// インデックスでブール値を取得する
    pub fn get(&self, index: usize) -> Result<Option<bool>> {
        if index >= self.length {
            return Err(Error::IndexOutOfBounds {
                index,
                size: self.length,
            });
        }
        
        // NULL値のチェック
        if let Some(ref mask) = self.null_mask {
            let byte_idx = index / 8;
            let bit_idx = index % 8;
            if byte_idx < mask.len() && (mask[byte_idx] & (1 << bit_idx)) != 0 {
                return Ok(None);
            }
        }
        
        self.data.get(index).map(Some)
    }
    
    /// 列内のすべてのブール値を取得する
    pub fn to_bools(&self) -> Vec<Option<bool>> {
        let mut result = Vec::with_capacity(self.length);
        
        for i in 0..self.length {
            result.push(self.get(i).unwrap_or(None));
        }
        
        result
    }
    
    /// true値の数を数える
    pub fn count_true(&self) -> usize {
        let mut count = 0;
        
        for i in 0..self.length {
            if let Ok(Some(true)) = self.get(i) {
                count += 1;
            }
        }
        
        count
    }
    
    /// false値の数を数える
    pub fn count_false(&self) -> usize {
        let mut count = 0;
        
        for i in 0..self.length {
            if let Ok(Some(false)) = self.get(i) {
                count += 1;
            }
        }
        
        count
    }
    
    /// マッピング関数を適用した新しい列を作成する
    pub fn map<F>(&self, f: F) -> Self 
    where
        F: Fn(bool) -> bool
    {
        let mut mapped_data = Vec::with_capacity(self.length);
        let mut has_nulls = false;
        
        for i in 0..self.length {
            match self.get(i) {
                Ok(Some(b)) => mapped_data.push(f(b)),
                Ok(None) => {
                    has_nulls = true;
                    mapped_data.push(false); // ダミー値
                },
                Err(_) => {
                    has_nulls = true;
                    mapped_data.push(false); // ダミー値
                },
            }
        }
        
        if has_nulls {
            let nulls = (0..self.length)
                .map(|i| self.get(i).map(|opt| opt.is_none()).unwrap_or(true))
                .collect();
            
            Self::with_nulls(mapped_data, nulls)
        } else {
            Self::new(mapped_data)
        }
    }
    
    /// 論理NOT操作を適用した新しい列を作成する
    pub fn logical_not(&self) -> Self {
        self.map(|b| !b)
    }
    
    /// フィルタリング条件に基づいて新しい列を作成する
    pub fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(Option<bool>) -> bool
    {
        let mut filtered_data = Vec::new();
        let mut filtered_nulls = Vec::new();
        let has_nulls = self.null_mask.is_some();
        
        for i in 0..self.length {
            let value = self.get(i).unwrap_or(None);
            if predicate(value) {
                filtered_data.push(value.unwrap_or(false));
                if has_nulls {
                    filtered_nulls.push(value.is_none());
                }
            }
        }
        
        if has_nulls {
            Self::with_nulls(filtered_data, filtered_nulls)
        } else {
            Self::new(filtered_data)
        }
    }
}

impl ColumnTrait for BooleanColumn {
    fn len(&self) -> usize {
        self.length
    }
    
    fn column_type(&self) -> ColumnType {
        ColumnType::Boolean
    }
    
    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    
    fn clone_column(&self) -> Column {
        Column::Boolean(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}