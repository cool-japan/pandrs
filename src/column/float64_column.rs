use std::sync::Arc;
use std::any::Any;

use crate::column::common::{Column, ColumnTrait, ColumnType};
use crate::error::{Error, Result};

/// Float64型の列を表す構造体
#[derive(Debug, Clone)]
pub struct Float64Column {
    pub(crate) data: Arc<[f64]>,
    pub(crate) null_mask: Option<Arc<[u8]>>,
    pub(crate) name: Option<String>,
}

impl Float64Column {
    /// 新しいFloat64Columnを作成する
    pub fn new(data: Vec<f64>) -> Self {
        Self {
            data: data.into(),
            null_mask: None,
            name: None,
        }
    }
    
    /// 名前付きのFloat64Columnを作成する
    pub fn with_name(data: Vec<f64>, name: impl Into<String>) -> Self {
        Self {
            data: data.into(),
            null_mask: None,
            name: Some(name.into()),
        }
    }
    
    /// NULL値を含むFloat64Columnを作成する
    pub fn with_nulls(data: Vec<f64>, nulls: Vec<bool>) -> Self {
        let null_mask = if nulls.iter().any(|&is_null| is_null) {
            Some(crate::column::common::utils::create_bitmask(&nulls))
        } else {
            None
        };
        
        Self {
            data: data.into(),
            null_mask,
            name: None,
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
    
    /// インデックスでデータを取得する
    pub fn get(&self, index: usize) -> Result<Option<f64>> {
        if index >= self.data.len() {
            return Err(Error::IndexOutOfBounds {
                index,
                size: self.data.len(),
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
        
        Ok(Some(self.data[index]))
    }
    
    /// データの合計を計算する（NULL値を除く）
    pub fn sum(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        
        match &self.null_mask {
            None => {
                // NULLがない場合は単純に合計
                self.data.iter().sum()
            },
            Some(mask) => {
                // NULLを除いて合計
                let mut sum = 0.0;
                for i in 0..self.data.len() {
                    let byte_idx = i / 8;
                    let bit_idx = i % 8;
                    if byte_idx >= mask.len() || (mask[byte_idx] & (1 << bit_idx)) == 0 {
                        sum += self.data[i];
                    }
                }
                sum
            }
        }
    }
    
    /// データの平均を計算する（NULL値を除く）
    pub fn mean(&self) -> Option<f64> {
        if self.data.is_empty() {
            return None;
        }
        
        let (sum, count) = match &self.null_mask {
            None => {
                // NULLがない場合
                let sum: f64 = self.data.iter().sum();
                (sum, self.data.len())
            },
            Some(mask) => {
                // NULLを除いて計算
                let mut sum = 0.0;
                let mut count = 0;
                for i in 0..self.data.len() {
                    let byte_idx = i / 8;
                    let bit_idx = i % 8;
                    if byte_idx >= mask.len() || (mask[byte_idx] & (1 << bit_idx)) == 0 {
                        sum += self.data[i];
                        count += 1;
                    }
                }
                (sum, count)
            }
        };
        
        if count == 0 {
            None
        } else {
            Some(sum / count as f64)
        }
    }
    
    /// データの最小値を計算する（NULL値を除く）
    pub fn min(&self) -> Option<f64> {
        if self.data.is_empty() {
            return None;
        }
        
        match &self.null_mask {
            None => {
                // NULLがない場合
                self.data.iter()
                    .copied()
                    .filter(|x| x.is_finite())
                    .fold(None, |min, x| {
                        Some(min.map_or(x, |m| m.min(x)))
                    })
            },
            Some(mask) => {
                // NULLを除いて計算
                let mut min_val = None;
                for i in 0..self.data.len() {
                    let byte_idx = i / 8;
                    let bit_idx = i % 8;
                    if byte_idx >= mask.len() || (mask[byte_idx] & (1 << bit_idx)) == 0 {
                        let val = self.data[i];
                        if val.is_finite() {
                            min_val = Some(min_val.map_or(val, |m: f64| m.min(val)));
                        }
                    }
                }
                min_val
            }
        }
    }
    
    /// データの最大値を計算する（NULL値を除く）
    pub fn max(&self) -> Option<f64> {
        if self.data.is_empty() {
            return None;
        }
        
        match &self.null_mask {
            None => {
                // NULLがない場合
                self.data.iter()
                    .copied()
                    .filter(|x| x.is_finite())
                    .fold(None, |max, x| {
                        Some(max.map_or(x, |m| m.max(x)))
                    })
            },
            Some(mask) => {
                // NULLを除いて計算
                let mut max_val = None;
                for i in 0..self.data.len() {
                    let byte_idx = i / 8;
                    let bit_idx = i % 8;
                    if byte_idx >= mask.len() || (mask[byte_idx] & (1 << bit_idx)) == 0 {
                        let val = self.data[i];
                        if val.is_finite() {
                            max_val = Some(max_val.map_or(val, |m: f64| m.max(val)));
                        }
                    }
                }
                max_val
            }
        }
    }
    
    /// マッピング関数を適用した新しい列を作成する
    pub fn map<F>(&self, f: F) -> Self 
    where
        F: Fn(f64) -> f64
    {
        let mapped_data: Vec<f64> = self.data.iter().map(|&x| f(x)).collect();
        
        Self {
            data: mapped_data.into(),
            null_mask: self.null_mask.clone(),
            name: self.name.clone(),
        }
    }
    
    /// フィルタリング条件に基づいて新しい列を作成する
    pub fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(Option<f64>) -> bool
    {
        let mut filtered_data = Vec::new();
        let mut filtered_nulls = Vec::new();
        let has_nulls = self.null_mask.is_some();
        
        for i in 0..self.data.len() {
            let value = self.get(i).unwrap_or(None);
            if predicate(value) {
                filtered_data.push(value.unwrap_or(f64::NAN));
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

impl ColumnTrait for Float64Column {
    fn len(&self) -> usize {
        self.data.len()
    }
    
    fn column_type(&self) -> ColumnType {
        ColumnType::Float64
    }
    
    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    
    fn clone_column(&self) -> Column {
        Column::Float64(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}