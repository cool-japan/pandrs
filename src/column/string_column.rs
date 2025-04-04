use std::sync::Arc;
use std::any::Any;

use crate::column::common::{Column, ColumnTrait, ColumnType};
use crate::column::string_pool::StringPool;
use crate::error::{Error, Result};

/// 文字列型の列を表す構造体（文字列プールを使用）
#[derive(Debug, Clone)]
pub struct StringColumn {
    pub(crate) string_pool: Arc<StringPool>,
    pub(crate) indices: Arc<[u32]>,
    pub(crate) null_mask: Option<Arc<[u8]>>,
    pub(crate) name: Option<String>,
}

impl StringColumn {
    /// 文字列ベクトルから新しいStringColumnを作成する
    pub fn new(data: Vec<String>) -> Self {
        let pool = StringPool::from_strings(data.clone());
        let indices: Vec<u32> = data.iter()
            .map(|s| pool.find(s).unwrap_or(0))
            .collect();
        
        Self {
            string_pool: Arc::new(pool),
            indices: indices.into(),
            null_mask: None,
            name: None,
        }
    }
    
    /// 名前付きのStringColumnを作成する
    pub fn with_name(data: Vec<String>, name: impl Into<String>) -> Self {
        let pool = StringPool::from_strings(data.clone());
        let indices: Vec<u32> = data.iter()
            .map(|s| pool.find(s).unwrap_or(0))
            .collect();
        
        Self {
            string_pool: Arc::new(pool),
            indices: indices.into(),
            null_mask: None,
            name: Some(name.into()),
        }
    }
    
    /// NULL値を含むStringColumnを作成する
    pub fn with_nulls(data: Vec<String>, nulls: Vec<bool>) -> Self {
        let null_mask = if nulls.iter().any(|&is_null| is_null) {
            Some(crate::column::common::utils::create_bitmask(&nulls))
        } else {
            None
        };
        
        let pool = StringPool::from_strings(data.clone());
        let indices: Vec<u32> = data.iter()
            .map(|s| pool.find(s).unwrap_or(0))
            .collect();
        
        Self {
            string_pool: Arc::new(pool),
            indices: indices.into(),
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
    
    /// インデックスで文字列を取得する
    pub fn get(&self, index: usize) -> Result<Option<&str>> {
        if index >= self.indices.len() {
            return Err(Error::IndexOutOfBounds {
                index,
                size: self.indices.len(),
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
        
        let str_idx = self.indices[index];
        Ok(self.string_pool.get(str_idx))
    }
    
    /// 列内のすべての文字列を取得する
    pub fn to_strings(&self) -> Vec<Option<String>> {
        let mut result = Vec::with_capacity(self.indices.len());
        
        for i in 0..self.indices.len() {
            let value = match self.get(i) {
                Ok(Some(s)) => Some(s.to_string()),
                Ok(None) => None,
                Err(_) => None,
            };
            result.push(value);
        }
        
        result
    }
    
    /// 文字列を検索する
    pub fn contains(&self, search_str: &str) -> Vec<bool> {
        let mut result = vec![false; self.indices.len()];
        
        for i in 0..self.indices.len() {
            if let Ok(Some(s)) = self.get(i) {
                result[i] = s.contains(search_str);
            }
        }
        
        result
    }
    
    /// 正規表現でマッチングする
    pub fn matches(&self, pattern: &str) -> Result<Vec<bool>> {
        use regex::Regex;
        
        let re = Regex::new(pattern)
            .map_err(|e| Error::InvalidRegex(e.to_string()))?;
        
        let mut result = vec![false; self.indices.len()];
        
        for i in 0..self.indices.len() {
            if let Ok(Some(s)) = self.get(i) {
                result[i] = re.is_match(s);
            }
        }
        
        Ok(result)
    }
    
    /// マッピング関数を適用した新しい列を作成する
    pub fn map<F>(&self, f: F) -> Self 
    where
        F: Fn(&str) -> String
    {
        let mut mapped_data = Vec::with_capacity(self.indices.len());
        let mut has_nulls = false;
        
        for i in 0..self.indices.len() {
            match self.get(i) {
                Ok(Some(s)) => mapped_data.push(f(s)),
                Ok(None) => {
                    has_nulls = true;
                    mapped_data.push(String::new()); // ダミー値
                },
                Err(_) => {
                    has_nulls = true;
                    mapped_data.push(String::new()); // ダミー値
                },
            }
        }
        
        if has_nulls {
            let nulls = (0..self.indices.len())
                .map(|i| self.get(i).map(|opt| opt.is_none()).unwrap_or(true))
                .collect();
            
            Self::with_nulls(mapped_data, nulls)
        } else {
            Self::new(mapped_data)
        }
    }
    
    /// フィルタリング条件に基づいて新しい列を作成する
    pub fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(Option<&str>) -> bool
    {
        let mut filtered_data = Vec::new();
        let mut filtered_nulls = Vec::new();
        let has_nulls = self.null_mask.is_some();
        
        for i in 0..self.indices.len() {
            let value = self.get(i).unwrap_or(None);
            if predicate(value) {
                filtered_data.push(value.unwrap_or("").to_string());
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

impl ColumnTrait for StringColumn {
    fn len(&self) -> usize {
        self.indices.len()
    }
    
    fn column_type(&self) -> ColumnType {
        ColumnType::String
    }
    
    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    
    fn clone_column(&self) -> Column {
        Column::String(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}