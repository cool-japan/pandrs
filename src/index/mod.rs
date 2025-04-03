mod multi_index;

pub use multi_index::{MultiIndex, StringMultiIndex};

use crate::error::{PandRSError, Result};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::Range;

/// インデックス構造体
///
/// DataFrameやSeriesの行ラベルを表現する
#[derive(Debug, Clone)]
pub struct Index<T>
where
    T: Debug + Clone + Eq + Hash + Display,
{
    /// インデックス値
    values: Vec<T>,

    /// 値から位置へのマッピング
    map: HashMap<T, usize>,
    
    /// インデックスの名前（オプション）
    name: Option<String>,
}

impl<T> Index<T>
where
    T: Debug + Clone + Eq + Hash + Display,
{
    /// 新しいインデックスを作成
    pub fn new(values: Vec<T>) -> Result<Self> {
        Self::with_name(values, None)
    }
    
    /// 名前付きの新しいインデックスを作成
    pub fn with_name(values: Vec<T>, name: Option<String>) -> Result<Self> {
        let mut map = HashMap::with_capacity(values.len());

        // 一意性チェックしながらマップ構築
        for (i, value) in values.iter().enumerate() {
            if map.insert(value.clone(), i).is_some() {
                return Err(PandRSError::Index(format!(
                    "インデックス値 '{}' が重複しています",
                    value
                )));
            }
        }

        Ok(Index { values, map, name })
    }

    /// 整数範囲からインデックスを作成
    pub fn from_range(range: Range<usize>) -> Result<Index<usize>> {
        let values: Vec<usize> = range.collect();
        Index::<usize>::new(values)
    }

    /// インデックス長を取得
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// インデックスが空かどうか
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// 値から位置を取得
    pub fn get_loc(&self, key: &T) -> Option<usize> {
        self.map.get(key).copied()
    }

    /// 位置から値を取得
    pub fn get_value(&self, pos: usize) -> Option<&T> {
        self.values.get(pos)
    }

    /// 全ての値を取得
    pub fn values(&self) -> &[T] {
        &self.values
    }
    
    /// インデックス名を取得
    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }
    
    /// インデックス名を設定
    pub fn set_name(&mut self, name: Option<String>) {
        self.name = name;
    }
    
    /// 新しい名前でインデックスをコピー
    pub fn rename(&self, name: Option<String>) -> Self {
        let mut new_index = self.clone();
        new_index.name = name;
        new_index
    }
}

/// 整数インデックス型のエイリアス
pub type RangeIndex = Index<usize>;

/// 文字列インデックス型のエイリアス
pub type StringIndex = Index<String>;