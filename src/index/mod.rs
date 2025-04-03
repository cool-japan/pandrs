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

/// インデックス型の共通トレイト
pub trait IndexTrait {
    /// インデックスの長さを取得
    fn len(&self) -> usize;

    /// インデックスが空かどうか
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> IndexTrait for Index<T> 
where 
    T: Debug + Clone + Eq + Hash + Display 
{
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T> IndexTrait for MultiIndex<T> 
where 
    T: Debug + Clone + Eq + Hash + Display 
{
    fn len(&self) -> usize {
        self.len()
    }
}

/// DataFrameが使用するインデックス型
#[derive(Debug, Clone)]
pub enum DataFrameIndex<T> 
where 
    T: Debug + Clone + Eq + Hash + Display 
{
    /// 単一レベルのインデックス
    Simple(Index<T>),
    /// 複数レベルのインデックス
    Multi(MultiIndex<T>),
}

impl<T> IndexTrait for DataFrameIndex<T> 
where 
    T: Debug + Clone + Eq + Hash + Display 
{
    fn len(&self) -> usize {
        match self {
            DataFrameIndex::Simple(idx) => idx.len(),
            DataFrameIndex::Multi(idx) => idx.len(),
        }
    }
}

impl<T> DataFrameIndex<T> 
where 
    T: Debug + Clone + Eq + Hash + Display 
{
    /// シンプルインデックスから作成
    pub fn from_simple(index: Index<T>) -> Self {
        DataFrameIndex::Simple(index)
    }

    /// マルチインデックスから作成
    pub fn from_multi(index: MultiIndex<T>) -> Self {
        DataFrameIndex::Multi(index)
    }

    /// デフォルトインデックスを作成
    pub fn default_with_len(len: usize) -> Result<DataFrameIndex<usize>> {
        let idx = Index::<usize>::from_range(0..len)?;
        Ok(DataFrameIndex::Simple(idx))
    }

    /// インデックスの種類を判定
    pub fn is_multi(&self) -> bool {
        matches!(self, DataFrameIndex::Multi(_))
    }
}

/// 整数インデックス型のエイリアス
pub type RangeIndex = Index<usize>;

/// 文字列インデックス型のエイリアス
pub type StringIndex = Index<String>;