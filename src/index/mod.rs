mod multi_index;

pub use multi_index::{MultiIndex, StringMultiIndex};

use crate::error::{PandRSError, Result};
use crate::temporal::Temporal;
use chrono::{NaiveDate, NaiveDateTime};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::Range;

/// インデックス構造体
///
/// DataFrameやSeriesの行ラベルを表現する基本的なインデックス構造です。
/// 一意な値のコレクションとそれらの位置を保持します。
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
    /// 新しいインデックスを作成します
    ///
    /// # 引数
    /// * `values` - インデックス値のベクタ
    ///
    /// # 戻り値
    /// 成功した場合は新しいIndexインスタンス、失敗した場合はエラー
    ///
    /// # エラー
    /// 値に重複がある場合にエラーを返します
    pub fn new(values: Vec<T>) -> Result<Self> {
        Self::with_name(values, None)
    }
    
    /// 名前付きの新しいインデックスを作成します
    ///
    /// # 引数
    /// * `values` - インデックス値のベクタ
    /// * `name` - インデックスの名前（オプション）
    ///
    /// # 戻り値
    /// 成功した場合は新しいIndexインスタンス、失敗した場合はエラー
    ///
    /// # エラー
    /// 値に重複がある場合にエラーを返します
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

    /// 整数範囲からインデックスを作成します
    ///
    /// # 引数
    /// * `range` - 整数範囲
    ///
    /// # 戻り値
    /// 成功した場合は新しいRangeIndex、失敗した場合はエラー
    pub fn from_range(range: Range<usize>) -> Result<Index<usize>> {
        let values: Vec<usize> = range.collect();
        Index::<usize>::new(values)
    }

    /// インデックス長を取得します
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// インデックスが空かどうかを判定します
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// 値から位置を取得します
    ///
    /// # 引数
    /// * `key` - 検索する値
    ///
    /// # 戻り値
    /// 値が見つかった場合はその位置（インデックス）、見つからなかった場合はNone
    pub fn get_loc(&self, key: &T) -> Option<usize> {
        self.map.get(key).copied()
    }

    /// 位置から値を取得します
    ///
    /// # 引数
    /// * `pos` - 検索する位置
    ///
    /// # 戻り値
    /// 位置が有効な場合はその値への参照、範囲外の場合はNone
    pub fn get_value(&self, pos: usize) -> Option<&T> {
        self.values.get(pos)
    }

    /// 全ての値を取得します
    pub fn values(&self) -> &[T] {
        &self.values
    }
    
    /// インデックス名を取得します
    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }
    
    /// インデックス名を設定します
    pub fn set_name(&mut self, name: Option<String>) {
        self.name = name;
    }
    
    /// 新しい名前でインデックスをコピーします
    pub fn rename(&self, name: Option<String>) -> Self {
        let mut new_index = self.clone();
        new_index.name = name;
        new_index
    }
}

/// インデックス型の共通トレイト
///
/// 異なる種類のインデックスに共通の機能を提供します。
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
///
/// 単一レベルのインデックスとマルチレベルのインデックスを統一的に扱うための列挙型です。
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
    /// シンプルインデックスから作成します
    pub fn from_simple(index: Index<T>) -> Self {
        DataFrameIndex::Simple(index)
    }

    /// マルチインデックスから作成します
    pub fn from_multi(index: MultiIndex<T>) -> Self {
        DataFrameIndex::Multi(index)
    }

    /// デフォルトインデックスを作成します
    ///
    /// # 引数
    /// * `len` - インデックスの長さ
    ///
    /// # 戻り値
    /// 0からlen-1までの連番を持つデフォルトインデックス
    pub fn default_with_len(len: usize) -> Result<DataFrameIndex<String>> {
        let range_idx = Index::<usize>::from_range(0..len)?;
        let string_values: Vec<String> = range_idx.values().iter().map(|i| i.to_string()).collect();
        let string_idx = Index::<String>::new(string_values)?;
        Ok(DataFrameIndex::Simple(string_idx))
    }

    /// インデックスがマルチインデックスかどうかを判定します
    pub fn is_multi(&self) -> bool {
        matches!(self, DataFrameIndex::Multi(_))
    }
}

/// 整数インデックス型のエイリアス
pub type RangeIndex = Index<usize>;

/// 文字列インデックス型のエイリアス
pub type StringIndex = Index<String>;

/// 日付時刻変換のための拡張メソッド
impl StringIndex {
    /// インデックス値を日付時刻の配列に変換
    /// 
    /// インデックスが日付文字列を含む場合、それらをNaiveDateTimeに変換します。
    /// 変換できない場合は現在の日付時刻を使用します。
    pub fn to_datetime_vec(&self) -> Result<Vec<NaiveDate>> {
        let mut result = Vec::with_capacity(self.len());
        
        for value in &self.values {
            // 日付形式の文字列を解析
            match NaiveDate::parse_from_str(value, "%Y-%m-%d") {
                Ok(date) => result.push(date),
                Err(_) => {
                    // 日付解析に失敗した場合は現在の日付を使用
                    result.push(NaiveDate::parse_from_str("2023-01-01", "%Y-%m-%d").unwrap_or_else(|_| NaiveDate::now()));
                }
            }
        }
        
        Ok(result)
    }
}

/// DataFrameIndexの拡張メソッド
impl DataFrameIndex<String> {
    /// インデックス値を日付時刻の配列に変換
    pub fn to_datetime_vec(&self) -> Result<Vec<NaiveDate>> {
        match self {
            DataFrameIndex::Simple(idx) => idx.to_datetime_vec(),
            DataFrameIndex::Multi(_) => {
                // マルチインデックスの場合、最初のレベルを使用
                Err(PandRSError::NotImplemented(
                    "マルチインデックスの日付時刻変換は現在サポートされていません".to_string()
                ))
            }
        }
    }
    
    /// インデックスの文字列値を取得
    pub fn string_values(&self) -> Option<Vec<String>> {
        match self {
            DataFrameIndex::Simple(idx) => {
                Some(idx.values().iter().map(|v| v.clone()).collect())
            },
            DataFrameIndex::Multi(multi_idx) => {
                // マルチインデックスは簡易文字列表現を返す
                Some(multi_idx.tuples().iter()
                    .map(|tuple| tuple.join(", "))
                    .collect())
            }
        }
    }
}