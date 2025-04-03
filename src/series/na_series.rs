use num_traits::NumCast;
use std::cmp::PartialOrd;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Sub};

use crate::error::{PandRSError, Result};
use crate::index::{Index, RangeIndex};
use crate::na::NA;

/// 欠損値をサポートするSeries構造体
#[derive(Debug, Clone)]
pub struct NASeries<T>
where
    T: Debug + Clone,
{
    /// Seriesのデータ値（NA型でラップ）
    values: Vec<NA<T>>,

    /// インデックスラベル
    index: RangeIndex,

    /// 名前（オプション）
    name: Option<String>,
}

impl<T> NASeries<T>
where
    T: Debug + Clone,
{
    /// 新しいNASeriesをベクトルから作成
    pub fn new(values: Vec<NA<T>>, name: Option<String>) -> Result<Self> {
        let len = values.len();
        let index = RangeIndex::from_range(0..len)?;

        Ok(NASeries {
            values,
            index,
            name,
        })
    }
    
    /// 文字列ベクトルからNASeriesを作成するヘルパー関数
    pub fn from_strings(string_values: Vec<String>, name: Option<String>) -> Result<NASeries<String>> {
        let na_values = string_values.into_iter()
            .map(|s| {
                if s.contains("NA") {
                    NA::<String>::NA
                } else {
                    NA::Value(s)
                }
            })
            .collect();
        NASeries::<String>::new(na_values, name)
    }

    /// 通常のベクトルから作成（NAを含まない）
    pub fn from_vec(values: Vec<T>, name: Option<String>) -> Result<Self> {
        let na_values = values.into_iter().map(NA::Value).collect();
        Self::new(na_values, name)
    }

    /// Optionベクトルから作成（Noneを含む可能性あり）
    pub fn from_options(values: Vec<Option<T>>, name: Option<String>) -> Result<Self> {
        let na_values = values
            .into_iter()
            .map(|opt| match opt {
                Some(v) => NA::Value(v),
                None => NA::NA,
            })
            .collect();
        Self::new(na_values, name)
    }

    /// カスタムインデックス付きでNASeriesを作成
    pub fn with_index<I>(values: Vec<NA<T>>, index: Index<I>, name: Option<String>) -> Result<Self>
    where
        I: Debug + Clone + Eq + std::hash::Hash + std::fmt::Display,
    {
        if values.len() != index.len() {
            return Err(PandRSError::Consistency(format!(
                "値の長さ ({}) とインデックスの長さ ({}) が一致しません",
                values.len(),
                index.len()
            )));
        }

        // 現状では整数インデックスしかサポートしていない
        let range_index = RangeIndex::from_range(0..values.len())?;

        Ok(NASeries {
            values,
            index: range_index,
            name,
        })
    }

    /// NASeriesの長さを取得
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// NASeriesが空かどうか
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// 位置から値を取得
    pub fn get(&self, pos: usize) -> Option<&NA<T>> {
        self.values.get(pos)
    }

    /// 値の配列を取得
    pub fn values(&self) -> &[NA<T>] {
        &self.values
    }

    /// 名前を取得
    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }

    /// インデックスを取得
    pub fn index(&self) -> &RangeIndex {
        &self.index
    }

    /// 名前を設定
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }
    
    /// 名前を設定（可変参照）
    pub fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }

    /// NAの個数を取得
    pub fn na_count(&self) -> usize {
        self.values.iter().filter(|v| v.is_na()).count()
    }

    /// 値が存在する個数を取得
    pub fn value_count(&self) -> usize {
        self.values.iter().filter(|v| v.is_value()).count()
    }

    /// NAを含むかどうか
    pub fn has_na(&self) -> bool {
        self.values.iter().any(|v| v.is_na())
    }
    
    /// 各要素がNAかどうかのブール配列を取得
    pub fn is_na(&self) -> Vec<bool> {
        self.values.iter().map(|v| v.is_na()).collect()
    }

    /// NAを除去したSeriesを返す
    pub fn dropna(&self) -> Result<Self> {
        let filtered_values: Vec<NA<T>> = self
            .values
            .iter()
            .filter(|v| v.is_value())
            .cloned()
            .collect();

        Self::new(filtered_values, self.name.clone())
    }

    /// NAを指定した値で埋める
    pub fn fillna(&self, fill_value: T) -> Result<Self> {
        let filled_values: Vec<NA<T>> = self
            .values
            .iter()
            .map(|v| match v {
                NA::Value(_) => v.clone(),
                NA::NA => NA::Value(fill_value.clone()),
            })
            .collect();

        Self::new(filled_values, self.name.clone())
    }
}

// 数値型のNASeriesに対する特化実装
impl<T> NASeries<T>
where
    T: Debug
        + Clone
        + Copy
        + Sum<T>
        + PartialOrd
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + NumCast
        + Default,
{
    /// 合計を計算（NAは無視）
    pub fn sum(&self) -> NA<T> {
        let values: Vec<T> = self
            .values
            .iter()
            .filter_map(|v| match v {
                NA::Value(val) => Some(*val),
                NA::NA => None,
            })
            .collect();

        if values.is_empty() {
            NA::NA
        } else {
            NA::Value(values.into_iter().sum())
        }
    }

    /// 平均を計算（NAは無視）
    pub fn mean(&self) -> NA<T> {
        let values: Vec<T> = self
            .values
            .iter()
            .filter_map(|v| match v {
                NA::Value(val) => Some(*val),
                NA::NA => None,
            })
            .collect();

        if values.is_empty() {
            return NA::NA;
        }

        let sum: T = values.iter().copied().sum();
        let count: T = match num_traits::cast(values.len()) {
            Some(n) => n,
            None => return NA::NA,
        };

        NA::Value(sum / count)
    }

    /// 最小値を計算（NAは無視）
    pub fn min(&self) -> NA<T> {
        let values: Vec<T> = self
            .values
            .iter()
            .filter_map(|v| match v {
                NA::Value(val) => Some(*val),
                NA::NA => None,
            })
            .collect();

        if values.is_empty() {
            return NA::NA;
        }

        let min = values
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .unwrap();

        NA::Value(min)
    }

    /// 最大値を計算（NAは無視）
    pub fn max(&self) -> NA<T> {
        let values: Vec<T> = self
            .values
            .iter()
            .filter_map(|v| match v {
                NA::Value(val) => Some(*val),
                NA::NA => None,
            })
            .collect();

        if values.is_empty() {
            return NA::NA;
        }

        let max = values
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .unwrap();

        NA::Value(max)
    }
}
