mod na_series;
mod categorical;

use num_traits::NumCast;
use std::cmp::PartialOrd;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Sub};
use std::str::FromStr;

use crate::error::{PandRSError, Result};
use crate::index::{Index, RangeIndex};
use crate::na::NA;

pub use self::na_series::NASeries;
pub use self::categorical::{Categorical, CategoricalOrder, StringCategorical};

/// Series構造体: 一次元の値の配列
#[derive(Debug, Clone)]
pub struct Series<T>
where
    T: Debug + Clone,
{
    /// Seriesのデータ値
    values: Vec<T>,

    /// インデックスラベル
    index: RangeIndex,

    /// 名前（オプション）
    name: Option<String>,
}

// 基本実装
impl<T> Series<T>
where
    T: Debug + Clone,
{
    /// 新しいSeriesをベクトルから作成
    pub fn new(values: Vec<T>, name: Option<String>) -> Result<Self> {
        let len = values.len();
        let index = RangeIndex::from_range(0..len)?;

        Ok(Series {
            values,
            index,
            name,
        })
    }

    /// カスタムインデックス付きでSeriesを作成
    pub fn with_index<I>(values: Vec<T>, index: Index<I>, name: Option<String>) -> Result<Self>
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

        Ok(Series {
            values,
            index: range_index,
            name,
        })
    }

    /// Seriesの長さを取得
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Seriesが空かどうか
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// 位置から値を取得
    pub fn get(&self, pos: usize) -> Option<&T> {
        self.values.get(pos)
    }

    /// 値の配列を取得
    pub fn values(&self) -> &[T] {
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
}

// 数値型のSeriesに対する特化実装
impl<T> Series<T>
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
    /// 合計を計算
    pub fn sum(&self) -> T {
        if self.values.is_empty() {
            T::default()
        } else {
            self.values.iter().copied().sum()
        }
    }

    /// 平均を計算
    pub fn mean(&self) -> Result<T> {
        if self.values.is_empty() {
            return Err(PandRSError::Consistency(
                "空のSeriesの平均は計算できません".to_string(),
            ));
        }

        let sum = self.sum();
        let count = match num_traits::cast(self.len()) {
            Some(n) => n,
            None => {
                return Err(PandRSError::Cast(
                    "長さを数値型にキャストできません".to_string(),
                ))
            }
        };

        Ok(sum / count)
    }

    /// 最小値を計算
    pub fn min(&self) -> Result<T> {
        if self.values.is_empty() {
            return Err(PandRSError::Consistency(
                "空のSeriesの最小値は計算できません".to_string(),
            ));
        }

        let min = self
            .values
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .unwrap();

        Ok(min)
    }

    /// 最大値を計算
    pub fn max(&self) -> Result<T> {
        if self.values.is_empty() {
            return Err(PandRSError::Consistency(
                "空のSeriesの最大値は計算できません".to_string(),
            ));
        }

        let max = self
            .values
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .unwrap();

        Ok(max)
    }
}

// 文字列型のSeriesに対する特化実装
impl Series<String> {
    /// 文字列Seriesを数値のNA<f64>ベクトルに変換
    /// 
    /// 各要素を数値に変換します。変換できない場合はNAを返します。
    pub fn to_numeric_vec(&self) -> Result<Vec<NA<f64>>> {
        let mut result = Vec::with_capacity(self.len());
        
        for value in &self.values {
            match value.parse::<f64>() {
                Ok(num) => result.push(NA::Value(num)),
                Err(_) => result.push(NA::NA),
            }
        }
        
        Ok(result)
    }
    
    /// 各要素に関数を適用
    pub fn apply_map<F>(&self, f: F) -> Series<String>
    where
        F: Fn(&String) -> String,
    {
        let transformed: Vec<String> = self.values.iter()
            .map(|v| f(v))
            .collect();
        
        Series {
            values: transformed,
            index: self.index.clone(),
            name: self.name.clone(),
        }
    }
}
