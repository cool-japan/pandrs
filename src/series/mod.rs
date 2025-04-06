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
    
    /// 名前を設定（可変参照版）
    pub fn set_name(&mut self, name: String) {
        self.name = Some(name);
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
    
    /// Series<T>をSeries<String>に変換するメソッド
    pub fn to_string_series(&self) -> Result<Series<String>> 
    where 
        T: std::fmt::Display,
    {
        let string_values: Vec<String> = self.values.iter().map(|v| v.to_string()).collect();
        Series::new(string_values, self.name.clone())
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

// Seriesに対するDefaultトレイト実装
impl<T> Default for Series<T>
where
    T: Debug + Clone,
    Vec<T>: Default,
{
    fn default() -> Self {
        Series {
            values: Vec::default(),
            index: RangeIndex::from_range(0..0).unwrap(),
            name: None,
        }
    }
}

// 数値型Seriesに統計関数を追加（f64特化版）
impl Series<f64> {
    /// 分散を計算（標本分散）
    /// 
    /// 不偏分散を計算します（n-1で割る）
    pub fn var(&self) -> Result<f64> {
        if self.values.is_empty() {
            return Err(PandRSError::Consistency(
                "空のSeriesの分散は計算できません".to_string(),
            ));
        }
        
        if self.values.len() == 1 {
            return Err(PandRSError::Consistency(
                "1つの要素しかないSeriesの分散は定義されていません".to_string(),
            ));
        }
        
        let mean = self.mean()?;
        let sum_squared_diff: f64 = self.values.iter()
            .map(|&v| (v - mean).powi(2))
            .sum();
        
        // 不偏分散（標本分散）を計算
        Ok(sum_squared_diff / (self.values.len() - 1) as f64)
    }
    
    /// 母分散を計算
    /// 
    /// 母分散を計算します（nで割る）
    pub fn var_pop(&self) -> Result<f64> {
        if self.values.is_empty() {
            return Err(PandRSError::Consistency(
                "空のSeriesの分散は計算できません".to_string(),
            ));
        }
        
        let mean = self.mean()?;
        let sum_squared_diff: f64 = self.values.iter()
            .map(|&v| (v - mean).powi(2))
            .sum();
        
        // 母分散を計算
        Ok(sum_squared_diff / self.values.len() as f64)
    }
    
    /// 標準偏差を計算（標本標準偏差）
    pub fn std(&self) -> Result<f64> {
        Ok(self.var()?.sqrt())
    }
    
    /// 母標準偏差を計算
    pub fn std_pop(&self) -> Result<f64> {
        Ok(self.var_pop()?.sqrt())
    }
    
    /// 分位数を計算
    /// 
    /// q: 0.0から1.0の間の分位数（0.5は中央値）
    pub fn quantile(&self, q: f64) -> Result<f64> {
        if self.values.is_empty() {
            return Err(PandRSError::Consistency(
                "空のSeriesの分位数は計算できません".to_string(),
            ));
        }
        
        if q < 0.0 || q > 1.0 {
            return Err(PandRSError::InvalidInput(
                "分位数は0.0から1.0の間である必要があります".to_string(),
            ));
        }
        
        let mut sorted_values = self.values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        if q == 0.0 {
            return Ok(sorted_values[0]);
        }
        
        if q == 1.0 {
            return Ok(sorted_values[sorted_values.len() - 1]);
        }
        
        let pos = q * (sorted_values.len() - 1) as f64;
        let idx_lower = pos.floor() as usize;
        let idx_upper = pos.ceil() as usize;
        
        if idx_lower == idx_upper {
            Ok(sorted_values[idx_lower])
        } else {
            let weight_upper = pos - idx_lower as f64;
            let weight_lower = 1.0 - weight_upper;
            Ok(weight_lower * sorted_values[idx_lower] + weight_upper * sorted_values[idx_upper])
        }
    }
    
    /// 中央値を計算
    pub fn median(&self) -> Result<f64> {
        self.quantile(0.5)
    }
    
    /// 尖度を計算
    pub fn kurtosis(&self) -> Result<f64> {
        if self.values.len() < 4 {
            return Err(PandRSError::Consistency(
                "尖度を計算するには少なくとも4つのデータポイントが必要です".to_string(),
            ));
        }
        
        let mean = self.mean()?;
        let n = self.values.len() as f64;
        
        let m4: f64 = self.values.iter()
            .map(|&x| (x - mean).powi(4))
            .sum::<f64>() / n;
            
        let m2: f64 = self.values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / n;
            
        // エクセス尖度（正規分布の尖度3を引いた値）
        let kurtosis = m4 / m2.powi(2) - 3.0;
        
        Ok(kurtosis)
    }
    
    /// 歪度を計算
    pub fn skewness(&self) -> Result<f64> {
        if self.values.len() < 3 {
            return Err(PandRSError::Consistency(
                "歪度を計算するには少なくとも3つのデータポイントが必要です".to_string(),
            ));
        }
        
        let mean = self.mean()?;
        let n = self.values.len() as f64;
        
        let m3: f64 = self.values.iter()
            .map(|&x| (x - mean).powi(3))
            .sum::<f64>() / n;
            
        let m2: f64 = self.values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / n;
            
        let skewness = m3 / m2.powf(1.5);
        
        Ok(skewness)
    }
    
    /// 別のSeriesとの共分散を計算
    pub fn cov(&self, other: &Series<f64>) -> Result<f64> {
        if self.values.is_empty() || other.values.is_empty() {
            return Err(PandRSError::Consistency(
                "空のSeriesとの共分散は計算できません".to_string(),
            ));
        }
        
        if self.values.len() != other.values.len() {
            return Err(PandRSError::Consistency(
                "共分散を計算するには両方のSeriesが同じ長さである必要があります".to_string(),
            ));
        }
        
        let mean_x = self.mean()?;
        let mean_y = other.mean()?;
        let n = self.values.len() as f64;
        
        let sum_xy = self.values.iter().zip(other.values.iter())
            .map(|(&x, &y)| (x - mean_x) * (y - mean_y))
            .sum::<f64>();
            
        // 不偏共分散推定量を使用
        Ok(sum_xy / (n - 1.0))
    }
    
    /// 別のSeriesとの相関係数を計算
    pub fn corr(&self, other: &Series<f64>) -> Result<f64> {
        let cov = self.cov(other)?;
        let std_x = self.std()?;
        let std_y = other.std()?;
        
        if std_x == 0.0 || std_y == 0.0 {
            return Err(PandRSError::Consistency(
                "標準偏差が0の場合、相関係数は計算できません".to_string(),
            ));
        }
        
        Ok(cov / (std_x * std_y))
    }
    
    /// 記述統計を一括取得
    pub fn describe(&self) -> Result<crate::stats::DescriptiveStats> {
        crate::stats::describe(self.values())
    }
}

impl Series<String> {
    /// 文字列Seriesを数値の`NA<f64>`ベクトルに変換
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
