//! 並列処理機能を提供するモジュール

use rayon::prelude::*;
use crate::error::Result;
use crate::DataFrame;
use crate::Series;
use crate::na::NA;
use crate::series::NASeries;

/// 並列処理の拡張: Seriesの並列処理
impl<T> Series<T>
where
    T: Clone + Send + Sync + 'static + std::fmt::Debug,
{
    /// すべての要素に対して並列で関数を適用
    pub fn par_map<F, R>(&self, f: F) -> Series<R>
    where
        F: Fn(&T) -> R + Send + Sync,
        R: Clone + Send + Sync + 'static + std::fmt::Debug,
    {
        let new_values: Vec<R> = self.values()
            .par_iter()
            .map(|v| f(v))
            .collect();
        
        Series::new(new_values, self.name().cloned()).unwrap()
    }
    
    /// 条件関数に基づいて要素を並列でフィルタリング
    pub fn par_filter<F>(&self, f: F) -> Series<T>
    where
        F: Fn(&T) -> bool + Send + Sync,
    {
        let filtered_values: Vec<T> = self.values()
            .par_iter()
            .filter(|v| f(v))
            .cloned()
            .collect();
        
        Series::new(filtered_values, self.name().cloned()).unwrap()
    }
}

/// 並列処理の拡張: 欠損値を含むSeriesの並列処理
impl<T> NASeries<T>
where
    T: Clone + Send + Sync + 'static + std::fmt::Debug,
{
    /// すべての要素に対して並列で関数を適用（NAは無視）
    pub fn par_map<F, R>(&self, f: F) -> NASeries<R>
    where
        F: Fn(&T) -> R + Send + Sync,
        R: Clone + Send + Sync + 'static + std::fmt::Debug,
    {
        let new_values: Vec<NA<R>> = self.values()
            .par_iter()
            .map(|v| match v {
                NA::Value(val) => NA::Value(f(val)),
                NA::NA => NA::NA,
            })
            .collect();
        
        NASeries::new(new_values, self.name().cloned()).unwrap()
    }
    
    /// 条件関数に基づいて要素を並列でフィルタリング（NAは除外）
    pub fn par_filter<F>(&self, f: F) -> NASeries<T>
    where
        F: Fn(&T) -> bool + Send + Sync,
    {
        let filtered_values: Vec<NA<T>> = self.values()
            .par_iter()
            .filter(|v| match v {
                NA::Value(val) => f(val),
                NA::NA => false,
            })
            .cloned()
            .collect();
        
        NASeries::new(filtered_values, self.name().cloned()).unwrap()
    }
}

/// 並列処理の拡張: DataFrameの並列処理
impl DataFrame {
    /// すべての列に対して並列で関数を適用
    pub fn par_apply<F>(&self, f: F) -> Result<DataFrame>
    where
        F: Fn(&str, usize, &str) -> String + Send + Sync,
    {
        let mut result = DataFrame::new();
        
        // 各列を並列処理
        let column_names = self.column_names().to_vec();
        
        // 行数と列名の準備
        let n_rows = self.row_count();
        
        // 各列を処理
        for col_name in &column_names {
            // 文字列値を取得
            let values = self.get_column_string_values(col_name)?;
            
            // 並列処理で新しい値を作成
            let new_values: Vec<String> = (0..n_rows)
                .into_par_iter()
                .map(|i| {
                    let val = if i < values.len() { &values[i] } else { "" };
                    f(col_name, i, val)
                })
                .collect();
            
            // 新しい列を追加
            let new_series = Series::new(new_values, Some(col_name.clone()))?;
            result.add_column(col_name.clone(), new_series)?;
        }
        
        Ok(result)
    }
    
    /// 行のフィルタリングを並列で実行
    pub fn par_filter_rows<F>(&self, f: F) -> Result<DataFrame>
    where
        F: Fn(usize) -> bool + Send + Sync,
    {
        let mut result = DataFrame::new();
        
        // 列名を取得
        let column_names = self.column_names().to_vec();
        
        // パラレルで行インデックスをフィルタリング
        let row_indices: Vec<usize> = (0..self.row_count())
            .into_par_iter()
            .filter(|&i| f(i))
            .collect();
        
        // 各列をフィルタリング
        for col_name in &column_names {
            let values = self.get_column_string_values(col_name)?;
            
            // フィルタリングされた値を取得
            let filtered_values: Vec<String> = row_indices
                .par_iter()
                .filter_map(|&i| {
                    if i < values.len() {
                        Some(values[i].clone())
                    } else {
                        None
                    }
                })
                .collect();
            
            // 新しい列を追加
            let new_series = Series::new(filtered_values, Some(col_name.clone()))?;
            result.add_column(col_name.clone(), new_series)?;
        }
        
        Ok(result)
    }
    
    /// グループ化操作を並列で実行
    pub fn par_groupby<K>(&self, key_func: K) -> Result<HashMap<String, DataFrame>>
    where
        K: Fn(usize) -> String + Send + Sync,
    {
        // グループマップ
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        
        // 行ごとにキーを計算して行インデックスをグループ化
        (0..self.row_count())
            .into_par_iter()
            .map(|i| (key_func(i), i))
            .collect::<Vec<_>>()
            .into_iter() // 結果を直列で処理
            .for_each(|(key, idx)| {
                groups.entry(key).or_insert_with(Vec::new).push(idx);
            });
        
        // 各グループに対してDataFrameを作成
        let mut result = HashMap::new();
        
        for (key, indices) in groups {
            let group_df = self.par_filter_rows(|i| indices.contains(&i))?;
            result.insert(key, group_df);
        }
        
        Ok(result)
    }
}

/// データの並列操作をサポートするためのユーティリティ
pub struct ParallelUtils;

impl ParallelUtils {
    /// ベクトルを並列で並べ替え
    pub fn par_sort<T>(mut values: Vec<T>) -> Vec<T>
    where
        T: Ord + Send,
    {
        values.par_sort();
        values
    }
    
    /// ベクトルの要素を並列で集計
    pub fn par_sum<T>(values: &[T]) -> T
    where
        T: Send + Sync + std::iter::Sum + Copy,
    {
        values.par_iter().copied().sum()
    }
    
    /// ベクトルの平均を並列で計算
    pub fn par_mean<T>(values: &[T]) -> Option<f64>
    where
        T: Send + Sync + Copy + Into<f64>,
    {
        if values.is_empty() {
            return None;
        }
        
        let sum: f64 = values.par_iter()
            .map(|&v| v.into())
            .sum();
        
        Some(sum / values.len() as f64)
    }
    
    /// ベクトルの最小値を並列で検索
    pub fn par_min<T>(values: &[T]) -> Option<T>
    where
        T: Send + Sync + Copy + Ord,
    {
        values.par_iter()
            .min()
            .copied()
    }
    
    /// ベクトルの最大値を並列で検索
    pub fn par_max<T>(values: &[T]) -> Option<T>
    where
        T: Send + Sync + Copy + Ord,
    {
        values.par_iter()
            .max()
            .copied()
    }
}

use std::collections::HashMap;