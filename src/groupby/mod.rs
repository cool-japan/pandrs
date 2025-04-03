use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use crate::error::{PandRSError, Result};
use crate::series::Series;
use crate::dataframe::DataFrame;

/// グループ化した結果を表す構造体
#[derive(Debug)]
pub struct GroupBy<'a, K, T>
where
    K: Debug + Eq + Hash + Clone,
    T: Debug + Clone,
{
    // アンダースコアをつけて未使用フィールドの警告を抑制
    #[allow(dead_code)]
    /// グループのキー
    keys: Vec<K>,
    
    /// グループ化された値
    groups: HashMap<K, Vec<usize>>,
    
    /// 元のシリーズ
    source: &'a Series<T>,
    
    // アンダースコアをつけて未使用フィールドの警告を抑制
    #[allow(dead_code)]
    /// グループ名
    name: Option<String>,
}

impl<'a, K, T> GroupBy<'a, K, T>
where
    K: Debug + Eq + Hash + Clone,
    T: Debug + Clone,
{
    /// 新しいグループを作成
    pub fn new(keys: Vec<K>, source: &'a Series<T>, name: Option<String>) -> Result<Self> {
        // キーとソースの長さが一致するか確認
        if keys.len() != source.len() {
            return Err(PandRSError::Consistency(format!(
                "キーの長さ ({}) とソースの長さ ({}) が一致しません",
                keys.len(), source.len()
            )));
        }
        
        // グループを作成
        let mut groups = HashMap::new();
        for (i, key) in keys.iter().enumerate() {
            groups.entry(key.clone())
                .or_insert_with(Vec::new)
                .push(i);
        }
        
        Ok(GroupBy {
            keys,
            groups,
            source,
            name,
        })
    }
    
    /// グループ数を取得
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }
    
    /// 各グループのサイズを返す
    pub fn size(&self) -> HashMap<K, usize> {
        self.groups.iter()
            .map(|(k, indices)| (k.clone(), indices.len()))
            .collect()
    }
    
    /// 各グループの合計を計算
    pub fn sum(&self) -> Result<HashMap<K, T>>
    where
        T: Copy + std::iter::Sum,
    {
        let mut results = HashMap::new();
        
        for (key, indices) in &self.groups {
            let values: Vec<T> = indices.iter()
                .filter_map(|&i| self.source.get(i).cloned())
                .collect();
            
            if !values.is_empty() {
                results.insert(key.clone(), values.into_iter().sum());
            }
        }
        
        Ok(results)
    }
    
    /// 各グループの平均を計算
    pub fn mean(&self) -> Result<HashMap<K, f64>>
    where
        T: Copy + Into<f64>,
    {
        let mut results = HashMap::new();
        
        for (key, indices) in &self.groups {
            let values: Vec<f64> = indices.iter()
                .filter_map(|&i| self.source.get(i).map(|&v| v.into()))
                .collect();
            
            if !values.is_empty() {
                let sum: f64 = values.iter().sum();
                let mean = sum / values.len() as f64;
                results.insert(key.clone(), mean);
            }
        }
        
        Ok(results)
    }
}

/// DataFrameのグループ化機能
pub struct DataFrameGroupBy<'a, K>
where
    K: Debug + Eq + Hash + Clone,
{
    // アンダースコアをつけて未使用フィールドの警告を抑制
    #[allow(dead_code)]
    /// グループのキー
    keys: Vec<K>,
    
    /// グループ化された行インデックス
    groups: HashMap<K, Vec<usize>>,
    
    // アンダースコアをつけて未使用フィールドの警告を抑制
    #[allow(dead_code)]
    /// 元のDataFrame
    source: &'a DataFrame,
    
    // アンダースコアをつけて未使用フィールドの警告を抑制
    #[allow(dead_code)]
    /// グループ化に使用した列名
    by: String,
}

impl<'a, K> DataFrameGroupBy<'a, K>
where
    K: Debug + Eq + Hash + Clone,
{
    /// 新しいDataFrameグループを作成
    pub fn new(keys: Vec<K>, source: &'a DataFrame, by: String) -> Result<Self> {
        // キーと行数が一致するか確認
        if keys.len() != source.row_count() {
            return Err(PandRSError::Consistency(format!(
                "キーの長さ ({}) とDataFrameの行数 ({}) が一致しません",
                keys.len(), source.row_count()
            )));
        }
        
        // グループを作成
        let mut groups = HashMap::new();
        for (i, key) in keys.iter().enumerate() {
            groups.entry(key.clone())
                .or_insert_with(Vec::new)
                .push(i);
        }
        
        Ok(DataFrameGroupBy {
            keys,
            groups,
            source,
            by,
        })
    }
    
    /// グループ数を取得
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }
    
    /// 各グループのサイズを返す
    pub fn size(&self) -> HashMap<K, usize> {
        self.groups.iter()
            .map(|(k, indices)| (k.clone(), indices.len()))
            .collect()
    }
    
    // TODO: DataFrame操作のメソッドを追加
}