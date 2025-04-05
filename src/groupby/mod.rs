use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use crate::dataframe::DataFrame;
use crate::error::{PandRSError, Result};
use crate::series::Series;

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
                keys.len(),
                source.len()
            )));
        }

        // グループを作成
        let mut groups = HashMap::new();
        for (i, key) in keys.iter().enumerate() {
            groups.entry(key.clone()).or_insert_with(Vec::new).push(i);
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
        self.groups
            .iter()
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
            let values: Vec<T> = indices
                .iter()
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
            let values: Vec<f64> = indices
                .iter()
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
                keys.len(),
                source.row_count()
            )));
        }

        // グループを作成
        let mut groups = HashMap::new();
        for (i, key) in keys.iter().enumerate() {
            groups.entry(key.clone()).or_insert_with(Vec::new).push(i);
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
        self.groups
            .iter()
            .map(|(k, indices)| (k.clone(), indices.len()))
            .collect()
    }

    /// グループ別のサイズをDataFrameとして取得
    pub fn size_as_df(&self) -> Result<DataFrame> {
        // 結果用のDataFrameを作成
        let mut result = DataFrame::new();
        
        // グループキー列と値の作成
        let mut keys = Vec::new();
        let mut sizes = Vec::new();
        
        for (key, indices) in &self.groups {
            keys.push(format!("{:?}", key));  // キーを文字列に変換
            sizes.push(indices.len().to_string());  // サイズを文字列に変換
        }
        
        // グループキー列の追加
        let key_column = Series::new(keys, Some("group_key".to_string()))?;
        result.add_column("group_key".to_string(), key_column)?;
        
        // サイズ列の追加
        let size_column = Series::new(sizes, Some("size".to_string()))?;
        result.add_column("size".to_string(), size_column)?;
        
        Ok(result)
    }
    
    /// 簡易的な集計関数
    pub fn aggregate(&self, column_name: &str, func_name: &str) -> Result<DataFrame> {
        // 列が存在するか確認
        if !self.source.contains_column(column_name) {
            return Err(PandRSError::Column(
                format!("列 '{}' が見つかりません", column_name)
            ));
        }
        
        // 結果用のDataFrameを作成
        let mut result = DataFrame::new();
        
        // グループキー列と値の作成
        let mut keys = Vec::new();
        let mut aggregated_values = Vec::new();
        
        // 列のデータを取得
        let column_data = self.source.get_column_numeric_values(column_name)?;
        
        for (key, indices) in &self.groups {
            // グループのキーを追加
            keys.push(format!("{:?}", key));
            
            // このグループのデータを抽出
            let group_data: Vec<f64> = indices.iter()
                .filter_map(|&idx| {
                    if idx < column_data.len() {
                        Some(column_data[idx])
                    } else {
                        None
                    }
                })
                .collect();
            
            // 集計関数を適用
            let result_value = if group_data.is_empty() {
                "0.0".to_string()
            } else {
                match func_name {
                    "sum" => group_data.iter().sum::<f64>().to_string(),
                    "mean" => (group_data.iter().sum::<f64>() / group_data.len() as f64).to_string(),
                    "min" => group_data.iter().fold(f64::INFINITY, |a, &b| a.min(b)).to_string(),
                    "max" => group_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)).to_string(),
                    "count" => group_data.len().to_string(),
                    _ => "0.0".to_string()
                }
            };
            
            aggregated_values.push(result_value);
        }
        
        // グループキー列の追加
        let key_column = Series::new(keys, Some("group_key".to_string()))?;
        result.add_column("group_key".to_string(), key_column)?;
        
        // 集計結果列の追加
        let result_column_name = format!("{}_{}", column_name, func_name);
        let value_column = Series::new(aggregated_values, Some(result_column_name.clone()))?;
        result.add_column(result_column_name, value_column)?;
        
        Ok(result)
    }
}
