//! ピボットテーブル機能を提供するモジュール

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

use crate::dataframe::DataFrame;
use crate::error::{PandRSError, Result};
use crate::series::Series;

/// 集計関数の種類
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunction {
    /// 合計
    Sum,
    /// 平均
    Mean,
    /// 最小値
    Min,
    /// 最大値
    Max,
    /// カウント
    Count,
}

impl AggFunction {
    /// 関数名を文字列で取得
    pub fn name(&self) -> &'static str {
        match self {
            AggFunction::Sum => "sum",
            AggFunction::Mean => "mean",
            AggFunction::Min => "min",
            AggFunction::Max => "max",
            AggFunction::Count => "count",
        }
    }

    /// 文字列から集計関数を解析
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "sum" => Some(AggFunction::Sum),
            "mean" | "avg" | "average" => Some(AggFunction::Mean),
            "min" | "minimum" => Some(AggFunction::Min),
            "max" | "maximum" => Some(AggFunction::Max),
            "count" => Some(AggFunction::Count),
            _ => None,
        }
    }
}

/// ピボットテーブルを作成するための構造体
#[derive(Debug)]
pub struct PivotTable<'a> {
    /// 元のDataFrame
    df: &'a DataFrame,

    /// インデックスとなる列名
    index: String,

    /// カラムとなる列名
    columns: String,

    /// 値となる列名
    values: String,

    /// 集計関数
    aggfunc: AggFunction,
}

impl<'a> PivotTable<'a> {
    /// 新しいピボットテーブルを作成
    pub fn new(
        df: &'a DataFrame,
        index: String,
        columns: String,
        values: String,
        aggfunc: AggFunction,
    ) -> Result<Self> {
        // 必要な列が存在するか確認
        if !df.contains_column(&index) {
            return Err(PandRSError::Column(format!(
                "インデックス列 '{}' が見つかりません",
                index
            )));
        }
        if !df.contains_column(&columns) {
            return Err(PandRSError::Column(format!(
                "カラム列 '{}' が見つかりません",
                columns
            )));
        }
        if !df.contains_column(&values) {
            return Err(PandRSError::Column(format!(
                "値列 '{}' が見つかりません",
                values
            )));
        }

        Ok(PivotTable {
            df,
            index,
            columns,
            values,
            aggfunc,
        })
    }

    /// ピボットテーブルを実行して新しいDataFrameを生成
    pub fn execute(&self) -> Result<DataFrame> {
        // 一意のインデックス値とカラム値を収集
        let mut index_values: HashSet<String> = HashSet::new();
        let mut column_values: HashSet<String> = HashSet::new();

        // インデックス列と列データを取得
        let index_values_vec = self.df.get_column_string_values(&self.index)?;
        let column_values_vec = self.df.get_column_string_values(&self.columns)?;
        let values_data_vec = self.df.get_column_numeric_values(&self.values)?;

        // 一意の値を収集
        for val in &index_values_vec {
            index_values.insert(val.clone());
        }

        for val in &column_values_vec {
            column_values.insert(val.clone());
        }

        // 結果DataFrame用の列を作成
        let mut result_df = DataFrame::new();

        // インデックス列を追加
        let empty_index_values: Vec<String> = Vec::new();
        let empty_index_series = Series::new(empty_index_values, Some("index".to_string()))?;
        result_df.add_column(self.index.clone(), empty_index_series)?;

        // カラム値を結果DataFrameの列として追加
        for column_val in &column_values {
            let empty_column_values: Vec<String> = Vec::new();
            let empty_series = Series::new(empty_column_values, Some(column_val.clone()))?;
            result_df.add_column(column_val.clone(), empty_series)?;
        }

        // 集計データを格納するマップを作成
        // (インデックス値, カラム値) -> 集計値のリスト
        let mut aggregation_map: HashMap<(String, String), Vec<f64>> = HashMap::new();

        // データを収集
        for i in 0..self.df.row_count() {
            if i < index_values_vec.len()
                && i < column_values_vec.len()
                && i < values_data_vec.len()
            {
                let index_val = &index_values_vec[i];
                let column_val = &column_values_vec[i];
                let value = values_data_vec[i];

                let key = (index_val.clone(), column_val.clone());

                aggregation_map
                    .entry(key)
                    .or_insert_with(Vec::new)
                    .push(value);
            }
        }

        // インデックス行ごとにカラム列に対応する値を集計
        for index_val in &index_values {
            // 各カラム列の値について集計
            let mut row_data: HashMap<String, String> = HashMap::new();
            row_data.insert(self.index.clone(), index_val.clone());

            for column_val in &column_values {
                // 特定のインデックス値とカラム値に対応するデータを集計
                let key = (index_val.clone(), column_val.clone());

                if let Some(values) = aggregation_map.get(&key) {
                    let agg_value = self.aggregate_values_from_vec(values)?;
                    let agg_value_str = agg_value.to_string();

                    // 結果に追加
                    row_data.insert(column_val.clone(), agg_value_str);
                } else {
                    // データがない場合は空文字列
                    row_data.insert(column_val.clone(), String::new());
                }
            }

            // TODO: 行データをDataFrameに追加
            // 現在の実装では行追加の具体的なメソッドがないため、
            // 将来的には行追加のメソッドを実装する必要がある
        }

        Ok(result_df)
    }

    /// 特定のインデックス値とカラム値に対応するデータを集計
    fn aggregate_values_from_vec(&self, values: &[f64]) -> Result<f64> {
        if values.is_empty() {
            return Ok(0.0);
        }

        match self.aggfunc {
            AggFunction::Sum => Ok(values.iter().sum()),
            AggFunction::Mean => {
                let sum: f64 = values.iter().sum();
                Ok(sum / values.len() as f64)
            }
            AggFunction::Min => {
                if let Some(min) = values.iter().fold(None, |min, &x| match min {
                    None => Some(x),
                    Some(y) => Some(if x < y { x } else { y }),
                }) {
                    Ok(min)
                } else {
                    Ok(0.0)
                }
            }
            AggFunction::Max => {
                if let Some(max) = values.iter().fold(None, |max, &x| match max {
                    None => Some(x),
                    Some(y) => Some(if x > y { x } else { y }),
                }) {
                    Ok(max)
                } else {
                    Ok(0.0)
                }
            }
            AggFunction::Count => Ok(values.len() as f64),
        }
    }
}

/// DataFrameの拡張: ピボットテーブル機能
impl DataFrame {
    /// ピボットテーブルを作成
    pub fn pivot_table(
        &self,
        index: &str,
        columns: &str,
        values: &str,
        aggfunc: AggFunction,
    ) -> Result<DataFrame> {
        let pivot = PivotTable::new(
            self,
            index.to_string(),
            columns.to_string(),
            values.to_string(),
            aggfunc,
        )?;

        pivot.execute()
    }

    /// 指定された列でのグループ化
    pub fn groupby(&self, by: &str) -> Result<GroupBy> {
        if !self.contains_column(by) {
            return Err(PandRSError::Column(format!(
                "グループ化列 '{}' が見つかりません",
                by
            )));
        }

        Ok(GroupBy {
            df: self,
            by: by.to_string(),
        })
    }
}

/// グループ化操作を表す構造体
#[derive(Debug)]
pub struct GroupBy<'a> {
    /// 元のDataFrame
    df: &'a DataFrame,

    /// グループ化する列名
    by: String,
}

impl<'a> GroupBy<'a> {
    /// 集計操作を実行
    pub fn agg(&self, columns: &[&str], aggfunc: AggFunction) -> Result<DataFrame> {
        // 各カラムが存在するか確認
        for col in columns {
            if !self.df.contains_column(col) {
                return Err(PandRSError::Column(format!(
                    "集計列 '{}' が見つかりません",
                    col
                )));
            }
        }

        // 結果DataFrame用の列を作成
        let mut result_df = DataFrame::new();

        // グループ化キーの列を取得
        let group_keys = self.df.get_column_string_values(&self.by)?;

        // 一意のグループキーを収集
        let mut unique_keys: HashSet<String> = HashSet::new();
        for key in &group_keys {
            unique_keys.insert(key.clone());
        }

        // 結果DataFrameに列を追加
        // グループキー列を追加
        let empty_key_values: Vec<String> = Vec::new();
        let empty_key_series = Series::new(empty_key_values, Some(self.by.clone()))?;
        result_df.add_column(self.by.clone(), empty_key_series)?;

        // 集計する各列の結果列を追加
        for &col in columns {
            let col_name = format!("{}_{}", col, aggfunc.name());
            let empty_values: Vec<String> = Vec::new();
            let empty_series = Series::new(empty_values, Some(col_name.clone()))?;
            result_df.add_column(col_name, empty_series)?;
        }

        // グループごとに集計
        for group_key in &unique_keys {
            // 各グループの行インデックスを収集
            let mut group_indices = Vec::new();
            for (i, key) in group_keys.iter().enumerate() {
                if key == group_key {
                    group_indices.push(i);
                }
            }

            // 各列について集計
            let mut row_data = HashMap::new();
            row_data.insert(self.by.clone(), group_key.clone());

            for &col in columns {
                let values = self.df.get_column_numeric_values(col)?;

                // グループの値だけを取得
                let group_values: Vec<f64> = group_indices
                    .iter()
                    .filter_map(|&idx| {
                        if idx < values.len() {
                            Some(values[idx])
                        } else {
                            None
                        }
                    })
                    .collect();

                // 集計関数を適用
                let agg_value = match aggfunc {
                    AggFunction::Sum => group_values.iter().sum(),
                    AggFunction::Mean => {
                        if group_values.is_empty() {
                            0.0
                        } else {
                            group_values.iter().sum::<f64>() / group_values.len() as f64
                        }
                    }
                    AggFunction::Min => group_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                    AggFunction::Max => group_values
                        .iter()
                        .fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                    AggFunction::Count => group_values.len() as f64,
                };

                // 結果を格納
                let col_name = format!("{}_{}", col, aggfunc.name());
                row_data.insert(col_name, agg_value.to_string());
            }

            // TODO: 行データをDataFrameに追加
            // 現在の実装では行の直接追加はサポートされていないため、
            // 将来的には行追加のメソッドを実装する必要がある
        }

        Ok(result_df)
    }

    /// 合計を計算
    pub fn sum(&self, columns: &[&str]) -> Result<DataFrame> {
        self.agg(columns, AggFunction::Sum)
    }

    /// 平均を計算
    pub fn mean(&self, columns: &[&str]) -> Result<DataFrame> {
        self.agg(columns, AggFunction::Mean)
    }

    /// 最小値を計算
    pub fn min(&self, columns: &[&str]) -> Result<DataFrame> {
        self.agg(columns, AggFunction::Min)
    }

    /// 最大値を計算
    pub fn max(&self, columns: &[&str]) -> Result<DataFrame> {
        self.agg(columns, AggFunction::Max)
    }

    /// カウントを計算
    pub fn count(&self, columns: &[&str]) -> Result<DataFrame> {
        self.agg(columns, AggFunction::Count)
    }
}
