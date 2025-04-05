//! 前処理モジュール
//!
//! 機械学習のための特徴量エンジニアリングと前処理機能を提供します。

use crate::dataframe::DataFrame;
use crate::error::Result;
use crate::ml::pipeline::Transformer;
use crate::series::Series;
use std::collections::HashMap;

/// 数値データを標準化するための変換器
pub struct StandardScaler {
    /// 各列の平均値
    means: HashMap<String, f64>,
    /// 各列の標準偏差
    stds: HashMap<String, f64>,
    /// 変換対象の列
    columns: Vec<String>,
}

impl StandardScaler {
    /// 新しいStandardScalerを作成
    pub fn new(columns: Vec<String>) -> Self {
        StandardScaler {
            means: HashMap::new(),
            stds: HashMap::new(),
            columns,
        }
    }
}

impl Transformer for StandardScaler {
    fn fit(&mut self, df: &DataFrame) -> Result<()> {
        for col_name in &self.columns {
            if let Some(series) = df.column(col_name) {
                let mean = series.mean()?;
                let std = series.std()?;
                self.means.insert(col_name.clone(), mean);
                self.stds.insert(col_name.clone(), std);
            }
        }
        Ok(())
    }
    
    fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        let mut result = df.clone();
        
        for col_name in &self.columns {
            if let (Some(mean), Some(std)) = (self.means.get(col_name), self.stds.get(col_name)) {
                if let Some(series) = df.column(col_name) {
                    let scaled_series = series.map(|x| match x {
                        crate::na::DataValue::Float64(v) => {
                            if *std > 0.0 {
                                crate::na::DataValue::Float64((v - mean) / std)
                            } else {
                                crate::na::DataValue::Float64(0.0)
                            }
                        }
                        crate::na::DataValue::Int64(v) => {
                            if *std > 0.0 {
                                crate::na::DataValue::Float64((v as f64 - mean) / std)
                            } else {
                                crate::na::DataValue::Float64(0.0)
                            }
                        }
                        x => x,
                    })?;
                    
                    result.replace_column(col_name.clone(), scaled_series)?;
                }
            }
        }
        
        Ok(result)
    }
    
    fn fit_transform(&mut self, df: &DataFrame) -> Result<DataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

/// 数値データを[0,1]の範囲に正規化するための変換器
pub struct MinMaxScaler {
    /// 各列の最小値
    mins: HashMap<String, f64>,
    /// 各列の最大値
    maxs: HashMap<String, f64>,
    /// 変換対象の列
    columns: Vec<String>,
}

impl MinMaxScaler {
    /// 新しいMinMaxScalerを作成
    pub fn new(columns: Vec<String>) -> Self {
        MinMaxScaler {
            mins: HashMap::new(),
            maxs: HashMap::new(),
            columns,
        }
    }
}

impl Transformer for MinMaxScaler {
    fn fit(&mut self, df: &DataFrame) -> Result<()> {
        for col_name in &self.columns {
            if let Some(series) = df.column(col_name) {
                let min = series.min()?;
                let max = series.max()?;
                
                match (min, max) {
                    (crate::na::DataValue::Float64(min_val), crate::na::DataValue::Float64(max_val)) => {
                        self.mins.insert(col_name.clone(), min_val);
                        self.maxs.insert(col_name.clone(), max_val);
                    }
                    (crate::na::DataValue::Int64(min_val), crate::na::DataValue::Int64(max_val)) => {
                        self.mins.insert(col_name.clone(), min_val as f64);
                        self.maxs.insert(col_name.clone(), max_val as f64);
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }
    
    fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        let mut result = df.clone();
        
        for col_name in &self.columns {
            if let (Some(min), Some(max)) = (self.mins.get(col_name), self.maxs.get(col_name)) {
                let range = max - min;
                
                if let Some(series) = df.column(col_name) {
                    let scaled_series = series.map(|x| match x {
                        crate::na::DataValue::Float64(v) => {
                            if range > 0.0 {
                                crate::na::DataValue::Float64((v - min) / range)
                            } else {
                                crate::na::DataValue::Float64(0.5)
                            }
                        }
                        crate::na::DataValue::Int64(v) => {
                            if range > 0.0 {
                                crate::na::DataValue::Float64((v as f64 - min) / range)
                            } else {
                                crate::na::DataValue::Float64(0.5)
                            }
                        }
                        x => x,
                    })?;
                    
                    result.replace_column(col_name.clone(), scaled_series)?;
                }
            }
        }
        
        Ok(result)
    }
    
    fn fit_transform(&mut self, df: &DataFrame) -> Result<DataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

/// カテゴリカルデータをダミー変数に変換するための変換器（One-Hot Encoding）
pub struct OneHotEncoder {
    /// 各列のカテゴリリスト
    categories: HashMap<String, Vec<String>>,
    /// 変換対象の列
    columns: Vec<String>,
    /// 最初のカテゴリを除外するかどうか（ダミー変数トラップ回避）
    drop_first: bool,
}

impl OneHotEncoder {
    /// 新しいOneHotEncoderを作成
    pub fn new(columns: Vec<String>, drop_first: bool) -> Self {
        OneHotEncoder {
            categories: HashMap::new(),
            columns,
            drop_first,
        }
    }
}

impl Transformer for OneHotEncoder {
    fn fit(&mut self, df: &DataFrame) -> Result<()> {
        for col_name in &self.columns {
            if let Some(series) = df.column(col_name) {
                let mut unique_vals = series
                    .iter()
                    .filter_map(|x| match x {
                        crate::na::DataValue::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                
                // 重複を削除
                unique_vals.sort();
                unique_vals.dedup();
                
                self.categories.insert(col_name.clone(), unique_vals);
            }
        }
        Ok(())
    }
    
    fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        let mut result = df.clone();
        
        for col_name in &self.columns {
            if let Some(categories) = self.categories.get(col_name) {
                let series = df.column(col_name).unwrap();
                
                // カテゴリごとに新しい列を作成
                let start_idx = if self.drop_first { 1 } else { 0 };
                
                for (i, category) in categories.iter().enumerate().skip(start_idx) {
                    let new_col_name = format!("{}_{}", col_name, category);
                    
                    let dummy_series = series.map(|x| match x {
                        crate::na::DataValue::String(s) => {
                            if s == category {
                                crate::na::DataValue::Int64(1)
                            } else {
                                crate::na::DataValue::Int64(0)
                            }
                        }
                        _ => crate::na::DataValue::NA,
                    })?;
                    
                    result.add_column(new_col_name, dummy_series)?;
                }
                
                // 元の列を削除
                result.drop_column(col_name)?;
            }
        }
        
        Ok(result)
    }
    
    fn fit_transform(&mut self, df: &DataFrame) -> Result<DataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}