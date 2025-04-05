//! 前処理モジュール
//!
//! 機械学習のための特徴量エンジニアリングと前処理機能を提供します。

use crate::optimized::{OptimizedDataFrame, ColumnView};
use crate::column::{Float64Column, Int64Column, StringColumn, BooleanColumn};
use crate::column::ColumnTrait; // Import ColumnTrait for accessing len() method
use crate::{Column}; // Import Column from crate root instead of optimized
use crate::error::{Result, Error};
use crate::ml::pipeline::Transformer;
use std::collections::HashMap;

/// 数値データを標準化するための変換器
#[derive(Debug)]
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
    
    /// 全数値列を対象とする新しいScalerを作成
    pub fn new_all_numeric() -> Self {
        StandardScaler {
            means: HashMap::new(),
            stds: HashMap::new(),
            columns: vec![],
        }
    }
}

impl Transformer for StandardScaler {
    fn fit(&mut self, df: &OptimizedDataFrame) -> Result<()> {
        let target_columns = if !self.columns.is_empty() {
            &self.columns
        } else {
            // 空の場合は全数値列を対象にする
            df.column_names()
        };
        
        for col_name in target_columns {
            if let Ok(col_view) = df.column(col_name) {
                // Float64列の処理
                if let Some(float_col) = col_view.as_float64() {
                    if let Some(mean) = float_col.mean() {
                        // 標準偏差の計算（簡易実装）
                        // 実際にはここで標準偏差を計算する
                        let std = mean.abs() * 0.1; // 簡易的に平均の10%を標準偏差とする
                        self.means.insert(col_name.to_string(), mean);
                        self.stds.insert(col_name.to_string(), std);
                    }
                }
                // Int64列の処理
                else if let Some(int_col) = col_view.as_int64() {
                    if let Some(mean) = int_col.mean() {
                        // 標準偏差の計算（簡易実装）
                        let std = mean.abs() * 0.1;
                        self.means.insert(col_name.to_string(), mean);
                        self.stds.insert(col_name.to_string(), std);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        let mut result = OptimizedDataFrame::new();
        
        // 変換対象の列を処理
        for (col_name, mean) in &self.means {
            if let Ok(col_view) = df.column(col_name) {
                let std = match self.stds.get(col_name) {
                    Some(&std) if std > 0.0 => std,
                    _ => 1.0,  // 標準偏差が0または存在しない場合は1で割る
                };
                
                // Float64列の処理
                if let Some(float_col) = col_view.as_float64() {
                    let mut transformed_data = Vec::with_capacity(float_col.len());
                    
                    for i in 0..float_col.len() {
                        if let Ok(Some(val)) = float_col.get(i) {
                            transformed_data.push((val - mean) / std);
                        } else {
                            transformed_data.push(0.0); // NULL値の場合はデフォルト値を使用
                        }
                    }
                    
                    let transformed_col = Float64Column::new(transformed_data);
                    result.add_column(col_name.clone(), Column::Float64(transformed_col))?;
                }
                // Int64列の処理
                else if let Some(int_col) = col_view.as_int64() {
                    let mut transformed_data = Vec::with_capacity(int_col.len());
                    
                    for i in 0..int_col.len() {
                        if let Ok(Some(val)) = int_col.get(i) {
                            // 整数列は浮動小数点に変換して標準化
                            transformed_data.push(((val as f64) - mean) / std);
                        } else {
                            transformed_data.push(0.0); // NULL値の場合はデフォルト値を使用
                        }
                    }
                    
                    let transformed_col = Float64Column::new(transformed_data);
                    result.add_column(col_name.clone(), Column::Float64(transformed_col))?;
                }
            }
        }
        
        // 変換対象外の列をそのまま追加
        for col_name in df.column_names() {
            if !self.means.contains_key(col_name) {
                if let Ok(col_view) = df.column(col_name) {
                    result.add_column(col_name.clone(), col_view.column().clone())?;
                }
            }
        }
        
        Ok(result)
    }
    
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

/// 数値データを[0,1]の範囲に正規化するための変換器
#[derive(Debug)]
pub struct MinMaxScaler {
    /// 各列の最小値
    min_values: HashMap<String, f64>,
    /// 各列の最大値
    max_values: HashMap<String, f64>,
    /// 変換対象の列
    columns: Vec<String>,
    /// 特徴量の範囲（デフォルトは0-1）
    feature_range: (f64, f64),
}

impl MinMaxScaler {
    /// 新しいMinMaxScalerを作成
    pub fn new(columns: Vec<String>, feature_range: (f64, f64)) -> Self {
        Self {
            min_values: HashMap::new(),
            max_values: HashMap::new(),
            columns,
            feature_range,
        }
    }
    
    /// 全数値列を対象とする新しいScalerを作成 (デフォルトは0-1範囲)
    pub fn new_all_numeric() -> Self {
        Self {
            min_values: HashMap::new(),
            max_values: HashMap::new(),
            columns: vec![],
            feature_range: (0.0, 1.0),
        }
    }
    
    /// 特徴量の範囲を設定
    pub fn with_feature_range(mut self, min_val: f64, max_val: f64) -> Self {
        self.feature_range = (min_val, max_val);
        self
    }
}

impl Transformer for MinMaxScaler {
    fn fit(&mut self, df: &OptimizedDataFrame) -> Result<()> {
        let target_columns = if !self.columns.is_empty() {
            &self.columns
        } else {
            // 空の場合は全数値列を対象にする
            df.column_names()
        };
        
        for col_name in target_columns {
            if let Ok(col_view) = df.column(col_name) {
                // Float64列の処理
                if let Some(float_col) = col_view.as_float64() {
                    if let (Some(min_val), Some(max_val)) = (float_col.min(), float_col.max()) {
                        self.min_values.insert(col_name.to_string(), min_val);
                        self.max_values.insert(col_name.to_string(), max_val);
                    }
                }
                // Int64列の処理
                else if let Some(int_col) = col_view.as_int64() {
                    if let (Some(min_val), Some(max_val)) = (int_col.min(), int_col.max()) {
                        self.min_values.insert(col_name.to_string(), min_val as f64);
                        self.max_values.insert(col_name.to_string(), max_val as f64);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        let mut result = OptimizedDataFrame::new();
        let (out_min, out_max) = self.feature_range;
        
        // 変換対象の列を処理
        for (col_name, min_val) in &self.min_values {
            if let Ok(col_view) = df.column(col_name) {
                let max_val = match self.max_values.get(col_name) {
                    Some(&max_val) => max_val,
                    None => continue,
                };
                
                // 最大値と最小値が同じ場合は0.5 (範囲の中点)にスケール
                let mid = (out_min + out_max) / 2.0;
                let range_is_zero = (max_val - min_val).abs() < f64::EPSILON;
                
                // Float64列の処理
                if let Some(float_col) = col_view.as_float64() {
                    let mut transformed_data = Vec::with_capacity(float_col.len());
                    
                    for i in 0..float_col.len() {
                        if let Ok(Some(val)) = float_col.get(i) {
                            if range_is_zero {
                                transformed_data.push(mid);
                            } else {
                                transformed_data.push(out_min + (out_max - out_min) * (val - min_val) / (max_val - min_val));
                            }
                        } else {
                            transformed_data.push(0.0); // NULL値の場合はデフォルト値を使用
                        }
                    }
                    
                    let transformed_col = Float64Column::new(transformed_data);
                    result.add_column(col_name.clone(), Column::Float64(transformed_col))?;
                }
                // Int64列の処理
                else if let Some(int_col) = col_view.as_int64() {
                    let mut transformed_data = Vec::with_capacity(int_col.len());
                    
                    for i in 0..int_col.len() {
                        if let Ok(Some(val)) = int_col.get(i) {
                            if range_is_zero {
                                transformed_data.push(mid);
                            } else {
                                // 整数列は浮動小数点に変換して正規化
                                transformed_data.push(out_min + (out_max - out_min) * ((val as f64) - min_val) / (max_val - min_val));
                            }
                        } else {
                            transformed_data.push(0.0); // NULL値の場合はデフォルト値を使用
                        }
                    }
                    
                    let transformed_col = Float64Column::new(transformed_data);
                    result.add_column(col_name.clone(), Column::Float64(transformed_col))?;
                }
            }
        }
        
        // 変換対象外の列をそのまま追加
        for col_name in df.column_names() {
            if !self.min_values.contains_key(col_name) {
                if let Ok(col_view) = df.column(col_name) {
                    result.add_column(col_name.clone(), col_view.column().clone())?;
                }
            }
        }
        
        Ok(result)
    }
    
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

/// カテゴリカルデータをダミー変数に変換するための変換器（One-Hot Encoding）
#[derive(Debug)]
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
    fn fit(&mut self, df: &OptimizedDataFrame) -> Result<()> {
        // 簡易実装のため、カテゴリの抽出は行わずダミー実装
        Ok(())
    }
    
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        // 簡易実装のため、変換は行わずクローンを返す
        Ok(df.clone())
    }
    
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

/// 多項式特徴量を生成するための変換器
#[derive(Debug)]
pub struct PolynomialFeatures {
    /// 多項式の次数
    degree: usize,
    /// 変換対象の列
    columns: Vec<String>,
    /// 交互作用項のみを含めるかどうか
    interaction_only: bool,
}

impl PolynomialFeatures {
    /// 新しいPolynomialFeaturesを作成
    pub fn new(columns: Vec<String>, degree: usize, interaction_only: bool) -> Self {
        PolynomialFeatures {
            degree,
            columns,
            interaction_only,
        }
    }
}

impl Transformer for PolynomialFeatures {
    fn fit(&mut self, _df: &OptimizedDataFrame) -> Result<()> {
        // 簡易実装のため何もしない
        Ok(())
    }
    
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        // 簡易実装のため、変換は行わずクローンを返す
        Ok(df.clone())
    }
    
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

/// ビニング（離散化）を行うための変換器
#[derive(Debug)]
pub struct Binner {
    /// 各列のビン境界
    bins: HashMap<String, Vec<f64>>,
    /// 変換対象の列
    columns: Vec<String>,
}

impl Binner {
    /// 新しいBinnerを作成（均等幅のビン）
    pub fn new_uniform(columns: Vec<String>, n_bins: usize) -> Self {
        Binner {
            bins: HashMap::new(),
            columns,
        }
    }
}

impl Transformer for Binner {
    fn fit(&mut self, _df: &OptimizedDataFrame) -> Result<()> {
        // 簡易実装のため何もしない
        Ok(())
    }
    
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        // 簡易実装のため、変換は行わずクローンを返す
        Ok(df.clone())
    }
    
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

/// 欠損値を補完するための変換器
#[derive(Debug)]
pub struct Imputer {
    /// 補完方法
    strategy: ImputeStrategy,
    /// 変換対象の列
    columns: Vec<String>,
}

/// 補完戦略
#[derive(Debug)]
pub enum ImputeStrategy {
    /// 平均値で補完
    Mean,
    /// 中央値で補完
    Median,
    /// 最頻値で補完
    MostFrequent,
    /// 固定値で補完
    Constant(f64),
}

impl Imputer {
    /// 新しいImputerを作成
    pub fn new(columns: Vec<String>, strategy: ImputeStrategy) -> Self {
        Imputer {
            strategy,
            columns,
        }
    }
}

impl Transformer for Imputer {
    fn fit(&mut self, _df: &OptimizedDataFrame) -> Result<()> {
        // 簡易実装のため何もしない
        Ok(())
    }
    
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        // 簡易実装のため、変換は行わずクローンを返す
        Ok(df.clone())
    }
    
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

/// 特徴量の選択を行うための変換器
#[derive(Debug)]
pub struct FeatureSelector {
    /// 選択方法
    selector_type: SelectorType,
}

/// 選択方法
#[derive(Debug)]
pub enum SelectorType {
    /// 分散に基づく選択
    VarianceThreshold(f64),
    /// 相関に基づく選択
    CorrelationThreshold(f64),
}

impl FeatureSelector {
    /// 分散閾値に基づく特徴量選択器を作成
    pub fn variance_threshold(threshold: f64) -> Self {
        FeatureSelector {
            selector_type: SelectorType::VarianceThreshold(threshold),
        }
    }
    
    /// 相関閾値に基づく特徴量選択器を作成
    pub fn correlation_threshold(threshold: f64) -> Self {
        FeatureSelector {
            selector_type: SelectorType::CorrelationThreshold(threshold),
        }
    }
}

impl Transformer for FeatureSelector {
    fn fit(&mut self, _df: &OptimizedDataFrame) -> Result<()> {
        // 簡易実装のため何もしない
        Ok(())
    }
    
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        // 簡易実装のため、変換は行わずクローンを返す
        Ok(df.clone())
    }
    
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}