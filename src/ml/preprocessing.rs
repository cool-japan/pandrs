//! 前処理モジュール
//!
//! 機械学習のための特徴量エンジニアリングと前処理機能を提供します。

use crate::dataframe::DataFrame;
use crate::error::Result;
use crate::ml::pipeline::Transformer;
use crate::series::Series;
use crate::na::DataValue;
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

/// 多項式特徴量を生成するための変換器
pub struct PolynomialFeatures {
    /// 多項式の次数
    degree: usize,
    /// 変換対象の列
    columns: Vec<String>,
    /// 交互作用項のみを含めるかどうか
    interaction_only: bool,
    /// 特徴量の組み合わせ
    feature_combinations: Vec<Vec<(String, usize)>>,
}

impl PolynomialFeatures {
    /// 新しいPolynomialFeaturesを作成
    pub fn new(columns: Vec<String>, degree: usize, interaction_only: bool) -> Self {
        PolynomialFeatures {
            degree,
            columns,
            interaction_only,
            feature_combinations: Vec::new(),
        }
    }
    
    /// 特徴量の組み合わせを生成
    fn generate_combinations(&mut self) {
        self.feature_combinations.clear();
        
        // 定数項（次数0）は含めない
        
        // 一次の項を追加
        for col in &self.columns {
            self.feature_combinations.push(vec![(col.clone(), 1)]);
        }
        
        // 高次の項を追加
        if self.degree >= 2 {
            let n = self.columns.len();
            
            // 2次以上の組み合わせを生成
            for d in 2..=self.degree {
                // 各特徴量の組み合わせを生成
                let mut queue = Vec::new();
                
                // 初期組み合わせとして1次の項を使用
                for i in 0..n {
                    queue.push(vec![(self.columns[i].clone(), 1)]);
                }
                
                while let Some(current) = queue.pop() {
                    // 現在の組み合わせの次数を計算
                    let current_degree: usize = current.iter().map(|(_, power)| power).sum();
                    
                    // 最後の特徴量のインデックスを取得
                    let last_feature_idx = self.columns.iter().position(|col| col == &current.last().unwrap().0).unwrap_or(0);
                    
                    // 次の特徴量を追加
                    for i in last_feature_idx..n {
                        let col = &self.columns[i];
                        
                        // 新しい組み合わせを作成
                        let mut new_combination = current.clone();
                        
                        // 既存の特徴量かどうかを確認
                        let existing_idx = new_combination.iter().position(|(feature, _)| feature == col);
                        
                        if let Some(idx) = existing_idx {
                            // 既存の特徴量の次数を増やす
                            new_combination[idx].1 += 1;
                        } else {
                            // 新しい特徴量を追加
                            new_combination.push((col.clone(), 1));
                        }
                        
                        // 新しい組み合わせの次数を計算
                        let new_degree: usize = new_combination.iter().map(|(_, power)| power).sum();
                        
                        // 次数がdを超えない場合、組み合わせを追加
                        if new_degree <= d {
                            // interaction_onlyがtrueの場合、同じ特徴量の2次以上の項は含めない
                            let is_valid = !self.interaction_only || 
                                new_combination.iter().all(|(_, power)| *power <= 1);
                            
                            if is_valid {
                                // 次数がdに等しい場合、結果に追加
                                if new_degree == d {
                                    self.feature_combinations.push(new_combination.clone());
                                }
                                
                                // キューに追加して続けて処理
                                queue.push(new_combination);
                            }
                        }
                    }
                }
            }
        }
        
        // 重複を削除
        self.feature_combinations.sort();
        self.feature_combinations.dedup();
    }
    
    /// 組み合わせから列名を生成
    fn get_column_name(combination: &[(String, usize)]) -> String {
        combination
            .iter()
            .map(|(col, power)| {
                if *power == 1 {
                    col.clone()
                } else {
                    format!("{}^{}", col, power)
                }
            })
            .collect::<Vec<_>>()
            .join("_")
    }
}

impl Transformer for PolynomialFeatures {
    fn fit(&mut self, _df: &DataFrame) -> Result<()> {
        self.generate_combinations();
        Ok(())
    }
    
    fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        let mut result = df.clone();
        
        for combination in &self.feature_combinations {
            if combination.len() == 1 && combination[0].1 == 1 {
                // 一次の項はすでにデータフレームに含まれているのでスキップ
                continue;
            }
            
            let new_col_name = Self::get_column_name(combination);
            
            // 各行に対して多項式特徴量を計算
            let mut values = Vec::with_capacity(df.nrows());
            
            for row_idx in 0..df.nrows() {
                let mut row_value = 1.0;
                let mut is_valid = true;
                
                for (col, power) in combination {
                    if let Some(series) = df.column(col) {
                        match series.get(row_idx) {
                            DataValue::Float64(v) => {
                                row_value *= v.powi(*power as i32);
                            }
                            DataValue::Int64(v) => {
                                row_value *= (v as f64).powi(*power as i32);
                            }
                            _ => {
                                is_valid = false;
                                break;
                            }
                        }
                    } else {
                        is_valid = false;
                        break;
                    }
                }
                
                if is_valid {
                    values.push(DataValue::Float64(row_value));
                } else {
                    values.push(DataValue::NA);
                }
            }
            
            let new_series = Series::from_vec(values)?;
            result.add_column(new_col_name, new_series)?;
        }
        
        Ok(result)
    }
    
    fn fit_transform(&mut self, df: &DataFrame) -> Result<DataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

/// ビニング（離散化）を行うための変換器
pub struct Binner {
    /// 各列のビン境界
    bins: HashMap<String, Vec<f64>>,
    /// 変換対象の列
    columns: Vec<String>,
    /// ビンのラベル
    labels: Option<Vec<String>>,
    /// ビンの数
    n_bins: usize,
    /// 均等幅のビンを使用するかどうか
    uniform: bool,
}

impl Binner {
    /// 新しいBinnerを作成（均等幅のビン）
    pub fn new_uniform(columns: Vec<String>, n_bins: usize) -> Self {
        Binner {
            bins: HashMap::new(),
            columns,
            labels: None,
            n_bins,
            uniform: true,
        }
    }
    
    /// 新しいBinnerを作成（カスタムビン境界）
    pub fn new_custom(columns: Vec<String>, bins: HashMap<String, Vec<f64>>) -> Self {
        Binner {
            bins,
            columns,
            labels: None,
            n_bins: 0,
            uniform: false,
        }
    }
    
    /// ビンのラベルを設定
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = Some(labels);
        self
    }
    
    /// 値がどのビンに属するかを判定
    fn get_bin_index(&self, value: f64, bins: &[f64]) -> usize {
        for (i, &bin_edge) in bins.iter().enumerate() {
            if value <= bin_edge {
                return i;
            }
        }
        bins.len()
    }
}

impl Transformer for Binner {
    fn fit(&mut self, df: &DataFrame) -> Result<()> {
        if self.uniform {
            for col_name in &self.columns {
                if let Some(series) = df.column(col_name) {
                    match (series.min()?, series.max()?) {
                        (DataValue::Float64(min), DataValue::Float64(max)) => {
                            let step = (max - min) / self.n_bins as f64;
                            let mut bin_edges = Vec::with_capacity(self.n_bins);
                            
                            for i in 1..self.n_bins {
                                bin_edges.push(min + step * i as f64);
                            }
                            
                            self.bins.insert(col_name.clone(), bin_edges);
                        }
                        (DataValue::Int64(min), DataValue::Int64(max)) => {
                            let min = min as f64;
                            let max = max as f64;
                            let step = (max - min) / self.n_bins as f64;
                            let mut bin_edges = Vec::with_capacity(self.n_bins);
                            
                            for i in 1..self.n_bins {
                                bin_edges.push(min + step * i as f64);
                            }
                            
                            self.bins.insert(col_name.clone(), bin_edges);
                        }
                        _ => {}
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        let mut result = df.clone();
        
        for col_name in &self.columns {
            if let Some(bin_edges) = self.bins.get(col_name) {
                if let Some(series) = df.column(col_name) {
                    let mut binned_values = Vec::with_capacity(series.len());
                    
                    for value in series.iter() {
                        match value {
                            DataValue::Float64(v) => {
                                let bin_idx = self.get_bin_index(*v, bin_edges);
                                
                                if let Some(labels) = &self.labels {
                                    if bin_idx < labels.len() {
                                        binned_values.push(DataValue::String(labels[bin_idx].clone()));
                                    } else {
                                        binned_values.push(DataValue::NA);
                                    }
                                } else {
                                    binned_values.push(DataValue::Int64(bin_idx as i64));
                                }
                            }
                            DataValue::Int64(v) => {
                                let bin_idx = self.get_bin_index(*v as f64, bin_edges);
                                
                                if let Some(labels) = &self.labels {
                                    if bin_idx < labels.len() {
                                        binned_values.push(DataValue::String(labels[bin_idx].clone()));
                                    } else {
                                        binned_values.push(DataValue::NA);
                                    }
                                } else {
                                    binned_values.push(DataValue::Int64(bin_idx as i64));
                                }
                            }
                            _ => {
                                binned_values.push(DataValue::NA);
                            }
                        }
                    }
                    
                    let new_series = Series::from_vec(binned_values)?;
                    let new_col_name = format!("{}_binned", col_name);
                    result.add_column(new_col_name, new_series)?;
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

/// 欠損値を補完するための変換器
pub struct Imputer {
    /// 補完方法
    strategy: ImputeStrategy,
    /// 変換対象の列
    columns: Vec<String>,
    /// 各列の補完値
    fill_values: HashMap<String, DataValue>,
}

/// 補完戦略
pub enum ImputeStrategy {
    /// 平均値で補完
    Mean,
    /// 中央値で補完
    Median,
    /// 最頻値で補完
    MostFrequent,
    /// 固定値で補完
    Constant(DataValue),
}

impl Imputer {
    /// 新しいImputerを作成
    pub fn new(columns: Vec<String>, strategy: ImputeStrategy) -> Self {
        Imputer {
            strategy,
            columns,
            fill_values: HashMap::new(),
        }
    }
    
    /// 最頻値を計算
    fn compute_most_frequent(series: &Series) -> Option<DataValue> {
        let mut value_counts = HashMap::new();
        
        for value in series.iter() {
            if value != &DataValue::NA {
                *value_counts.entry(value.clone()).or_insert(0) += 1;
            }
        }
        
        value_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(value, _)| value)
    }
}

impl Transformer for Imputer {
    fn fit(&mut self, df: &DataFrame) -> Result<()> {
        for col_name in &self.columns {
            if let Some(series) = df.column(col_name) {
                let fill_value = match &self.strategy {
                    ImputeStrategy::Mean => {
                        match series.mean()? {
                            mean => DataValue::Float64(mean),
                        }
                    }
                    ImputeStrategy::Median => {
                        match series.median()? {
                            median => match median {
                                DataValue::Float64(m) => DataValue::Float64(m),
                                DataValue::Int64(m) => DataValue::Int64(m),
                                _ => DataValue::NA,
                            },
                        }
                    }
                    ImputeStrategy::MostFrequent => {
                        Self::compute_most_frequent(series).unwrap_or(DataValue::NA)
                    }
                    ImputeStrategy::Constant(value) => value.clone(),
                };
                
                self.fill_values.insert(col_name.clone(), fill_value);
            }
        }
        
        Ok(())
    }
    
    fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        let mut result = df.clone();
        
        for col_name in &self.columns {
            if let Some(fill_value) = self.fill_values.get(col_name) {
                if let Some(series) = df.column(col_name) {
                    let filled_series = series.map(|x| {
                        if x == &DataValue::NA {
                            fill_value.clone()
                        } else {
                            x.clone()
                        }
                    })?;
                    
                    result.replace_column(col_name.clone(), filled_series)?;
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

/// 特徴量の選択を行うための変換器
pub struct FeatureSelector {
    /// 選択方法
    selector_type: SelectorType,
    /// 選択する特徴量の数または割合
    k: usize,
}

/// 選択方法
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
            k: 0,
        }
    }
    
    /// 相関閾値に基づく特徴量選択器を作成
    pub fn correlation_threshold(threshold: f64) -> Self {
        FeatureSelector {
            selector_type: SelectorType::CorrelationThreshold(threshold),
            k: 0,
        }
    }
}

impl Transformer for FeatureSelector {
    fn fit(&mut self, df: &DataFrame) -> Result<()> {
        // この実装では特に前処理は必要ないため、何もしない
        Ok(())
    }
    
    fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        let mut result = DataFrame::new();
        let columns = df.column_names();
        
        match &self.selector_type {
            SelectorType::VarianceThreshold(threshold) => {
                for col_name in columns {
                    if let Some(series) = df.column(&col_name) {
                        let variance = series.var()?;
                        
                        if variance >= *threshold {
                            result.add_column(col_name.clone(), series.clone())?;
                        }
                    }
                }
            }
            SelectorType::CorrelationThreshold(threshold) => {
                // まず、すべての数値列を追加
                let numeric_columns: Vec<String> = columns
                    .into_iter()
                    .filter(|col_name| {
                        if let Some(series) = df.column(col_name) {
                            matches!(series.get(0), DataValue::Float64(_) | DataValue::Int64(_))
                        } else {
                            false
                        }
                    })
                    .collect();
                
                let mut selected_columns = Vec::new();
                
                // 列間の相関係数を確認
                for i in 0..numeric_columns.len() {
                    let col_i = &numeric_columns[i];
                    let mut keep_column = true;
                    
                    for j in 0..i {
                        let col_j = &numeric_columns[j];
                        
                        if let (Some(series_i), Some(series_j)) = (df.column(col_i), df.column(col_j)) {
                            if let Ok(corr) = crate::stats::correlation(series_i, series_j) {
                                if corr.abs() >= *threshold {
                                    // 相関が閾値以上なら、この列を除外
                                    keep_column = false;
                                    break;
                                }
                            }
                        }
                    }
                    
                    if keep_column {
                        selected_columns.push(col_i.clone());
                    }
                }
                
                // 選択された列をデータフレームに追加
                for col_name in selected_columns {
                    if let Some(series) = df.column(&col_name) {
                        result.add_column(col_name.clone(), series.clone())?;
                    }
                }
                
                // 数値列以外の列も追加
                for col_name in df.column_names() {
                    if let Some(series) = df.column(&col_name) {
                        if !matches!(series.get(0), DataValue::Float64(_) | DataValue::Int64(_)) {
                            result.add_column(col_name.clone(), series.clone())?;
                        }
                    }
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