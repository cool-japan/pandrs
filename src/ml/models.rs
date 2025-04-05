//! 機械学習モデルモジュール
//!
//! 機械学習モデルの実装とモデル評価のためのユーティリティを提供します。

use crate::optimized::OptimizedDataFrame;
use crate::error::{Result, Error};
use crate::column::{Float64Column, Column, ColumnTrait};
use crate::dataframe::DataValue;
use crate::stats;
use std::collections::HashMap;

/// 教師あり学習モデルに共通するトレイト
pub trait SupervisedModel {
    /// モデルを訓練データでフィットさせる
    fn fit(&mut self, df: &OptimizedDataFrame, target: &str, features: &[&str]) -> Result<()>;
    
    /// 新しいデータに対して予測を行う
    fn predict(&self, df: &OptimizedDataFrame) -> Result<Float64Column>;
    
    /// モデルのスコアを計算（デフォルトはR^2）
    fn score(&self, df: &DataFrame, target: &str) -> Result<f64> {
        let y_true = df.column(target).ok_or_else(|| {
            crate::error::Error::InvalidOperation(format!("Target column '{}' not found", target))
        })?;
        
        let y_pred = self.predict(df)?;
        
        // デフォルトでR^2スコアを使用
        crate::ml::metrics::regression::r2_score(y_true, &y_pred)
    }
}

/// 線形回帰モデル
pub struct LinearRegression {
    /// 回帰係数
    coefficients: Vec<f64>,
    /// 切片
    intercept: f64,
    /// 特徴量の名前
    feature_names: Vec<String>,
    /// 学習済みかどうか
    fitted: bool,
}

impl LinearRegression {
    /// 新しい線形回帰モデルを作成
    pub fn new() -> Self {
        LinearRegression {
            coefficients: Vec::new(),
            intercept: 0.0,
            feature_names: Vec::new(),
            fitted: false,
        }
    }
    
    /// 係数を取得
    pub fn coefficients(&self) -> HashMap<String, f64> {
        self.feature_names
            .iter()
            .zip(self.coefficients.iter())
            .map(|(name, coef)| (name.clone(), *coef))
            .collect()
    }
    
    /// 切片を取得
    pub fn intercept(&self) -> f64 {
        self.intercept
    }
    
    /// モデルの決定係数（R^2）を計算
    pub fn r_squared(&self, df: &DataFrame, target: &str) -> Result<f64> {
        self.score(df, target)
    }
}

impl SupervisedModel for LinearRegression {
    fn fit(&mut self, df: &DataFrame, target: &str, features: &[&str]) -> Result<()> {
        // statsモジュールの線形回帰を使用
        let result = stats::regression::linear_regression(df, target, features)?;
        
        self.intercept = result.intercept();
        self.coefficients = result.coefficients().values().cloned().collect();
        self.feature_names = result.coefficients().keys().cloned().collect();
        self.fitted = true;
        
        Ok(())
    }
    
    fn predict(&self, df: &DataFrame) -> Result<Series> {
        if !self.fitted {
            return Err(crate::error::Error::InvalidOperation(
                "Model has not been fitted yet".to_string()
            ));
        }
        
        let n_rows = df.nrows();
        let mut predictions = Vec::with_capacity(n_rows);
        
        for row_idx in 0..n_rows {
            let mut pred = self.intercept;
            
            for (feature_idx, feature_name) in self.feature_names.iter().enumerate() {
                if let Some(series) = df.column(feature_name) {
                    match series.get(row_idx) {
                        DataValue::Float64(v) => {
                            pred += v * self.coefficients[feature_idx];
                        }
                        DataValue::Int64(v) => {
                            pred += *v as f64 * self.coefficients[feature_idx];
                        }
                        _ => {
                            return Err(crate::error::Error::InvalidOperation(
                                format!("Non-numeric value in feature '{}'", feature_name)
                            ));
                        }
                    }
                } else {
                    return Err(crate::error::Error::InvalidOperation(
                        format!("Feature '{}' not found in DataFrame", feature_name)
                    ));
                }
            }
            
            predictions.push(DataValue::Float64(pred));
        }
        
        Series::from_vec(predictions)
    }
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// ロジスティック回帰モデル
pub struct LogisticRegression {
    /// 回帰係数
    coefficients: Vec<f64>,
    /// 切片
    intercept: f64,
    /// 特徴量の名前
    feature_names: Vec<String>,
    /// クラスラベル
    classes: Vec<String>,
    /// 正則化強度
    regularization: f64,
    /// 最大イテレーション数
    max_iter: usize,
    /// 収束閾値
    tol: f64,
    /// 学習済みかどうか
    fitted: bool,
}

impl LogisticRegression {
    /// 新しいロジスティック回帰モデルを作成
    pub fn new(regularization: f64, max_iter: usize, tol: f64) -> Self {
        LogisticRegression {
            coefficients: Vec::new(),
            intercept: 0.0,
            feature_names: Vec::new(),
            classes: Vec::new(),
            regularization,
            max_iter,
            tol,
            fitted: false,
        }
    }
    
    /// シグモイド関数
    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }
    
    /// 係数を取得
    pub fn coefficients(&self) -> HashMap<String, f64> {
        self.feature_names
            .iter()
            .zip(self.coefficients.iter())
            .map(|(name, coef)| (name.clone(), *coef))
            .collect()
    }
    
    /// 切片を取得
    pub fn intercept(&self) -> f64 {
        self.intercept
    }
    
    /// クラス確率を予測
    pub fn predict_proba(&self, df: &DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(crate::error::Error::InvalidOperation(
                "Model has not been fitted yet".to_string()
            ));
        }
        
        let n_rows = df.nrows();
        
        // 二項分類の場合
        if self.classes.len() == 2 {
            let mut proba_class_0 = Vec::with_capacity(n_rows);
            let mut proba_class_1 = Vec::with_capacity(n_rows);
            
            for row_idx in 0..n_rows {
                let mut z = self.intercept;
                
                for (feature_idx, feature_name) in self.feature_names.iter().enumerate() {
                    if let Some(series) = df.column(feature_name) {
                        match series.get(row_idx) {
                            DataValue::Float64(v) => {
                                z += v * self.coefficients[feature_idx];
                            }
                            DataValue::Int64(v) => {
                                z += *v as f64 * self.coefficients[feature_idx];
                            }
                            _ => {
                                return Err(crate::error::Error::InvalidOperation(
                                    format!("Non-numeric value in feature '{}'", feature_name)
                                ));
                            }
                        }
                    } else {
                        return Err(crate::error::Error::InvalidOperation(
                            format!("Feature '{}' not found in DataFrame", feature_name)
                        ));
                    }
                }
                
                let prob_1 = Self::sigmoid(z);
                proba_class_0.push(DataValue::Float64(1.0 - prob_1));
                proba_class_1.push(DataValue::Float64(prob_1));
            }
            
            let mut result = DataFrame::new();
            result.add_column(
                format!("proba_{}", self.classes[0]),
                Series::from_vec(proba_class_0)?,
            )?;
            result.add_column(
                format!("proba_{}", self.classes[1]),
                Series::from_vec(proba_class_1)?,
            )?;
            
            Ok(result)
        } else {
            // 多クラス分類はこの簡易実装では扱っていない
            Err(crate::error::Error::InvalidOperation(
                "Multi-class logistic regression is not implemented yet".to_string()
            ))
        }
    }
}

impl SupervisedModel for LogisticRegression {
    fn fit(&mut self, df: &DataFrame, target: &str, features: &[&str]) -> Result<()> {
        // 簡易的な実装として、二値分類のみサポート
        let target_series = df.column(target).ok_or_else(|| {
            crate::error::Error::InvalidOperation(format!("Target column '{}' not found", target))
        })?;
        
        // クラスを抽出
        let mut classes = Vec::new();
        for value in target_series.iter() {
            match value {
                DataValue::String(s) => {
                    if !classes.contains(s) {
                        classes.push(s.clone());
                    }
                }
                DataValue::Int64(i) => {
                    let s = i.to_string();
                    if !classes.contains(&s) {
                        classes.push(s);
                    }
                }
                _ => {
                    return Err(crate::error::Error::InvalidOperation(
                        "Target column must contain string or integer values".to_string()
                    ));
                }
            }
        }
        
        if classes.len() != 2 {
            return Err(crate::error::Error::InvalidOperation(
                "This implementation of LogisticRegression only supports binary classification".to_string()
            ));
        }
        
        self.classes = classes;
        self.feature_names = features.iter().map(|s| s.to_string()).collect();
        
        // 特徴量行列を構築
        let n_samples = df.nrows();
        let n_features = features.len();
        
        // X行列: n_samples x n_features
        let mut x = vec![vec![0.0; n_features]; n_samples];
        
        // Y: n_samples
        let mut y = vec![0.0; n_samples];
        
        // データの準備
        for (row_idx, row) in x.iter_mut().enumerate() {
            // 特徴量
            for (col_idx, &feature) in features.iter().enumerate() {
                let series = df.column(feature).ok_or_else(|| {
                    crate::error::Error::InvalidOperation(format!("Feature '{}' not found", feature))
                })?;
                
                match series.get(row_idx) {
                    DataValue::Float64(v) => {
                        row[col_idx] = *v;
                    }
                    DataValue::Int64(v) => {
                        row[col_idx] = *v as f64;
                    }
                    _ => {
                        return Err(crate::error::Error::InvalidOperation(
                            format!("Non-numeric value in feature '{}'", feature)
                        ));
                    }
                }
            }
            
            // ターゲット
            match target_series.get(row_idx) {
                DataValue::String(s) => {
                    if s == &self.classes[1] {
                        y[row_idx] = 1.0;
                    }
                }
                DataValue::Int64(i) => {
                    if i.to_string() == self.classes[1] {
                        y[row_idx] = 1.0;
                    }
                }
                _ => {
                    return Err(crate::error::Error::InvalidOperation(
                        "Target column must contain string or integer values".to_string()
                    ));
                }
            }
        }
        
        // 勾配降下法でパラメータを学習
        let mut coef = vec![0.0; n_features];
        let mut intercept = 0.0;
        
        let learning_rate = 0.01;
        let mut prev_loss = f64::INFINITY;
        
        for iter in 0..self.max_iter {
            // 予測値を計算
            let mut y_pred = vec![0.0; n_samples];
            for i in 0..n_samples {
                let mut z = intercept;
                for j in 0..n_features {
                    z += coef[j] * x[i][j];
                }
                y_pred[i] = Self::sigmoid(z);
            }
            
            // 勾配を計算
            let mut grad_coef = vec![0.0; n_features];
            let mut grad_intercept = 0.0;
            
            for i in 0..n_samples {
                let error = y_pred[i] - y[i];
                
                for j in 0..n_features {
                    grad_coef[j] += error * x[i][j];
                }
                
                grad_intercept += error;
            }
            
            // 正則化項の追加（L2正則化）
            for j in 0..n_features {
                grad_coef[j] = grad_coef[j] / n_samples as f64 + self.regularization * coef[j];
            }
            grad_intercept /= n_samples as f64;
            
            // パラメータの更新
            for j in 0..n_features {
                coef[j] -= learning_rate * grad_coef[j];
            }
            intercept -= learning_rate * grad_intercept;
            
            // ロスの計算
            let mut loss = 0.0;
            for i in 0..n_samples {
                if y[i] == 1.0 {
                    loss -= (y_pred[i]).ln();
                } else {
                    loss -= (1.0 - y_pred[i]).ln();
                }
            }
            loss /= n_samples as f64;
            
            // 正則化項の追加
            let l2_norm: f64 = coef.iter().map(|c| c.powi(2)).sum();
            loss += 0.5 * self.regularization * l2_norm;
            
            // 収束判定
            let diff = (prev_loss - loss).abs();
            if diff < self.tol {
                break;
            }
            
            prev_loss = loss;
        }
        
        self.coefficients = coef;
        self.intercept = intercept;
        self.fitted = true;
        
        Ok(())
    }
    
    fn predict(&self, df: &DataFrame) -> Result<Series> {
        if !self.fitted {
            return Err(crate::error::Error::InvalidOperation(
                "Model has not been fitted yet".to_string()
            ));
        }
        
        let proba_df = self.predict_proba(df)?;
        let n_rows = df.nrows();
        let mut predictions = Vec::with_capacity(n_rows);
        
        let proba_col = format!("proba_{}", self.classes[1]);
        let proba_series = proba_df.column(&proba_col).unwrap();
        
        for row_idx in 0..n_rows {
            let prob = match proba_series.get(row_idx) {
                DataValue::Float64(v) => *v,
                _ => 0.0,
            };
            
            if prob >= 0.5 {
                predictions.push(DataValue::String(self.classes[1].clone()));
            } else {
                predictions.push(DataValue::String(self.classes[0].clone()));
            }
        }
        
        Series::from_vec(predictions)
    }
    
    fn score(&self, df: &DataFrame, target: &str) -> Result<f64> {
        let y_true = df.column(target).ok_or_else(|| {
            crate::error::Error::InvalidOperation(format!("Target column '{}' not found", target))
        })?;
        
        let y_pred = self.predict(df)?;
        
        // 分類問題なのでAccuracyを使用
        crate::ml::metrics::classification::accuracy_score(y_true, &y_pred)
    }
}

/// モデル選択と評価のためのユーティリティ
pub mod model_selection {
    use super::*;
    use rand::seq::SliceRandom;
    use rand::thread_rng;
    
    /// 訓練データとテストデータに分割する
    pub fn train_test_split(
        df: &DataFrame,
        test_size: f64,
        random_seed: Option<u64>,
    ) -> Result<(DataFrame, DataFrame)> {
        let n_rows = df.nrows();
        let test_rows = (n_rows as f64 * test_size).round() as usize;
        
        // インデックスをシャッフル
        let mut indices: Vec<usize> = (0..n_rows).collect();
        
        let mut rng = match random_seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };
        
        indices.shuffle(&mut rng);
        
        // 訓練用インデックスとテスト用インデックス
        let train_indices = indices.iter().skip(test_rows).cloned().collect::<Vec<_>>();
        let test_indices = indices.iter().take(test_rows).cloned().collect::<Vec<_>>();
        
        // インデックスを使ってデータを取得
        let train_df = df.take_rows(&train_indices)?;
        let test_df = df.take_rows(&test_indices)?;
        
        Ok((train_df, test_df))
    }
    
    /// 交差検証のための分割を生成
    pub fn k_fold_split(df: &DataFrame, k: usize) -> Result<Vec<(DataFrame, DataFrame)>> {
        let n_rows = df.nrows();
        let fold_size = n_rows / k;
        
        let mut folds = Vec::with_capacity(k);
        
        for i in 0..k {
            let start = i * fold_size;
            let end = if i == k - 1 { n_rows } else { (i + 1) * fold_size };
            
            let test_indices: Vec<usize> = (start..end).collect();
            let train_indices: Vec<usize> = (0..n_rows)
                .filter(|&idx| !test_indices.contains(&idx))
                .collect();
            
            let train_df = df.take_rows(&train_indices)?;
            let test_df = df.take_rows(&test_indices)?;
            
            folds.push((train_df, test_df));
        }
        
        Ok(folds)
    }
    
    /// 交差検証によるモデル評価
    pub fn cross_val_score<M: SupervisedModel + Clone>(
        model: &M,
        df: &DataFrame,
        target: &str,
        features: &[&str],
        cv: usize,
    ) -> Result<Vec<f64>> {
        let folds = k_fold_split(df, cv)?;
        let mut scores = Vec::with_capacity(cv);
        
        for (train_df, test_df) in folds {
            let mut fold_model = model.clone();
            fold_model.fit(&train_df, target, features)?;
            let score = fold_model.score(&test_df, target)?;
            scores.push(score);
        }
        
        Ok(scores)
    }
    
    /// グリッドサーチによるハイパーパラメータチューニング
    pub struct GridSearchCV<M: SupervisedModel + Clone> {
        /// 対象モデル
        base_model: M,
        /// パラメータグリッド
        param_grid: Vec<HashMap<String, Vec<f64>>>,
        /// 分割数
        cv: usize,
        /// スコアが高いほど良いか
        higher_is_better: bool,
        /// 最良のパラメータ
        best_params: Option<HashMap<String, f64>>,
        /// 最良のスコア
        best_score: f64,
        /// 最良のモデル
        best_model: Option<M>,
    }
    
    impl<M: SupervisedModel + Clone> GridSearchCV<M> {
        /// 新しいGridSearchCVを作成
        pub fn new(base_model: M, param_grid: Vec<HashMap<String, Vec<f64>>>, cv: usize, higher_is_better: bool) -> Self {
            GridSearchCV {
                base_model,
                param_grid,
                cv,
                higher_is_better,
                best_params: None,
                best_score: if higher_is_better { f64::NEG_INFINITY } else { f64::INFINITY },
                best_model: None,
            }
        }
        
        /// パラメータの全組み合わせを生成
        fn generate_param_combinations(&self) -> Vec<HashMap<String, f64>> {
            let mut combinations = Vec::new();
            
            for param_set in &self.param_grid {
                let mut new_combinations = Vec::new();
                
                if combinations.is_empty() {
                    // 初期状態
                    for (param_name, param_values) in param_set {
                        for &value in param_values {
                            let mut param_map = HashMap::new();
                            param_map.insert(param_name.clone(), value);
                            new_combinations.push(param_map);
                        }
                    }
                } else {
                    // 既存の組み合わせと新しいパラメータの組み合わせ
                    for combo in &combinations {
                        for (param_name, param_values) in param_set {
                            for &value in param_values {
                                let mut param_map = combo.clone();
                                param_map.insert(param_name.clone(), value);
                                new_combinations.push(param_map);
                            }
                        }
                    }
                }
                
                combinations = new_combinations;
            }
            
            combinations
        }
        
        /// グリッドサーチを実行
        pub fn fit(&mut self, df: &DataFrame, target: &str, features: &[&str]) -> Result<()> {
            let param_combinations = self.generate_param_combinations();
            
            for params in param_combinations {
                let mut model = self.base_model.clone();
                
                // TODO: パラメータを設定する方法は、モデルの実装によって異なる
                // ここでは簡略化のため、実際のパラメータ設定は行っていない
                
                // 交差検証でスコアを計算
                let scores = cross_val_score(&model, df, target, features, self.cv)?;
                let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
                
                // スコアを評価し、最良のパラメータとモデルを更新
                let is_better = if self.higher_is_better {
                    mean_score > self.best_score
                } else {
                    mean_score < self.best_score
                };
                
                if is_better {
                    self.best_score = mean_score;
                    self.best_params = Some(params.clone());
                    
                    // 最良のモデルを再学習
                    let mut best_model = self.base_model.clone();
                    // TODO: パラメータを設定
                    best_model.fit(df, target, features)?;
                    self.best_model = Some(best_model);
                }
            }
            
            Ok(())
        }
        
        /// 最良のパラメータを取得
        pub fn best_params(&self) -> Option<&HashMap<String, f64>> {
            self.best_params.as_ref()
        }
        
        /// 最良のスコアを取得
        pub fn best_score(&self) -> f64 {
            self.best_score
        }
        
        /// 最良のモデルを取得
        pub fn best_model(&self) -> Option<&M> {
            self.best_model.as_ref()
        }
    }
}

/// モデル保存と読み込みのためのユーティリティ
pub mod model_persistence {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::fs::File;
    use std::io::{Read, Write};
    use std::path::Path;
    
    /// 線形回帰モデルをシリアライズできる形式
    #[derive(Serialize, Deserialize)]
    struct LinearRegressionData {
        coefficients: Vec<f64>,
        intercept: f64,
        feature_names: Vec<String>,
    }
    
    /// ロジスティック回帰モデルをシリアライズできる形式
    #[derive(Serialize, Deserialize)]
    struct LogisticRegressionData {
        coefficients: Vec<f64>,
        intercept: f64,
        feature_names: Vec<String>,
        classes: Vec<String>,
        regularization: f64,
        max_iter: usize,
        tol: f64,
    }
    
    /// モデルの保存と読み込みのためのトレイト
    pub trait ModelPersistence {
        /// モデルをファイルに保存
        fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<()>;
        
        /// モデルをファイルから読み込み
        fn load_model<P: AsRef<Path>>(path: P) -> Result<Self>
        where
            Self: Sized;
    }
    
    impl ModelPersistence for LinearRegression {
        fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<()> {
            let data = LinearRegressionData {
                coefficients: self.coefficients.clone(),
                intercept: self.intercept,
                feature_names: self.feature_names.clone(),
            };
            
            let json = serde_json::to_string(&data)
                .map_err(|e| crate::error::Error::IoError(e.to_string()))?;
            
            let mut file = File::create(path)
                .map_err(|e| crate::error::Error::IoError(e.to_string()))?;
            
            file.write_all(json.as_bytes())
                .map_err(|e| crate::error::Error::IoError(e.to_string()))?;
            
            Ok(())
        }
        
        fn load_model<P: AsRef<Path>>(path: P) -> Result<Self> {
            let mut file = File::open(path)
                .map_err(|e| crate::error::Error::IoError(e.to_string()))?;
            
            let mut json = String::new();
            file.read_to_string(&mut json)
                .map_err(|e| crate::error::Error::IoError(e.to_string()))?;
            
            let data: LinearRegressionData = serde_json::from_str(&json)
                .map_err(|e| crate::error::Error::IoError(e.to_string()))?;
            
            let model = LinearRegression {
                coefficients: data.coefficients,
                intercept: data.intercept,
                feature_names: data.feature_names,
                fitted: true,
            };
            
            Ok(model)
        }
    }
    
    impl ModelPersistence for LogisticRegression {
        fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<()> {
            let data = LogisticRegressionData {
                coefficients: self.coefficients.clone(),
                intercept: self.intercept,
                feature_names: self.feature_names.clone(),
                classes: self.classes.clone(),
                regularization: self.regularization,
                max_iter: self.max_iter,
                tol: self.tol,
            };
            
            let json = serde_json::to_string(&data)
                .map_err(|e| crate::error::Error::IoError(e.to_string()))?;
            
            let mut file = File::create(path)
                .map_err(|e| crate::error::Error::IoError(e.to_string()))?;
            
            file.write_all(json.as_bytes())
                .map_err(|e| crate::error::Error::IoError(e.to_string()))?;
            
            Ok(())
        }
        
        fn load_model<P: AsRef<Path>>(path: P) -> Result<Self> {
            let mut file = File::open(path)
                .map_err(|e| crate::error::Error::IoError(e.to_string()))?;
            
            let mut json = String::new();
            file.read_to_string(&mut json)
                .map_err(|e| crate::error::Error::IoError(e.to_string()))?;
            
            let data: LogisticRegressionData = serde_json::from_str(&json)
                .map_err(|e| crate::error::Error::IoError(e.to_string()))?;
            
            let model = LogisticRegression {
                coefficients: data.coefficients,
                intercept: data.intercept,
                feature_names: data.feature_names,
                classes: data.classes,
                regularization: data.regularization,
                max_iter: data.max_iter,
                tol: data.tol,
                fitted: true,
            };
            
            Ok(model)
        }
    }
}