//! 機械学習モデルモジュール
//!
//! 機械学習モデルの実装とモデル評価のためのユーティリティを提供します。
//! OptimizedDataFrameに対応した実装です。

use crate::optimized::{OptimizedDataFrame, ColumnView};
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
    fn predict(&self, df: &OptimizedDataFrame) -> Result<Vec<f64>>;
    
    /// モデルのスコアを計算（デフォルトはR^2）
    fn score(&self, df: &OptimizedDataFrame, target: &str) -> Result<f64> {
        // 目標変数を取得
        let y_true = match df.column(target) {
            Some(col) => {
                // Float64Columnに変換
                let numeric_data = self.extract_numeric_values(&col)?;
                numeric_data
            },
            None => return Err(Error::Column(format!("Target column '{}' not found", target)))
        };
        
        // 予測を取得
        let y_pred = self.predict(df)?;
        
        // デフォルトでR^2スコアを使用
        crate::ml::metrics::regression::r2_score(&y_true, &y_pred)
    }
    
    /// カラムから数値データを抽出するヘルパーメソッド
    fn extract_numeric_values(&self, col: &ColumnView) -> Result<Vec<f64>> {
        match col.column_type() {
            crate::column::ColumnType::Float64 => {
                let mut values = Vec::with_capacity(col.len());
                for i in 0..col.len() {
                    if let Some(value) = col.get_f64(i)? {
                        values.push(value);
                    } else {
                        values.push(0.0); // NAは0として扱う（または適切な戦略を実装）
                    }
                }
                Ok(values)
            },
            crate::column::ColumnType::Int64 => {
                let mut values = Vec::with_capacity(col.len());
                for i in 0..col.len() {
                    if let Some(value) = col.get_i64(i)? {
                        values.push(value as f64);
                    } else {
                        values.push(0.0); // NAは0として扱う
                    }
                }
                Ok(values)
            },
            _ => Err(Error::Type(format!("Column type {:?} cannot be converted to numeric", col.column_type())))
        }
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
    pub fn r_squared(&self, df: &OptimizedDataFrame, target: &str) -> Result<f64> {
        self.score(df, target)
    }
    
    // データフレームをstatsモジュールの線形回帰に変換
    fn prepare_data_for_regression(&self, df: &OptimizedDataFrame, target: &str, features: &[&str]) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
        // 特徴量データの抽出
        let mut x_data: Vec<Vec<f64>> = Vec::with_capacity(df.row_count());
        for _ in 0..df.row_count() {
            x_data.push(Vec::with_capacity(features.len()));
        }
        
        // 特徴量の値を取得
        for &feature in features {
            let column = df.column(feature).ok_or_else(|| {
                Error::Column(format!("Feature column '{}' not found", feature))
            })?;
            
            let values = self.extract_numeric_values(&column)?;
            
            // 行列に追加
            for (i, &value) in values.iter().enumerate() {
                if i < x_data.len() {
                    x_data[i].push(value);
                }
            }
        }
        
        // 目標変数の値を取得
        let target_column = df.column(target).ok_or_else(|| {
            Error::Column(format!("Target column '{}' not found", target))
        })?;
        
        let y_data = self.extract_numeric_values(&target_column)?;
        
        Ok((x_data, y_data))
    }
    
    // 最小二乗法による線形回帰の実装
    fn fit_linear_regression(&mut self, x: &[Vec<f64>], y: &[f64], feature_names: &[&str]) -> Result<()> {
        if x.is_empty() || y.is_empty() {
            return Err(Error::Empty("Empty data for linear regression".to_string()));
        }
        
        let n_samples = x.len();
        let n_features = x[0].len();
        
        if n_features != feature_names.len() {
            return Err(Error::Consistency(format!(
                "Number of features ({}) doesn't match feature names length ({})",
                n_features, feature_names.len()
            )));
        }
        
        // 行列計算のためのXTX（X^T * X）行列を計算
        let mut xtx = vec![vec![0.0; n_features + 1]; n_features + 1];
        let mut xty = vec![0.0; n_features + 1];
        
        // バイアス項のXTX部分を設定
        xtx[0][0] = n_samples as f64;
        
        // バイアス項のXTY部分を設定
        xty[0] = y.iter().sum();
        
        // 特徴量部分のXTX行列を計算
        for i in 0..n_features {
            // バイアス項と特徴量の相互作用
            let sum_xi = x.iter().map(|sample| sample[i]).sum::<f64>();
            xtx[0][i + 1] = sum_xi;
            xtx[i + 1][0] = sum_xi;
            
            // 特徴量同士の相互作用
            for j in 0..n_features {
                let sum_xixj = x.iter().map(|sample| sample[i] * sample[j]).sum::<f64>();
                xtx[i + 1][j + 1] = sum_xixj;
            }
            
            // XTY部分を計算
            let sum_xiy = x.iter().zip(y.iter()).map(|(sample, &yi)| sample[i] * yi).sum::<f64>();
            xty[i + 1] = sum_xiy;
        }
        
        // ガウスの消去法で連立方程式を解く
        let mut coeffs = self.solve_linear_system(&xtx, &xty)?;
        
        // 切片と係数を設定
        self.intercept = coeffs.remove(0);
        self.coefficients = coeffs;
        self.feature_names = feature_names.iter().map(|&s| s.to_string()).collect();
        self.fitted = true;
        
        Ok(())
    }
    
    // ガウスの消去法による連立方程式の解法
    fn solve_linear_system(&self, a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
        let n = a.len();
        
        // 拡大係数行列を作成
        let mut augmented = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(n + 1);
            row.extend_from_slice(&a[i]);
            row.push(b[i]);
            augmented.push(row);
        }
        
        // 前進消去
        for i in 0..n {
            // ピボット選択（部分ピボット選択）
            let mut max_idx = i;
            let mut max_val = augmented[i][i].abs();
            
            for j in (i + 1)..n {
                let abs_val = augmented[j][i].abs();
                if abs_val > max_val {
                    max_idx = j;
                    max_val = abs_val;
                }
            }
            
            // 特異行列チェック
            if max_val < 1e-10 {
                return Err(Error::Computation("Singular matrix, cannot solve linear system".to_string()));
            }
            
            // 行の交換
            if max_idx != i {
                augmented.swap(i, max_idx);
            }
            
            // 対角成分で正規化
            let pivot = augmented[i][i];
            for j in i..=n {
                augmented[i][j] /= pivot;
            }
            
            // 他の行を消去
            for j in 0..n {
                if j != i {
                    let factor = augmented[j][i];
                    for k in i..=n {
                        augmented[j][k] -= factor * augmented[i][k];
                    }
                }
            }
        }
        
        // 解を取得
        let mut x = vec![0.0; n];
        for i in 0..n {
            x[i] = augmented[i][n];
        }
        
        Ok(x)
    }
}

impl SupervisedModel for LinearRegression {
    fn fit(&mut self, df: &OptimizedDataFrame, target: &str, features: &[&str]) -> Result<()> {
        // データを準備
        let (x_data, y_data) = self.prepare_data_for_regression(df, target, features)?;
        
        // 線形回帰を実行
        self.fit_linear_regression(&x_data, &y_data, features)?;
        
        Ok(())
    }
    
    fn predict(&self, df: &OptimizedDataFrame) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(Error::InvalidOperation(
                "Model has not been fitted yet".to_string()
            ));
        }
        
        let n_rows = df.row_count();
        let mut predictions = Vec::with_capacity(n_rows);
        
        // 特徴量の存在確認と列インデックスのマッピング
        let mut feature_columns = Vec::with_capacity(self.feature_names.len());
        for feature_name in &self.feature_names {
            let column = df.column(feature_name).ok_or_else(|| {
                Error::Column(format!("Feature column '{}' not found", feature_name))
            })?;
            feature_columns.push(column);
        }
        
        // 各行に対して予測
        for row_idx in 0..n_rows {
            let mut pred = self.intercept;
            
            // 各特徴量の寄与を加算
            for (i, column) in feature_columns.iter().enumerate() {
                let feature_value = match column.column_type() {
                    crate::column::ColumnType::Float64 => {
                        column.get_f64(row_idx)?.unwrap_or(0.0)
                    },
                    crate::column::ColumnType::Int64 => {
                        column.get_i64(row_idx)?.map(|v| v as f64).unwrap_or(0.0)
                    },
                    _ => 0.0 // 数値型以外は0として扱う（または適切な戦略を実装）
                };
                
                pred += self.coefficients[i] * feature_value;
            }
            
            predictions.push(pred);
        }
        
        Ok(predictions)
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
    /// 学習率
    learning_rate: f64,
    /// 最大イテレーション数
    max_iter: usize,
    /// 収束判定基準
    tol: f64,
    /// 学習済みかどうか
    fitted: bool,
}

impl LogisticRegression {
    /// 新しいロジスティック回帰モデルを作成
    pub fn new(learning_rate: f64, max_iter: usize, tol: f64) -> Self {
        LogisticRegression {
            coefficients: Vec::new(),
            intercept: 0.0,
            feature_names: Vec::new(),
            learning_rate,
            max_iter,
            tol,
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
    
    /// シグモイド関数
    fn sigmoid(&self, z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }
    
    /// ロジスティック回帰の予測確率を計算
    fn predict_proba_row(&self, features: &[f64]) -> f64 {
        if features.len() != self.coefficients.len() {
            return 0.5; // エラー時はデフォルト確率を返す
        }
        
        let mut z = self.intercept;
        for i in 0..features.len() {
            z += features[i] * self.coefficients[i];
        }
        
        self.sigmoid(z)
    }
    
    /// 確率モデルとしての予測
    pub fn predict_proba(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        if !self.fitted {
            return Err(Error::InvalidOperation(
                "Model has not been fitted yet".to_string()
            ));
        }
        
        let n_rows = df.row_count();
        let mut proba_class0 = Vec::with_capacity(n_rows);
        let mut proba_class1 = Vec::with_capacity(n_rows);
        
        // 特徴量の列を取得
        let mut feature_columns = Vec::with_capacity(self.feature_names.len());
        for feature_name in &self.feature_names {
            let column = df.column(feature_name).ok_or_else(|| {
                Error::Column(format!("Feature column '{}' not found", feature_name))
            })?;
            feature_columns.push(column);
        }
        
        // 各行の確率を計算
        for row_idx in 0..n_rows {
            let mut features = Vec::with_capacity(self.feature_names.len());
            
            // 行の特徴量を取得
            for column in &feature_columns {
                let value = match column.column_type() {
                    crate::column::ColumnType::Float64 => {
                        column.get_f64(row_idx)?.unwrap_or(0.0)
                    },
                    crate::column::ColumnType::Int64 => {
                        column.get_i64(row_idx)?.map(|v| v as f64).unwrap_or(0.0)
                    },
                    _ => 0.0
                };
                
                features.push(value);
            }
            
            // 確率を計算
            let prob = self.predict_proba_row(&features);
            proba_class1.push(prob);
            proba_class0.push(1.0 - prob);
        }
        
        // 結果を新しいデータフレームとして返す
        let mut result_df = OptimizedDataFrame::new();
        
        // 確率列を追加
        let class0_col = Float64Column::new(proba_class0, false, "probability_0".to_string())?;
        let class1_col = Float64Column::new(proba_class1, false, "probability_1".to_string())?;
        
        result_df.add_float_column("probability_0", class0_col)?;
        result_df.add_float_column("probability_1", class1_col)?;
        
        Ok(result_df)
    }
    
    /// 確率勾配降下法によるロジスティック回帰の学習
    fn fit_logistic_regression(&mut self, x: &[Vec<f64>], y: &[f64], feature_names: &[&str]) -> Result<()> {
        if x.is_empty() || y.is_empty() {
            return Err(Error::Empty("Empty data for logistic regression".to_string()));
        }
        
        let n_samples = x.len();
        let n_features = x[0].len();
        
        if n_features != feature_names.len() {
            return Err(Error::Consistency(format!(
                "Number of features ({}) doesn't match feature names length ({})",
                n_features, feature_names.len()
            )));
        }
        
        // パラメータの初期化
        let mut weights = vec![0.0; n_features];
        let mut intercept = 0.0;
        
        // 確率的勾配降下法
        for _ in 0..self.max_iter {
            let mut weights_grad = vec![0.0; n_features];
            let mut intercept_grad = 0.0;
            let mut loss = 0.0;
            
            // 勾配の計算
            for i in 0..n_samples {
                let mut z = intercept;
                for j in 0..n_features {
                    z += weights[j] * x[i][j];
                }
                
                let y_pred = self.sigmoid(z);
                let error = y_pred - y[i];
                
                // 損失関数（交差エントロピー）の計算
                if y[i] > 0.5 {
                    loss -= (y_pred + 1e-15).ln();
                } else {
                    loss -= (1.0 - y_pred + 1e-15).ln();
                }
                
                // 勾配の更新
                intercept_grad += error;
                for j in 0..n_features {
                    weights_grad[j] += error * x[i][j];
                }
            }
            
            // 平均損失
            loss /= n_samples as f64;
            
            // パラメータの更新
            intercept -= self.learning_rate * intercept_grad / n_samples as f64;
            for j in 0..n_features {
                weights[j] -= self.learning_rate * weights_grad[j] / n_samples as f64;
            }
            
            // 収束判定
            if weights_grad.iter().all(|&g| g.abs() < self.tol) && intercept_grad.abs() < self.tol {
                break;
            }
        }
        
        // モデルパラメータを設定
        self.intercept = intercept;
        self.coefficients = weights;
        self.feature_names = feature_names.iter().map(|&s| s.to_string()).collect();
        self.fitted = true;
        
        Ok(())
    }
}

impl SupervisedModel for LogisticRegression {
    fn fit(&mut self, df: &OptimizedDataFrame, target: &str, features: &[&str]) -> Result<()> {
        // 特徴量データを抽出
        let mut x_data: Vec<Vec<f64>> = Vec::with_capacity(df.row_count());
        for _ in 0..df.row_count() {
            x_data.push(Vec::with_capacity(features.len()));
        }
        
        // 特徴量の値を取得
        for &feature in features {
            let column = df.column(feature).ok_or_else(|| {
                Error::Column(format!("Feature column '{}' not found", feature))
            })?;
            
            let values = self.extract_numeric_values(&column)?;
            
            // 行列に追加
            for (i, &value) in values.iter().enumerate() {
                if i < x_data.len() {
                    x_data[i].push(value);
                }
            }
        }
        
        // 目標変数の値を取得（バイナリ分類用に0/1に変換）
        let target_column = df.column(target).ok_or_else(|| {
            Error::Column(format!("Target column '{}' not found", target))
        })?;
        
        let target_values = self.extract_target_values(&target_column)?;
        
        // ロジスティック回帰を実行
        self.fit_logistic_regression(&x_data, &target_values, features)?;
        
        Ok(())
    }
    
    fn predict(&self, df: &OptimizedDataFrame) -> Result<Vec<f64>> {
        // 確率を計算
        let proba_df = self.predict_proba(df)?;
        
        // 確率が0.5以上なら1、そうでなければ0
        let proba_col = proba_df.column("probability_1").ok_or_else(|| {
            Error::Column("Probability column not found".to_string())
        })?;
        
        let mut predictions = Vec::with_capacity(proba_col.len());
        
        for i in 0..proba_col.len() {
            let prob = proba_col.get_f64(i)?.unwrap_or(0.0);
            let prediction = if prob >= 0.5 { 1.0 } else { 0.0 };
            predictions.push(prediction);
        }
        
        Ok(predictions)
    }
}

impl LogisticRegression {
    // 分類の目標値（カテゴリカル）を数値に変換
    fn extract_target_values(&self, col: &ColumnView) -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(col.len());
        
        match col.column_type() {
            // 数値型の場合はそのまま使用
            crate::column::ColumnType::Float64 => {
                for i in 0..col.len() {
                    if let Some(value) = col.get_f64(i)? {
                        values.push(if value > 0.5 { 1.0 } else { 0.0 });
                    } else {
                        values.push(0.0); // 欠損値は0として扱う
                    }
                }
            },
            crate::column::ColumnType::Int64 => {
                for i in 0..col.len() {
                    if let Some(value) = col.get_i64(i)? {
                        values.push(if value != 0 { 1.0 } else { 0.0 });
                    } else {
                        values.push(0.0); // 欠損値は0として扱う
                    }
                }
            },
            // 文字列型の場合は"1", "true", "yes"などをポジティブクラスとみなす
            crate::column::ColumnType::String => {
                for i in 0..col.len() {
                    if let Some(value) = col.get_string(i)? {
                        let lower_val = value.to_lowercase();
                        let is_positive = lower_val == "1" || 
                                        lower_val == "true" || 
                                        lower_val == "yes" || 
                                        lower_val == "t" || 
                                        lower_val == "y";
                        values.push(if is_positive { 1.0 } else { 0.0 });
                    } else {
                        values.push(0.0); // 欠損値は0として扱う
                    }
                }
            },
            // ブール型の場合はtrueを1、falseを0に変換
            crate::column::ColumnType::Boolean => {
                for i in 0..col.len() {
                    if let Some(value) = col.get_bool(i)? {
                        values.push(if value { 1.0 } else { 0.0 });
                    } else {
                        values.push(0.0); // 欠損値は0として扱う
                    }
                }
            },
            _ => return Err(Error::Type(format!("Column type {:?} cannot be used as target for classification", col.column_type())))
        }
        
        Ok(values)
    }
}

/// モデル選択モジュール - 訓練/テスト分割と交差検証
pub mod model_selection {
    use std::marker::PhantomData;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    use crate::optimized::OptimizedDataFrame;
    use crate::error::{Result, Error};
    use crate::column::{Column, ColumnTrait, Float64Column};
    use super::SupervisedModel;
    
    /// データセットを訓練セットとテストセットに分割
    pub fn train_test_split(
        df: &OptimizedDataFrame,
        test_size: f64,
        random_state: Option<u64>
    ) -> Result<(OptimizedDataFrame, OptimizedDataFrame)> {
        if test_size <= 0.0 || test_size >= 1.0 {
            return Err(Error::InvalidValue(format!(
                "Invalid test_size: {}, must be between 0 and 1", test_size
            )));
        }
        
        let n_rows = df.row_count();
        let n_test = (n_rows as f64 * test_size).round() as usize;
        let n_train = n_rows - n_test;
        
        if n_train == 0 || n_test == 0 {
            return Err(Error::InvalidOperation(
                "Both train and test splits must have at least one sample".to_string()
            ));
        }
        
        // インデックスの作成
        let mut indices: Vec<usize> = (0..n_rows).collect();
        
        // ランダムにシャッフル
        let mut rng = match random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        
        // Fisher-Yatesアルゴリズムでシャッフル
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
        
        // 訓練セットとテストセットに分割
        let train_indices = &indices[0..n_train];
        let test_indices = &indices[n_train..];
        
        // 新しいデータフレームを作成
        let mut train_df = OptimizedDataFrame::new();
        let mut test_df = OptimizedDataFrame::new();
        
        // 列を追加
        for col_name in df.column_names() {
            let column = df.column(col_name).unwrap();
            
            // 訓練データ用の列を作成
            let train_column = match column.column_type() {
                crate::column::ColumnType::Float64 => {
                    let mut train_values = Vec::with_capacity(n_train);
                    for &idx in train_indices {
                        train_values.push(column.get_f64(idx)?.unwrap_or(0.0));
                    }
                    Column::Float64(Float64Column::new(train_values, false, col_name.clone())?)
                },
                crate::column::ColumnType::Int64 => {
                    let mut train_values = Vec::with_capacity(n_train);
                    for &idx in train_indices {
                        train_values.push(column.get_i64(idx)?.unwrap_or(0));
                    }
                    Column::Int64(crate::column::Int64Column::new(train_values, false, col_name.clone())?)
                },
                crate::column::ColumnType::String => {
                    let mut train_values = Vec::with_capacity(n_train);
                    for &idx in train_indices {
                        train_values.push(column.get_string(idx)?.unwrap_or_default());
                    }
                    Column::String(crate::column::StringColumn::new(train_values, col_name.clone())?)
                },
                crate::column::ColumnType::Boolean => {
                    let mut train_values = Vec::with_capacity(n_train);
                    for &idx in train_indices {
                        train_values.push(column.get_bool(idx)?.unwrap_or(false));
                    }
                    Column::Boolean(crate::column::BooleanColumn::new(train_values, false, col_name.clone())?)
                },
            };
            
            // テストデータ用の列を作成
            let test_column = match column.column_type() {
                crate::column::ColumnType::Float64 => {
                    let mut test_values = Vec::with_capacity(n_test);
                    for &idx in test_indices {
                        test_values.push(column.get_f64(idx)?.unwrap_or(0.0));
                    }
                    Column::Float64(Float64Column::new(test_values, false, col_name.clone())?)
                },
                crate::column::ColumnType::Int64 => {
                    let mut test_values = Vec::with_capacity(n_test);
                    for &idx in test_indices {
                        test_values.push(column.get_i64(idx)?.unwrap_or(0));
                    }
                    Column::Int64(crate::column::Int64Column::new(test_values, false, col_name.clone())?)
                },
                crate::column::ColumnType::String => {
                    let mut test_values = Vec::with_capacity(n_test);
                    for &idx in test_indices {
                        test_values.push(column.get_string(idx)?.unwrap_or_default());
                    }
                    Column::String(crate::column::StringColumn::new(test_values, col_name.clone())?)
                },
                crate::column::ColumnType::Boolean => {
                    let mut test_values = Vec::with_capacity(n_test);
                    for &idx in test_indices {
                        test_values.push(column.get_bool(idx)?.unwrap_or(false));
                    }
                    Column::Boolean(crate::column::BooleanColumn::new(test_values, false, col_name.clone())?)
                },
            };
            
            // データフレームに列を追加
            train_df.add_column(col_name.clone(), train_column)?;
            test_df.add_column(col_name.clone(), test_column)?;
        }
        
        Ok((train_df, test_df))
    }
    
    /// K分割交差検証によるモデル評価
    pub fn cross_val_score<M>(
        model: &M,
        df: &OptimizedDataFrame,
        target: &str,
        features: &[&str],
        k_folds: usize
    ) -> Result<Vec<f64>>
    where
        M: SupervisedModel + Clone,
    {
        if k_folds < 2 {
            return Err(Error::InvalidValue(
                "Number of folds must be at least 2".to_string()
            ));
        }
        
        let n_rows = df.row_count();
        if n_rows < k_folds {
            return Err(Error::InvalidOperation(format!(
                "Cannot perform {}-fold cross validation with only {} samples",
                k_folds, n_rows
            )));
        }
        
        // フォールド分割の準備
        let fold_size = n_rows / k_folds;
        let remainder = n_rows % k_folds;
        
        // 各フォールドのスコアを格納
        let mut scores = Vec::with_capacity(k_folds);
        
        // 各フォールドでモデルをトレーニング＆評価
        for fold_idx in 0..k_folds {
            // テストデータの範囲を決定
            let test_start = fold_idx * fold_size + fold_idx.min(remainder);
            let test_end = test_start + fold_size + if fold_idx < remainder { 1 } else { 0 };
            
            // 訓練・テストデータフレームを作成
            let mut train_df = OptimizedDataFrame::new();
            let mut test_df = OptimizedDataFrame::new();
            
            // 各列についてデータを分割
            for col_name in df.column_names() {
                let column = df.column(col_name).unwrap();
                
                // 訓練データとテストデータの列を作成
                match column.column_type() {
                    crate::column::ColumnType::Float64 => {
                        let mut train_values = Vec::with_capacity(n_rows - (test_end - test_start));
                        let mut test_values = Vec::with_capacity(test_end - test_start);
                        
                        for i in 0..n_rows {
                            let value = column.get_f64(i)?.unwrap_or(0.0);
                            if i < test_start || i >= test_end {
                                train_values.push(value);
                            } else {
                                test_values.push(value);
                            }
                        }
                        
                        let train_col = Float64Column::new(train_values, false, col_name.clone())?;
                        let test_col = Float64Column::new(test_values, false, col_name.clone())?;
                        
                        train_df.add_float_column(col_name, train_col)?;
                        test_df.add_float_column(col_name, test_col)?;
                    },
                    crate::column::ColumnType::Int64 => {
                        let mut train_values = Vec::with_capacity(n_rows - (test_end - test_start));
                        let mut test_values = Vec::with_capacity(test_end - test_start);
                        
                        for i in 0..n_rows {
                            let value = column.get_i64(i)?.unwrap_or(0);
                            if i < test_start || i >= test_end {
                                train_values.push(value);
                            } else {
                                test_values.push(value);
                            }
                        }
                        
                        let train_col = crate::column::Int64Column::new(train_values, false, col_name.clone())?;
                        let test_col = crate::column::Int64Column::new(test_values, false, col_name.clone())?;
                        
                        train_df.add_int_column(col_name, train_col)?;
                        test_df.add_int_column(col_name, test_col)?;
                    },
                    crate::column::ColumnType::String => {
                        let mut train_values = Vec::with_capacity(n_rows - (test_end - test_start));
                        let mut test_values = Vec::with_capacity(test_end - test_start);
                        
                        for i in 0..n_rows {
                            let value = column.get_string(i)?.unwrap_or_default();
                            if i < test_start || i >= test_end {
                                train_values.push(value);
                            } else {
                                test_values.push(value);
                            }
                        }
                        
                        let train_col = crate::column::StringColumn::new(train_values, col_name.clone())?;
                        let test_col = crate::column::StringColumn::new(test_values, col_name.clone())?;
                        
                        train_df.add_string_column(col_name, train_col)?;
                        test_df.add_string_column(col_name, test_col)?;
                    },
                    crate::column::ColumnType::Boolean => {
                        let mut train_values = Vec::with_capacity(n_rows - (test_end - test_start));
                        let mut test_values = Vec::with_capacity(test_end - test_start);
                        
                        for i in 0..n_rows {
                            let value = column.get_bool(i)?.unwrap_or(false);
                            if i < test_start || i >= test_end {
                                train_values.push(value);
                            } else {
                                test_values.push(value);
                            }
                        }
                        
                        let train_col = crate::column::BooleanColumn::new(train_values, false, col_name.clone())?;
                        let test_col = crate::column::BooleanColumn::new(test_values, false, col_name.clone())?;
                        
                        train_df.add_bool_column(col_name, train_col)?;
                        test_df.add_bool_column(col_name, test_col)?;
                    },
                }
            }
            
            // モデルをクローンして訓練
            let mut fold_model = model.clone();
            fold_model.fit(&train_df, target, features)?;
            
            // テストデータでスコア計算
            let score = fold_model.score(&test_df, target)?;
            scores.push(score);
        }
        
        Ok(scores)
    }
    
    /// データセットをK分割するヘルパー関数
    pub fn k_fold_split(
        df: &OptimizedDataFrame,
        k_folds: usize,
        random_state: Option<u64>
    ) -> Result<Vec<(OptimizedDataFrame, OptimizedDataFrame)>> {
        if k_folds < 2 {
            return Err(Error::InvalidValue(
                "Number of folds must be at least 2".to_string()
            ));
        }
        
        let n_rows = df.row_count();
        if n_rows < k_folds {
            return Err(Error::InvalidOperation(format!(
                "Cannot perform {}-fold cross validation with only {} samples",
                k_folds, n_rows
            )));
        }
        
        // インデックスの作成
        let mut indices: Vec<usize> = (0..n_rows).collect();
        
        // ランダムにシャッフル
        let mut rng = match random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        
        // Fisher-Yatesアルゴリズムでシャッフル
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
        
        // フォールドサイズを計算
        let fold_size = n_rows / k_folds;
        let remainder = n_rows % k_folds;
        
        // 各フォールドの分割を作成
        let mut folds = Vec::with_capacity(k_folds);
        
        for fold_idx in 0..k_folds {
            // テストデータの範囲を決定
            let test_start = fold_idx * fold_size + fold_idx.min(remainder);
            let test_end = test_start + fold_size + if fold_idx < remainder { 1 } else { 0 };
            
            // 訓練・テストインデックスを抽出
            let test_indices: Vec<usize> = indices[test_start..test_end].to_vec();
            let train_indices: Vec<usize> = indices.iter()
                .enumerate()
                .filter(|&(i, _)| i < test_start || i >= test_end)
                .map(|(_, &idx)| idx)
                .collect();
            
            // 訓練・テストデータフレームを作成
            let mut train_df = OptimizedDataFrame::new();
            let mut test_df = OptimizedDataFrame::new();
            
            // 各列についてデータを分割
            for col_name in df.column_names() {
                let column = df.column(col_name).unwrap();
                
                // インデックスに基づいて値を抽出
                match column.column_type() {
                    crate::column::ColumnType::Float64 => {
                        let mut train_values = Vec::with_capacity(train_indices.len());
                        let mut test_values = Vec::with_capacity(test_indices.len());
                        
                        for &idx in &train_indices {
                            train_values.push(column.get_f64(idx)?.unwrap_or(0.0));
                        }
                        
                        for &idx in &test_indices {
                            test_values.push(column.get_f64(idx)?.unwrap_or(0.0));
                        }
                        
                        let train_col = Float64Column::new(train_values, false, col_name.clone())?;
                        let test_col = Float64Column::new(test_values, false, col_name.clone())?;
                        
                        train_df.add_float_column(col_name, train_col)?;
                        test_df.add_float_column(col_name, test_col)?;
                    },
                    // 他のデータ型も同様に処理
                    _ => {
                        // 簡略化のため省略
                    }
                }
            }
            
            folds.push((train_df, test_df));
        }
        
        Ok(folds)
    }
}

/// モデル永続化モジュール - モデルの保存と読み込み
pub mod model_persistence {
    use serde::{Serialize, Deserialize};
    use std::path::Path;
    use std::fs::{File, create_dir_all};
    use std::io::{BufReader, BufWriter, Write};
    use crate::error::{Result, Error};
    
    /// モデル永続化トレイト
    pub trait ModelPersistence: Sized {
        /// モデルをJSONファイルとして保存
        fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<()>;
        
        /// JSONファイルからモデルを読み込み
        fn load_model<P: AsRef<Path>>(path: P) -> Result<Self>;
    }
    
    /// 線形回帰モデルの永続化用データ構造
    #[derive(Serialize, Deserialize)]
    struct LinearRegressionData {
        coefficients: Vec<f64>,
        intercept: f64,
        feature_names: Vec<String>,
    }
    
    /// ロジスティック回帰モデルの永続化用データ構造
    #[derive(Serialize, Deserialize)]
    struct LogisticRegressionData {
        coefficients: Vec<f64>,
        intercept: f64,
        feature_names: Vec<String>,
        learning_rate: f64,
        max_iter: usize,
        tol: f64,
    }
    
    impl ModelPersistence for super::LinearRegression {
        fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<()> {
            if !self.fitted {
                return Err(Error::InvalidOperation(
                    "Cannot save unfitted model".to_string()
                ));
            }
            
            // モデルデータの作成
            let model_data = LinearRegressionData {
                coefficients: self.coefficients.clone(),
                intercept: self.intercept,
                feature_names: self.feature_names.clone(),
            };
            
            // ディレクトリが存在しない場合は作成
            if let Some(parent) = path.as_ref().parent() {
                if !parent.exists() {
                    create_dir_all(parent)?;
                }
            }
            
            // JSONとして保存
            let file = File::create(path)?;
            let writer = BufWriter::new(file);
            serde_json::to_writer_pretty(writer, &model_data)?;
            
            Ok(())
        }
        
        fn load_model<P: AsRef<Path>>(path: P) -> Result<Self> {
            // ファイルが存在するか確認
            if !path.as_ref().exists() {
                return Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Model file not found: {:?}", path.as_ref())
                )));
            }
            
            // JSONからデータを読み込み
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            let model_data: LinearRegressionData = serde_json::from_reader(reader)?;
            
            // モデルの再構築
            let mut model = super::LinearRegression::new();
            model.coefficients = model_data.coefficients;
            model.intercept = model_data.intercept;
            model.feature_names = model_data.feature_names;
            model.fitted = true;
            
            Ok(model)
        }
    }
    
    impl ModelPersistence for super::LogisticRegression {
        fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<()> {
            if !self.fitted {
                return Err(Error::InvalidOperation(
                    "Cannot save unfitted model".to_string()
                ));
            }
            
            // モデルデータの作成
            let model_data = LogisticRegressionData {
                coefficients: self.coefficients.clone(),
                intercept: self.intercept,
                feature_names: self.feature_names.clone(),
                learning_rate: self.learning_rate,
                max_iter: self.max_iter,
                tol: self.tol,
            };
            
            // ディレクトリが存在しない場合は作成
            if let Some(parent) = path.as_ref().parent() {
                if !parent.exists() {
                    create_dir_all(parent)?;
                }
            }
            
            // JSONとして保存
            let file = File::create(path)?;
            let writer = BufWriter::new(file);
            serde_json::to_writer_pretty(writer, &model_data)?;
            
            Ok(())
        }
        
        fn load_model<P: AsRef<Path>>(path: P) -> Result<Self> {
            // ファイルが存在するか確認
            if !path.as_ref().exists() {
                return Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Model file not found: {:?}", path.as_ref())
                )));
            }
            
            // JSONからデータを読み込み
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            let model_data: LogisticRegressionData = serde_json::from_reader(reader)?;
            
            // モデルの再構築
            let mut model = super::LogisticRegression::new(
                model_data.learning_rate,
                model_data.max_iter,
                model_data.tol
            );
            model.coefficients = model_data.coefficients;
            model.intercept = model_data.intercept;
            model.feature_names = model_data.feature_names;
            model.fitted = true;
            
            Ok(model)
        }
    }
}