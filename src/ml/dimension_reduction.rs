//! 次元削減モジュール
//!
//! 高次元データの可視化と分析のための次元削減アルゴリズムを提供します。

use crate::optimized::{OptimizedDataFrame, ColumnView};
use crate::column::{Float64Column, Int64Column, Column};
use crate::column::ColumnTrait;
use crate::error::{Result, Error};
use crate::ml::pipeline::Transformer;
use std::collections::HashMap;
use rand::Rng;

/// 主成分分析（PCA）の実装
#[derive(Debug)]
pub struct PCA {
    /// 削減後の次元数
    n_components: usize,
    /// 各主成分の分散説明率
    explained_variance_ratio: Vec<f64>,
    /// 累積分散説明率
    cumulative_explained_variance: Vec<f64>,
    /// 主成分の固有ベクトル
    components: Vec<Vec<f64>>,
    /// 各特徴量の平均値
    mean: Vec<f64>,
    /// 各特徴量の標準偏差
    std: Vec<f64>,
    /// 特徴量の名前
    feature_names: Vec<String>,
    /// 学習済みかどうか
    fitted: bool,
}

impl PCA {
    /// 新しいPCAインスタンスを作成
    pub fn new(n_components: usize) -> Self {
        PCA {
            n_components,
            explained_variance_ratio: Vec::new(),
            cumulative_explained_variance: Vec::new(),
            components: Vec::new(),
            mean: Vec::new(),
            std: Vec::new(),
            feature_names: Vec::new(),
            fitted: false,
        }
    }
    
    /// 分散説明率を取得
    pub fn explained_variance_ratio(&self) -> &[f64] {
        &self.explained_variance_ratio
    }
    
    /// 累積分散説明率を取得
    pub fn cumulative_explained_variance(&self) -> &[f64] {
        &self.cumulative_explained_variance
    }
    
    /// 主成分の固有ベクトルを取得
    pub fn components(&self) -> &[Vec<f64>] {
        &self.components
    }
    
    /// データ行列からの共分散行列の計算
    fn compute_covariance_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n_samples = data.len();
        let n_features = data[0].len();
        
        // 特徴量ごとの平均を計算
        let mut mean = vec![0.0; n_features];
        for sample in data {
            for (j, &val) in sample.iter().enumerate() {
                mean[j] += val;
            }
        }
        
        for j in 0..n_features {
            mean[j] /= n_samples as f64;
        }
        
        // 中心化したデータを作成
        let centered_data: Vec<Vec<f64>> = data
            .iter()
            .map(|sample| {
                sample
                    .iter()
                    .enumerate()
                    .map(|(j, &val)| val - mean[j])
                    .collect()
            })
            .collect();
        
        // 共分散行列の計算
        let mut cov = vec![vec![0.0; n_features]; n_features];
        
        for i in 0..n_features {
            for j in 0..n_features {
                let mut sum = 0.0;
                for sample in &centered_data {
                    sum += sample[i] * sample[j];
                }
                cov[i][j] = sum / (n_samples as f64 - 1.0);
            }
        }
        
        cov
    }
    
    /// べき乗法による最大固有値と対応する固有ベクトルの計算
    fn power_iteration(matrix: &[Vec<f64>], tol: f64, max_iter: usize) -> (f64, Vec<f64>) {
        let n = matrix.len();
        
        // ランダムな初期ベクトル（単位ベクトル）
        let mut vec = vec![1.0 / (n as f64).sqrt(); n];
        
        // べき乗法の反復
        for _ in 0..max_iter {
            // 行列ベクトル積
            let mut new_vec = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    new_vec[i] += matrix[i][j] * vec[j];
                }
            }
            
            // ノルムの計算
            let norm: f64 = new_vec.iter().map(|&x| x * x).sum::<f64>().sqrt();
            
            // 収束判定
            let mut converged = true;
            for i in 0..n {
                let v = new_vec[i] / norm;
                if (v - vec[i]).abs() > tol {
                    converged = false;
                }
                vec[i] = v;
            }
            
            if converged {
                break;
            }
        }
        
        // 固有値の計算（レイリー商）
        let mut eigenvalue = 0.0;
        let mut denom = 0.0;
        
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += matrix[i][j] * vec[j];
            }
            eigenvalue += vec[i] * sum;
            denom += vec[i] * vec[i];
        }
        
        eigenvalue /= denom;
        
        (eigenvalue, vec)
    }
    
    /// デフレーション処理：行列から固有ベクトルの寄与を取り除く
    fn deflate(matrix: &mut [Vec<f64>], eigenvalue: f64, eigenvector: &[f64]) {
        let n = matrix.len();
        
        for i in 0..n {
            for j in 0..n {
                matrix[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
            }
        }
    }
    
    // 標準偏差を計算する関数（カラムから）
    fn compute_std(values: &[f64], mean: f64) -> f64 {
        if values.is_empty() {
            return 1.0;  // デフォルト値
        }
        
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (values.len() as f64);
        
        variance.sqrt()
    }
}

impl Transformer for PCA {
    fn fit(&mut self, df: &OptimizedDataFrame) -> Result<()> {
        // 数値列のみ抽出
        let numeric_columns: Vec<String> = df.column_names()
            .into_iter()
            .filter(|col_name| {
                if let Ok(col_view) = df.column(col_name) {
                    col_view.as_float64().is_some() || col_view.as_int64().is_some()
                } else {
                    false
                }
            })
            .map(|s| s.to_string())  // ここを修正: &Stringをクローンして所有権を取得
            .collect();
        
        if numeric_columns.is_empty() {
            return Err(Error::InvalidOperation(
                "DataFrame must contain at least one numeric column for PCA".to_string()
            ));
        }
        
        self.feature_names = numeric_columns.clone();
        let n_features = self.feature_names.len();
        
        // n_componentsが特徴量数を超えないように調整
        let n_components = self.n_components.min(n_features);
        self.n_components = n_components;
        
        // データの準備
        let n_samples = df.row_count();
        let mut data = vec![vec![0.0; n_features]; n_samples];
        self.mean = vec![0.0; n_features];
        self.std = vec![1.0; n_features];
        
        // データの読み込みと標準化
        for (col_idx, col_name) in self.feature_names.iter().enumerate() {
            let col_view = df.column(col_name)?;
            
            if let Some(float_col) = col_view.as_float64() {
                // データの読み込み
                let mut values = Vec::with_capacity(n_samples);
                for row_idx in 0..n_samples {
                    if let Ok(Some(value)) = float_col.get(row_idx) {
                        values.push(value);
                        data[row_idx][col_idx] = value;
                    }
                }
                
                // 平均と標準偏差を計算
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let std = Self::compute_std(&values, mean);
                
                self.mean[col_idx] = mean;
                self.std[col_idx] = std;
                
                // データの標準化
                for row_idx in 0..n_samples {
                    if std > 0.0 {
                        data[row_idx][col_idx] = (data[row_idx][col_idx] - mean) / std;
                    }
                }
            } else if let Some(int_col) = col_view.as_int64() {
                // データの読み込み
                let mut values = Vec::with_capacity(n_samples);
                for row_idx in 0..n_samples {
                    if let Ok(Some(value)) = int_col.get(row_idx) {
                        values.push(value as f64);
                        data[row_idx][col_idx] = value as f64;
                    }
                }
                
                // 平均と標準偏差を計算
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let std = Self::compute_std(&values, mean);
                
                self.mean[col_idx] = mean;
                self.std[col_idx] = std;
                
                // データの標準化
                for row_idx in 0..n_samples {
                    if std > 0.0 {
                        data[row_idx][col_idx] = (data[row_idx][col_idx] - mean) / std;
                    }
                }
            }
        }
        
        // 共分散行列の計算
        let mut cov_matrix = Self::compute_covariance_matrix(&data);
        
        // 固有値分解
        let mut eigenvalues = Vec::with_capacity(n_components);
        self.components = Vec::with_capacity(n_components);
        
        // べき乗法で上位n_components個の固有値と固有ベクトルを計算
        for _ in 0..n_components {
            let (eigenvalue, eigenvector) = Self::power_iteration(&cov_matrix, 1e-10, 100);
            eigenvalues.push(eigenvalue);
            self.components.push(eigenvector.clone());
            
            // デフレーション
            Self::deflate(&mut cov_matrix, eigenvalue, &eigenvector);
        }
        
        // 固有値の合計（全分散）
        let total_variance: f64 = eigenvalues.iter().sum();
        
        // 分散説明率の計算
        self.explained_variance_ratio = eigenvalues
            .iter()
            .map(|&val| val / total_variance)
            .collect();
        
        // 累積分散説明率の計算
        self.cumulative_explained_variance = Vec::with_capacity(n_components);
        let mut cum_sum = 0.0;
        for &ratio in &self.explained_variance_ratio {
            cum_sum += ratio;
            self.cumulative_explained_variance.push(cum_sum);
        }
        
        self.fitted = true;
        
        Ok(())
    }
    
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        if !self.fitted {
            return Err(Error::InvalidOperation(
                "PCA has not been fitted yet".to_string()
            ));
        }
        
        let n_samples = df.row_count();
        let n_features = self.feature_names.len();
        
        // 標準化されたデータを格納する行列
        let mut data = vec![vec![0.0; n_features]; n_samples];
        
        // データの読み込みと標準化
        for (col_idx, col_name) in self.feature_names.iter().enumerate() {
            let col_view = df.column(col_name).map_err(|_| {
                Error::InvalidOperation(
                    format!("Column '{}' not found in DataFrame", col_name)
                )
            })?;
            
            if let Some(float_col) = col_view.as_float64() {
                // データの読み込みと標準化
                for row_idx in 0..n_samples {
                    if let Ok(Some(value)) = float_col.get(row_idx) {
                        if self.std[col_idx] > 0.0 {
                            data[row_idx][col_idx] = (value - self.mean[col_idx]) / self.std[col_idx];
                        }
                    }
                }
            } else if let Some(int_col) = col_view.as_int64() {
                // データの読み込みと標準化
                for row_idx in 0..n_samples {
                    if let Ok(Some(value)) = int_col.get(row_idx) {
                        if self.std[col_idx] > 0.0 {
                            data[row_idx][col_idx] = ((value as f64) - self.mean[col_idx]) / self.std[col_idx];
                        }
                    }
                }
            }
        }
        
        // 主成分に変換
        let mut transformed_data = vec![vec![0.0; self.n_components]; n_samples];
        
        for i in 0..n_samples {
            for j in 0..self.n_components {
                let mut pc_value = 0.0;
                for k in 0..n_features {
                    pc_value += data[i][k] * self.components[j][k];
                }
                transformed_data[i][j] = pc_value;
            }
        }
        
        // OptimizedDataFrameに変換
        let mut result_df = OptimizedDataFrame::new();
        
        // 主成分列を追加
        for j in 0..self.n_components {
            let mut pc_values = Vec::with_capacity(n_samples);
            
            for i in 0..n_samples {
                pc_values.push(transformed_data[i][j]);
            }
            
            let pc_col = Float64Column::new(pc_values);
            result_df.add_column(format!("PC{}", j + 1), Column::Float64(pc_col))?;
        }
        
        // 非数値列があれば、そのまま追加
        for col_name in df.column_names() {
            if !self.feature_names.contains(&col_name) {
                if let Ok(col_view) = df.column(&col_name) {
                    if let Some(str_col) = col_view.as_string() {
                        // 文字列列の場合
                        let mut values = Vec::with_capacity(n_samples);
                        for i in 0..n_samples {
                            if let Ok(Some(value)) = str_col.get(i) {
                                values.push(value.to_string());
                            } else {
                                values.push("".to_string());
                            }
                        }
                        let string_col = Column::String(crate::column::StringColumn::new(values));
                        result_df.add_column(col_name.clone(), string_col)?;
                    } else if let Some(bool_col) = col_view.as_boolean() {
                        // ブール列の場合
                        let mut values = Vec::with_capacity(n_samples);
                        for i in 0..n_samples {
                            if let Ok(Some(value)) = bool_col.get(i) {
                                values.push(value);
                            } else {
                                values.push(false);
                            }
                        }
                        let bool_col = Column::Boolean(crate::column::BooleanColumn::new(values));
                        result_df.add_column(col_name.clone(), bool_col)?;
                    }
                }
            }
        }
        
        Ok(result_df)
    }
    
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

/// t-SNE (t-distributed Stochastic Neighbor Embedding) の実装
#[derive(Debug)]
pub struct TSNE {
    /// 削減後の次元数（通常は2または3）
    n_components: usize,
    /// 学習率
    learning_rate: f64,
    /// パープレキシティ（近傍サイズを制御するパラメータ）
    perplexity: f64,
    /// 最大反復回数
    max_iter: usize,
    /// 初期化方法
    init: TSNEInit,
    /// 埋め込み後の座標
    embedding: Vec<Vec<f64>>,
    /// 特徴量の名前
    feature_names: Vec<String>,
    /// 学習済みかどうか
    fitted: bool,
}

/// t-SNEの初期化方法
#[derive(Debug)]
pub enum TSNEInit {
    /// ランダム初期化
    Random,
    /// PCAによる初期化
    PCA,
}

impl TSNE {
    /// 新しいTSNEインスタンスを作成
    pub fn new(
        n_components: usize,
        perplexity: f64,
        learning_rate: f64,
        max_iter: usize,
        init: TSNEInit,
    ) -> Self {
        TSNE {
            n_components,
            learning_rate,
            perplexity,
            max_iter,
            init,
            embedding: Vec::new(),
            feature_names: Vec::new(),
            fitted: false,
        }
    }
    
    /// 埋め込みを取得
    pub fn embedding(&self) -> &[Vec<f64>] {
        &self.embedding
    }
    
    /// ユークリッド距離の二乗を計算
    fn squared_euclidean_distance(x: &[f64], y: &[f64]) -> f64 {
        x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - yi).powi(2))
            .sum()
    }
    
    /// 条件付き確率行列P（高次元空間での類似度）を計算
    fn compute_pairwise_affinities(data: &[Vec<f64>], perplexity: f64) -> Vec<Vec<f64>> {
        let n_samples = data.len();
        let mut p = vec![vec![0.0; n_samples]; n_samples];
        
        // 条件付き確率を計算
        for i in 0..n_samples {
            // 二分探索でシグマを求める
            let mut beta = 1.0; // ベータ = 1 / (2 * シグマ^2)
            let mut beta_min = 0.0;
            let mut beta_max = f64::INFINITY;
            
            let target_entropy = perplexity.ln();
            let tol = 1e-5;
            let max_iter = 50;
            
            for _ in 0..max_iter {
                // 条件付き確率を計算
                let mut sum_pi = 0.0;
                for j in 0..n_samples {
                    if i != j {
                        let dist = Self::squared_euclidean_distance(&data[i], &data[j]);
                        p[i][j] = (-beta * dist).exp();
                        sum_pi += p[i][j];
                    }
                }
                
                // 確率分布の正規化
                for j in 0..n_samples {
                    if i != j && sum_pi > 0.0 {
                        p[i][j] /= sum_pi;
                    }
                }
                
                // エントロピーの計算
                let mut entropy = 0.0;
                for j in 0..n_samples {
                    if i != j && p[i][j] > 1e-7 {
                        entropy -= p[i][j] * p[i][j].ln();
                    }
                }
                
                // パープレキシティとの差
                let entropy_diff = entropy - target_entropy;
                
                if entropy_diff.abs() < tol {
                    break;
                }
                
                // ベータの更新
                if entropy_diff > 0.0 {
                    beta_min = beta;
                    if beta_max == f64::INFINITY {
                        beta *= 2.0;
                    } else {
                        beta = (beta + beta_max) / 2.0;
                    }
                } else {
                    beta_max = beta;
                    beta = (beta + beta_min) / 2.0;
                }
            }
        }
        
        // 対称化とスケーリング
        let mut symmetric_p = vec![vec![0.0; n_samples]; n_samples];
        for i in 0..n_samples {
            for j in 0..n_samples {
                symmetric_p[i][j] = (p[i][j] + p[j][i]) / (2.0 * n_samples as f64);
            }
        }
        
        symmetric_p
    }
    
    /// t分布による低次元空間での類似度（Q）を計算
    fn compute_q_matrix(embedding: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n_samples = embedding.len();
        let mut q = vec![vec![0.0; n_samples]; n_samples];
        let mut sum_q = 0.0;
        
        for i in 0..n_samples {
            for j in 0..i {
                let dist = Self::squared_euclidean_distance(&embedding[i], &embedding[j]);
                let q_ij = 1.0 / (1.0 + dist);
                q[i][j] = q_ij;
                q[j][i] = q_ij;
                sum_q += 2.0 * q_ij;
            }
        }
        
        // 正規化
        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j {
                    q[i][j] /= sum_q;
                }
            }
        }
        
        q
    }
    
    /// 勾配を計算
    fn compute_gradient(p: &[Vec<f64>], q: &[Vec<f64>], embedding: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n_samples = embedding.len();
        let n_components = embedding[0].len();
        let mut grad = vec![vec![0.0; n_components]; n_samples];
        
        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j {
                    // (p_ij - q_ij) * (1 + ||y_i - y_j||^2)^-1
                    let factor = (p[i][j] - q[i][j]) * (1.0 + Self::squared_euclidean_distance(&embedding[i], &embedding[j])).powi(-1);
                    
                    for k in 0..n_components {
                        grad[i][k] += 4.0 * factor * (embedding[i][k] - embedding[j][k]);
                    }
                }
            }
        }
        
        grad
    }
}

impl Transformer for TSNE {
    fn fit(&mut self, df: &OptimizedDataFrame) -> Result<()> {
        // 数値列のみ抽出
        let numeric_columns: Vec<String> = df.column_names()
            .into_iter()
            .filter(|col_name| {
                if let Ok(col_view) = df.column(col_name) {
                    col_view.as_float64().is_some() || col_view.as_int64().is_some()
                } else {
                    false
                }
            })
            .map(|s| s.to_string())  // ここを修正
            .collect();
        
        if numeric_columns.is_empty() {
            return Err(Error::InvalidOperation(
                "DataFrame must contain at least one numeric column for t-SNE".to_string()
            ));
        }
        
        self.feature_names = numeric_columns.clone();
        
        // データの準備
        let n_samples = df.row_count();
        let n_features = self.feature_names.len();
        let mut data = vec![vec![0.0; n_features]; n_samples];
        
        // データの読み込み
        for (col_idx, col_name) in self.feature_names.iter().enumerate() {
            let col_view = df.column(col_name)?;
            
            if let Some(float_col) = col_view.as_float64() {
                for row_idx in 0..n_samples {
                    if let Ok(Some(value)) = float_col.get(row_idx) {
                        data[row_idx][col_idx] = value;
                    }
                }
            } else if let Some(int_col) = col_view.as_int64() {
                for row_idx in 0..n_samples {
                    if let Ok(Some(value)) = int_col.get(row_idx) {
                        data[row_idx][col_idx] = value as f64;
                    }
                }
            }
        }
        
        // 条件付き確率行列の計算
        let p = Self::compute_pairwise_affinities(&data, self.perplexity);
        
        // 初期埋め込みの生成
        self.embedding = match self.init {
            TSNEInit::Random => {
                // ランダム初期化
                let mut rng = rand::rng();
                (0..n_samples)
                    .map(|_| {
                        (0..self.n_components)
                            .map(|_| 1e-4 * rng.random_range(-1.0..1.0))
                            .collect()
                    })
                    .collect()
            }
            TSNEInit::PCA => {
                // PCAによる初期化
                let mut pca = PCA::new(self.n_components);
                let pca_result = pca.fit_transform(df)?;
                
                let mut embedding = vec![vec![0.0; self.n_components]; n_samples];
                for i in 0..n_samples {
                    for j in 0..self.n_components {
                        let pc_col = format!("PC{}", j + 1);
                        let col_view = pca_result.column(&pc_col)?;
                        
                        if let Some(float_col) = col_view.as_float64() {
                            if let Ok(Some(value)) = float_col.get(i) {
                                embedding[i][j] = value * 1e-4;
                            }
                        }
                    }
                }
                embedding
            }
        };
        
        // 勾配降下法によるt-SNEの最適化
        let mut gains = vec![vec![1.0; self.n_components]; n_samples];
        let mut velocities = vec![vec![0.0; self.n_components]; n_samples];
        let mut momentum = 0.5;
        
        for iter in 0..self.max_iter {
            // t分布による低次元空間での類似度（Q）を計算
            let q = Self::compute_q_matrix(&self.embedding);
            
            // 勾配を計算
            let grad = Self::compute_gradient(&p, &q, &self.embedding);
            
            // 更新
            if iter == 20 {
                momentum = 0.8;  // 20回の反復後、モメンタムを増加
            }
            
            for i in 0..n_samples {
                for j in 0..self.n_components {
                    // 調整学習率（gains）の更新
                    if grad[i][j] * velocities[i][j] > 0.0 {
                        gains[i][j] = gains[i][j] * 0.8;
                    } else {
                        gains[i][j] = gains[i][j] + 0.2;
                    }
                    
                    gains[i][j] = f64::max(gains[i][j], 0.01);
                    
                    // 速度の更新
                    velocities[i][j] = momentum * velocities[i][j] - 
                                      self.learning_rate * gains[i][j] * grad[i][j];
                    
                    // 埋め込みの更新
                    self.embedding[i][j] += velocities[i][j];
                }
            }
            
            // 埋め込みの正規化（中心を原点に）
            let mut mean = vec![0.0; self.n_components];
            for i in 0..n_samples {
                for j in 0..self.n_components {
                    mean[j] += self.embedding[i][j];
                }
            }
            
            for j in 0..self.n_components {
                mean[j] /= n_samples as f64;
            }
            
            for i in 0..n_samples {
                for j in 0..self.n_components {
                    self.embedding[i][j] -= mean[j];
                }
            }
        }
        
        self.fitted = true;
        
        Ok(())
    }
    
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        if !self.fitted {
            return Err(Error::InvalidOperation(
                "t-SNE has not been fitted yet".to_string()
            ));
        }
        
        // t-SNEは新しいデータポイントを既存の埋め込みに追加することができないため、
        // トレーニングデータと同じデータでなければエラーを返す
        return Err(Error::InvalidOperation(
            "t-SNE does not support the transform method on new data. Use fit_transform instead.".to_string()
        ));
    }
    
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        
        // OptimizedDataFrameに変換
        let n_samples = df.row_count();
        let mut result_df = OptimizedDataFrame::new();
        
        // TSNE次元列を追加
        for j in 0..self.n_components {
            let mut values = Vec::with_capacity(n_samples);
            
            for i in 0..n_samples {
                values.push(self.embedding[i][j]);
            }
            
            let col = Float64Column::new(values);
            result_df.add_column(format!("TSNE{}", j + 1), Column::Float64(col))?;
        }
        
        // 非数値列があれば、そのまま追加
        for col_name in df.column_names() {
            if !self.feature_names.contains(&col_name) {
                if let Ok(col_view) = df.column(&col_name) {
                    if let Some(str_col) = col_view.as_string() {
                        // 文字列列の場合
                        let mut values = Vec::with_capacity(n_samples);
                        for i in 0..n_samples {
                            if let Ok(Some(value)) = str_col.get(i) {
                                values.push(value.to_string());
                            } else {
                                values.push("".to_string());
                            }
                        }
                        let string_col = Column::String(crate::column::StringColumn::new(values));
                        result_df.add_column(col_name.clone(), string_col)?;
                    } else if let Some(bool_col) = col_view.as_boolean() {
                        // ブール列の場合
                        let mut values = Vec::with_capacity(n_samples);
                        for i in 0..n_samples {
                            if let Ok(Some(value)) = bool_col.get(i) {
                                values.push(value);
                            } else {
                                values.push(false);
                            }
                        }
                        let bool_col = Column::Boolean(crate::column::BooleanColumn::new(values));
                        result_df.add_column(col_name.clone(), bool_col)?;
                    }
                }
            }
        }
        
        Ok(result_df)
    }
}