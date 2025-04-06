//! 異常検出モジュール
//!
//! データセットから外れ値や異常パターンを検出するためのアルゴリズムを提供します。

use crate::optimized::{OptimizedDataFrame, ColumnView};
use crate::column::{Float64Column, Int64Column, Column, ColumnTrait};
use crate::error::{Result, Error};
use crate::ml::pipeline::Transformer;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::{HashMap, HashSet};

/// 孤立森（Isolation Forest）異常検出アルゴリズム
pub struct IsolationForest {
    /// 決定木の数
    n_estimators: usize,
    /// サブサンプリングのサイズ
    max_samples: Option<usize>,
    /// 特徴量のサブサンプリング（割合）
    max_features: Option<f64>,
    /// 乱数シード
    random_seed: Option<u64>,
    /// 汚染率（異常値の期待割合）
    contamination: f64,
    /// 各サンプルの異常スコア
    anomaly_scores: Vec<f64>,
    /// 異常のしきい値
    threshold: f64,
    /// 異常フラグ（1: 異常, -1: 正常）
    labels: Vec<i64>,
    /// 特徴量の名前
    feature_names: Vec<String>,
    /// 学習済みかどうか
    fitted: bool,
    /// 木のコレクション
    trees: Vec<ITree>,
}

/// Isolation Forestの決定木
struct ITree {
    /// 木の深さ上限
    height_limit: usize,
    /// 木の根ノード
    root: Option<Box<ITreeNode>>,
}

/// Isolation Forestの決定木のノード
struct ITreeNode {
    /// 分割特徴量のインデックス
    split_feature: Option<usize>,
    /// 分割閾値
    split_threshold: Option<f64>,
    /// 左の子ノード
    left: Option<Box<ITreeNode>>,
    /// 右の子ノード
    right: Option<Box<ITreeNode>>,
    /// 分割を行わない場合の木の深さ
    depth: usize,
    /// このノードのサンプル数
    size: usize,
}

impl IsolationForest {
    /// 新しいIsolationForestインスタンスを作成
    pub fn new(
        n_estimators: usize,
        max_samples: Option<usize>,
        max_features: Option<f64>,
        contamination: f64,
        random_seed: Option<u64>,
    ) -> Self {
        if contamination <= 0.0 || contamination >= 0.5 {
            panic!("Contamination must be in (0, 0.5)");
        }
        
        IsolationForest {
            n_estimators,
            max_samples,
            max_features,
            random_seed,
            contamination,
            anomaly_scores: Vec::new(),
            threshold: 0.0,
            labels: Vec::new(),
            feature_names: Vec::new(),
            fitted: false,
            trees: Vec::new(),
        }
    }
    
    /// 異常スコアを取得
    pub fn anomaly_scores(&self) -> &[f64] {
        &self.anomaly_scores
    }
    
    /// 異常フラグを取得（1: 異常, -1: 正常）
    pub fn labels(&self) -> &[i64] {
        &self.labels
    }
    
    /// 決定木を構築
    fn build_tree(
        &self,
        data: &[Vec<f64>],
        indices: &[usize],
        height_limit: usize,
        depth: usize,
        rng: &mut StdRng,
    ) -> Option<Box<ITreeNode>> {
        // 終了条件
        if indices.is_empty() {
            return None;
        }
        
        if depth >= height_limit || indices.len() <= 1 {
            return Some(Box::new(ITreeNode {
                split_feature: None,
                split_threshold: None,
                left: None,
                right: None,
                depth,
                size: indices.len(),
            }));
        }
        
        // 特徴量のサンプリング
        let n_features = data[0].len();
        let n_features_to_use = match self.max_features {
            Some(ratio) => (ratio * n_features as f64).round() as usize,
            None => n_features,
        };
        
        let feature_indices: Vec<usize> = (0..n_features).collect();
        let sampled_features: Vec<usize> = feature_indices
            .iter()
            .copied()
            .filter(|_| rng.random_bool(n_features_to_use as f64 / n_features as f64))
            .collect();
        
        if sampled_features.is_empty() {
            // 最低1つの特徴量を選択
            return Some(Box::new(ITreeNode {
                split_feature: Some(rng.random_range(0..n_features)),
                split_threshold: Some(rng.random()),
                left: None,
                right: None,
                depth,
                size: indices.len(),
            }));
        }
        
        // ランダムに特徴量と閾値を選択
        let split_feature = sampled_features[rng.random_range(0..sampled_features.len())];
        
        // 選択した特徴量の最小値と最大値を求める
        let min_val = indices.iter().map(|&i| data[i][split_feature]).fold(f64::INFINITY, f64::min);
        let max_val = indices.iter().map(|&i| data[i][split_feature]).fold(f64::NEG_INFINITY, f64::max);
        
        // 最小値と最大値が同じ場合は分割できない
        if (max_val - min_val).abs() < f64::EPSILON {
            return Some(Box::new(ITreeNode {
                split_feature: None,
                split_threshold: None,
                left: None,
                right: None,
                depth,
                size: indices.len(),
            }));
        }
        
        // 閾値をランダムに選択
        let split_threshold = min_val + rng.random::<f64>() * (max_val - min_val);
        
        // データを分割
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        
        for &idx in indices {
            if data[idx][split_feature] < split_threshold {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }
        
        // 左右の子ノードを再帰的に構築
        let left = self.build_tree(data, &left_indices, height_limit, depth + 1, rng);
        let right = self.build_tree(data, &right_indices, height_limit, depth + 1, rng);
        
        Some(Box::new(ITreeNode {
            split_feature: Some(split_feature),
            split_threshold: Some(split_threshold),
            left,
            right,
            depth,
            size: indices.len(),
        }))
    }
    
    /// サンプルの経路長を計算
    fn compute_path_length(node: &Option<Box<ITreeNode>>, x: &[f64], current_height: usize) -> usize {
        match node {
            None => current_height,
            Some(node) => {
                match (node.split_feature, node.split_threshold) {
                    (Some(feature), Some(threshold)) => {
                        if x[feature] < threshold {
                            Self::compute_path_length(&node.left, x, current_height + 1)
                        } else {
                            Self::compute_path_length(&node.right, x, current_height + 1)
                        }
                    }
                    _ => current_height + Self::c_factor(node.size),
                }
            }
        }
    }
    
    /// 調整係数c(n)を計算
    fn c_factor(n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        
        let n = n as f64;
        let h = 2.0 * (n - 1.0).ln() + 0.5772156649; // オイラー定数
        let c = 2.0 * h - (2.0 * (n - 1.0) / n);
        
        c.round() as usize
    }
    
    /// カラムから数値データを抽出するヘルパーメソッド
    fn extract_numeric_values(&self, col: &ColumnView) -> Result<Vec<f64>> {
        match col.column_type() {
            crate::column::ColumnType::Float64 => {
                let float_col = col.as_float64().ok_or_else(|| 
                    Error::ColumnTypeMismatch {
                        name: col.column().name().unwrap_or("").to_string(),
                        expected: crate::column::ColumnType::Float64,
                        found: col.column_type(),
                    }
                )?;
                
                let mut values = Vec::with_capacity(col.len());
                for i in 0..col.len() {
                    if let Ok(Some(value)) = float_col.get(i) {
                        values.push(value);
                    } else {
                        values.push(0.0); // NAは0として扱う（または適切な戦略を実装）
                    }
                }
                Ok(values)
            },
            crate::column::ColumnType::Int64 => {
                let int_col = col.as_int64().ok_or_else(|| 
                    Error::ColumnTypeMismatch {
                        name: col.column().name().unwrap_or("").to_string(),
                        expected: crate::column::ColumnType::Int64,
                        found: col.column_type(),
                    }
                )?;
                
                let mut values = Vec::with_capacity(col.len());
                for i in 0..col.len() {
                    if let Ok(Some(value)) = int_col.get(i) {
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

impl Transformer for IsolationForest {
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
            .map(|s| s.to_string())
            .collect();
        
        if numeric_columns.is_empty() {
            return Err(Error::InvalidOperation(
                "DataFrame must contain at least one numeric column for IsolationForest".to_string()
            ));
        }
        
        self.feature_names = numeric_columns.clone();
        
        // データの準備
        let n_samples = df.row_count();
        let n_features = self.feature_names.len();
        let mut data = vec![vec![0.0; n_features]; n_samples];
        
        // データの読み込み
        for (col_idx, col_name) in self.feature_names.iter().enumerate() {
            let column = df.column(col_name)?;
            
            let values = self.extract_numeric_values(&column)?;
            
            for row_idx in 0..n_samples {
                if row_idx < values.len() {
                    data[row_idx][col_idx] = values[row_idx];
                }
            }
        }
        
        // 乱数生成器を初期化
        let mut rng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(rand::random()),
        };
        
        // サブサンプリングサイズを決定
        let sub_sample_size = match self.max_samples {
            Some(size) => size.min(n_samples),
            None => (n_samples as f64 * 0.632).min(256.0).max(1.0) as usize, // 経験則（paper推奨）
        };
        
        // 高さの制限を計算
        let height_limit = (sub_sample_size as f64).log2().ceil() as usize;
        
        // 決定木を構築
        self.trees.clear();
        for _ in 0..self.n_estimators {
            // サブサンプリング
            let mut indices: Vec<usize> = (0..n_samples).collect();
            
            // インデックスをシャッフル
            for i in (1..indices.len()).rev() {
                let j = rng.random_range(0..=i);
                indices.swap(i, j);
            }
            
            indices.truncate(sub_sample_size);
            
            // 木を構築
            let mut tree = ITree {
                height_limit,
                root: None,
            };
            
            tree.root = self.build_tree(&data, &indices, height_limit, 0, &mut rng);
            self.trees.push(tree);
        }
        
        // 異常スコアを計算
        self.anomaly_scores = vec![0.0; n_samples];
        
        for i in 0..n_samples {
            let mut path_length_sum = 0.0;
            
            for tree in &self.trees {
                let path_length = Self::compute_path_length(&tree.root, &data[i], 0) as f64;
                path_length_sum += path_length;
            }
            
            let avg_path_length = path_length_sum / self.n_estimators as f64;
            let expected_path_length = Self::c_factor(sub_sample_size) as f64;
            
            // 正規化した異常スコア
            // スコアが高いほど異常である（0〜1の範囲）
            self.anomaly_scores[i] = 2.0_f64.powf(-avg_path_length / expected_path_length);
        }
        
        // 閾値を計算（contamination率に基づく）
        let mut sorted_scores = self.anomaly_scores.clone();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap()); // 降順
        
        let threshold_idx = (self.contamination * n_samples as f64).round() as usize;
        self.threshold = sorted_scores.get(threshold_idx.max(1) - 1).copied().unwrap_or(0.5);
        
        // ラベルを割り当て
        self.labels = self.anomaly_scores
            .iter()
            .map(|&score| if score >= self.threshold { 1 } else { -1 })
            .collect();
        
        self.fitted = true;
        
        Ok(())
    }
    
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        if !self.fitted {
            return Err(Error::InvalidOperation(
                "IsolationForest has not been fitted yet".to_string()
            ));
        }
        
        // 元のデータフレームをコピー
        let mut result = df.clone();
        
        // 異常スコアと予測ラベルをデータフレームに追加
        let mut scores_float_col = Float64Column::new(self.anomaly_scores.clone());
        let mut labels_int_col = Int64Column::new(self.labels.clone());
        
        scores_float_col.set_name("anomaly_score");
        labels_int_col.set_name("anomaly");
        
        let scores_column = Column::Float64(scores_float_col);
        let labels_column = Column::Int64(labels_int_col);
        
        result.add_column("anomaly_score".to_string(), scores_column)?;
        result.add_column("anomaly".to_string(), labels_column)?;
        
        Ok(result)
    }
    
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

/// LOF (Local Outlier Factor) 異常検出アルゴリズム
pub struct LocalOutlierFactor {
    /// 近傍数
    n_neighbors: usize,
    /// 汚染率（異常値の期待割合）
    contamination: f64,
    /// メトリック
    metric: DistanceMetric,
    /// 各サンプルのLOFスコア
    lof_scores: Vec<f64>,
    /// 異常のしきい値
    threshold: f64,
    /// 異常フラグ（1: 異常, -1: 正常）
    labels: Vec<i64>,
    /// 特徴量の名前
    feature_names: Vec<String>,
    /// 訓練データ
    data: Vec<Vec<f64>>,
    /// 学習済みかどうか
    fitted: bool,
}

/// 距離メトリック
pub enum DistanceMetric {
    /// ユークリッド距離
    Euclidean,
    /// マンハッタン距離
    Manhattan,
    /// コサイン距離
    Cosine,
}

impl LocalOutlierFactor {
    /// 新しいLocalOutlierFactorインスタンスを作成
    pub fn new(
        n_neighbors: usize,
        contamination: f64,
        metric: DistanceMetric,
    ) -> Self {
        if contamination <= 0.0 || contamination >= 0.5 {
            panic!("Contamination must be in (0, 0.5)");
        }
        
        LocalOutlierFactor {
            n_neighbors,
            contamination,
            metric,
            lof_scores: Vec::new(),
            threshold: 0.0,
            labels: Vec::new(),
            feature_names: Vec::new(),
            data: Vec::new(),
            fitted: false,
        }
    }
    
    /// LOFスコアを取得
    pub fn lof_scores(&self) -> &[f64] {
        &self.lof_scores
    }
    
    /// 異常フラグを取得（1: 異常, -1: 正常）
    pub fn labels(&self) -> &[i64] {
        &self.labels
    }
    
    /// 2つのデータ点間の距離を計算
    fn compute_distance(&self, x: &[f64], y: &[f64]) -> f64 {
        match self.metric {
            DistanceMetric::Euclidean => {
                // ユークリッド距離
                x.iter()
                    .zip(y.iter())
                    .map(|(&xi, &yi)| (xi - yi).powi(2))
                    .sum::<f64>()
                    .sqrt()
            }
            DistanceMetric::Manhattan => {
                // マンハッタン距離
                x.iter()
                    .zip(y.iter())
                    .map(|(&xi, &yi)| (xi - yi).abs())
                    .sum()
            }
            DistanceMetric::Cosine => {
                // コサイン距離
                let dot_product: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
                let norm_x: f64 = x.iter().map(|&xi| xi.powi(2)).sum::<f64>().sqrt();
                let norm_y: f64 = y.iter().map(|&yi| yi.powi(2)).sum::<f64>().sqrt();
                
                if norm_x > 0.0 && norm_y > 0.0 {
                    1.0 - dot_product / (norm_x * norm_y)
                } else {
                    1.0 // 最大距離
                }
            }
        }
    }
    
    /// k-近傍を見つける
    fn find_neighbors(&self, point_idx: usize, k: usize) -> Vec<(usize, f64)> {
        let n_samples = self.data.len();
        let mut distances = Vec::with_capacity(n_samples - 1);
        
        for i in 0..n_samples {
            if i != point_idx {
                let dist = self.compute_distance(&self.data[point_idx], &self.data[i]);
                distances.push((i, dist));
            }
        }
        
        // 距離でソート
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        // 上位k個を返す
        distances.into_iter().take(k.min(n_samples - 1)).collect()
    }
    
    /// 到達可能距離を計算
    fn reachability_distance(&self, point_a_idx: usize, point_b_idx: usize, k_distance: f64) -> f64 {
        let direct_distance = self.compute_distance(&self.data[point_a_idx], &self.data[point_b_idx]);
        direct_distance.max(k_distance)
    }
    
    /// カラムから数値データを抽出するヘルパーメソッド
    fn extract_numeric_values(&self, col: &ColumnView) -> Result<Vec<f64>> {
        match col.column_type() {
            crate::column::ColumnType::Float64 => {
                let float_col = col.as_float64().ok_or_else(|| 
                    Error::ColumnTypeMismatch {
                        name: col.column().name().unwrap_or("").to_string(),
                        expected: crate::column::ColumnType::Float64,
                        found: col.column_type(),
                    }
                )?;
                
                let mut values = Vec::with_capacity(col.len());
                for i in 0..col.len() {
                    if let Ok(Some(value)) = float_col.get(i) {
                        values.push(value);
                    } else {
                        values.push(0.0); // NAは0として扱う（または適切な戦略を実装）
                    }
                }
                Ok(values)
            },
            crate::column::ColumnType::Int64 => {
                let int_col = col.as_int64().ok_or_else(|| 
                    Error::ColumnTypeMismatch {
                        name: col.column().name().unwrap_or("").to_string(),
                        expected: crate::column::ColumnType::Int64,
                        found: col.column_type(),
                    }
                )?;
                
                let mut values = Vec::with_capacity(col.len());
                for i in 0..col.len() {
                    if let Ok(Some(value)) = int_col.get(i) {
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

impl Transformer for LocalOutlierFactor {
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
            .map(|s| s.to_string())
            .collect();
        
        if numeric_columns.is_empty() {
            return Err(Error::InvalidOperation(
                "DataFrame must contain at least one numeric column for LocalOutlierFactor".to_string()
            ));
        }
        
        self.feature_names = numeric_columns.clone();
        
        // データの準備
        let n_samples = df.row_count();
        let n_features = self.feature_names.len();
        self.data = vec![vec![0.0; n_features]; n_samples];
        
        // データの読み込み
        for (col_idx, col_name) in self.feature_names.iter().enumerate() {
            let column = df.column(col_name)?;
            
            let values = self.extract_numeric_values(&column)?;
            
            for row_idx in 0..n_samples {
                if row_idx < values.len() {
                    self.data[row_idx][col_idx] = values[row_idx];
                }
            }
        }
        
        // 1. 各点のk-近傍を見つける
        let mut neighbors = Vec::with_capacity(n_samples);
        let mut k_distances = Vec::with_capacity(n_samples);
        
        for i in 0..n_samples {
            let k_neighbors = self.find_neighbors(i, self.n_neighbors);
            // k番目の点との距離
            let k_dist = k_neighbors.last().map(|&(_, dist)| dist).unwrap_or(0.0);
            
            neighbors.push(k_neighbors);
            k_distances.push(k_dist);
        }
        
        // 2. 局所到達可能密度を計算
        let mut lrd = vec![0.0; n_samples];
        
        for i in 0..n_samples {
            let mut sum_reachability = 0.0;
            
            for &(neighbor_idx, _) in &neighbors[i] {
                let reach_dist = self.reachability_distance(i, neighbor_idx, k_distances[neighbor_idx]);
                sum_reachability += reach_dist;
            }
            
            if !neighbors[i].is_empty() {
                lrd[i] = neighbors[i].len() as f64 / sum_reachability;
            } else {
                lrd[i] = 0.0;
            }
        }
        
        // 3. LOFスコアを計算
        self.lof_scores = vec![0.0; n_samples];
        
        for i in 0..n_samples {
            let mut lof_sum = 0.0;
            
            for &(neighbor_idx, _) in &neighbors[i] {
                lof_sum += lrd[neighbor_idx] / lrd[i];
            }
            
            if !neighbors[i].is_empty() {
                self.lof_scores[i] = lof_sum / neighbors[i].len() as f64;
            } else {
                self.lof_scores[i] = 1.0;  // デフォルト値
            }
        }
        
        // 閾値を計算（contamination率に基づく）
        let mut sorted_scores = self.lof_scores.clone();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap()); // 降順
        
        let threshold_idx = (self.contamination * n_samples as f64).round() as usize;
        self.threshold = sorted_scores.get(threshold_idx.max(1) - 1).copied().unwrap_or(1.0);
        
        // ラベルを割り当て（LOFが閾値以上なら異常）
        self.labels = self.lof_scores
            .iter()
            .map(|&score| if score >= self.threshold { 1 } else { -1 })
            .collect();
        
        self.fitted = true;
        
        Ok(())
    }
    
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        if !self.fitted {
            return Err(Error::InvalidOperation(
                "LocalOutlierFactor has not been fitted yet".to_string()
            ));
        }
        
        // データサイズが一致しているか確認
        if df.row_count() != self.lof_scores.len() {
            return Err(Error::InvalidOperation(
                "Number of samples in the input DataFrame does not match the number of samples used during fitting".to_string()
            ));
        }
        
        // 元のデータフレームをコピー
        let mut result = df.clone();
        
        // LOFスコアと予測ラベルをデータフレームに追加
        let mut scores_float_col = Float64Column::new(self.lof_scores.clone());
        let mut labels_int_col = Int64Column::new(self.labels.clone());
        
        scores_float_col.set_name("lof_score");
        labels_int_col.set_name("anomaly");
        
        let scores_column = Column::Float64(scores_float_col);
        let labels_column = Column::Int64(labels_int_col);
        
        result.add_column("lof_score".to_string(), scores_column)?;
        result.add_column("anomaly".to_string(), labels_column)?;
        
        Ok(result)
    }
    
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

/// 一クラスSVM異常検出アルゴリズム
pub struct OneClassSVM {
    /// ニュー（閾値調整パラメータ）
    nu: f64,
    /// カーネル係数
    gamma: f64,
    /// 最大イテレーション数
    max_iter: usize,
    /// 収束閾値
    tol: f64,
    /// サポートベクター
    support_vectors: Vec<Vec<f64>>,
    /// ラグランジュ乗数
    alphas: Vec<f64>,
    /// バイアス
    rho: f64,
    /// 決定スコア
    decision_scores: Vec<f64>,
    /// 異常フラグ（1: 異常, -1: 正常）
    labels: Vec<i64>,
    /// 特徴量の名前
    feature_names: Vec<String>,
    /// 学習済みかどうか
    fitted: bool,
}

impl OneClassSVM {
    /// 新しいOneClassSVMインスタンスを作成
    pub fn new(
        nu: f64,
        gamma: f64,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        if nu <= 0.0 || nu >= 1.0 {
            panic!("Nu must be in (0, 1)");
        }
        
        OneClassSVM {
            nu,
            gamma,
            max_iter,
            tol,
            support_vectors: Vec::new(),
            alphas: Vec::new(),
            rho: 0.0,
            decision_scores: Vec::new(),
            labels: Vec::new(),
            feature_names: Vec::new(),
            fitted: false,
        }
    }
    
    /// 決定スコアを取得
    pub fn decision_scores(&self) -> &[f64] {
        &self.decision_scores
    }
    
    /// 異常フラグを取得（1: 異常, -1: 正常）
    pub fn labels(&self) -> &[i64] {
        &self.labels
    }
    
    /// RBFカーネルを計算
    fn rbf_kernel(&self, x: &[f64], y: &[f64]) -> f64 {
        let squared_distance = x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - yi).powi(2))
            .sum::<f64>();
        
        (-self.gamma * squared_distance).exp()
    }
    
    /// 新しいサンプルの決定関数値を計算
    fn decision_function(&self, x: &[f64]) -> f64 {
        let mut sum = 0.0;
        
        for (i, support_vector) in self.support_vectors.iter().enumerate() {
            sum += self.alphas[i] * self.rbf_kernel(x, support_vector);
        }
        
        sum - self.rho
    }
    
    /// カラムから数値データを抽出するヘルパーメソッド
    fn extract_numeric_values(&self, col: &ColumnView) -> Result<Vec<f64>> {
        match col.column_type() {
            crate::column::ColumnType::Float64 => {
                let float_col = col.as_float64().ok_or_else(|| 
                    Error::ColumnTypeMismatch {
                        name: col.column().name().unwrap_or("").to_string(),
                        expected: crate::column::ColumnType::Float64,
                        found: col.column_type(),
                    }
                )?;
                
                let mut values = Vec::with_capacity(col.len());
                for i in 0..col.len() {
                    if let Ok(Some(value)) = float_col.get(i) {
                        values.push(value);
                    } else {
                        values.push(0.0); // NAは0として扱う（または適切な戦略を実装）
                    }
                }
                Ok(values)
            },
            crate::column::ColumnType::Int64 => {
                let int_col = col.as_int64().ok_or_else(|| 
                    Error::ColumnTypeMismatch {
                        name: col.column().name().unwrap_or("").to_string(),
                        expected: crate::column::ColumnType::Int64,
                        found: col.column_type(),
                    }
                )?;
                
                let mut values = Vec::with_capacity(col.len());
                for i in 0..col.len() {
                    if let Ok(Some(value)) = int_col.get(i) {
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

impl Transformer for OneClassSVM {
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
            .map(|s| s.to_string())
            .collect();
        
        if numeric_columns.is_empty() {
            return Err(Error::InvalidOperation(
                "DataFrame must contain at least one numeric column for OneClassSVM".to_string()
            ));
        }
        
        self.feature_names = numeric_columns.clone();
        
        // データの準備
        let n_samples = df.row_count();
        let n_features = self.feature_names.len();
        let mut data = vec![vec![0.0; n_features]; n_samples];
        
        // データの読み込み
        for (col_idx, col_name) in self.feature_names.iter().enumerate() {
            let column = df.column(col_name)?;
            
            let values = self.extract_numeric_values(&column)?;
            
            for row_idx in 0..n_samples {
                if row_idx < values.len() {
                    data[row_idx][col_idx] = values[row_idx];
                }
            }
        }
        
        // ここでは簡易版の実装を提供
        // 実際の本格的な実装ではSMOアルゴリズムなど、より効率的な方法を使用する
        
        // カーネル行列を計算
        let mut kernel_matrix = vec![vec![0.0; n_samples]; n_samples];
        for i in 0..n_samples {
            for j in 0..=i {
                let k_ij = self.rbf_kernel(&data[i], &data[j]);
                kernel_matrix[i][j] = k_ij;
                kernel_matrix[j][i] = k_ij;
            }
        }
        
        // 簡易的な最適化手順（実際にはSMOアルゴリズムを使うべき）
        let mut alphas = vec![0.0; n_samples];
        let mut g = vec![0.0; n_samples]; // 勾配
        
        // 初期化
        for i in 0..n_samples {
            alphas[i] = 1.0 / (n_samples as f64 * self.nu); // 均等初期化
            g[i] = 0.0;
            for j in 0..n_samples {
                g[i] -= alphas[j] * kernel_matrix[i][j];
            }
        }
        
        // 最適化
        for _ in 0..self.max_iter {
            let mut max_diff = 0.0;
            
            for i in 0..n_samples {
                let old_alpha_i = alphas[i];
                
                // 勾配降下ステップ
                let new_alpha_i = old_alpha_i - g[i] / kernel_matrix[i][i];
                
                // クリッピング
                alphas[i] = new_alpha_i.max(0.0).min(1.0 / (n_samples as f64 * self.nu));
                
                let diff = alphas[i] - old_alpha_i;
                if diff.abs() > max_diff {
                    max_diff = diff.abs();
                }
                
                // 勾配の更新
                for j in 0..n_samples {
                    g[j] -= diff * kernel_matrix[i][j];
                }
            }
            
            // 収束判定
            if max_diff < self.tol {
                break;
            }
        }
        
        // サポートベクターとラグランジュ乗数を抽出
        let mut support_vector_indices = Vec::new();
        for i in 0..n_samples {
            if alphas[i] > 1e-5 {
                support_vector_indices.push(i);
            }
        }
        
        self.support_vectors = support_vector_indices
            .iter()
            .map(|&i| data[i].clone())
            .collect();
        
        self.alphas = support_vector_indices
            .iter()
            .map(|&i| alphas[i])
            .collect();
        
        // バイアス（rho）の計算
        let mut rho_sum = 0.0;
        let mut count = 0;
        
        for &i in &support_vector_indices {
            let mut f_i = 0.0;
            for (j, &sv_j) in support_vector_indices.iter().enumerate() {
                f_i += self.alphas[j] * kernel_matrix[i][sv_j];
            }
            
            if alphas[i] < (1.0 / (n_samples as f64 * self.nu)) - 1e-5 {
                rho_sum += f_i;
                count += 1;
            }
        }
        
        if count > 0 {
            self.rho = rho_sum / count as f64;
        } else {
            // バックアッププラン
            let mut f_sum = 0.0;
            for i in 0..n_samples {
                let mut f_i = 0.0;
                for (j, &sv_j) in support_vector_indices.iter().enumerate() {
                    f_i += self.alphas[j] * kernel_matrix[i][sv_j];
                }
                f_sum += f_i;
            }
            self.rho = f_sum / n_samples as f64;
        }
        
        // 決定スコアを計算
        self.decision_scores = Vec::with_capacity(n_samples);
        
        for i in 0..n_samples {
            let mut score = 0.0;
            for (j, &sv_j) in support_vector_indices.iter().enumerate() {
                score += self.alphas[j] * kernel_matrix[i][sv_j];
            }
            score -= self.rho;
            self.decision_scores.push(score);
        }
        
        // ラベルを割り当て（スコアが0未満なら異常）
        self.labels = self.decision_scores
            .iter()
            .map(|&score| if score < 0.0 { 1 } else { -1 })
            .collect();
        
        self.fitted = true;
        
        Ok(())
    }
    
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        if !self.fitted {
            return Err(Error::InvalidOperation(
                "OneClassSVM has not been fitted yet".to_string()
            ));
        }
        
        // データサイズが一致しているか確認
        if df.row_count() != self.decision_scores.len() {
            return Err(Error::InvalidOperation(
                "Number of samples in the input DataFrame does not match the number of samples used during fitting".to_string()
            ));
        }
        
        // 元のデータフレームをコピー
        let mut result = df.clone();
        
        // 決定スコアと予測ラベルをデータフレームに追加
        let mut scores_float_col = Float64Column::new(self.decision_scores.clone());
        let mut labels_int_col = Int64Column::new(self.labels.clone());
        
        scores_float_col.set_name("decision_score");
        labels_int_col.set_name("anomaly");
        
        let scores_column = Column::Float64(scores_float_col);
        let labels_column = Column::Int64(labels_int_col);
        
        result.add_column("decision_score".to_string(), scores_column)?;
        result.add_column("anomaly".to_string(), labels_column)?;
        
        Ok(result)
    }
    
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}