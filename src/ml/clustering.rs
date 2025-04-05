//! クラスタリングモジュール
//!
//! データ点をグループ（クラスタ）に分類するためのアルゴリズムを提供します。

use crate::optimized::{OptimizedDataFrame, ColumnView};
use crate::error::{Result, Error};
use crate::ml::pipeline::Transformer;
use crate::column::{Float64Column, Column, ColumnTrait};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::{HashMap, HashSet};

/// k-means クラスタリングアルゴリズム
pub struct KMeans {
    /// クラスタ数
    k: usize,
    /// 最大反復回数
    max_iter: usize,
    /// 収束閾値
    tol: f64,
    /// 乱数シード
    random_seed: Option<u64>,
    /// クラスタ中心
    centroids: Vec<Vec<f64>>,
    /// 各データポイントが属するクラスタ
    labels: Vec<usize>,
    /// 特徴量の名前
    feature_names: Vec<String>,
    /// イナーシャ（クラスタ内の二乗距離の合計）
    inertia: f64,
    /// 収束までの反復回数
    n_iter: usize,
    /// 学習済みかどうか
    fitted: bool,
}

impl KMeans {
    /// 新しいKMeansインスタンスを作成
    pub fn new(k: usize, max_iter: usize, tol: f64, random_seed: Option<u64>) -> Self {
        KMeans {
            k,
            max_iter,
            tol,
            random_seed,
            centroids: Vec::new(),
            labels: Vec::new(),
            feature_names: Vec::new(),
            inertia: 0.0,
            n_iter: 0,
            fitted: false,
        }
    }
    
    /// クラスタラベルを取得
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }
    
    /// クラスタ中心を取得
    pub fn centroids(&self) -> &[Vec<f64>] {
        &self.centroids
    }
    
    /// イナーシャを取得
    pub fn inertia(&self) -> f64 {
        self.inertia
    }
    
    /// 収束までの反復回数を取得
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
    
    /// ユークリッド距離の二乗を計算
    fn squared_euclidean_distance(x: &[f64], y: &[f64]) -> f64 {
        x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - yi).powi(2))
            .sum()
    }
    
    /// k-means++による初期クラスタ中心の選択
    fn kmeans_plus_plus_init(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n_samples = data.len();
        let n_features = data[0].len();
        
        let mut rng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        
        // 最初の中心をランダムに選択
        let first_idx = rng.gen_range(0..n_samples);
        let mut centroids = vec![data[first_idx].clone()];
        
        // 残りのk-1個の中心を選択
        for _ in 1..self.k {
            // 各データポイントから最も近い中心までの距離の二乗を計算
            let mut distances = vec![0.0; n_samples];
            let mut sum_distances = 0.0;
            
            for (i, point) in data.iter().enumerate() {
                // 最も近い中心までの距離の二乗
                let closest_dist = centroids
                    .iter()
                    .map(|c| Self::squared_euclidean_distance(point, c))
                    .fold(f64::INFINITY, |a, b| a.min(b));
                
                distances[i] = closest_dist;
                sum_distances += closest_dist;
            }
            
            // 距離の二乗に比例した確率で次の中心を選択
            let mut cumsum = 0.0;
            let threshold = rng.gen_range(0.0..sum_distances);
            
            for (i, &dist) in distances.iter().enumerate() {
                cumsum += dist;
                if cumsum >= threshold {
                    centroids.push(data[i].clone());
                    break;
                }
            }
        }
        
        centroids
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

impl Transformer for KMeans {
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
                "DataFrame must contain at least one numeric column for KMeans".to_string()
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
        
        // k-means++による初期クラスタ中心の選択
        self.centroids = self.kmeans_plus_plus_init(&data);
        
        // k-meansアルゴリズムのメインループ
        let mut prev_inertia = f64::INFINITY;
        self.labels = vec![0; n_samples];
        
        for iter in 0..self.max_iter {
            // 各データポイントに最も近いクラスタを割り当て
            let mut new_labels = vec![0; n_samples];
            let mut inertia = 0.0;
            
            for (i, point) in data.iter().enumerate() {
                let mut min_dist = f64::INFINITY;
                let mut closest_centroid = 0;
                
                for (j, centroid) in self.centroids.iter().enumerate() {
                    let dist = Self::squared_euclidean_distance(point, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        closest_centroid = j;
                    }
                }
                
                new_labels[i] = closest_centroid;
                inertia += min_dist;
            }
            
            self.labels = new_labels;
            self.inertia = inertia;
            
            // クラスタ中心の更新
            let mut new_centroids = vec![vec![0.0; n_features]; self.k];
            let mut counts = vec![0; self.k];
            
            for (i, point) in data.iter().enumerate() {
                let cluster = self.labels[i];
                counts[cluster] += 1;
                
                for j in 0..n_features {
                    new_centroids[cluster][j] += point[j];
                }
            }
            
            for i in 0..self.k {
                if counts[i] > 0 {
                    for j in 0..n_features {
                        new_centroids[i][j] /= counts[i] as f64;
                    }
                }
            }
            
            // 収束判定
            let mut centroid_shift = 0.0;
            for (old, new) in self.centroids.iter().zip(new_centroids.iter()) {
                centroid_shift += Self::squared_euclidean_distance(old, new);
            }
            
            self.centroids = new_centroids;
            
            // イナーシャの変化が閾値以下なら収束
            let inertia_change = (prev_inertia - self.inertia).abs();
            if inertia_change / prev_inertia < self.tol {
                self.n_iter = iter + 1;
                break;
            }
            
            if centroid_shift < self.tol {
                self.n_iter = iter + 1;
                break;
            }
            
            prev_inertia = self.inertia;
            
            // 最後の反復の場合
            if iter == self.max_iter - 1 {
                self.n_iter = self.max_iter;
            }
        }
        
        self.fitted = true;
        
        Ok(())
    }
    
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        if !self.fitted {
            return Err(Error::InvalidOperation(
                "KMeans has not been fitted yet".to_string()
            ));
        }
        
        // 元のデータフレームをコピー
        let mut result = df.clone();
        
        // 各データポイントへのクラスタ割り当て
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
        
        // 各データポイントに最も近いクラスタを割り当て
        let mut labels = Vec::with_capacity(n_samples);
        let mut distances = Vec::with_capacity(n_samples);
        
        for point in &data {
            let mut min_dist = f64::INFINITY;
            let mut closest_centroid = 0;
            
            for (j, centroid) in self.centroids.iter().enumerate() {
                let dist = Self::squared_euclidean_distance(point, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    closest_centroid = j;
                }
            }
            
            labels.push(closest_centroid as i64);
            distances.push(min_dist.sqrt()); // ユークリッド距離
        }
        
        // クラスタラベルと距離を新しい列として追加
        let labels_column = Column::Int64(crate::column::Int64Column::new(labels, false, "cluster".to_string())?);
        let distances_column = Column::Float64(Float64Column::new(distances, false, "distance_to_centroid".to_string())?);
        
        result.add_column("cluster".to_string(), labels_column)?;
        result.add_column("distance_to_centroid".to_string(), distances_column)?;
        
        Ok(result)
    }
    
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

/// 階層的クラスタリングアルゴリズム
pub struct AgglomerativeClustering {
    /// クラスタ数
    n_clusters: usize,
    /// リンケージ方法
    linkage: Linkage,
    /// 距離メトリック
    metric: DistanceMetric,
    /// クラスタラベル
    labels: Vec<usize>,
    /// 特徴量の名前
    feature_names: Vec<String>,
    /// 学習済みかどうか
    fitted: bool,
}

/// リンケージ方法
pub enum Linkage {
    /// 単連結法（最小距離法）
    Single,
    /// 完全連結法（最大距離法）
    Complete,
    /// 群平均法
    Average,
    /// ウォード法
    Ward,
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

impl AgglomerativeClustering {
    /// 新しいAgglomerativeClusteringインスタンスを作成
    pub fn new(n_clusters: usize, linkage: Linkage, metric: DistanceMetric) -> Self {
        AgglomerativeClustering {
            n_clusters,
            linkage,
            metric,
            labels: Vec::new(),
            feature_names: Vec::new(),
            fitted: false,
        }
    }
    
    /// クラスタラベルを取得
    pub fn labels(&self) -> &[usize] {
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
    
    /// 2つのクラスタ間の距離を計算
    fn compute_cluster_distance(
        &self,
        cluster1: &[usize],
        cluster2: &[usize],
        data: &[Vec<f64>],
        distances: &HashMap<(usize, usize), f64>,
    ) -> f64 {
        match self.linkage {
            Linkage::Single => {
                // 単連結法：最小距離
                let mut min_dist = f64::INFINITY;
                
                for &i in cluster1 {
                    for &j in cluster2 {
                        let dist = if i < j {
                            *distances.get(&(i, j)).unwrap_or(&f64::INFINITY)
                        } else {
                            *distances.get(&(j, i)).unwrap_or(&f64::INFINITY)
                        };
                        
                        min_dist = min_dist.min(dist);
                    }
                }
                
                min_dist
            }
            Linkage::Complete => {
                // 完全連結法：最大距離
                let mut max_dist = 0.0;
                
                for &i in cluster1 {
                    for &j in cluster2 {
                        let dist = if i < j {
                            *distances.get(&(i, j)).unwrap_or(&0.0)
                        } else {
                            *distances.get(&(j, i)).unwrap_or(&0.0)
                        };
                        
                        max_dist = max_dist.max(dist);
                    }
                }
                
                max_dist
            }
            Linkage::Average => {
                // 群平均法：平均距離
                let mut sum_dist = 0.0;
                let mut count = 0;
                
                for &i in cluster1 {
                    for &j in cluster2 {
                        let dist = if i < j {
                            *distances.get(&(i, j)).unwrap_or(&0.0)
                        } else {
                            *distances.get(&(j, i)).unwrap_or(&0.0)
                        };
                        
                        sum_dist += dist;
                        count += 1;
                    }
                }
                
                if count > 0 {
                    sum_dist / count as f64
                } else {
                    f64::INFINITY
                }
            }
            Linkage::Ward => {
                // ウォード法：分散の増加
                let n1 = cluster1.len();
                let n2 = cluster2.len();
                
                if n1 == 0 || n2 == 0 {
                    return f64::INFINITY;
                }
                
                // クラスタ1の重心
                let mut centroid1 = vec![0.0; data[0].len()];
                for &i in cluster1 {
                    for j in 0..data[0].len() {
                        centroid1[j] += data[i][j];
                    }
                }
                for j in 0..centroid1.len() {
                    centroid1[j] /= n1 as f64;
                }
                
                // クラスタ2の重心
                let mut centroid2 = vec![0.0; data[0].len()];
                for &i in cluster2 {
                    for j in 0..data[0].len() {
                        centroid2[j] += data[i][j];
                    }
                }
                for j in 0..centroid2.len() {
                    centroid2[j] /= n2 as f64;
                }
                
                // 重心間の距離
                let mut dist = 0.0;
                for j in 0..centroid1.len() {
                    dist += (centroid1[j] - centroid2[j]).powi(2);
                }
                dist = dist.sqrt();
                
                // ウォード法の距離
                (n1 * n2) as f64 * dist / (n1 + n2) as f64
            }
        }
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

impl Transformer for AgglomerativeClustering {
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
                "DataFrame must contain at least one numeric column for AgglomerativeClustering".to_string()
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
        
        // すべてのペア間の距離を計算
        let mut distances = HashMap::new();
        for i in 0..n_samples {
            for j in i+1..n_samples {
                let dist = self.compute_distance(&data[i], &data[j]);
                distances.insert((i, j), dist);
            }
        }
        
        // 各データポイントを独自のクラスタとして初期化
        let mut clusters: Vec<Vec<usize>> = (0..n_samples).map(|i| vec![i]).collect();
        
        // クラスタが目標数になるまで繰り返し
        while clusters.len() > self.n_clusters {
            // 最も近い2つのクラスタを見つける
            let mut min_dist = f64::INFINITY;
            let mut merge_i = 0;
            let mut merge_j = 0;
            
            for i in 0..clusters.len() {
                for j in i+1..clusters.len() {
                    let dist = self.compute_cluster_distance(&clusters[i], &clusters[j], &data, &distances);
                    
                    if dist < min_dist {
                        min_dist = dist;
                        merge_i = i;
                        merge_j = j;
                    }
                }
            }
            
            // クラスタをマージ
            let cluster_j = clusters.remove(merge_j); // 高いインデックスから削除
            let cluster_i = &mut clusters[merge_i];
            cluster_i.extend(cluster_j);
        }
        
        // 最終的なクラスタラベルを割り当て
        self.labels = vec![0; n_samples];
        for (cluster_idx, cluster) in clusters.iter().enumerate() {
            for &sample_idx in cluster {
                self.labels[sample_idx] = cluster_idx;
            }
        }
        
        self.fitted = true;
        
        Ok(())
    }
    
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        if !self.fitted {
            return Err(Error::InvalidOperation(
                "AgglomerativeClustering has not been fitted yet".to_string()
            ));
        }
        
        // データサイズが一致しているか確認
        if df.row_count() != self.labels.len() {
            return Err(Error::InvalidOperation(
                "Number of samples in the input DataFrame does not match the number of samples used during fitting".to_string()
            ));
        }
        
        // 元のデータフレームをコピー
        let mut result = df.clone();
        
        // クラスタラベルを新しい列として追加
        let labels: Vec<i64> = self.labels.iter().map(|&l| l as i64).collect();
        let labels_column = Column::Int64(crate::column::Int64Column::new(labels, false, "cluster".to_string())?);
        
        result.add_column("cluster".to_string(), labels_column)?;
        
        Ok(result)
    }
    
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

/// DBSCAN (Density-Based Spatial Clustering of Applications with Noise) アルゴリズム
pub struct DBSCAN {
    /// イプシロン（近傍半径）
    eps: f64,
    /// 最小点数
    min_samples: usize,
    /// 距離メトリック
    metric: DistanceMetric,
    /// クラスタラベル
    labels: Vec<i64>,
    /// 特徴量の名前
    feature_names: Vec<String>,
    /// 学習済みかどうか
    fitted: bool,
}

impl DBSCAN {
    /// 新しいDBSCANインスタンスを作成
    pub fn new(eps: f64, min_samples: usize, metric: DistanceMetric) -> Self {
        DBSCAN {
            eps,
            min_samples,
            metric,
            labels: Vec::new(),
            feature_names: Vec::new(),
            fitted: false,
        }
    }
    
    /// クラスタラベルを取得
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
    
    /// 指定された点の近傍点を探す
    fn region_query(&self, point_idx: usize, data: &[Vec<f64>]) -> Vec<usize> {
        let mut neighbors = Vec::new();
        
        for (i, point) in data.iter().enumerate() {
            if i != point_idx && self.compute_distance(&data[point_idx], point) <= self.eps {
                neighbors.push(i);
            }
        }
        
        neighbors
    }
    
    /// 点を中心としてクラスタを拡張
    fn expand_cluster(
        &self,
        point_idx: usize,
        neighbors: &[usize],
        cluster_id: i64,
        labels: &mut [i64],
        data: &[Vec<f64>],
        visited: &mut HashSet<usize>,
    ) {
        labels[point_idx] = cluster_id;
        
        let mut i = 0;
        let mut neighbors_vec = neighbors.to_vec();
        
        while i < neighbors_vec.len() {
            let current_point = neighbors_vec[i];
            
            // 未訪問の点を処理
            if !visited.contains(&current_point) {
                visited.insert(current_point);
                
                let current_neighbors = self.region_query(current_point, data);
                
                if current_neighbors.len() >= self.min_samples {
                    // 密度到達可能な点を追加
                    for &neighbor in &current_neighbors {
                        if !neighbors_vec.contains(&neighbor) {
                            neighbors_vec.push(neighbor);
                        }
                    }
                }
            }
            
            // ラベルがまだ割り当てられていない場合、このクラスタに追加
            if labels[current_point] == -1 {
                labels[current_point] = cluster_id;
            }
            
            i += 1;
        }
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

impl Transformer for DBSCAN {
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
                "DataFrame must contain at least one numeric column for DBSCAN".to_string()
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
        
        // ラベルの初期化（未割り当て: -1）
        self.labels = vec![-1; n_samples];
        let mut visited = HashSet::new();
        let mut cluster_id = 0;
        
        // 各点を処理
        for i in 0..n_samples {
            if !visited.contains(&i) {
                visited.insert(i);
                
                // 近傍点を探す
                let neighbors = self.region_query(i, &data);
                
                if neighbors.len() < self.min_samples {
                    // ノイズとして扱う
                    self.labels[i] = -1;
                } else {
                    // 新しいクラスタを開始
                    self.expand_cluster(i, &neighbors, cluster_id, &mut self.labels, &data, &mut visited);
                    cluster_id += 1;
                }
            }
        }
        
        self.fitted = true;
        
        Ok(())
    }
    
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        if !self.fitted {
            return Err(Error::InvalidOperation(
                "DBSCAN has not been fitted yet".to_string()
            ));
        }
        
        // データサイズが一致しているか確認
        if df.row_count() != self.labels.len() {
            return Err(Error::InvalidOperation(
                "Number of samples in the input DataFrame does not match the number of samples used during fitting".to_string()
            ));
        }
        
        // 元のデータフレームをコピー
        let mut result = df.clone();
        
        // クラスタラベルを新しい列として追加
        let labels_column = Column::Int64(crate::column::Int64Column::new(self.labels.clone(), false, "cluster".to_string())?);
        
        result.add_column("cluster".to_string(), labels_column)?;
        
        Ok(result)
    }
    
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}