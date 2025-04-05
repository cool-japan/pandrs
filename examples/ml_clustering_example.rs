use pandrs::*;
use pandrs::ml::clustering::{KMeans, AgglomerativeClustering, DBSCAN, Linkage, DistanceMetric};
use pandrs::ml::dimension_reduction::PCA;
use rand::prelude::*;

fn main() -> Result<(), PandRSError> {
    println!("PandRS クラスタリングアルゴリズムの例");
    println!("================================");
    
    // サンプルデータの生成
    let df = create_sample_data()?;
    println!("元のデータフレーム（最初の5行）:");
    println!("{}", df.head(5)?);
    
    // K-meansクラスタリングの例
    kmeans_example(&df)?;
    
    // 階層的クラスタリングの例
    hierarchical_example(&df)?;
    
    // DBSCANクラスタリングの例
    dbscan_example(&df)?;
    
    // クラスタリング結果の可視化（PCAを使用）
    visualization_example(&df)?;
    
    Ok(())
}

// K-meansクラスタリングの例
fn kmeans_example(df: &DataFrame) -> Result<(), PandRSError> {
    println!("\n==== K-means クラスタリング ====");
    
    // K-meansインスタンスの作成（クラスタ数=3）
    let mut kmeans = KMeans::new(3, 100, 1e-4, Some(42));
    
    // クラスタリングの実行
    let kmeans_result = kmeans.fit_transform(df)?;
    
    // クラスタの分布を表示
    let cluster_counts = count_clusters(&kmeans_result, "cluster")?;
    
    println!("\nK-meansクラスタリング結果:");
    for (cluster, count) in cluster_counts {
        println!("クラスタ {}: {} サンプル", cluster, count);
    }
    
    // クラスタ中心を取得して表示
    println!("\nクラスタ中心:");
    for (i, centroid) in kmeans.centroids().iter().enumerate() {
        println!("クラスタ {}: {:?}", i, centroid);
    }
    
    // イナーシャ（クラスタ内の分散）を表示
    println!("\nイナーシャ（クラスタ内二乗距離の合計）: {}", kmeans.inertia());
    println!("収束までの反復回数: {}", kmeans.n_iter());
    
    // クラスタリング結果を表示
    println!("\nクラスタリング結果（最初の5行）:");
    println!("{}", kmeans_result.head(5)?);
    
    // 真のクラスタラベルとの比較（もし存在すれば）
    if df.column("true_cluster").is_some() {
        let accuracy = compare_clustering(&kmeans_result, "cluster", "true_cluster")?;
        println!("\n真のラベルとの一致率: {:.2}%", accuracy * 100.0);
    }
    
    Ok(())
}

// 階層的クラスタリングの例
fn hierarchical_example(df: &DataFrame) -> Result<(), PandRSError> {
    println!("\n==== 階層的クラスタリング ====");
    
    // 階層的クラスタリングインスタンスの作成（クラスタ数=3, ウォード法, ユークリッド距離）
    let mut hierarchical = AgglomerativeClustering::new(
        3,
        Linkage::Ward,
        DistanceMetric::Euclidean,
    );
    
    // クラスタリングの実行
    let hierarchical_result = hierarchical.fit_transform(df)?;
    
    // クラスタの分布を表示
    let cluster_counts = count_clusters(&hierarchical_result, "cluster")?;
    
    println!("\n階層的クラスタリング結果:");
    for (cluster, count) in cluster_counts {
        println!("クラスタ {}: {} サンプル", cluster, count);
    }
    
    // クラスタリング結果を表示
    println!("\nクラスタリング結果（最初の5行）:");
    println!("{}", hierarchical_result.head(5)?);
    
    // 真のクラスタラベルとの比較（もし存在すれば）
    if df.column("true_cluster").is_some() {
        let accuracy = compare_clustering(&hierarchical_result, "cluster", "true_cluster")?;
        println!("\n真のラベルとの一致率: {:.2}%", accuracy * 100.0);
    }
    
    // 異なるリンケージ方法の比較
    println!("\n異なるリンケージ方法の比較:");
    
    let linkages = vec![
        (Linkage::Single, "単連結法（最小距離法）"),
        (Linkage::Complete, "完全連結法（最大距離法）"),
        (Linkage::Average, "群平均法"),
        (Linkage::Ward, "ウォード法"),
    ];
    
    for (linkage, name) in linkages {
        let mut model = AgglomerativeClustering::new(3, linkage, DistanceMetric::Euclidean);
        let result = model.fit_transform(df)?;
        let cluster_counts = count_clusters(&result, "cluster")?;
        
        println!("\n{} の結果:", name);
        for (cluster, count) in cluster_counts {
            println!("クラスタ {}: {} サンプル", cluster, count);
        }
        
        if df.column("true_cluster").is_some() {
            let accuracy = compare_clustering(&result, "cluster", "true_cluster")?;
            println!("真のラベルとの一致率: {:.2}%", accuracy * 100.0);
        }
    }
    
    Ok(())
}

// DBSCANクラスタリングの例
fn dbscan_example(df: &DataFrame) -> Result<(), PandRSError> {
    println!("\n==== DBSCAN クラスタリング ====");
    
    // DBSCANインスタンスの作成（eps=1.0, min_samples=5, ユークリッド距離）
    let mut dbscan = DBSCAN::new(1.0, 5, DistanceMetric::Euclidean);
    
    // クラスタリングの実行
    let dbscan_result = dbscan.fit_transform(df)?;
    
    // クラスタの分布を表示
    let cluster_counts = count_clusters(&dbscan_result, "cluster")?;
    
    println!("\nDBSCANクラスタリング結果:");
    for (cluster, count) in cluster_counts {
        if cluster < 0 {
            println!("ノイズ: {} サンプル", count);
        } else {
            println!("クラスタ {}: {} サンプル", cluster, count);
        }
    }
    
    // クラスタリング結果を表示
    println!("\nクラスタリング結果（最初の5行）:");
    println!("{}", dbscan_result.head(5)?);
    
    // 異なるeps値での比較
    println!("\n異なるeps値での比較:");
    
    let eps_values = vec![0.5, 1.0, 1.5, 2.0];
    
    for eps in eps_values {
        let mut model = DBSCAN::new(eps, 5, DistanceMetric::Euclidean);
        let result = model.fit_transform(df)?;
        let cluster_counts = count_clusters(&result, "cluster")?;
        
        println!("\neps = {} の結果:", eps);
        let mut n_clusters = 0;
        let mut n_noise = 0;
        
        for (cluster, count) in &cluster_counts {
            if *cluster < 0 {
                n_noise = *count;
            } else {
                n_clusters += 1;
            }
        }
        
        println!("クラスタ数: {}", n_clusters);
        println!("ノイズポイント: {} サンプル", n_noise);
        
        for (cluster, count) in cluster_counts {
            if cluster < 0 {
                println!("ノイズ: {} サンプル", count);
            } else {
                println!("クラスタ {}: {} サンプル", cluster, count);
            }
        }
    }
    
    Ok(())
}

// クラスタリング結果の可視化（PCAを使用）
fn visualization_example(df: &DataFrame) -> Result<(), PandRSError> {
    println!("\n==== クラスタリング結果の可視化 (PCA) ====");
    
    // PCAで2次元に削減
    let mut pca = PCA::new(2);
    let pca_result = pca.fit_transform(df)?;
    
    // K-meansクラスタリングを実行
    let mut kmeans = KMeans::new(3, 100, 1e-4, Some(42));
    let kmeans_result = kmeans.fit_transform(&pca_result)?;
    
    // クラスタごとの平均位置を計算
    println!("\nPCA空間でのクラスタ中心:");
    
    let cluster_series = kmeans_result.column("cluster").unwrap();
    let pc1_series = kmeans_result.column("PC1").unwrap();
    let pc2_series = kmeans_result.column("PC2").unwrap();
    
    let mut cluster_positions = std::collections::HashMap::new();
    
    for row_idx in 0..kmeans_result.nrows() {
        let cluster = match cluster_series.get(row_idx) {
            DataValue::Int64(i) => *i,
            _ => 0,
        };
        
        let x = match pc1_series.get(row_idx) {
            DataValue::Float64(v) => *v,
            _ => 0.0,
        };
        
        let y = match pc2_series.get(row_idx) {
            DataValue::Float64(v) => *v,
            _ => 0.0,
        };
        
        let entry = cluster_positions.entry(cluster).or_insert((0.0, 0.0, 0));
        entry.0 += x;
        entry.1 += y;
        entry.2 += 1;
    }
    
    for (cluster, (sum_x, sum_y, count)) in cluster_positions {
        let avg_x = sum_x / count as f64;
        let avg_y = sum_y / count as f64;
        println!("クラスタ {}: 中心 ({:.4}, {:.4})", cluster, avg_x, avg_y);
    }
    
    // PCA空間での分散説明率
    println!("\nPCA分散説明率: {:?}", pca.explained_variance_ratio());
    println!("累積分散説明率: {:?}", pca.cumulative_explained_variance());
    
    Ok(())
}

// クラスタの分布をカウント
fn count_clusters(df: &DataFrame, cluster_col: &str) -> Result<Vec<(i64, usize)>, PandRSError> {
    let cluster_series = df.column(cluster_col).ok_or_else(|| {
        PandRSError::InvalidOperation(format!("Column {} not found", cluster_col))
    })?;
    
    let mut counts = std::collections::HashMap::new();
    
    for value in cluster_series.iter() {
        match value {
            DataValue::Int64(c) => {
                *counts.entry(*c).or_insert(0) += 1;
            }
            _ => {}
        }
    }
    
    // ソートした結果を返す
    let mut result: Vec<(i64, usize)> = counts.into_iter().collect();
    result.sort_by_key(|&(cluster, _)| cluster);
    
    Ok(result)
}

// クラスタリング結果を真のラベルと比較
fn compare_clustering(df: &DataFrame, pred_col: &str, true_col: &str) -> Result<f64, PandRSError> {
    let pred_series = df.column(pred_col).ok_or_else(|| {
        PandRSError::InvalidOperation(format!("Column {} not found", pred_col))
    })?;
    
    let true_series = df.column(true_col).ok_or_else(|| {
        PandRSError::InvalidOperation(format!("Column {} not found", true_col))
    })?;
    
    // クラスタラベルとトゥルークラスタの対応関係を見つける
    let mut cluster_mapping = std::collections::HashMap::new();
    let mut cluster_counts = std::collections::HashMap::new();
    
    for i in 0..df.nrows() {
        let pred_value = match pred_series.get(i) {
            DataValue::Int64(c) => *c,
            _ => continue,
        };
        
        let true_value = match true_series.get(i) {
            DataValue::Int64(c) => *c,
            DataValue::String(s) => s.parse::<i64>().unwrap_or(0),
            _ => continue,
        };
        
        let entry = cluster_counts
            .entry(pred_value)
            .or_insert_with(std::collections::HashMap::new);
        *entry.entry(true_value).or_insert(0) += 1;
    }
    
    // 各予測クラスタに対して、最も頻度の高い真のクラスタを割り当て
    for (&pred_cluster, counts) in &cluster_counts {
        let max_entry = counts.iter().max_by_key(|&(_, count)| count);
        if let Some((&true_cluster, _)) = max_entry {
            cluster_mapping.insert(pred_cluster, true_cluster);
        }
    }
    
    // 一致率を計算
    let mut correct = 0;
    let total = df.nrows();
    
    for i in 0..total {
        let pred_value = match pred_series.get(i) {
            DataValue::Int64(c) => *c,
            _ => continue,
        };
        
        let true_value = match true_series.get(i) {
            DataValue::Int64(c) => *c,
            DataValue::String(s) => s.parse::<i64>().unwrap_or(0),
            _ => continue,
        };
        
        if let Some(&mapped_cluster) = cluster_mapping.get(&pred_value) {
            if mapped_cluster == true_value {
                correct += 1;
            }
        }
    }
    
    Ok(correct as f64 / total as f64)
}

// サンプルデータの生成
fn create_sample_data() -> Result<DataFrame, PandRSError> {
    let mut rng = rand::thread_rng();
    
    // 300サンプルのデータを生成
    let n_samples = 300;
    let n_features = 5; // 5次元データ
    
    // 3つのクラスタを生成
    let mut features = Vec::new();
    for _ in 0..n_features {
        features.push(Vec::with_capacity(n_samples));
    }
    
    let mut true_clusters = Vec::with_capacity(n_samples);
    
    // クラスタ1: 多変量正規分布 (中心 [0, 0, ..., 0], 小さな分散)
    for _ in 0..n_samples/3 {
        for j in 0..n_features {
            features[j].push(rng.gen_range(-1.0..1.0));
        }
        true_clusters.push(0);
    }
    
    // クラスタ2: 多変量正規分布 (中心 [5, 5, ..., 5], 大きな分散)
    for _ in 0..n_samples/3 {
        for j in 0..n_features {
            features[j].push(5.0 + rng.gen_range(-1.5..1.5));
        }
        true_clusters.push(1);
    }
    
    // クラスタ3: 多変量正規分布 (中心 [-5, -5, ..., -5], 中程度の分散)
    for _ in 0..n_samples/3 {
        for j in 0..n_features {
            features[j].push(-5.0 + rng.gen_range(-1.0..1.0));
        }
        true_clusters.push(2);
    }
    
    // DataFrame作成
    let mut df = DataFrame::new();
    
    for (j, feature) in features.iter().enumerate() {
        df.add_column(format!("feature{}", j + 1), Series::from_vec(feature.clone())?)?;
    }
    
    // 真のクラスタラベルを追加
    df.add_column("true_cluster".to_string(), Series::from_vec(true_clusters.iter().map(|&c| DataValue::Int64(c)).collect())?)?;
    
    Ok(df)
}