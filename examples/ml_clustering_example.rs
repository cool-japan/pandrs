extern crate pandrs;

use pandrs::optimized::OptimizedDataFrame;
use pandrs::ml::clustering::{KMeans, DBSCAN, AgglomerativeClustering, DistanceMetric, Linkage};
use pandrs::column::{Float64Column, Column};
use pandrs::ml::pipeline::Transformer;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 擬似的なクラスタリングデータを生成
    println!("✅ クラスタリングアルゴリズムの例");
    println!("=================================");
    println!("1. 擬似データの生成");
    
    let mut rng = StdRng::seed_from_u64(42);
    let n_samples = 1000;
    let n_clusters = 3;
    
    // クラスタごとに異なる中心点からデータを生成
    let cluster_centers = vec![
        (0.0, 0.0),       // クラスタ1の中心
        (5.0, 5.0),       // クラスタ2の中心
        (-5.0, 5.0),     // クラスタ3の中心
    ];
    
    let mut x_values = Vec::with_capacity(n_samples);
    let mut y_values = Vec::with_capacity(n_samples);
    let mut true_labels = Vec::with_capacity(n_samples);
    
    for i in 0..n_samples {
        // 各サンプルを均等にクラスタに割り当て
        let cluster_idx = i % n_clusters;
        let (center_x, center_y) = cluster_centers[cluster_idx];
        
        // クラスタ中心からのばらつきを生成（正規分布）
        let x = center_x + rng.random_range(-1.5..1.5);
        let y = center_y + rng.random_range(-1.5..1.5);
        
        x_values.push(x);
        y_values.push(y);
        true_labels.push(cluster_idx as i64);
    }
    
    // データフレームの作成
    let mut df = OptimizedDataFrame::new();
    
    let x_col = Column::Float64(Float64Column::with_name(x_values, "x"));
    let y_col = Column::Float64(Float64Column::with_name(y_values, "y"));
    let true_labels_col = Column::Int64(pandrs::column::Int64Column::with_name(true_labels, "true_cluster"));
    
    df.add_column("x".to_string(), x_col)?;
    df.add_column("y".to_string(), y_col)?;
    df.add_column("true_cluster".to_string(), true_labels_col)?;
    
    println!("データ生成完了: {} サンプル, {} クラスタ", n_samples, n_clusters);
    println!("データフレームの最初の数行:");
    // df.head() の代わりに最初の5行を直接表示
    println!("データフレーム (最初の5行):");
    for i in 0..std::cmp::min(5, df.row_count()) {
        if let (Ok(Some(x)), Ok(Some(y)), Ok(Some(cluster))) = (
            df.column("x").unwrap().as_float64().unwrap().get(i),
            df.column("y").unwrap().as_float64().unwrap().get(i),
            df.column("true_cluster").unwrap().as_int64().unwrap().get(i)
        ) {
            println!("行 {}: x={:.4}, y={:.4}, cluster={}", i, x, y, cluster);
        }
    }
    
    // K-means クラスタリング
    println!("\n2. k-means クラスタリング");
    let mut kmeans = KMeans::new(3, 100, 1e-4, Some(42));
    let kmeans_result = kmeans.fit_transform(&df)?;
    
    println!("k-means クラスタリング完了");
    println!("クラスタ中心: {:?}", kmeans.centroids());
    println!("イナーシャ: {:.4}", kmeans.inertia());
    println!("反復回数: {}", kmeans.n_iter());
    println!("結果の最初の数行:");
    // kmeans_result.head() の代わりに最初の5行を直接表示
    println!("k-means結果 (最初の5行):");
    for i in 0..std::cmp::min(5, kmeans_result.row_count()) {
        if let (Ok(Some(x)), Ok(Some(y)), Ok(Some(cluster))) = (
            kmeans_result.column("x").unwrap().as_float64().unwrap().get(i),
            kmeans_result.column("y").unwrap().as_float64().unwrap().get(i),
            kmeans_result.column("cluster").unwrap().as_int64().unwrap().get(i)
        ) {
            println!("行 {}: x={:.4}, y={:.4}, cluster={}", i, x, y, cluster);
        }
    }
    
    // クラスタの分布を表示
    let cluster_counts = count_clusters(&kmeans_result, "cluster")?;
    println!("クラスタの分布:");
    for (cluster, count) in cluster_counts {
        println!("クラスタ {} に含まれるサンプル数: {}", cluster, count);
    }
    
    // DBSCAN クラスタリング
    println!("\n3. DBSCAN クラスタリング");
    let mut dbscan = DBSCAN::new(1.0, 5, DistanceMetric::Euclidean);
    let dbscan_result = dbscan.fit_transform(&df)?;
    
    println!("DBSCAN クラスタリング完了");
    println!("結果の最初の数行:");
    // dbscan_result.head() の代わりに最初の5行を直接表示
    println!("DBSCAN結果 (最初の5行):");
    for i in 0..std::cmp::min(5, dbscan_result.row_count()) {
        if let (Ok(Some(x)), Ok(Some(y)), Ok(Some(cluster))) = (
            dbscan_result.column("x").unwrap().as_float64().unwrap().get(i),
            dbscan_result.column("y").unwrap().as_float64().unwrap().get(i),
            dbscan_result.column("cluster").unwrap().as_int64().unwrap().get(i)
        ) {
            println!("行 {}: x={:.4}, y={:.4}, cluster={}", i, x, y, cluster);
        }
    }
    
    // クラスタの分布を表示
    let cluster_counts = count_clusters(&dbscan_result, "cluster")?;
    println!("クラスタの分布:");
    for (cluster, count) in cluster_counts {
        println!("クラスタ {} に含まれるサンプル数: {}", cluster, count);
    }
    
    // 階層的クラスタリング
    println!("\n4. 階層的クラスタリング");
    let mut agg_clustering = AgglomerativeClustering::new(3, Linkage::Ward, DistanceMetric::Euclidean);
    let agg_result = agg_clustering.fit_transform(&df)?;
    
    println!("階層的クラスタリング完了");
    println!("結果の最初の数行:");
    // agg_result.head() の代わりに最初の5行を直接表示
    println!("階層的クラスタリング結果 (最初の5行):");
    for i in 0..std::cmp::min(5, agg_result.row_count()) {
        if let (Ok(Some(x)), Ok(Some(y)), Ok(Some(cluster))) = (
            agg_result.column("x").unwrap().as_float64().unwrap().get(i),
            agg_result.column("y").unwrap().as_float64().unwrap().get(i),
            agg_result.column("cluster").unwrap().as_int64().unwrap().get(i)
        ) {
            println!("行 {}: x={:.4}, y={:.4}, cluster={}", i, x, y, cluster);
        }
    }
    
    // クラスタの分布を表示
    let cluster_counts = count_clusters(&agg_result, "cluster")?;
    println!("クラスタの分布:");
    for (cluster, count) in cluster_counts {
        println!("クラスタ {} に含まれるサンプル数: {}", cluster, count);
    }
    
    println!("\n=================================");
    println!("✅ クラスタリングの例が正常に完了しました");
    
    Ok(())
}

// クラスタごとのサンプル数をカウントする関数
fn count_clusters(df: &OptimizedDataFrame, column: &str) -> Result<Vec<(i64, usize)>, Box<dyn std::error::Error>> {
    let mut counts = std::collections::HashMap::new();
    
    // クラスタごとの数をカウント
    let cluster_col = df.column(column)?;
    
    if let Some(int_col) = cluster_col.as_int64() {
        for i in 0..cluster_col.len() {
            if let Ok(Some(cluster)) = int_col.get(i) {
                *counts.entry(cluster).or_insert(0) += 1;
            }
        }
    }
    
    // ソートされた結果を返す
    let mut result: Vec<(i64, usize)> = counts.into_iter().collect();
    result.sort_by_key(|(cluster, _)| *cluster);
    
    Ok(result)
}