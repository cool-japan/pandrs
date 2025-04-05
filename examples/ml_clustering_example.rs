extern crate pandrs;

use pandrs::optimized::OptimizedDataFrame;
use pandrs::ml::clustering::{KMeans, DBSCAN, AgglomerativeClustering, DistanceMetric, Linkage};
use pandrs::column::{Float64Column, Column};
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
        let x = center_x + rng.gen_range(-1.5..1.5);
        let y = center_y + rng.gen_range(-1.5..1.5);
        
        x_values.push(x);
        y_values.push(y);
        true_labels.push(cluster_idx as i64);
    }
    
    // データフレームの作成
    let mut df = OptimizedDataFrame::new();
    
    let x_col = Column::Float64(Float64Column::new(x_values, false, "x".to_string())?);
    let y_col = Column::Float64(Float64Column::new(y_values, false, "y".to_string())?);
    let true_labels_col = Column::Int64(pandrs::column::Int64Column::new(true_labels, false, "true_cluster".to_string())?);
    
    df.add_column("x".to_string(), x_col)?;
    df.add_column("y".to_string(), y_col)?;
    df.add_column("true_cluster".to_string(), true_labels_col)?;
    
    println!("データ生成完了: {} サンプル, {} クラスタ", n_samples, n_clusters);
    println!("データフレームの最初の数行:");
    println!("{}", df.head(5)?);
    
    // K-means クラスタリング
    println!("\n2. k-means クラスタリング");
    let mut kmeans = KMeans::new(3, 100, 1e-4, Some(42));
    let kmeans_result = kmeans.fit_transform(&df)?;
    
    println!("k-means クラスタリング完了");
    println!("クラスタ中心: {:?}", kmeans.centroids());
    println!("イナーシャ: {:.4}", kmeans.inertia());
    println!("反復回数: {}", kmeans.n_iter());
    println!("結果の最初の数行:");
    println!("{}", kmeans_result.head(5)?);
    
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
    println!("{}", dbscan_result.head(5)?);
    
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
    println!("{}", agg_result.head(5)?);
    
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
    
    for i in 0..cluster_col.len() {
        if let Some(cluster) = cluster_col.get_i64(i)? {
            *counts.entry(cluster).or_insert(0) += 1;
        }
    }
    
    // ソートされた結果を返す
    let mut result: Vec<(i64, usize)> = counts.into_iter().collect();
    result.sort_by_key(|(cluster, _)| *cluster);
    
    Ok(result)
}