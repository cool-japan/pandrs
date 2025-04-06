use pandrs::{PandRSError, OptimizedDataFrame};
use pandrs::column::{Float64Column, StringColumn, Column, ColumnTrait};
use pandrs::ml::dimension_reduction::{PCA, TSNE, TSNEInit};
use pandrs::ml::pipeline::Transformer;
use rand::Rng;

fn main() -> Result<(), PandRSError> {
    println!("PandRS 次元削減アルゴリズムの例");
    println!("============================");
    
    // サンプルデータの生成
    let df = create_sample_data()?;
    println!("元のデータフレーム: {:?} 行 x {:?} 列", df.row_count(), df.column_names().len());
    
    // PCAの例
    pca_example(&df)?;
    
    // t-SNEの例
    tsne_example(&df)?;
    
    Ok(())
}

// PCAの例
fn pca_example(df: &OptimizedDataFrame) -> Result<(), PandRSError> {
    println!("\n==== PCA (主成分分析) ====");
    
    // PCAインスタンスの作成（2次元に削減）
    let mut pca = PCA::new(2);
    
    // PCAの実行
    let pca_result = pca.fit_transform(df)?;
    
    println!("\nPCA実行後のデータフレーム: {:?} 行 x {:?} 列", pca_result.row_count(), pca_result.column_names().len());
    
    // 主成分1と主成分2の値を表示
    if let Ok(pc1_view) = pca_result.column("PC1") {
        if let Some(pc1) = pc1_view.as_float64() {
            // 平均と標準偏差を手動で計算
            let values: Vec<f64> = (0..pc1.len())
                .filter_map(|i| pc1.get(i).ok().flatten())
                .collect();
            
            let mean = if !values.is_empty() {
                values.iter().sum::<f64>() / values.len() as f64
            } else {
                0.0
            };
            
            let std_dev = if values.len() > 1 {
                let var = values.iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>() / (values.len() - 1) as f64;
                var.sqrt()
            } else {
                0.0
            };
            
            println!("\n主成分1の平均: {:.4}", mean);
            println!("主成分1の標準偏差: {:.4}", std_dev);
        }
    }
    
    if let Ok(pc2_view) = pca_result.column("PC2") {
        if let Some(pc2) = pc2_view.as_float64() {
            // 平均と標準偏差を手動で計算
            let values: Vec<f64> = (0..pc2.len())
                .filter_map(|i| pc2.get(i).ok().flatten())
                .collect();
            
            let mean = if !values.is_empty() {
                values.iter().sum::<f64>() / values.len() as f64
            } else {
                0.0
            };
            
            let std_dev = if values.len() > 1 {
                let var = values.iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>() / (values.len() - 1) as f64;
                var.sqrt()
            } else {
                0.0
            };
            
            println!("主成分2の平均: {:.4}", mean);
            println!("主成分2の標準偏差: {:.4}", std_dev);
        }
    }
    
    // 分散説明率の表示
    println!("\n分散説明率:");
    for (i, ratio) in pca.explained_variance_ratio().iter().enumerate() {
        println!("PC{}: {:.4} ({:.2}%)", i + 1, ratio, ratio * 100.0);
    }
    
    println!("\n累積分散説明率:");
    for (i, ratio) in pca.cumulative_explained_variance().iter().enumerate() {
        println!("PC1-PC{}: {:.4} ({:.2}%)", i + 1, ratio, ratio * 100.0);
    }
    
    // クラスタごとの主成分値の平均を計算
    calculate_cluster_means(&pca_result, "PC1", "PC2")?;
    
    // 3次元（高次元）へのPCA
    println!("\n==== 3次元PCA ====");
    let mut pca3d = PCA::new(3);
    let pca3d_result = pca3d.fit_transform(df)?;
    
    println!("\n3次元PCA実行後のデータフレーム: {:?} 行 x {:?} 列", pca3d_result.row_count(), pca3d_result.column_names().len());
    
    // 分散説明率の表示
    println!("\n3次元PCAの分散説明率:");
    for (i, ratio) in pca3d.explained_variance_ratio().iter().enumerate() {
        println!("PC{}: {:.4} ({:.2}%)", i + 1, ratio, ratio * 100.0);
    }
    println!("累積分散説明率: {:.4} ({:.2}%)", 
        pca3d.cumulative_explained_variance().last().unwrap_or(&0.0),
        pca3d.cumulative_explained_variance().last().unwrap_or(&0.0) * 100.0);
    
    Ok(())
}

// t-SNEの例
fn tsne_example(df: &OptimizedDataFrame) -> Result<(), PandRSError> {
    println!("\n==== t-SNE (t-distributed Stochastic Neighbor Embedding) ====");
    
    // t-SNEインスタンスの作成（2次元に削減）
    let mut tsne = TSNE::new(
        2,         // 次元数
        30.0,      // パープレキシティ
        200.0,     // 学習率
        100,       // 最大反復回数
        TSNEInit::PCA, // PCAで初期化
    );
    
    println!("\nt-SNEを実行中（しばらく時間がかかります）...");
    
    // t-SNEの実行
    let tsne_result = tsne.fit_transform(df)?;
    
    println!("t-SNE実行後のデータフレーム: {:?} 行 x {:?} 列", tsne_result.row_count(), tsne_result.column_names().len());
    
    // クラスタごとの埋め込み座標の平均を計算
    calculate_cluster_means(&tsne_result, "TSNE1", "TSNE2")?;
    
    Ok(())
}

// クラスタごとの平均座標を計算して表示
fn calculate_cluster_means(df: &OptimizedDataFrame, x_col: &str, y_col: &str) -> Result<(), PandRSError> {
    // クラスタカラムがない場合は何もしない
    if !df.column_names().contains(&"cluster".to_string()) {
        return Ok(());
    }
    
    println!("\nクラスタごとの平均座標:");
    
    let cluster_view = df.column("cluster")?;
    let x_view = df.column(x_col)?;
    let y_view = df.column(y_col)?;
    
    if let Some(cluster_col) = cluster_view.as_string() {
        if let (Some(x_col), Some(y_col)) = (x_view.as_float64(), y_view.as_float64()) {
            // クラスタの種類を取得
            let mut clusters = Vec::new();
            for i in 0..df.row_count() {
                if let Ok(Some(cluster)) = cluster_col.get(i) {
                    let cluster_str = cluster.to_string();
                    if !clusters.contains(&cluster_str) {
                        clusters.push(cluster_str);
                    }
                }
            }
            
            // クラスタごとの合計と数を計算
            let mut cluster_sums = std::collections::HashMap::new();
            
            for i in 0..df.row_count() {
                if let (Ok(Some(cluster)), Ok(Some(x)), Ok(Some(y))) = 
                    (cluster_col.get(i), x_col.get(i), y_col.get(i)) {
                    let entry = cluster_sums.entry(cluster.to_string())
                        .or_insert((0.0, 0.0, 0));
                    entry.0 += x;
                    entry.1 += y;
                    entry.2 += 1;
                }
            }
            
            // クラスタごとの平均を計算して表示
            for cluster in clusters {
                if let Some(&(x_sum, y_sum, count)) = cluster_sums.get(&cluster) {
                    let x_mean = x_sum / count as f64;
                    let y_mean = y_sum / count as f64;
                    println!("クラスタ {}: ({:.4}, {:.4}), {} サンプル", 
                        cluster, x_mean, y_mean, count);
                }
            }
        }
    }
    
    Ok(())
}

// サンプルデータの生成（3つのクラスタを持つ多次元データ）
fn create_sample_data() -> Result<OptimizedDataFrame, PandRSError> {
    let mut rng = rand::rng();
    let mut df = OptimizedDataFrame::new();
    
    // 300サンプルのデータを生成
    let n_samples = 300;
    let n_features = 10; // 10次元データ
    
    // クラスタの定義
    let clusters = ["A", "B", "C"];
    let centers = [
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    // クラスタAの中心
        vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],    // クラスタBの中心
        vec![-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0]  // クラスタCの中心
    ];
    
    // 各特徴量の値を生成
    for j in 0..n_features {
        let mut feature_values = Vec::with_capacity(n_samples);
        
        // クラスタAのデータ (中心 [0, 0, ...], 小さな分散)
        for _ in 0..n_samples/3 {
            feature_values.push(centers[0][j] + rng.random_range(-1.0..1.0));
        }
        
        // クラスタBのデータ (中心 [5, 5, ...], 大きな分散)
        for _ in 0..n_samples/3 {
            feature_values.push(centers[1][j] + rng.random_range(-2.0..2.0));
        }
        
        // クラスタCのデータ (中心 [-5, -5, ...], 中程度の分散)
        for _ in 0..n_samples/3 {
            feature_values.push(centers[2][j] + rng.random_range(-1.5..1.5));
        }
        
        // 特徴量列をデータフレームに追加
        let column = Float64Column::with_name(feature_values, format!("feature{}", j + 1));
        df.add_column(format!("feature{}", j + 1), Column::Float64(column))?;
    }
    
    // クラスタラベルを追加
    let mut cluster_labels = Vec::with_capacity(n_samples);
    
    for cluster_idx in 0..3 {
        for _ in 0..n_samples/3 {
            cluster_labels.push(clusters[cluster_idx].to_string());
        }
    }
    
    let cluster_column = StringColumn::with_name(cluster_labels, "cluster");
    df.add_column("cluster", Column::String(cluster_column))?;
    
    Ok(df)
}