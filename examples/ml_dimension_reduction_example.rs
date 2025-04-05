use pandrs::*;
use pandrs::ml::dimension_reduction::{PCA, TSNE, TSNEInit};
use rand::prelude::*;

fn main() -> Result<(), PandRSError> {
    println!("PandRS 次元削減アルゴリズムの例");
    println!("============================");
    
    // サンプルデータの生成
    let df = create_sample_data()?;
    println!("元のデータフレーム（最初の5行）:");
    println!("{}", df.head(5)?);
    
    // PCAの例
    pca_example(&df)?;
    
    // t-SNEの例
    tsne_example(&df)?;
    
    Ok(())
}

// PCAの例
fn pca_example(df: &DataFrame) -> Result<(), PandRSError> {
    println!("\n==== PCA (主成分分析) ====");
    
    // PCAインスタンスの作成（2次元に削減）
    let mut pca = PCA::new(2);
    
    // PCAの実行
    let pca_result = pca.fit_transform(df)?;
    
    println!("\nPCA実行後のデータフレーム（最初の5行）:");
    println!("{}", pca_result.head(5)?);
    
    // 分散説明率の表示
    println!("\n分散説明率:");
    println!("{:?}", pca.explained_variance_ratio());
    
    println!("\n累積分散説明率:");
    println!("{:?}", pca.cumulative_explained_variance());
    
    // 主成分の係数（固有ベクトル）の表示
    println!("\n主成分の係数:");
    for (i, component) in pca.components().iter().enumerate() {
        println!("PC{}: {:?}", i + 1, component);
    }
    
    // 3次元（高次元）へのPCA
    let mut pca3d = PCA::new(3);
    let pca3d_result = pca3d.fit_transform(df)?;
    
    println!("\n3次元PCA実行後のデータフレーム（最初の5行）:");
    println!("{}", pca3d_result.head(5)?);
    
    // 分散説明率の表示
    println!("\n3次元PCAの分散説明率:");
    println!("{:?}", pca3d.explained_variance_ratio());
    println!("累積分散説明率: {:?}", pca3d.cumulative_explained_variance());
    
    Ok(())
}

// t-SNEの例
fn tsne_example(df: &DataFrame) -> Result<(), PandRSError> {
    println!("\n==== t-SNE (t-distributed Stochastic Neighbor Embedding) ====");
    
    // t-SNEインスタンスの作成（2次元に削減）
    let mut tsne = TSNE::new(
        2, // 次元数
        30.0, // パープレキシティ
        200.0, // 学習率
        500, // 最大反復回数
        TSNEInit::PCA, // PCAで初期化
    );
    
    // t-SNEの実行
    let tsne_result = tsne.fit_transform(df)?;
    
    println!("\nt-SNE実行後のデータフレーム（最初の5行）:");
    println!("{}", tsne_result.head(5)?);
    
    // クラスタ分析の例（t-SNEの結果を使用）
    if df.column("cluster").is_some() {
        // クラスタごとの平均位置を計算
        let cluster_series = df.column("cluster").unwrap();
        
        let mut cluster_positions = std::collections::HashMap::new();
        
        for row_idx in 0..tsne_result.nrows() {
            let cluster = match cluster_series.get(row_idx) {
                DataValue::String(s) => s.clone(),
                DataValue::Int64(i) => i.to_string(),
                _ => "unknown".to_string(),
            };
            
            let x = match tsne_result.column("TSNE1").unwrap().get(row_idx) {
                DataValue::Float64(v) => *v,
                _ => 0.0,
            };
            
            let y = match tsne_result.column("TSNE2").unwrap().get(row_idx) {
                DataValue::Float64(v) => *v,
                _ => 0.0,
            };
            
            let entry = cluster_positions.entry(cluster).or_insert((0.0, 0.0, 0));
            entry.0 += x;
            entry.1 += y;
            entry.2 += 1;
        }
        
        println!("\nクラスタごとの平均位置（t-SNE空間）:");
        for (cluster, (sum_x, sum_y, count)) in cluster_positions {
            let avg_x = sum_x / count as f64;
            let avg_y = sum_y / count as f64;
            println!("クラスタ {}: ({:.4}, {:.4})", cluster, avg_x, avg_y);
        }
    }
    
    Ok(())
}

// サンプルデータの生成（3つのクラスタを持つ多次元データ）
fn create_sample_data() -> Result<DataFrame, PandRSError> {
    let mut rng = rand::thread_rng();
    
    // 300サンプルのデータを生成
    let n_samples = 300;
    let n_features = 10; // 10次元データ
    
    // 3つのクラスタを生成
    let mut features = Vec::new();
    for _ in 0..n_features {
        features.push(Vec::with_capacity(n_samples));
    }
    
    let mut clusters = Vec::with_capacity(n_samples);
    
    // クラスタ1: 多変量正規分布 (中心 [0, 0, ..., 0], 小さな分散)
    for i in 0..n_samples/3 {
        for j in 0..n_features {
            features[j].push(rng.gen_range(-1.0..1.0));
        }
        clusters.push("A".to_string());
    }
    
    // クラスタ2: 多変量正規分布 (中心 [5, 5, ..., 5], 大きな分散)
    for i in 0..n_samples/3 {
        for j in 0..n_features {
            features[j].push(5.0 + rng.gen_range(-2.0..2.0));
        }
        clusters.push("B".to_string());
    }
    
    // クラスタ3: 多変量正規分布 (中心 [-5, -5, ..., -5], 中程度の分散)
    for i in 0..n_samples/3 {
        for j in 0..n_features {
            features[j].push(-5.0 + rng.gen_range(-1.5..1.5));
        }
        clusters.push("C".to_string());
    }
    
    // DataFrame作成
    let mut df = DataFrame::new();
    
    for (j, feature) in features.iter().enumerate() {
        df.add_column(format!("feature{}", j + 1), Series::from_vec(feature.clone())?)?;
    }
    
    // クラスタラベルを追加
    df.add_column("cluster".to_string(), Series::from_vec(clusters)?)?;
    
    Ok(df)
}