use pandrs::*;
use pandrs::ml::pipeline::{Pipeline, Transformer};
use pandrs::ml::preprocessing::{StandardScaler, MinMaxScaler, OneHotEncoder};
use pandrs::ml::metrics::regression::{mean_squared_error, r2_score};
use rand::prelude::*;

fn main() -> Result<(), PandRSError> {
    println!("PandRS 機械学習パイプラインの例");
    println!("==============================");
    
    // サンプルデータの作成
    let mut df = create_sample_data()?;
    println!("元のデータフレーム:");
    println!("{}", df);
    
    // パイプラインの構築
    let mut pipeline = Pipeline::new();
    
    // カテゴリカルデータをOne-Hot Encodingで変換
    pipeline.add_transformer(OneHotEncoder::new(vec!["category".to_string()], true));
    
    // 数値データを標準化
    pipeline.add_transformer(StandardScaler::new(vec!["value1".to_string(), "value2".to_string()]));
    
    // パイプラインによるデータ変換
    let transformed_df = pipeline.fit_transform(&df)?;
    
    println!("\n変換後のデータフレーム:");
    println!("{}", transformed_df);
    
    // 回帰分析の例示
    // 教師データとテストデータに分割
    let (train_df, test_df) = split_train_test(&transformed_df, 0.7)?;
    
    // 簡易な線形回帰（statsモジュールを使用）
    // 説明変数: value1, value2
    // 目的変数: target
    let features = vec!["value1", "value2"];
    let target = "target";
    
    let model = stats::regression::linear_regression(&train_df, target, &features)?;
    
    println!("\n線形回帰の結果:");
    println!("係数: {:?}", model.coefficients());
    println!("切片: {}", model.intercept());
    println!("決定係数: {}", model.r_squared());
    
    // テストデータで予測
    let predictions = model.predict(&test_df)?;
    
    // 評価指標の計算
    let y_true = test_df.column(target).unwrap();
    let mse = mean_squared_error(y_true, &predictions)?;
    let r2 = r2_score(y_true, &predictions)?;
    
    println!("\nテストデータでの評価:");
    println!("MSE: {}", mse);
    println!("R2スコア: {}", r2);
    
    Ok(())
}

// サンプルデータの作成
fn create_sample_data() -> Result<DataFrame, PandRSError> {
    let mut rng = rand::thread_rng();
    
    // 50行のデータを生成
    let n = 50;
    
    // カテゴリカルデータ
    let categories = vec!["A", "B", "C"];
    let cat_data: Vec<String> = (0..n)
        .map(|_| categories[rng.gen_range(0..categories.len())].to_string())
        .collect();
    
    // 数値データ
    let value1: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
    let value2: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..100.0)).collect();
    
    // 目的変数（線形関係 + ノイズ）
    let target: Vec<f64> = value1
        .iter()
        .zip(value2.iter())
        .map(|(x, y)| 2.0 * x + 0.5 * y + rng.gen_range(-5.0..5.0))
        .collect();
    
    // DataFrame作成
    let mut df = DataFrame::new();
    df.add_column("category".to_string(), Series::from_vec(cat_data)?)?;
    df.add_column("value1".to_string(), Series::from_vec(value1)?)?;
    df.add_column("value2".to_string(), Series::from_vec(value2)?)?;
    df.add_column("target".to_string(), Series::from_vec(target)?)?;
    
    Ok(df)
}

// 教師データとテストデータに分割する補助関数
fn split_train_test(df: &DataFrame, train_ratio: f64) -> Result<(DataFrame, DataFrame), PandRSError> {
    let n = df.nrows();
    let train_size = (n as f64 * train_ratio) as usize;
    
    // インデックスをシャッフル
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rand::thread_rng());
    
    // 訓練用インデックスとテスト用インデックス
    let train_indices = indices.iter().take(train_size).cloned().collect::<Vec<_>>();
    let test_indices = indices.iter().skip(train_size).cloned().collect::<Vec<_>>();
    
    // インデックスを使ってデータを取得
    let train_df = df.take_rows(&train_indices)?;
    let test_df = df.take_rows(&test_indices)?;
    
    Ok((train_df, test_df))
}