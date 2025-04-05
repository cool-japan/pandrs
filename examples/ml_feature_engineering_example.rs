use pandrs::*;
use pandrs::ml::pipeline::{Pipeline, Transformer};
use pandrs::ml::preprocessing::{StandardScaler, MinMaxScaler, OneHotEncoder, PolynomialFeatures, Binner, Imputer, ImputeStrategy, FeatureSelector};
use pandrs::ml::metrics::regression::{mean_squared_error, r2_score};
use rand::prelude::*;

fn main() -> Result<(), PandRSError> {
    println!("PandRS 特徴量エンジニアリングの例");
    println!("================================");
    
    // サンプルデータの作成
    let mut df = create_sample_data()?;
    println!("元のデータフレーム:");
    println!("{}", df);
    
    // 1. 多項式特徴量の生成
    let mut poly_features = PolynomialFeatures::new(vec!["value1".to_string(), "value2".to_string()], 2, false);
    let poly_df = poly_features.fit_transform(&df)?;
    
    println!("\n多項式特徴量を追加したデータフレーム:");
    println!("{}", poly_df);
    
    // 2. ビニング（離散化）
    let mut binner = Binner::new_uniform(vec!["value1".to_string()], 4)
        .with_labels(vec!["低".to_string(), "中低".to_string(), "中高".to_string(), "高".to_string()]);
    let binned_df = binner.fit_transform(&df)?;
    
    println!("\nビニング適用後のデータフレーム:");
    println!("{}", binned_df);
    
    // 3. 欠損値の処理
    // サンプルデータに欠損値を追加
    let mut na_df = df.clone();
    let mut rng = rand::thread_rng();
    let n_rows = na_df.nrows();
    
    // value1列に欠損値を追加
    let mut value1_series = na_df.column("value1").unwrap().clone();
    for i in 0..10 {
        let idx = rng.gen_range(0..n_rows);
        value1_series.set_value(idx, DataValue::NA)?;
    }
    na_df.replace_column("value1".to_string(), value1_series)?;
    
    println!("\n欠損値を含むデータフレーム:");
    println!("{}", na_df);
    
    // 平均値で補完
    let mut imputer = Imputer::new(vec!["value1".to_string()], ImputeStrategy::Mean);
    let imputed_df = imputer.fit_transform(&na_df)?;
    
    println!("\n欠損値を平均値で補完したデータフレーム:");
    println!("{}", imputed_df);
    
    // 4. 特徴量選択
    // 分散に基づく選択
    let selector = FeatureSelector::variance_threshold(0.5);
    let selected_df = selector.fit_transform(&poly_df)?;
    
    println!("\n分散に基づいて選択された特徴量:");
    println!("{}", selected_df);
    
    // 5. パイプラインを使用した特徴量エンジニアリング
    let mut pipeline = Pipeline::new();
    
    // カテゴリカルデータをOne-Hot Encoding
    pipeline.add_transformer(OneHotEncoder::new(vec!["category".to_string()], true));
    
    // 欠損値を平均値で補完
    pipeline.add_transformer(Imputer::new(vec!["value1".to_string()], ImputeStrategy::Mean));
    
    // 多項式特徴量を生成
    pipeline.add_transformer(PolynomialFeatures::new(vec!["value1".to_string(), "value2".to_string()], 2, false));
    
    // 数値データを標準化
    pipeline.add_transformer(StandardScaler::new(vec!["value1".to_string(), "value2".to_string(),
                                                     "value1_value2".to_string(), "value1^2".to_string(), "value2^2".to_string()]));
    
    // パイプラインによるデータ変換
    let transformed_df = pipeline.fit_transform(&na_df)?;
    
    println!("\n特徴量エンジニアリングパイプライン適用後のデータフレーム:");
    println!("{}", transformed_df);
    
    // 回帰分析の例示
    // 教師データとテストデータに分割
    let (train_df, test_df) = split_train_test(&transformed_df, 0.7)?;
    
    // 説明変数として多項式特徴量を含むすべての数値列を使用
    let all_columns = train_df.column_names();
    let target = "target";
    
    // 目的変数以外のすべての列を特徴量として使用
    let features: Vec<&str> = all_columns.iter()
        .filter(|&col| col != target)
        .map(|s| s.as_str())
        .collect();
    
    let model = stats::regression::linear_regression(&train_df, target, &features)?;
    
    println!("\n多項式特徴量を使用した線形回帰の結果:");
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
    
    // 2つの特徴量 x1, x2を生成
    let value1: Vec<f64> = (0..n).map(|_| rng.gen_range(-10.0..10.0)).collect();
    let value2: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..100.0)).collect();
    
    // 非線形関係のある目的変数 y = 2*x1 + 0.5*x2 + 3*x1^2 + 0.1*x1*x2 + noise
    let target: Vec<f64> = value1
        .iter()
        .zip(value2.iter())
        .map(|(x1, x2)| {
            2.0 * x1 + 0.5 * x2 + 3.0 * x1.powi(2) + 0.1 * x1 * x2 + rng.gen_range(-5.0..5.0)
        })
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