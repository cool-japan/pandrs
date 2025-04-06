use pandrs::*;
use pandrs::dataframe::DataValue;
use pandrs::ml::models::{SupervisedModel, LinearRegression, LogisticRegression};
use pandrs::ml::models::model_selection::{train_test_split, cross_val_score};
use pandrs::ml::models::model_persistence::ModelPersistence;
use pandrs::ml::pipeline::Pipeline;
use pandrs::ml::preprocessing::{StandardScaler, PolynomialFeatures};
use pandrs::ml::metrics::regression::{mean_squared_error, r2_score};
use pandrs::ml::metrics::classification::{accuracy_score, precision_score, recall_score, f1_score};
use rand::Rng;
use std::collections::HashMap;

fn main() -> Result<(), PandRSError> {
    println!("PandRS モデル学習と評価の例");
    println!("==========================");
    
    // 回帰モデルの例
    regression_example()?;
    
    // 分類モデルの例
    classification_example()?;
    
    // モデル選択と評価の例
    model_selection_example()?;
    
    // モデルの保存と読み込み例
    model_persistence_example()?;
    
    Ok(())
}

// 回帰モデルの例
fn regression_example() -> Result<(), PandRSError> {
    println!("\n==== 回帰モデルの例 ====");
    
    // 回帰データの生成
    let reg_df = create_regression_data()?;
    println!("回帰データサンプル:");
    println!("{}", reg_df);
    
    // 訓練データとテストデータに分割
    let (train_df, test_df) = train_test_split(&reg_df, 0.3, Some(42))?;
    println!("訓練データサイズ: {}, テストデータサイズ: {}", train_df.nrows(), test_df.nrows());
    
    // 特徴量のリスト
    let features = vec!["feature1", "feature2", "feature3"];
    let target = "target";
    
    // 線形回帰モデルの作成と学習
    let mut model = LinearRegression::new();
    model.fit(&train_df, target, &features)?;
    
    // 係数と切片の表示
    println!("\n線形回帰モデルの結果:");
    println!("係数: {:?}", model.coefficients());
    println!("切片: {}", model.intercept());
    
    // テストデータでの予測
    let predictions = model.predict(&test_df)?;
    
    // モデル評価
    let mse = mean_squared_error(test_df.column(target).unwrap(), &predictions)?;
    let r2 = r2_score(test_df.column(target).unwrap(), &predictions)?;
    
    println!("\nモデル評価:");
    println!("MSE: {}", mse);
    println!("R^2: {}", r2);
    
    // パイプラインを使った特徴量エンジニアリングと回帰
    println!("\n特徴量エンジニアリングパイプラインを使った回帰:");
    
    let mut pipeline = Pipeline::new();
    
    // 多項式特徴量の追加
    pipeline.add_transformer(PolynomialFeatures::new(vec![features[0].to_string(), features[1].to_string()], 2, false));
    
    // 特徴量の標準化
    pipeline.add_transformer(StandardScaler::new(vec![
        features[0].to_string(), features[1].to_string(), features[2].to_string(),
        format!("{}_{}^2", features[0], features[0]),
        format!("{}_{}", features[0], features[1]),
        format!("{}_{}^2", features[1], features[1]),
    ]));
    
    // パイプラインの適用
    let transformed_train_df = pipeline.fit_transform(&train_df)?;
    let transformed_test_df = pipeline.transform(&test_df)?;
    
    // 新しい特徴量リストの取得
    let poly_features: Vec<&str> = transformed_train_df.column_names().iter()
        .filter(|&name| name != target)
        .map(|s| s.as_str())
        .collect();
    
    // 変換後のデータで線形回帰
    let mut poly_model = LinearRegression::new();
    poly_model.fit(&transformed_train_df, target, &poly_features)?;
    
    // テストデータでの予測
    let poly_predictions = poly_model.predict(&transformed_test_df)?;
    
    // モデル評価
    let poly_mse = mean_squared_error(transformed_test_df.column(target).unwrap(), &poly_predictions)?;
    let poly_r2 = r2_score(transformed_test_df.column(target).unwrap(), &poly_predictions)?;
    
    println!("多項式特徴量を使用した線形回帰の評価:");
    println!("MSE: {}", poly_mse);
    println!("R^2: {}", poly_r2);
    
    Ok(())
}

// 分類モデルの例
fn classification_example() -> Result<(), PandRSError> {
    println!("\n==== 分類モデルの例 ====");
    
    // 分類データの生成
    let cls_df = create_classification_data()?;
    println!("分類データサンプル:");
    println!("{}", cls_df);
    
    // 訓練データとテストデータに分割
    let (train_df, test_df) = train_test_split(&cls_df, 0.3, Some(42))?;
    
    // 特徴量のリスト
    let features = vec!["feature1", "feature2"];
    let target = "target";
    
    // ロジスティック回帰モデルの作成と学習
    let mut model = LogisticRegression::new(0.1, 1000, 1e-5);
    model.fit(&train_df, target, &features)?;
    
    // 係数と切片の表示
    println!("\nロジスティック回帰モデルの結果:");
    println!("係数: {:?}", model.coefficients());
    println!("切片: {}", model.intercept());
    
    // テストデータでの予測
    let predictions = model.predict(&test_df)?;
    
    // モデル評価
    let y_true = test_df.column(target).unwrap();
    let accuracy = accuracy_score(y_true, &predictions)?;
    let precision = precision_score(y_true, &predictions, &DataValue::String("1".to_string()))?;
    let recall = recall_score(y_true, &predictions, &DataValue::String("1".to_string()))?;
    let f1 = f1_score(y_true, &predictions, &DataValue::String("1".to_string()))?;
    
    println!("\nモデル評価:");
    println!("正解率: {}", accuracy);
    println!("適合率: {}", precision);
    println!("再現率: {}", recall);
    println!("F1スコア: {}", f1);
    
    // 確率予測
    let proba_df = model.predict_proba(&test_df)?;
    println!("\n確率予測サンプル:");
    println!("{}", proba_df.head(5)?);
    
    Ok(())
}

// モデル選択と評価の例
fn model_selection_example() -> Result<(), PandRSError> {
    println!("\n==== モデル選択と評価の例 ====");
    
    // 回帰データの生成
    let reg_df = create_regression_data()?;
    
    // 特徴量のリスト
    let features = vec!["feature1", "feature2", "feature3"];
    let target = "target";
    
    // 交差検証によるモデル評価
    println!("\n交差検証（5分割）の結果:");
    let model = LinearRegression::new();
    let cv_scores = cross_val_score(&model, &reg_df, target, &features, 5)?;
    
    println!("各分割のスコア: {:?}", cv_scores);
    println!("平均スコア: {}", cv_scores.iter().sum::<f64>() / cv_scores.len() as f64);
    
    Ok(())
}

// モデルの保存と読み込み例
fn model_persistence_example() -> Result<(), PandRSError> {
    println!("\n==== モデルの保存と読み込み例 ====");
    
    // 回帰データの生成
    let reg_df = create_regression_data()?;
    
    // 特徴量のリスト
    let features = vec!["feature1", "feature2", "feature3"];
    let target = "target";
    
    // モデルの学習
    let mut model = LinearRegression::new();
    model.fit(&reg_df, target, &features)?;
    
    // モデルの保存
    let model_path = "/tmp/linear_regression_model.json";
    model.save_model(model_path)?;
    println!("モデルを保存しました: {}", model_path);
    
    // モデルの読み込み
    let loaded_model = LinearRegression::load_model(model_path)?;
    println!("モデルを読み込みました");
    
    // 読み込んだモデルのパラメータ確認
    println!("読み込んだモデルの係数: {:?}", loaded_model.coefficients());
    println!("読み込んだモデルの切片: {}", loaded_model.intercept());
    
    // 予測の検証
    let orig_pred = model.predict(&reg_df.head(5)?)?;
    let loaded_pred = loaded_model.predict(&reg_df.head(5)?)?;
    
    println!("元のモデルの予測: {:?}", orig_pred.iter().take(5).collect::<Vec<_>>());
    println!("読み込んだモデルの予測: {:?}", loaded_pred.iter().take(5).collect::<Vec<_>>());
    
    Ok(())
}

// 回帰データの生成
fn create_regression_data() -> Result<DataFrame, PandRSError> {
    let mut rng = rand::thread_rng();
    
    // 100行のデータを生成
    let n = 100;
    
    // 3つの特徴量
    let feature1: Vec<f64> = (0..n).map(|_| rng.random_range(-10.0..10.0)).collect();
    let feature2: Vec<f64> = (0..n).map(|_| rng.random_range(0.0..100.0)).collect();
    let feature3: Vec<f64> = (0..n).map(|_| rng.random_range(-5.0..15.0)).collect();
    
    // 線形関係のある目的変数: y = 2*x1 + 0.5*x2 - 1.5*x3 + noise
    let target: Vec<f64> = feature1
        .iter()
        .zip(feature2.iter())
        .zip(feature3.iter())
        .map(|((x1, x2), x3)| {
            2.0 * x1 + 0.5 * x2 - 1.5 * x3 + rng.random_range(-5.0..5.0)
        })
        .collect();
    
    // DataFrame作成
    let mut df = DataFrame::new();
    df.add_column("feature1".to_string(), Series::from_vec(feature1)?)?;
    df.add_column("feature2".to_string(), Series::from_vec(feature2)?)?;
    df.add_column("feature3".to_string(), Series::from_vec(feature3)?)?;
    df.add_column("target".to_string(), Series::from_vec(target)?)?;
    
    Ok(df)
}

// 分類データの生成
fn create_classification_data() -> Result<DataFrame, PandRSError> {
    let mut rng = rand::thread_rng();
    
    // 100行のデータを生成
    let n = 100;
    
    // 2つの特徴量
    let feature1: Vec<f64> = (0..n).map(|_| rng.random_range(-5.0..5.0)).collect();
    let feature2: Vec<f64> = (0..n).map(|_| rng.random_range(-5.0..5.0)).collect();
    
    // ロジスティックモデルを使った二値分類
    // P(y=1) = sigmoid(1.5*x1 - 2*x2)
    let target: Vec<String> = feature1
        .iter()
        .zip(feature2.iter())
        .map(|(x1, x2)| {
            let z = 1.5 * x1 - 2.0 * x2;
            let p = 1.0 / (1.0 + (-z).exp());
            
            if rng.random::<f64>() < p {
                "1".to_string()
            } else {
                "0".to_string()
            }
        })
        .collect();
    
    // DataFrame作成
    let mut df = DataFrame::new();
    df.add_column("feature1".to_string(), Series::from_vec(feature1)?)?;
    df.add_column("feature2".to_string(), Series::from_vec(feature2)?)?;
    df.add_column("target".to_string(), Series::from_vec(target)?)?;
    
    Ok(df)
}