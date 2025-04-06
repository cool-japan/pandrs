use pandrs::*;
use pandrs::stats;
use pandrs::ml::metrics::regression::{mean_squared_error, root_mean_squared_error, r2_score};
use pandrs::ml::metrics::classification::{accuracy_score, precision_score, recall_score, f1_score};
use rand::Rng;

// サンプルコードのため、OptimizedDataFrameにしか対応していないパイプラインを
// 通常のDataFrameで示すためのシンプルな例

fn main() -> Result<(), PandRSError> {
    println!("PandRS 機械学習関連機能の例");
    println!("========================");
    
    // サンプルデータの作成
    let df = create_sample_data()?;
    
    println!("元のデータフレーム:");
    println!("{:?}", df);
    
    // 線形回帰（statsモジュールを使用）
    let x_columns = &["value1", "value2"];
    let y_column = "target";
    
    // 回帰分析の実行
    let model = stats::linear_regression(&df, y_column, x_columns)?;
    
    println!("\n線形回帰の結果:");
    println!("係数: {:?}", model.coefficients);
    println!("切片: {}", model.intercept);
    println!("決定係数: {}", model.r_squared);
    
    // 値の取得と実際の値と予測値の比較
    let x1 = 2.0;
    let x2 = 50.0;
    
    // 予測値の計算（手動で実装）
    let predicted = model.intercept + model.coefficients[0] * x1 + model.coefficients[1] * x2;
    
    // 実際の値の計算（ノイズなし）
    let actual = 2.0 * x1 + 0.5 * x2;
    
    println!("\n予測の例:");
    println!("x1 = {}, x2 = {} のとき", x1, x2);
    println!("予測値: {}", predicted);
    println!("実際の値（ノイズなし）: {}", actual);
    println!("誤差: {}", (predicted - actual).abs());
    
    // 回帰メトリクスのサンプル
    let y_true = vec![3.0, 4.0, 5.0, 6.0, 7.0];
    let y_pred = vec![2.8, 4.2, 5.1, 5.8, 7.4];
    
    println!("\n回帰評価指標:");
    println!("MSE: {}", mean_squared_error(&y_true, &y_pred)?);
    println!("RMSE: {}", root_mean_squared_error(&y_true, &y_pred)?);
    println!("R2: {}", r2_score(&y_true, &y_pred)?);
    
    // 分類メトリクスのサンプル
    let y_true_class = vec![true, false, true, true, false];
    let y_pred_class = vec![true, false, false, true, true];
    
    println!("\n分類評価指標:");
    println!("精度: {}", accuracy_score(&y_true_class, &y_pred_class)?);
    println!("適合率: {}", precision_score(&y_true_class, &y_pred_class)?);
    println!("再現率: {}", recall_score(&y_true_class, &y_pred_class)?);
    println!("F1スコア: {}", f1_score(&y_true_class, &y_pred_class)?);
    
    // OptimizedDataFrameを使う機械学習パイプラインの紹介
    println!("\n機械学習パイプラインの機能:");
    println!("- StandardScaler: 数値データの標準化");
    println!("- MinMaxScaler: 数値データの0-1正規化");
    println!("- OneHotEncoder: カテゴリデータのダミー変数化");
    println!("- PolynomialFeatures: 多項式特徴量の生成");
    println!("- Imputer: 欠損値の補完");
    println!("- FeatureSelector: 特徴量の選択");
    println!("- Pipeline: 変換ステップの連鎖");
    
    Ok(())
}

// サンプルデータの作成
fn create_sample_data() -> Result<DataFrame, PandRSError> {
    let mut rng = rand::rng();
    
    // 10行のデータを生成
    let n = 10;
    
    // カテゴリカルデータ
    let categories = vec!["A", "B", "C"];
    let cat_data: Vec<String> = (0..n)
        .map(|_| categories[rng.random_range(0..categories.len())].to_string())
        .collect();
    
    // 数値データ
    let value1: Vec<f64> = (0..n).map(|_| rng.random_range(-10.0..10.0)).collect();
    let value2: Vec<f64> = (0..n).map(|_| rng.random_range(0.0..100.0)).collect();
    
    // 目的変数（線形関係 + ノイズ）
    let target: Vec<f64> = value1
        .iter()
        .zip(value2.iter())
        .map(|(x, y)| 2.0 * x + 0.5 * y + rng.random_range(-5.0..5.0))
        .collect();
    
    // DataFrame作成
    let mut df = DataFrame::new();
    df.add_column("category".to_string(), Series::new(cat_data, Some("category".to_string()))?)?;
    df.add_column("value1".to_string(), Series::new(value1, Some("value1".to_string()))?)?;
    df.add_column("value2".to_string(), Series::new(value2, Some("value2".to_string()))?)?;
    df.add_column("target".to_string(), Series::new(target, Some("target".to_string()))?)?;
    
    Ok(df)
}