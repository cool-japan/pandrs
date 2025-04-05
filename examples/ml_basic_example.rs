// 最小限のサンプル例 (OptimizedDataFrame版)
use pandrs::{OptimizedDataFrame, Column, Float64Column, Int64Column};
use pandrs::column::ColumnTrait; // Add ColumnTrait import
use pandrs::error::Error;
use pandrs::ml::preprocessing::{StandardScaler, MinMaxScaler};
use pandrs::ml::pipeline::{Pipeline, Transformer};

fn main() -> Result<(), Error> {
    println!("PandRS 機械学習モジュール基本例 (OptimizedDataFrame版)");
    println!("===================================================");
    
    // サンプルデータの作成
    let mut df = OptimizedDataFrame::new();
    
    // 特徴量1: Float64型
    let feature1 = Float64Column::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    df.add_column("feature1", Column::Float64(feature1))?;
    
    // 特徴量2: Float64型
    let feature2 = Float64Column::new(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    df.add_column("feature2", Column::Float64(feature2))?;
    
    // 特徴量3: Int64型
    let feature3 = Int64Column::new(vec![100, 200, 300, 400, 500]);
    df.add_column("feature3", Column::Int64(feature3))?;
    
    println!("元のデータフレーム:");
    println!("{:?}", df);
    
    // 基本情報の確認
    println!("\n基本情報:");
    println!("行数: {}", df.row_count());
    println!("列数: {}", df.column_count());
    println!("列名: {:?}", df.column_names());
    
    // StandardScalerを使用してみる
    println!("\n=== StandardScalerの適用 ===");
    let mut scaler = StandardScaler::new(vec!["feature1".to_string(), "feature2".to_string()]);
    let scaled_df = scaler.fit_transform(&df)?;
    println!("標準化されたデータフレーム:");
    println!("{:?}", scaled_df);
    
    // feature1列の値を確認
    if let Ok(col) = scaled_df.column("feature1") {
        if let Some(float_col) = col.as_float64() {
            println!("\nfeature1列の標準化された値:");
            for i in 0..float_col.len() {
                if let Ok(val) = float_col.get(i) {
                    println!("行 {}: {:?}", i, val);
                }
            }
        }
    }
    
    // MinMaxScalerを使用してみる
    println!("\n=== MinMaxScalerの適用 ===");
    let mut minmax = MinMaxScaler::new(vec!["feature1".to_string(), "feature2".to_string()], (0.0, 1.0));
    let normalized_df = minmax.fit_transform(&df)?;
    println!("正規化されたデータフレーム:");
    println!("{:?}", normalized_df);
    
    // feature1列の値を確認
    if let Ok(col) = normalized_df.column("feature1") {
        if let Some(float_col) = col.as_float64() {
            println!("\nfeature1列の正規化された値:");
            for i in 0..float_col.len() {
                if let Ok(val) = float_col.get(i) {
                    println!("行 {}: {:?}", i, val);
                }
            }
        }
    }
    
    // パイプラインを使用してみる
    println!("\n=== パイプラインの使用 ===");
    let mut pipeline = Pipeline::new();
    pipeline.add_transformer(StandardScaler::new(vec!["feature1".to_string()]))
           .add_transformer(MinMaxScaler::new(vec!["feature2".to_string()], (0.0, 1.0)));
    
    let result_df = pipeline.fit_transform(&df)?;
    println!("パイプライン適用後のデータフレーム:");
    println!("{:?}", result_df);
    
    println!("\n === サンプル実行完了 ===");
    Ok(())
}