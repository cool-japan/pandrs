use pandrs::{OptimizedDataFrame, LazyFrame, Column, StringColumn, Float64Column};
use pandrs::error::Error;

fn main() -> Result<(), Error> {
    println!("=== 最適化版 MultiIndex 使用例 ===\n");

    // =========================================
    // マルチレベルインデックスをシミュレート
    // =========================================

    println!("--- 複数列によるマルチレベルインデックスのシミュレーション ---");

    // 最適化されたDataFrameを作成
    let mut df = OptimizedDataFrame::new();
    
    // レベル1の列（製品カテゴリ）
    let category = vec![
        "電子機器".to_string(), "電子機器".to_string(), "電子機器".to_string(), "電子機器".to_string(),
        "家具".to_string(), "家具".to_string(), "家具".to_string(), "家具".to_string(),
    ];
    let category_col = StringColumn::new(category);
    df.add_column("category", Column::String(category_col))?;
    
    // レベル2の列（製品名）
    let product = vec![
        "テレビ".to_string(), "テレビ".to_string(), "パソコン".to_string(), "パソコン".to_string(),
        "テーブル".to_string(), "テーブル".to_string(), "椅子".to_string(), "椅子".to_string(),
    ];
    let product_col = StringColumn::new(product);
    df.add_column("product", Column::String(product_col))?;
    
    // レベル3の列（年）
    let year = vec![
        "2022".to_string(), "2023".to_string(), "2022".to_string(), "2023".to_string(),
        "2022".to_string(), "2023".to_string(), "2022".to_string(), "2023".to_string(),
    ];
    let year_col = StringColumn::new(year);
    df.add_column("year", Column::String(year_col))?;
    
    // 値列（売上）
    let sales = vec![1000.0, 1200.0, 1500.0, 1800.0, 800.0, 900.0, 600.0, 700.0];
    let sales_col = Float64Column::new(sales);
    df.add_column("sales", Column::Float64(sales_col))?;
    
    println!("マルチレベルインデックスを持つデータフレーム:");
    println!("{:?}", df);
    
    // =========================================
    // マルチレベルを使った集計
    // =========================================
    
    println!("\n--- マルチレベルを使った集計 ---");
    
    // カテゴリと製品ごとの売上集計
    let result = LazyFrame::new(df.clone())
        .aggregate(
            vec!["category".to_string(), "product".to_string()],
            vec![("sales".to_string(), pandrs::AggregateOp::Sum, "total_sales".to_string())]
        )
        .execute()?;
    
    println!("カテゴリと製品ごとの売上:");
    println!("{:?}", result);
    
    // カテゴリごとの売上集計
    let category_result = LazyFrame::new(df.clone())
        .aggregate(
            vec!["category".to_string()],
            vec![("sales".to_string(), pandrs::AggregateOp::Sum, "total_sales".to_string())]
        )
        .execute()?;
    
    println!("\nカテゴリごとの売上:");
    println!("{:?}", category_result);
    
    // 年ごとの売上集計
    let year_result = LazyFrame::new(df)
        .aggregate(
            vec!["year".to_string()],
            vec![("sales".to_string(), pandrs::AggregateOp::Sum, "total_sales".to_string())]
        )
        .execute()?;
    
    println!("\n年ごとの売上:");
    println!("{:?}", year_result);
    
    println!("\n=== MultiIndex 使用例完了 ===");
    Ok(())
}