use pandrs::{OptimizedDataFrame, LazyFrame, Column, Int64Column, StringColumn, AggregateOp};
use pandrs::error::Error;

fn main() -> Result<(), Error> {
    println!("=== 最適化版グループ操作のサンプル ===");

    // データの準備
    let mut df = OptimizedDataFrame::new();
    
    // 値の列
    let values = Int64Column::new(vec![10, 20, 15, 30, 25, 15]);
    df.add_column("values", Column::Int64(values))?;
    
    // グループ化のためのキー
    let categories = StringColumn::new(vec![
        "A".to_string(), "B".to_string(), "A".to_string(), 
        "C".to_string(), "B".to_string(), "A".to_string()
    ]);
    df.add_column("category", Column::String(categories))?;
    
    println!("元のデータ:");
    println!("{:?}", df);
    
    // グループサイズを計算
    let sizes = LazyFrame::new(df.clone())
        .aggregate(
            vec!["category".to_string()],
            vec![("values".to_string(), AggregateOp::Count, "size".to_string())]
        )
        .execute()?;
    
    println!("\n--- グループサイズ ---");
    println!("{:?}", sizes);
    
    // 各グループの合計を計算
    let sums = LazyFrame::new(df.clone())
        .aggregate(
            vec!["category".to_string()],
            vec![("values".to_string(), AggregateOp::Sum, "sum".to_string())]
        )
        .execute()?;
    
    println!("\n--- グループごとの合計 ---");
    println!("{:?}", sums);
    
    // 各グループの平均を計算
    let means = LazyFrame::new(df.clone())
        .aggregate(
            vec!["category".to_string()],
            vec![("values".to_string(), AggregateOp::Mean, "mean".to_string())]
        )
        .execute()?;
    
    println!("\n--- グループごとの平均 ---");
    println!("{:?}", means);
    
    // 複数の集計を一度に計算
    let all_stats = LazyFrame::new(df.clone())
        .aggregate(
            vec!["category".to_string()],
            vec![
                ("values".to_string(), AggregateOp::Count, "count".to_string()),
                ("values".to_string(), AggregateOp::Sum, "sum".to_string()),
                ("values".to_string(), AggregateOp::Mean, "mean".to_string()),
                ("values".to_string(), AggregateOp::Min, "min".to_string()),
                ("values".to_string(), AggregateOp::Max, "max".to_string())
            ]
        )
        .execute()?;
    
    println!("\n--- グループごとの複数の統計量 ---");
    println!("{:?}", all_stats);
    
    // 異なるデータ型でのグループ化
    println!("\n--- 異なるデータ型でのグループ化 ---");
    
    let mut age_df = OptimizedDataFrame::new();
    let ages = Int64Column::new(vec![25, 30, 25, 40, 30, 25]);
    let values = Int64Column::new(vec![10, 20, 15, 30, 25, 15]);
    
    age_df.add_column("age", Column::Int64(ages))?;
    age_df.add_column("values", Column::Int64(values))?;
    
    let age_means = LazyFrame::new(age_df)
        .aggregate(
            vec!["age".to_string()],
            vec![("values".to_string(), AggregateOp::Mean, "mean".to_string())]
        )
        .execute()?;
    
    println!("年齢別の平均値:");
    println!("{:?}", age_means);
    
    println!("=== グループ操作サンプル完了 ===");
    Ok(())
}