use pandrs::{OptimizedDataFrame, LazyFrame, Column, StringColumn, Int64Column};
use pandrs::error::Error;

fn main() -> Result<(), Error> {
    println!("=== 最適化版 Categorical データ型使用例 ===\n");
    
    // ===========================================================
    // 基本的なカテゴリカルデータの作成
    // ===========================================================
    
    println!("--- 基本的なカテゴリカルデータの作成 ---");
    
    // 最適化されたDataFrameを作成
    let mut df = OptimizedDataFrame::new();
    
    // 都市データ（カテゴリカル）
    let cities = vec![
        "東京".to_string(), 
        "大阪".to_string(), 
        "東京".to_string(), 
        "名古屋".to_string(), 
        "大阪".to_string(), 
        "東京".to_string()
    ];
    
    // 最適化版では文字列プールを使用してカテゴリカルデータを効率的に格納
    let city_col = StringColumn::new(cities);
    df.add_column("city", Column::String(city_col))?;
    
    // 人口データを追加
    let population = vec![1350, 980, 1380, 550, 990, 1360];
    let pop_col = Int64Column::new(population);
    df.add_column("population", Column::Int64(pop_col))?;
    
    println!("カテゴリカルデータを含むDataFrame:");
    println!("{:?}", df);
    
    // ===========================================================
    // カテゴリカルデータの分析
    // ===========================================================
    
    println!("\n--- カテゴリカルデータの分析 ---");
    
    // カテゴリごとの集計を行う
    let result = LazyFrame::new(df.clone())
        .aggregate(
            vec!["city".to_string()],
            vec![
                ("population".to_string(), pandrs::AggregateOp::Count, "count".to_string()),
                ("population".to_string(), pandrs::AggregateOp::Sum, "total_population".to_string()),
                ("population".to_string(), pandrs::AggregateOp::Mean, "avg_population".to_string())
            ]
        )
        .execute()?;
    
    println!("カテゴリごとの集計結果:");
    println!("{:?}", result);
    
    // ===========================================================
    // カテゴリカルデータのフィルタリング例
    // ===========================================================
    
    println!("\n--- カテゴリカルデータのフィルタリング ---");
    
    // 「東京」のデータだけをフィルタリングする例
    // 注: 実際のフィルタリングはboolean列を作成して行う必要がある
    
    // 東京かどうかのブール列を作成
    let mut is_tokyo = vec![false; df.row_count()];
    let city_view = df.column("city")?;
    
    if let Some(str_col) = city_view.as_string() {
        for i in 0..df.row_count() {
            if let Ok(Some(city)) = str_col.get(i) {
                is_tokyo[i] = city == "東京";
            }
        }
    }
    
    // ブール列をDataFrameに追加
    let bool_col = pandrs::BooleanColumn::new(is_tokyo);
    let mut filtered_df = df.clone();
    filtered_df.add_column("is_tokyo", Column::Boolean(bool_col))?;
    
    // フィルタリングを実行
    let tokyo_df = filtered_df.filter("is_tokyo")?;
    
    println!("「東京」のデータのみ:");
    println!("{:?}", tokyo_df);
    
    println!("\n=== Categorical データ型使用例完了 ===");
    Ok(())
}