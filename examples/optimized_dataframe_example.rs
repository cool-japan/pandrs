use pandrs::{OptimizedDataFrame, LazyFrame, AggregateOp, Column, Int64Column, Float64Column, StringColumn};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("最適化されたDataFrameと遅延評価のサンプル\n");
    
    // 最適化されたDataFrameの作成
    println!("1. 最適化されたDataFrameの作成");
    let mut df = OptimizedDataFrame::new();
    
    // 整数列を追加
    let id_data = Int64Column::new(vec![1, 2, 3, 4, 5]);
    df.add_column("id", Column::Int64(id_data))?;
    
    // 浮動小数点列を追加
    let value_data = Float64Column::new(vec![10.1, 20.2, 30.3, 40.4, 50.5]);
    df.add_column("value", Column::Float64(value_data))?;
    
    // 文字列列を追加
    let category_data = StringColumn::new(
        vec!["A".to_string(), "B".to_string(), "A".to_string(), "C".to_string(), "B".to_string()]
    );
    df.add_column("category", Column::String(category_data))?;
    
    // DataFrameの表示
    println!("\n{:?}\n", df);
    
    // 列の取得と操作
    println!("2. 列の取得と操作");
    let value_col = df.column("value")?;
    if let Some(float_col) = value_col.as_float64() {
        let sum = float_col.sum();
        let mean = float_col.mean().unwrap_or(0.0);
        println!("value列の合計: {}", sum);
        println!("value列の平均: {:.2}", mean);
    }
    
    // フィルタリング
    println!("\n3. カテゴリが 'A' の行を選択");
    // まず、ブール列を作成
    let category_col = df.column("category")?;
    let mut is_a = vec![false; df.row_count()];
    if let Some(str_col) = category_col.as_string() {
        for i in 0..df.row_count() {
            if let Ok(Some(val)) = str_col.get(i) {
                is_a[i] = val == "A";
            }
        }
    }
    
    // ブール列をDataFrameに追加
    let bool_data = pandrs::BooleanColumn::new(is_a);
    df.add_column("is_a", Column::Boolean(bool_data))?;
    
    // フィルタリングの実行
    let filtered_df = df.filter("is_a")?;
    println!("\n{:?}\n", filtered_df);
    
    // 遅延評価を使用したデータ処理
    println!("4. 遅延評価を使用したデータ処理");
    
    let lazy_df = LazyFrame::new(df);
    
    // 処理の定義（まだ実行されない）
    let result_lazy = lazy_df
        .select(&["id", "value", "category"])
        .filter("is_a");
    
    // 実行計画の説明
    println!("\n実行計画:");
    println!("{}", result_lazy.explain());
    
    // 計算の実行
    println!("\n遅延評価の結果:");
    let result_df = result_lazy.execute()?;
    println!("{:?}\n", result_df);
    
    // グループ化と集計の例
    println!("5. グループ化と集計");
    
    // 新しいデータフレームを作成
    let mut sales_df = OptimizedDataFrame::new();
    
    // 商品カテゴリ列
    let category_data = StringColumn::new(vec![
        "電子機器".to_string(), "家具".to_string(), "電子機器".to_string(), 
        "家具".to_string(), "電子機器".to_string(), "食品".to_string(),
        "食品".to_string(), "電子機器".to_string(), "家具".to_string(),
    ]);
    sales_df.add_column("カテゴリ", Column::String(category_data))?;
    
    // 売上額列
    let amount_data = Float64Column::new(vec![
        150.0, 230.5, 120.0, 450.5, 300.0, 50.0, 75.5, 200.0, 175.0
    ]);
    sales_df.add_column("売上額", Column::Float64(amount_data))?;
    
    println!("\n元の売上データ:");
    println!("{:?}\n", sales_df);
    
    // 遅延評価を使用してグループ化と集計を行う
    let lazy_sales = LazyFrame::new(sales_df);
    let agg_result = lazy_sales
        .aggregate(
            vec!["カテゴリ".to_string()],
            vec![
                ("売上額".to_string(), AggregateOp::Sum, "合計売上".to_string()),
                ("売上額".to_string(), AggregateOp::Mean, "平均売上".to_string()),
                ("売上額".to_string(), AggregateOp::Count, "売上回数".to_string()),
            ]
        )
        .execute()?;
    
    println!("カテゴリ別集計結果:");
    println!("{:?}\n", agg_result);
    
    Ok(())
}