use pandrs::{OptimizedDataFrame, LazyFrame, Column, StringColumn, Int64Column, AggregateOp};
use pandrs::error::Error;

fn main() -> Result<(), Error> {
    println!("最適化版 データフレーム形状変換のサンプル");

    // サンプルデータフレームを作成
    let mut df = OptimizedDataFrame::new();

    // ID列
    let id_col = StringColumn::new(vec![
        "1".to_string(), "2".to_string(), "3".to_string()
    ]);
    df.add_column("id", Column::String(id_col))?;

    // 製品カテゴリ
    let category_col = StringColumn::new(vec![
        "食品".to_string(), "電化製品".to_string(), "衣類".to_string()
    ]);
    df.add_column("category", Column::String(category_col))?;

    // 月別売上データ
    let jan_col = Int64Column::new(vec![1000, 1500, 800]);
    df.add_column("1月", Column::Int64(jan_col))?;

    let feb_col = Int64Column::new(vec![1200, 1300, 1100]);
    df.add_column("2月", Column::Int64(feb_col))?;

    let mar_col = Int64Column::new(vec![900, 1800, 1400]);
    df.add_column("3月", Column::Int64(mar_col))?;

    println!("元のデータフレーム:");
    println!("{:?}", df);

    // melt操作 - ワイド形式から長形式へ変換
    println!("\n----- melt操作（ワイド形式から長形式へ変換） -----");
    
    let melted_df = df.melt(
        &["id", "category"],
        Some(&["1月", "2月", "3月"]),
        Some("月"),
        Some("売上")
    )?;
    
    println!("{:?}", melted_df);

    // データフレームの結合
    println!("\n----- データフレームの結合 -----");
    
    // 追加のデータフレームを作成
    let mut df2 = OptimizedDataFrame::new();
    
    let id_col2 = StringColumn::new(vec![
        "4".to_string(), "5".to_string()
    ]);
    df2.add_column("id", Column::String(id_col2))?;
    
    let category_col2 = StringColumn::new(vec![
        "文房具".to_string(), "家具".to_string()
    ]);
    df2.add_column("category", Column::String(category_col2))?;
    
    let jan_col2 = Int64Column::new(vec![500, 2000]);
    df2.add_column("1月", Column::Int64(jan_col2))?;
    
    let feb_col2 = Int64Column::new(vec![600, 2200]);
    df2.add_column("2月", Column::Int64(feb_col2))?;
    
    let mar_col2 = Int64Column::new(vec![700, 1900]);
    df2.add_column("3月", Column::Int64(mar_col2))?;

    println!("追加のデータフレーム:");
    println!("{:?}", df2);

    // データフレームを縦に結合
    // 注: ここではOptimizedDataFrameのconcatメソッドがあると仮定しています
    // 実際には実装が必要な場合があります
    let concat_df = df.append(&df2)?;
    println!("結合後のデータフレーム:");
    println!("{:?}", concat_df);

    // 条件付き集計 - LazyFrameとフィルタリングを使用
    println!("\n----- 条件付き集計 (LazyFrameを使用) -----");
    
    // 条件: 2月の売上が1000以上の行のみを対象にカテゴリ別の3月の合計売上を計算
    // まず、条件に合う行をフィルタリングするためのブール列を作成
    let feb_sales = df.column("2月")?;
    let mut is_high_sales = vec![false; df.row_count()];
    
    if let Some(int_col) = feb_sales.as_int64() {
        for i in 0..df.row_count() {
            if let Ok(Some(value)) = int_col.get(i) {
                is_high_sales[i] = value >= 1000;
            }
        }
    }
    
    // ブール列をDataFrameに追加
    let bool_col = pandrs::BooleanColumn::new(is_high_sales);
    let mut filtered_df = df.clone();
    filtered_df.add_column("is_high_sales", Column::Boolean(bool_col))?;
    
    // フィルタリングとグループ集計
    let result = LazyFrame::new(filtered_df)
        .filter("is_high_sales")
        .aggregate(
            vec!["category".to_string()],
            vec![("3月".to_string(), AggregateOp::Sum, "sum_march".to_string())]
        )
        .execute()?;
    
    println!("条件付き集計結果 (2月の売上が1000以上のカテゴリ別3月売上合計):");
    println!("{:?}", result);

    Ok(())
}