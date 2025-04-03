use pandrs::{DataFrame, Series, MeltOptions, StackOptions, UnstackOptions, NA};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("データフレーム形状変換のサンプル");

    // サンプルデータフレームを作成
    let mut df = DataFrame::new();

    // ID列
    df.add_column(
        "id".to_string(),
        Series::new(vec!["1", "2", "3"], Some("id".to_string()))?,
    )?;

    // 製品カテゴリ
    df.add_column(
        "category".to_string(),
        Series::new(vec!["食品", "電化製品", "衣類"], Some("category".to_string()))?,
    )?;

    // 月別売上データ
    df.add_column(
        "1月".to_string(),
        Series::new(vec!["1000", "1500", "800"], Some("1月".to_string()))?,
    )?;

    df.add_column(
        "2月".to_string(),
        Series::new(vec!["1200", "1300", "1100"], Some("2月".to_string()))?,
    )?;

    df.add_column(
        "3月".to_string(),
        Series::new(vec!["900", "1800", "1400"], Some("3月".to_string()))?,
    )?;

    println!("元のデータフレーム:");
    println!("{:?}", df);

    // melt操作 - ワイド形式から長形式へ変換
    println!("\n----- melt操作（ワイド形式から長形式へ変換） -----");
    let melt_options = MeltOptions {
        id_vars: Some(vec!["id".to_string(), "category".to_string()]),
        value_vars: Some(vec![
            "1月".to_string(),
            "2月".to_string(),
            "3月".to_string(),
        ]),
        var_name: Some("月".to_string()),
        value_name: Some("売上".to_string()),
    };

    let melted_df = df.melt(&melt_options)?;
    println!("{:?}", melted_df);

    // stack操作
    println!("\n----- stack操作（列から行へスタック） -----");
    let stack_options = StackOptions {
        columns: Some(vec![
            "1月".to_string(),
            "2月".to_string(),
            "3月".to_string(),
        ]),
        var_name: Some("月".to_string()),
        value_name: Some("売上".to_string()),
        dropna: false,
    };

    let stacked_df = df.stack(&stack_options)?;
    println!("{:?}", stacked_df);

    // unstack操作（melted_dfを使用）
    println!("\n----- unstack操作（行から列へ変換） -----");
    let unstack_options = UnstackOptions {
        var_column: "月".to_string(),
        value_column: "売上".to_string(),
        index_columns: Some(vec!["id".to_string(), "category".to_string()]),
        fill_value: None,
    };

    let unstacked_df = melted_df.unstack(&unstack_options)?;
    println!("{:?}", unstacked_df);

    // データフレームの結合
    println!("\n----- データフレームの結合（concat） -----");
    
    // 追加のデータフレームを作成
    let mut df2 = DataFrame::new();
    df2.add_column(
        "id".to_string(),
        Series::new(vec!["4", "5"], Some("id".to_string()))?,
    )?;
    df2.add_column(
        "category".to_string(),
        Series::new(vec!["文房具", "家具"], Some("category".to_string()))?,
    )?;
    df2.add_column(
        "1月".to_string(),
        Series::new(vec!["500", "2000"], Some("1月".to_string()))?,
    )?;
    df2.add_column(
        "2月".to_string(),
        Series::new(vec!["600", "2200"], Some("2月".to_string()))?,
    )?;
    df2.add_column(
        "3月".to_string(),
        Series::new(vec!["700", "1900"], Some("3月".to_string()))?,
    )?;

    println!("追加のデータフレーム:");
    println!("{:?}", df2);

    // データフレームを結合
    let concat_df = DataFrame::concat(&[&df, &df2], true)?;
    println!("結合後のデータフレーム:");
    println!("{:?}", concat_df);

    // 条件付き集計
    println!("\n----- 条件付き集計 -----");
    
    // 条件: 2月の売上が1000以上の行のみを対象にカテゴリ別の3月の合計売上を計算
    let result = df.conditional_aggregate(
        "category",
        "3月",
        |row| {
            if let Some(sales_str) = row.get("2月") {
                if let Ok(sales) = sales_str.parse::<i32>() {
                    return sales >= 1000;
                }
            }
            false
        },
        |values| {
            let sum: i32 = values
                .iter()
                .filter_map(|v| v.parse::<i32>().ok())
                .sum();
            sum.to_string()
        },
    )?;

    println!("条件付き集計結果 (2月の売上が1000以上のカテゴリ別3月売上合計):");
    println!("{:?}", result);

    Ok(())
}