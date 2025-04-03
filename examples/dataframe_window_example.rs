use pandrs::dataframe::DataFrame;
use pandrs::error::PandRSError;
use pandrs::series::Series;

fn main() -> Result<(), PandRSError> {
    println!("=== DataFrameのウィンドウ操作の例 ===\n");

    // サンプルデータを作成
    let mut df = create_sample_dataframe()?;

    // データフレームの内容を表示
    println!("元のDataFrame:");
    println!("{:?}\n", df);

    // 1. 移動平均（rolling mean）の例
    println!("1. 移動平均（rolling mean）の例");
    let df_rolling = df.rolling(3, "価格", "mean", None)?;
    println!("{:?}\n", df_rolling);

    // 2. 移動合計（rolling sum）の例
    println!("2. 移動合計（rolling sum）の例");
    let df_sum = df.rolling(3, "数量", "sum", Some("数量_移動合計_3"))?;
    println!("{:?}\n", df_sum);

    // 3. 移動標準偏差（rolling std）の例
    println!("3. 移動標準偏差（rolling std）の例");
    let df_std = df.rolling(3, "価格", "std", None)?;
    println!("{:?}\n", df_std);
    
    // 4. 移動最小値（rolling min）の例
    println!("4. 移動最小値（rolling min）の例");
    let df_min = df.rolling(3, "価格", "min", None)?;
    println!("{:?}\n", df_min);

    // 5. 移動最大値（rolling max）の例
    println!("5. 移動最大値（rolling max）の例");
    let df_max = df.rolling(3, "価格", "max", None)?;
    println!("{:?}\n", df_max);

    // 6. 拡大平均（expanding mean）の例
    println!("6. 拡大平均（expanding mean）の例");
    let df_expanding = df.expanding(2, "価格", "mean", None)?;
    println!("{:?}\n", df_expanding);

    // 7. 指数加重移動平均（EWM）の例
    println!("7. 指数加重移動平均（EWM）の例");
    
    // spanを指定した場合（span = 3）
    let df_ewm_span = df.ewm("価格", "mean", Some(3), None, Some("価格_ewm_span3"))?;
    println!("7.1 span=3を指定した場合:");
    println!("{:?}\n", df_ewm_span);
    
    // alphaを指定した場合（alpha = 0.5）
    let df_ewm_alpha = df.ewm("価格", "mean", None, Some(0.5), Some("価格_ewm_alpha0.5"))?;
    println!("7.2 alpha=0.5を指定した場合:");
    println!("{:?}\n", df_ewm_alpha);

    // 8. 複数の操作を一度に適用した例
    println!("8. 複数の操作を組み合わせた例:");
    
    // 順番に複数の操作を適用
    let mut result_df = df.clone();
    
    // 移動平均を追加
    result_df = result_df.rolling(3, "価格", "mean", None)?;
    
    // 拡大平均を追加
    result_df = result_df.expanding(2, "価格", "mean", None)?;
    
    // 指数加重移動平均を追加
    result_df = result_df.ewm("価格", "mean", Some(3), None, None)?;
    
    println!("{:?}\n", result_df);

    println!("=== DataFrameのウィンドウ操作の例を完了 ===");
    Ok(())
}

// サンプルデータフレームを作成するヘルパー関数
fn create_sample_dataframe() -> Result<DataFrame, PandRSError> {
    let mut df = DataFrame::new();
    
    // 日付列を追加
    let dates = vec![
        "2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", 
        "2023-01-05", "2023-01-06", "2023-01-07", "2023-01-08",
        "2023-01-09", "2023-01-10"
    ];
    let date_series = Series::new(dates, Some("日付".to_string()))?;
    df.add_column("日付".to_string(), date_series)?;
    
    // 商品列を追加
    let products = vec![
        "商品A", "商品B", "商品A", "商品C", 
        "商品B", "商品A", "商品C", "商品A",
        "商品B", "商品C"
    ];
    let product_series = Series::new(products, Some("商品".to_string()))?;
    df.add_column("商品".to_string(), product_series)?;
    
    // 価格列を追加
    let prices = vec![
        "100", "150", "110", "200", 
        "160", "120", "210", "115",
        "165", "220"
    ];
    let price_series = Series::new(prices, Some("価格".to_string()))?;
    df.add_column("価格".to_string(), price_series)?;
    
    // 数量列を追加
    let quantities = vec![
        "5", "3", "6", "2", 
        "4", "7", "3", "8",
        "5", "4"
    ];
    let quantity_series = Series::new(quantities, Some("数量".to_string()))?;
    df.add_column("数量".to_string(), quantity_series)?;
    
    Ok(df)
}