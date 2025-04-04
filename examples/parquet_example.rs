use pandrs::{
    optimized::dataframe::OptimizedDataFrame,
    column::{Column, Int64Column, Float64Column, StringColumn}
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // サンプルのデータフレームを作成
    let mut df = OptimizedDataFrame::new();
    
    // 整数列を追加
    let int_data = vec![1, 2, 3, 4, 5];
    df.add_column("id", Column::Int64(Int64Column::new(int_data)))?;
    
    // 浮動小数点列を追加
    let float_data = vec![1.1, 2.2, 3.3, 4.4, 5.5];
    df.add_column("value", Column::Float64(Float64Column::new(float_data)))?;
    
    // 文字列列を追加
    let string_data = vec![
        "A".to_string(), "B".to_string(), "C".to_string(), 
        "D".to_string(), "E".to_string()
    ];
    df.add_column("category", Column::String(StringColumn::new(string_data)))?;
    
    println!("元のデータフレーム:");
    println!("{:?}", df);
    
    // Parquetサポートはまだ実装中
    println!("\n注意: Parquet対応は現在実装中です。");
    println!("将来のリリースで利用可能になる予定です。");
    
    /*
    // Parquetファイルに書き込み
    let path = "example.parquet";
    match write_parquet(&df, path, Some(ParquetCompression::Snappy)) {
        Ok(_) => {
            println!("データフレームを {} に書き込みました", path);
            
            // Parquetファイルから読み込み
            match read_parquet(path) {
                Ok(df_read) => {
                    println!("\n読み込んだデータフレーム:");
                    println!("{:?}", df_read);
                    
                    // 結果を検証
                    assert_eq!(df.row_count(), df_read.row_count());
                    assert_eq!(df.column_count(), df_read.column_count());
                    
                    println!("\n検証成功: データが一致しました");
                },
                Err(e) => println!("Parquetファイルの読み込みエラー: {}", e),
            }
        },
        Err(e) => println!("Parquetファイルの書き込みエラー: {}", e),
    }
    */
    
    Ok(())
}