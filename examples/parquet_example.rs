use pandrs::{DataFrame, Series};
use pandrs::io::{read_parquet, write_parquet, ParquetCompression};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // サンプルのデータフレームを作成
    let mut df = DataFrame::new();
    
    // 整数列を追加
    let int_data = Series::new(vec![1, 2, 3, 4, 5], Some("id".to_string()))?;
    df.add_column("id".to_string(), int_data)?;
    
    // 浮動小数点列を追加
    let float_data = Series::new(vec![1.1, 2.2, 3.3, 4.4, 5.5], Some("value".to_string()))?;
    df.add_column("value".to_string(), float_data)?;
    
    // 文字列列を追加
    let string_data = Series::new(
        vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string(), "E".to_string()],
        Some("category".to_string())
    )?;
    df.add_column("category".to_string(), string_data)?;
    
    println!("元のデータフレーム:");
    println!("{:?}", df);
    
    // Parquetサポートはまだ実装中
    println!("\n注意: Parquet対応は現在実装中です。");
    println!("将来のリリースで利用可能になる予定です。");
    
    /*
    // 現在はParquet機能は未実装ですが、依存関係を導入しました
    // 以下のコードは将来のバージョンで使えるようになる予定です

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