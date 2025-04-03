use pandrs::{NASeries, PandRSError, NA};

fn main() -> Result<(), PandRSError> {
    println!("=== NA値（欠損値）のサポート ===");

    // 欠損値を含むデータを作成
    let data = vec![
        NA::Value(10),
        NA::Value(20),
        NA::NA, // 欠損値
        NA::Value(40),
        NA::NA, // 欠損値
    ];

    // NASeriesを作成
    let series = NASeries::new(data, Some("numbers".to_string()))?;

    println!("NAを含むシリーズ: {:?}", series);
    println!("NAの数: {}", series.na_count());
    println!("値の数: {}", series.value_count());
    println!("NAを含むか: {}", series.has_na());

    // 集計関数の挙動
    println!("\n--- NA値の扱い ---");
    println!("合計（NAは無視）: {:?}", series.sum());
    println!("平均（NAは無視）: {:?}", series.mean());
    println!("最小値（NAは無視）: {:?}", series.min());
    println!("最大値（NAは無視）: {:?}", series.max());

    // NAの処理
    println!("\n--- NA値の処理 ---");
    let dropped = series.dropna()?;
    println!("NAを削除したシリーズ: {:?}", dropped);
    println!("NAを削除した後の長さ: {}", dropped.len());

    let filled = series.fillna(0)?;
    println!("NAを0で埋めたシリーズ: {:?}", filled);

    // Optionからの変換
    println!("\n--- Optionからの変換 ---");
    let option_data = vec![Some(100), Some(200), None, Some(400), None];
    let option_series = NASeries::from_options(option_data, Some("from_options".to_string()))?;
    println!("Optionからのシリーズ: {:?}", option_series);

    // 数値演算
    println!("\n--- NA値を含む数値演算 ---");
    let a = NA::Value(10);
    let b = NA::Value(5);
    let na = NA::<i32>::NA;

    println!("{:?} + {:?} = {:?}", a, b, a + b);
    println!("{:?} - {:?} = {:?}", a, b, a - b);
    println!("{:?} * {:?} = {:?}", a, b, a * b);
    println!("{:?} / {:?} = {:?}", a, b, a / b);

    println!("{:?} + {:?} = {:?}", a, na, a + na);
    println!("{:?} * {:?} = {:?}", b, na, b * na);

    println!("=== NA値サンプル完了 ===");
    Ok(())
}
