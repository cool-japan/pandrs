use pandrs::{DataFrame, NASeries, ParallelUtils, Series, NA};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== 並列処理機能の例 ===\n");

    // サンプルデータの作成
    let numbers = Series::new(
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        Some("numbers".to_string()),
    )?;

    // Series.par_map のテスト
    println!("並列マップ処理の例:");
    let squared = numbers.par_map(|x| x * x);
    println!("元の値: {:?}", numbers.values());
    println!("二乗: {:?}", squared.values());

    // Series.par_filter のテスト
    println!("\n並列フィルタリングの例:");
    let even_numbers = numbers.par_filter(|x| x % 2 == 0);
    println!("偶数のみ: {:?}", even_numbers.values());

    // NAを含むデータの処理
    let na_data = vec![
        NA::Value(10),
        NA::Value(20),
        NA::NA,
        NA::Value(40),
        NA::NA,
        NA::Value(60),
    ];
    let na_series = NASeries::new(na_data, Some("na_numbers".to_string()))?;

    println!("\nNAを含むデータの並列処理:");
    let na_tripled = na_series.par_map(|x| x * 3);
    println!("元の値: {:?}", na_series.values());
    println!("3倍: {:?}", na_tripled.values());

    // DataFrame の並列処理
    println!("\nDataFrameの並列処理:");

    // サンプルDataFrameの作成
    let mut df = DataFrame::new();
    let names = Series::new(
        vec!["Alice", "Bob", "Charlie", "David", "Eve"],
        Some("name".to_string()),
    )?;
    let ages = Series::new(vec![25, 30, 35, 40, 45], Some("age".to_string()))?;
    let scores = Series::new(vec![85, 90, 78, 92, 88], Some("score".to_string()))?;

    df.add_column("name".to_string(), names)?;
    df.add_column("age".to_string(), ages)?;
    df.add_column("score".to_string(), scores)?;

    // DataFrameの並列変換
    println!("DataFrame.par_apply の例:");
    let transformed_df = df.par_apply(|col, _row, val| {
        match col {
            "age" => {
                // 年齢に+1
                let age: i32 = val.parse().unwrap_or(0);
                (age + 1).to_string()
            }
            "score" => {
                // スコアに+5
                let score: i32 = val.parse().unwrap_or(0);
                (score + 5).to_string()
            }
            _ => val.to_string(),
        }
    })?;

    println!(
        "元のDF 行数: {}, 列数: {}",
        df.row_count(),
        df.column_count()
    );
    println!(
        "変換後のDF 行数: {}, 列数: {}",
        transformed_df.row_count(),
        transformed_df.column_count()
    );

    // 行のフィルタリング
    println!("\nDataFrame.par_filter_rows の例:");
    let filtered_df = df.par_filter_rows(|row| {
        // スコアが85より大きい行のみを保持
        if let Ok(values) = df.get_column_numeric_values("score") {
            if row < values.len() {
                return values[row] > 85.0;
            }
        }
        false
    })?;

    println!("フィルタリング後の行数: {}", filtered_df.row_count());

    // ParallelUtils の使用例
    println!("\nParallelUtils機能の例:");

    let unsorted = vec![5, 3, 8, 1, 9, 4, 7, 2, 6];
    let sorted = ParallelUtils::par_sort(unsorted.clone());
    println!("ソート前: {:?}", unsorted);
    println!("ソート後: {:?}", sorted);

    let numbers_vec = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let sum = ParallelUtils::par_sum(&numbers_vec);
    let mean = ParallelUtils::par_mean(&numbers_vec);
    let min = ParallelUtils::par_min(&numbers_vec);
    let max = ParallelUtils::par_max(&numbers_vec);

    println!("合計: {}", sum);
    println!("平均: {}", mean.unwrap());
    println!("最小: {}", min.unwrap());
    println!("最大: {}", max.unwrap());

    println!("\n=== 並列処理機能の例完了 ===");
    Ok(())
}
