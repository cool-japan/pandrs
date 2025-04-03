use pandrs::pivot::AggFunction;
use pandrs::{DataFrame, PandRSError, Series};

fn main() -> Result<(), PandRSError> {
    println!("=== ピボットテーブルとグループ化の例 ===");

    // サンプルデータを作成
    let mut df = DataFrame::new();

    // 列データの作成
    let category = Series::new(
        vec![
            "A".to_string(),
            "B".to_string(),
            "A".to_string(),
            "C".to_string(),
            "B".to_string(),
            "A".to_string(),
            "C".to_string(),
            "B".to_string(),
        ],
        Some("category".to_string()),
    )?;

    let region = Series::new(
        vec![
            "East".to_string(),
            "West".to_string(),
            "West".to_string(),
            "East".to_string(),
            "East".to_string(),
            "West".to_string(),
            "West".to_string(),
            "East".to_string(),
        ],
        Some("region".to_string()),
    )?;

    let sales = Series::new(
        vec![100, 150, 200, 120, 180, 90, 250, 160],
        Some("sales".to_string()),
    )?;

    // DataFrameに列を追加
    df.add_column("category".to_string(), category)?;
    df.add_column("region".to_string(), region)?;
    df.add_column("sales".to_string(), sales)?;

    println!("DataFrame 情報:");
    println!("  列数: {}", df.column_count());
    println!("  行数: {}", df.row_count());
    println!("  列名: {:?}", df.column_names());

    // グループ化と集計
    println!("\n=== カテゴリーによるグループ化 ===");
    let category_group = df.groupby("category")?;

    println!("カテゴリー別合計（実装中）:");
    let _category_sum = category_group.sum(&["sales"])?;

    // ピボットテーブル（実装中）
    println!("\n=== ピボットテーブル ===");
    println!("カテゴリー別・地域別の売上合計（実装中）:");
    let _pivot_result = df.pivot_table("category", "region", "sales", AggFunction::Sum)?;

    // 注: ピボットテーブルとグループ化機能はまだ実装中のため、
    // 実際の結果は表示されません

    println!("\n=== 集計関数の例 ===");
    let functions = [
        AggFunction::Sum,
        AggFunction::Mean,
        AggFunction::Min,
        AggFunction::Max,
        AggFunction::Count,
    ];

    for func in &functions {
        println!(
            "集計関数: {} ({})",
            func.name(),
            match func {
                AggFunction::Sum => "合計",
                AggFunction::Mean => "平均",
                AggFunction::Min => "最小",
                AggFunction::Max => "最大",
                AggFunction::Count => "カウント",
            }
        );
    }

    println!("\n=== ピボットテーブルの例（完了）===");
    Ok(())
}
