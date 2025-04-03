use pandrs::series::{CategoricalOrder, StringCategorical, NASeries};
use pandrs::{DataFrame, NA};
use pandrs::error::Result;
use std::path::Path;

fn main() -> Result<()> {
    println!("=== カテゴリカルデータとNA（欠損値）統合の例 ===\n");
    
    // ===========================================================
    // 欠損値を含むカテゴリカルデータの作成
    // ===========================================================
    
    println!("--- 欠損値を含むカテゴリカルデータの作成 ---");
    let values = vec![
        NA::Value("東京".to_string()),
        NA::Value("大阪".to_string()),
        NA::NA,  // 欠損値
        NA::Value("名古屋".to_string()),
        NA::Value("大阪".to_string()),
    ];
    
    // NAベクトルからカテゴリカルデータを作成
    let cat = StringCategorical::from_na_vec(
        values, 
        None,  // カテゴリを自動検出
        Some(CategoricalOrder::Unordered),
    )?;
    
    println!("カテゴリ: {:?}", cat.categories());
    println!("コード: {:?}", cat.codes()); // -1がNA値を表す
    
    // カテゴリカルからNA値を含むベクトルに変換
    let na_values = cat.to_na_vec();
    println!("NA値を含むベクトル: {:?}", na_values);
    
    // NASeriesに変換
    let na_series = cat.to_na_series(Some("地域".to_string()))?;
    println!("NASeriesに変換: {:?}", na_series);
    println!("NA値の数: {}", na_series.na_count());
    
    // ===========================================================
    // DataFrameとの統合
    // ===========================================================
    
    println!("\n--- DataFrameとNA値の統合 ---");
    
    // NA値を含むシリーズを作成
    let regions = vec![
        NA::Value("北海道".to_string()),
        NA::Value("関東".to_string()),
        NA::NA,
        NA::Value("関西".to_string()),
        NA::Value("九州".to_string()),
    ];
    
    let populations = vec![
        NA::Value("少".to_string()),
        NA::Value("多".to_string()),
        NA::Value("中".to_string()),
        NA::NA,
        NA::Value("中".to_string()),
    ];
    
    let region_series = NASeries::new(regions, Some("地域".to_string()))?;
    let pop_series = NASeries::new(populations, Some("人口".to_string()))?;
    
    // DataFrameを作成
    let mut df = DataFrame::new();
    
    // NAシリーズをカテゴリカル列として追加
    df.add_na_series_as_categorical(
        "地域".to_string(),
        region_series,
        None,  // カテゴリを自動検出
        Some(CategoricalOrder::Unordered),
    )?;
    
    df.add_na_series_as_categorical(
        "人口".to_string(),
        pop_series,
        None,  // カテゴリを自動検出
        Some(CategoricalOrder::Ordered),
    )?;
    
    println!("\nデータフレーム:\n{:?}", df);
    println!("地域列はカテゴリカルか: {}", df.is_categorical("地域"));
    println!("人口列はカテゴリカルか: {}", df.is_categorical("人口"));
    
    // ===========================================================
    // カテゴリカル操作とNA値の処理
    // ===========================================================
    
    println!("\n--- カテゴリカル操作とNA値の処理 ---");
    
    // 地域カテゴリを取得
    let region_cat = df.get_categorical("地域")?;
    println!("地域カテゴリ: {:?}", region_cat.categories());
    
    // 人口カテゴリを取得
    let pop_cat = df.get_categorical("人口")?;
    println!("人口カテゴリ（順序付き）: {:?}", pop_cat.categories());
    
    // カテゴリの追加
    df.add_categories("地域", vec!["沖縄".to_string()])?;
    
    let updated_region_cat = df.get_categorical("地域")?;
    println!("地域カテゴリ（追加後）: {:?}", updated_region_cat.categories());
    
    // ===========================================================
    // CSV保存と読み込み
    // ===========================================================
    
    println!("\n--- カテゴリカルデータのCSV保存と読み込み ---");
    
    // CSVに保存
    let temp_path = Path::new("/tmp/categorical_na_example.csv");
    df.to_csv_with_categorical(temp_path)?;
    println!("CSVファイルに保存: {:?}", temp_path);
    
    // CSVから読み込み
    let df_loaded = DataFrame::from_csv_with_categorical(temp_path, true)?;
    println!("\n読み込み後のデータフレーム:\n{:?}", df_loaded);
    
    // カテゴリカル情報が保持されているか確認
    println!("地域列はカテゴリカルか: {}", df_loaded.is_categorical("地域"));
    println!("人口列はカテゴリカルか: {}", df_loaded.is_categorical("人口"));
    
    // ===========================================================
    // カテゴリカル演算
    // ===========================================================
    
    println!("\n--- カテゴリカル演算の例 ---");
    
    // 新しいカテゴリカルデータを作成
    let values1 = vec![
        NA::Value("A".to_string()),
        NA::Value("B".to_string()),
        NA::NA,
        NA::Value("C".to_string()),
    ];
    
    let values2 = vec![
        NA::Value("B".to_string()),
        NA::Value("C".to_string()),
        NA::Value("D".to_string()),
        NA::NA,
    ];
    
    let cat1 = StringCategorical::from_na_vec(values1, None, None)?;
    let cat2 = StringCategorical::from_na_vec(values2, None, None)?;
    
    println!("カテゴリカル1: {:?}", cat1.categories());
    println!("カテゴリカル2: {:?}", cat2.categories());
    
    // 和集合
    let union_cat = cat1.union(&cat2)?;
    println!("和集合のカテゴリ: {:?}", union_cat.categories());
    
    // 積集合
    let intersection_cat = cat1.intersection(&cat2)?;
    println!("積集合のカテゴリ: {:?}", intersection_cat.categories());
    
    // 差集合
    let difference_cat = cat1.difference(&cat2)?;
    println!("差集合のカテゴリ: {:?}", difference_cat.categories());
    
    println!("\n=== サンプル完了 ===");
    Ok(())
}