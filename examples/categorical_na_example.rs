use pandrs::{DataFrame, NA, Series};
use pandrs::series::{CategoricalOrder, StringCategorical};
use pandrs::error::Result;
use std::path::Path;

fn main() -> Result<()> {
    println!("=== カテゴリカルデータとNA（欠損値）統合の例 ===\n");
    
    // 1. カテゴリカルデータの作成
    println!("1. カテゴリカルデータの作成");
    
    // NA値を含むベクトルを作成
    let values = vec![
        NA::Value("赤".to_string()),
        NA::Value("青".to_string()),
        NA::NA,  // 欠損値
        NA::Value("緑".to_string()),
        NA::Value("赤".to_string()),  // 重複値
    ];
    
    // ベクトルからカテゴリカルデータ型を作成
    // 順序なしカテゴリとして作成
    let cat = StringCategorical::from_na_vec(
        values.clone(),
        None,  // カテゴリを自動検出
        Some(CategoricalOrder::Unordered)  // 順序なし
    )?;
    
    println!("カテゴリ一覧: {:?}", cat.categories());
    println!("カテゴリ数: {}", cat.categories().len());
    println!("データ数: {}", cat.len());
    
    // カテゴリコードの表示
    println!("内部コード: {:?}", cat.codes());
    println!();
    
    // 2. 順序付きカテゴリカルデータの作成
    println!("2. 順序付きカテゴリカルデータの作成");
    
    // 明示的な順序を持つカテゴリリスト
    let ordered_categories = vec!["低".to_string(), "中".to_string(), "高".to_string()];
    
    // NA値を含むベクトルを作成
    let values = vec![
        NA::Value("中".to_string()),
        NA::Value("低".to_string()),
        NA::NA,  // 欠損値
        NA::Value("高".to_string()),
        NA::Value("中".to_string()),  // 重複値
    ];
    
    // 順序付きカテゴリとして作成
    let ordered_cat = StringCategorical::from_na_vec(
        values.clone(),
        Some(ordered_categories),  // 明示的なカテゴリリスト
        Some(CategoricalOrder::Ordered)  // 順序あり
    )?;
    
    println!("順序付きカテゴリ一覧: {:?}", ordered_cat.categories());
    println!("カテゴリ数: {}", ordered_cat.categories().len());
    println!("データ数: {}", ordered_cat.len());
    
    // カテゴリコードの表示
    println!("内部コード: {:?}", ordered_cat.codes());
    println!();
    
    // 3. カテゴリカルデータの操作
    println!("3. カテゴリカルデータの操作");
    
    // 2つのカテゴリカルデータを作成
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
    
    // 集合演算
    let union = cat1.union(&cat2)?; // 和集合
    let intersection = cat1.intersection(&cat2)?; // 積集合
    let difference = cat1.difference(&cat2)?; // 差集合
    
    println!("集合1のカテゴリ: {:?}", cat1.categories());
    println!("集合2のカテゴリ: {:?}", cat2.categories());
    println!("和集合: {:?}", union.categories());
    println!("積集合: {:?}", intersection.categories());
    println!("差集合(集合1-集合2): {:?}", difference.categories());
    println!();
    
    // 4. DataFrameでのカテゴリカル列の使用
    println!("4. DataFrameでのカテゴリカル列の使用");
    
    // NA値を含むベクトルを作成 (カテゴリカル用にまず作成)
    let values = vec![
        NA::Value("高".to_string()),
        NA::Value("中".to_string()),
        NA::NA,
        NA::Value("低".to_string()),
    ];
    
    // サンプルコードなので簡略化
    let order_cats = vec!["低".to_string(), "中".to_string(), "高".to_string()];
    
    // カテゴリカルデータを作成
    let cat_eval = StringCategorical::from_na_vec(
        values.clone(), // クローンしておく
        Some(order_cats), 
        Some(CategoricalOrder::Ordered)
    )?;
    
    // カテゴリカルデータのサイズを出力
    println!("作成するカテゴリカルデータのサイズ: {}", cat_eval.len());
    
    // カテゴリカル列として追加
    let categoricals = vec![("評価".to_string(), cat_eval)];
    let mut df = DataFrame::from_categoricals(categoricals)?;
    
    // データの行数を確認して一致させる
    println!("DataFrame行数: {}", df.row_count());
    println!("注意: DataFrame作成時にNA値は除外されます");
    
    // 数値列を追加 (行数を合わせる)
    let scores = vec![95, 80, 0];  // DataFrame行数に合わせる
    println!("スコアのサイズ: {}", scores.len());
    
    df.add_column(
        "スコア".to_string(), 
        Series::new(scores, Some("スコア".to_string()))?
    )?;
    
    println!("DataFrame: ");
    println!("{:#?}", df);
    
    // カテゴリカルデータの取得と検証
    println!("「評価」列はカテゴリカルか: {}", df.is_categorical("評価"));
    
    // エラーハンドリングを明示的に行う
    match df.get_categorical("評価") {
        Ok(cat_col) => println!("「評価」列のカテゴリ: {:?}", cat_col.categories()),
        Err(_) => println!("「評価」列のカテゴリ取得に失敗")
    }
    println!();
    
    // 5. CSVファイルとの入出力
    println!("5. CSVファイルとの入出力");
    
    // 一時ファイルに保存
    let temp_path = Path::new("/tmp/categorical_example.csv");
    df.to_csv(temp_path)?;
    
    println!("CSVファイルに保存: {}", temp_path.display());
    
    // ファイルから読み込み
    let df_loaded = DataFrame::from_csv(temp_path, true)?;
    
    // CSV読み込み後はカテゴリカル情報は失われる（通常の文字列列として読み込まれる）
    println!("CSVから読み込まれたデータ:");
    println!("{:#?}", df_loaded);
    
    // CSVから読み込まれたデータの確認
    
    // CSVから読み込んだデータは特殊な形式になっている点に注意
    println!("CSVから読み込んだデータ形式の例:");
    println!("「評価」列の最初の値: {:?}", df_loaded.get_column("評価").unwrap().values()[0]);
    
    // このCSV読み込みデータからカテゴリカルデータを再構築する場合は、
    // より複雑な処理が必要になるため、以下は単純な例としています
    
    // 新しいカテゴリカルデータを作成して例とする
    let new_values = vec![
        NA::Value("高".to_string()),
        NA::Value("中".to_string()),
        NA::NA,
        NA::Value("低".to_string()),
    ];
    
    let new_cat = StringCategorical::from_na_vec(
        new_values,
        None,
        Some(CategoricalOrder::Ordered),
    )?;
    
    println!("新たに作成したカテゴリカルデータの例:");
    println!("カテゴリ: {:?}", new_cat.categories());
    println!("順序: {:?}", new_cat.ordered());
    
    println!("\n実際のCSV読み込みデータからカテゴリカルデータへの変換には、");
    println!("CSVの形式や文字列エスケープ方法に応じたパース処理が必要です。");
    
    println!("\n=== サンプル終了 ===");
    Ok(())
}