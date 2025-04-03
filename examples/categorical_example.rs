use pandrs::series::{CategoricalOrder, StringCategorical};
use pandrs::{DataFrame, Series};
use pandrs::error::Result;

fn main() -> Result<()> {
    println!("=== Categorical データ型使用例 ===\n");
    
    // ===========================================================
    // 基本的なカテゴリカルデータの作成
    // ===========================================================
    
    println!("--- 基本的なカテゴリカルデータの作成 ---");
    let values = vec!["東京", "大阪", "東京", "名古屋", "大阪", "東京"];
    let values_str: Vec<String> = values.iter().map(|s| s.to_string()).collect();
    
    // カテゴリカルデータを作成（自動的にユニークな値が抽出される）
    let cat = StringCategorical::new(
        values_str, 
        None,  // カテゴリを自動検出
        Some(CategoricalOrder::Unordered),
    )?;
    
    println!("オリジナルデータ: {:?}", values);
    println!("カテゴリ: {:?}", cat.categories());
    println!("順序タイプ: {:?}", cat.ordered());
    println!("データ長: {}", cat.len());
    
    // カテゴリカルから実際の値を取得
    println!("\n最初の3つの値: {} {} {}", 
        cat.get(0).unwrap(),
        cat.get(1).unwrap(),
        cat.get(2).unwrap()
    );
    
    // ===========================================================
    // 明示的なカテゴリリストを使用した作成
    // ===========================================================
    
    println!("\n--- 明示的なカテゴリリストでの作成 ---");
    let values2 = vec!["赤", "青", "赤"];
    let values2_str: Vec<String> = values2.iter().map(|s| s.to_string()).collect();
    
    // すべてのカテゴリを事前に定義
    let categories = vec!["赤", "青", "緑", "黄"];
    let categories_str: Vec<String> = categories.iter().map(|s| s.to_string()).collect();
    
    // 順序付きカテゴリカルを作成
    let cat2 = StringCategorical::new(
        values2_str,
        Some(categories_str),  // 明示的なカテゴリリスト
        Some(CategoricalOrder::Ordered),
    )?;
    
    println!("カテゴリ: {:?}", cat2.categories()); // 赤、青、緑、黄
    println!("データ: {:?}", 
        (0..cat2.len()).map(|i| cat2.get(i).unwrap()).collect::<Vec<_>>()
    ); // 赤、青、赤
    
    // ===========================================================
    // カテゴリカルデータの操作
    // ===========================================================
    
    println!("\n--- カテゴリ操作の例 ---");
    
    // ベースとなるカテゴリカルデータ
    let fruits = vec!["りんご", "バナナ", "りんご", "オレンジ"];
    let fruits_str: Vec<String> = fruits.iter().map(|s| s.to_string()).collect();
    let mut fruit_cat = StringCategorical::new(fruits_str, None, None)?;
    
    println!("元のカテゴリ: {:?}", fruit_cat.categories());
    
    // カテゴリの追加
    let new_cats = vec!["ぶどう", "いちご"];
    let new_cats_str: Vec<String> = new_cats.iter().map(|s| s.to_string()).collect();
    fruit_cat.add_categories(new_cats_str)?;
    
    println!("カテゴリ追加後: {:?}", fruit_cat.categories());
    
    // カテゴリの順序を変更
    let reordered = vec!["バナナ", "いちご", "オレンジ", "りんご", "ぶどう"];
    let reordered_str: Vec<String> = reordered.iter().map(|s| s.to_string()).collect();
    fruit_cat.reorder_categories(reordered_str)?;
    
    println!("順序変更後: {:?}", fruit_cat.categories());
    println!("データ: {:?}", 
        (0..fruit_cat.len()).map(|i| fruit_cat.get(i).unwrap()).collect::<Vec<_>>()
    );
    
    // ===========================================================
    // DataFrameとの連携
    // ===========================================================
    
    println!("\n--- DataFrameとカテゴリカルデータの連携 ---");
    
    // 基本的なDataFrameを作成
    let mut df = DataFrame::new();
    
    // 通常の列を追加
    let regions = vec!["北海道", "関東", "関西", "九州", "関東", "関西"];
    let regions_str: Vec<String> = regions.iter().map(|s| s.to_string()).collect();
    let pop = vec!["少", "多", "多", "中", "多", "多"];
    let pop_str: Vec<String> = pop.iter().map(|s| s.to_string()).collect();
    
    df.add_column("地域".to_string(), Series::new(regions_str, Some("地域".to_string()))?)?;
    df.add_column("人口".to_string(), Series::new(pop_str, Some("人口".to_string()))?)?;
    
    println!("オリジナルDataFrame:\n{:?}", df);
    
    // ===========================================================
    // 簡略化したカテゴリカルDataFrameの作成
    // ===========================================================
    
    // 直接カテゴリカルからDataFrameを作成
    println!("\n--- 直接カテゴリカルを含むDataFrameの作成 ---");
    
    // カテゴリカルデータを作成
    let populations = vec!["少", "中", "多"];
    let populations_str: Vec<String> = populations.iter().map(|s| s.to_string()).collect();
    let pop_cat = StringCategorical::new(
        populations_str,
        None,  // 自動検出
        Some(CategoricalOrder::Ordered)
    )?;
    
    // 地域データ
    let regions = vec!["北海道", "関東", "関西"];
    let regions_str: Vec<String> = regions.iter().map(|s| s.to_string()).collect();
    
    // 両方のカテゴリカルからDataFrameを作成
    let categoricals = vec![
        ("人口".to_string(), pop_cat),
    ];
    
    let mut df_cat = DataFrame::from_categoricals(categoricals)?;
    
    // 地域列を追加
    df_cat.add_column("地域".to_string(), Series::new(regions_str, Some("地域".to_string()))?)?;
    
    println!("\nカテゴリカル変換後のDataFrame:\n{:?}", df_cat);
    
    // カテゴリカル判定
    println!("\n人口列はカテゴリカルか: {}", df_cat.is_categorical("人口"));
    println!("地域列はカテゴリカルか: {}", df_cat.is_categorical("地域"));
    
    // ===========================================================
    // マルチカテゴリカルデータフレームの実装例
    // ===========================================================
    
    println!("\n--- マルチカテゴリカルDataFrameの例 ---");
    
    // 製品データと色データを別のカテゴリカルとして作成
    let products = vec!["A", "B", "C"];
    let products_str: Vec<String> = products.iter().map(|s| s.to_string()).collect();
    let product_cat = StringCategorical::new(products_str, None, None)?;
    
    let colors = vec!["赤", "青", "緑"];
    let colors_str: Vec<String> = colors.iter().map(|s| s.to_string()).collect();
    let color_cat = StringCategorical::new(colors_str, None, None)?;
    
    // 両方のカテゴリカルを含むデータフレームを作成
    let multi_categoricals = vec![
        ("製品".to_string(), product_cat),
        ("色".to_string(), color_cat),
    ];
    
    let multi_df = DataFrame::from_categoricals(multi_categoricals)?;
    
    println!("マルチカテゴリカルDataFrame:\n{:?}", multi_df);
    println!("\n製品列はカテゴリカルか: {}", multi_df.is_categorical("製品"));
    println!("色列はカテゴリカルか: {}", multi_df.is_categorical("色"));
    
    // ===========================================================
    // カテゴリカルデータの集計と分析
    // ===========================================================
    
    println!("\n--- カテゴリカルデータの集計とグループ化 ---");
    
    // 単純なデータフレームから始める
    let mut df_simple = DataFrame::new();
    
    // 製品データを追加
    let products = vec!["A", "B", "C", "A", "B"];
    let products_str: Vec<String> = products.iter().map(|s| s.to_string()).collect();
    let sales = vec!["100", "150", "200", "120", "180"];
    let sales_str: Vec<String> = sales.iter().map(|s| s.to_string()).collect();
    
    df_simple.add_column("製品".to_string(), Series::new(products_str.clone(), Some("製品".to_string()))?)?;
    df_simple.add_column("売上".to_string(), Series::new(sales_str, Some("売上".to_string()))?)?;
    
    println!("元のDataFrame:\n{:?}", df_simple);
    
    // 製品別に集計
    let product_counts = df_simple.value_counts("製品")?;
    println!("\n製品別カウント:\n{:?}", product_counts);
    
    // カテゴリカルシリーズの変換と相互運用
    println!("\n--- カテゴリカルとシリーズの相互変換 ---");
    
    // 単純なカテゴリカルシリーズの作成
    let letter_cat = StringCategorical::new(
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
        None,
        None
    )?;
    
    // シリーズに変換
    let letter_series = letter_cat.to_series(Some("文字".to_string()))?;
    println!("カテゴリカルからシリーズに変換したもの: {:?}", letter_series);
    
    // カテゴリカルデータについての追加情報
    println!("\n--- カテゴリカルデータの特性 ---");
    println!("カテゴリカルデータは値の出現に関係なく一度だけメモリに保存されます。");
    println!("このため、重複する文字列値の多いデータセットで特に効率的です。");
    println!("また、順序付きカテゴリカルを使うと意味のある順序でデータを並べられます。");
    
    println!("\n=== サンプル完了 ===");
    Ok(())
}