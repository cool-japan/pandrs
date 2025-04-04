use std::time::Instant;
use pandrs::optimized::dataframe::OptimizedDataFrame;
use pandrs::column::{Column, StringColumn, Int64Column};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OptimizedDataFrame カテゴリカル機能テスト ===\n");
    
    // 基本的なデータフレームを作成
    let mut df = OptimizedDataFrame::new();
    
    // 人口データ（カテゴリカルデータとして適切な少数の繰り返しパターン）
    let population_data = vec![
        "多".to_string(), "少".to_string(), "中".to_string(), "多".to_string(), 
        "中".to_string(), "少".to_string(), "多".to_string(), "多".to_string(),
        "少".to_string(), "中".to_string()
    ];
    
    // 都市データ
    let cities = vec![
        "東京".to_string(), "札幌".to_string(), "名古屋".to_string(), "大阪".to_string(),
        "福岡".to_string(), "仙台".to_string(), "広島".to_string(), "横浜".to_string(),
        "神戸".to_string(), "京都".to_string()
    ];
    
    // 数値データ（サンプル）
    let values = vec![100, 50, 75, 95, 60, 45, 80, 90, 55, 70];
    
    // ---------------------------------------------------
    // 普通の列の追加とカテゴリカル列の追加を比較
    // ---------------------------------------------------
    println!("--- 通常の文字列列とカテゴリカル列の追加 ---");
    
    // 都市列を通常の文字列列として追加
    let start = Instant::now();
    df.add_column("都市", Column::String(StringColumn::new(cities)))?;
    let regular_time = start.elapsed();
    println!("通常の文字列列追加時間: {:?}", regular_time);
    
    // 人口列をカテゴリカル列として追加（高度に最適化）
    let start = Instant::now();
    df.add_categorical_column("人口", population_data)?;
    let categorical_time = start.elapsed();
    println!("カテゴリカル列追加時間: {:?}", categorical_time);
    
    // 数値列を追加
    df.add_column("値", Column::Int64(Int64Column::new(values)))?;
    
    println!("\nデータフレーム概要:");
    println!("{:?}", df);
    
    // ---------------------------------------------------
    // カテゴリカルステータスのチェック
    // ---------------------------------------------------
    println!("\n--- カテゴリカルステータスの確認 ---");
    println!("都市列はカテゴリカルか: {}", df.is_categorical("都市"));
    println!("人口列はカテゴリカルか: {}", df.is_categorical("人口"));
    
    // ---------------------------------------------------
    // 通常列をカテゴリカルに変換
    // ---------------------------------------------------
    println!("\n--- 通常列のカテゴリカル変換 ---");
    let start = Instant::now();
    df.astype_categorical("都市", None, None)?;
    let conversion_time = start.elapsed();
    println!("通常列のカテゴリカル変換時間: {:?}", conversion_time);
    
    println!("変換後、都市列はカテゴリカルか: {}", df.is_categorical("都市"));
    
    // ---------------------------------------------------
    // カテゴリの取得
    // ---------------------------------------------------
    println!("\n--- カテゴリの取得 ---");
    let population_cats = df.get_categories("人口")?;
    println!("人口のカテゴリ: {:?}", population_cats);
    
    let city_cats = df.get_categories("都市")?;
    println!("都市のカテゴリ: {:?}", city_cats);
    
    // ---------------------------------------------------
    // カテゴリの追加
    // ---------------------------------------------------
    println!("\n--- カテゴリの追加 ---");
    df.add_categories("人口", &["極多".to_string(), "極少".to_string()])?;
    
    let new_cats = df.get_categories("人口")?;
    println!("追加後の人口カテゴリ: {:?}", new_cats);
    
    // ---------------------------------------------------
    // カテゴリカル順序の設定
    // ---------------------------------------------------
    println!("\n--- カテゴリカル順序の設定 ---");
    // カテゴリカルオーダーのインポートができていないのでここでは直接渡す
    let ordered = match true {
        true => pandrs::series::CategoricalOrder::Ordered,
        false => pandrs::series::CategoricalOrder::Unordered
    };
    df.set_categorical_ordered("人口", ordered)?;
    println!("人口列を順序付きカテゴリカルに設定しました");
    
    // ---------------------------------------------------
    // 出現回数の計算
    // ---------------------------------------------------
    println!("\n--- カテゴリカル出現回数の計算 ---");
    let pop_counts = df.value_counts("人口")?;
    println!("人口の出現回数: {} 種類", pop_counts.len());
    println!("  詳細: {:?}", pop_counts);
    
    let city_counts = df.value_counts("都市")?;
    println!("\n都市の出現回数: {} 種類", city_counts.len());
    println!("  詳細: {:?}", city_counts);
    
    // ---------------------------------------------------
    // パフォーマンス比較のための大規模データセット
    // ---------------------------------------------------
    println!("\n--- パフォーマンス比較（大規模データセット）---");
    
    // 10,000行のデータセットを作成（カテゴリ数が少ない高重複データ）
    let categories = ["A", "B", "C", "D", "E"];
    let n = 500_000;
    
    let mut large_data = Vec::with_capacity(n);
    for i in 0..n {
        large_data.push(categories[i % categories.len()].to_string());
    }
    
    // 非カテゴリカル列の作成
    println!("生成したデータサイズ: {}要素 ({}ユニーク値)", n, categories.len());
    let mut large_df = OptimizedDataFrame::new();
    let start = Instant::now();
    large_df.add_column("regular", Column::String(StringColumn::new(large_data.clone())))?;
    let large_regular_time = start.elapsed();
    println!("大規模データの通常列追加時間: {:?}", large_regular_time);
    
    // カテゴリカル列の作成
    let mut cat_df = OptimizedDataFrame::new();
    let start = Instant::now();
    cat_df.add_categorical_column("categorical", large_data)?;
    let large_cat_time = start.elapsed();
    println!("大規模データのカテゴリカル列追加時間: {:?}", large_cat_time);
    println!("高速化率: {:.2}倍", large_regular_time.as_secs_f64() / large_cat_time.as_secs_f64());
    
    // メモリ使用量の推定
    let memory_regular = n * 8; // 各文字列にポインタとキャパシティ用 8バイト程度
    let memory_categorical = categories.len() * 16 + n * 4; // カテゴリのみ格納 + インデックス
    println!("推定メモリ使用量比較:");
    println!("  通常列: 約{}MB", memory_regular / (1024*1024));
    println!("  カテゴリカル列: 約{}MB", memory_categorical / (1024*1024));
    println!("  メモリ削減率: {:.1}%", 100.0 * (1.0 - memory_categorical as f64 / memory_regular as f64));
    
    println!("\n=== テスト完了 ===");
    
    Ok(())
}